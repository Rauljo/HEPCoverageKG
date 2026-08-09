"""
Evaluation: the reference reader — an independent second opinion from the papers.

The problem it solves. The supervisor's gold is, in his own words, "graph-
agreement gold, not physics truth": every answer was computed by filtering his
pilot export. Scoring our graph against it measures whether two pipelines built
from the same extraction agree with each other. It cannot tell us whether either
agrees with the papers, and where the extraction dropped something both will
drop it identically and score 100%.

So this reads the PAPERS. Nothing here may touch `entity`, `assertion`,
`entity_facet` or any derived table — only `source_block`, the harvested text.
The moment the reader sees the graph, it stops being independent and the whole
exercise becomes a second measurement of the extraction (S-33 / the L2 layer).

Three things make it more than "ask a model twice".

**Every claim carries a quote, and the quote is checked mechanically.** A model
saying "yes, this paper uses an ABCD estimate" is an opinion; a model saying so
and citing a sentence that provably appears in the paper is evidence. The check
is a normalised substring match against `source_block.text` — a quote either is
in the paper or it is not. Fabricated citations are dropped, not believed. This
does not catch misreading, only invention; the honest name for the output is
therefore *corroborated gold*, not truth.

**Self-consistency, with disagreement surfaced rather than resolved.** Repeats at
non-zero temperature; unanimous verdicts become gold, split verdicts are flagged
for a human. Silently taking a majority would hide exactly the cases worth
looking at (Farquhar et al. 2024 on semantic entropy; the same pattern already
used in aliases/confidence.py).

**Sections first, whole paper on a miss.** Routing to the sections a question is
likely answered in costs ~3x less than reading everything. It is sound HERE
because every sweep question is an EXISTENCE question: a yes backed by a verified
quote is final regardless of what the rest of the paper says, so only a NO needs
escalating. For a "how many" question the cascade would be unsound — a partial
answer from one section would look complete.

Scope is decided in eval/supervisor.py by regex over the question text, never by
reading the gold answer. Scoping a check to the papers the answer names can only
confirm the answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from hepcoveragekg.aliases.context import clean_latex

logger = logging.getLogger(__name__)

# A yes/no plus one quoted sentence. Generous enough for a long quote, tight
# enough that a model cannot wander into an essay.
MAX_COMPLETION_TOKENS = 500

# In-flight requests. Matches the aliases layer's default so a shared endpoint
# is safe out of the box; DIAS raises it via LLM_CONCURRENCY.
DEFAULT_CONCURRENCY = 8

# A run whose endpoint has died should stop, not spend an hour writing errors.
ABORT_AFTER_CONSECUTIVE_ERRORS = 20

# Blocks that are never physics. The author list alone is 1.3M characters across
# the corpus -- 9% of everything -- and no question is ever answered from it.
_DROP_KINDS = ("bibliography",)
_DROP_SECTIONS = ("%Collaboration%", "Acknowledg%", "References%")

# Reading windows, sized from a MEASURED token ratio rather than a guess.
#
# The first real run lost roughly half its calls to
#     400 - This model's maximum context length is 8192 tokens
# because CHUNK_CHARS was set from an assumed 4 chars/token. Measured on this
# corpus with the served model's own tokenizer: **3.46 chars/token** -- physics
# prose is dense with LaTeX (`$\mathup{{{t}}}$`) and markup tokenises badly.
# 24,000 chars is therefore ~6,900 tokens, and prompt + window + completion sat
# right on the ceiling.
#
# Worse than the loss itself: the failures were SYSTEMATIC, not random. The
# biggest windows are the ones that overflow, and the biggest windows are the
# content-rich sections most likely to hold an answer. So the run did not just
# lose half its data, it lost the half that mattered, and reported the result as
# a confident "not found".
#
# Budget, against the 8192 the server actually serves:
#     8192 - 500 completion - ~450 prompt = ~7,240 usable
#     at 3.46 chars/token, and leaving real headroom for a denser-than-average
#     section: 12,000 chars ~= 3,470 tokens. Half the ceiling, deliberately.
CHARS_PER_TOKEN = 3.46          # measured, Mistral-Small-24B on this corpus
CHUNK_CHARS = 12_000
CHUNK_OVERLAP = 1_500

# Which prompt a question gets. Existence questions ("which analyses do X?")
# sweep the corpus and only need yes/no; the questions that name a paper ask
# WHAT or WHY, and a yes/no answer to "what is the observed 95% CL limit on the
# stop mass" throws away the number that makes it an answer.
#
# It also matters for the three disputed Tier 4 premises (Q12-Q14): yes/no would
# establish only that *something* relevant is in the text. The extracted value is
# what can be set beside his gold and beside the graph.
EXISTENCE, EXTRACTION = "existence", "extraction"


# A quote shorter than this matches by accident ("the", "40 GeV"), so it cannot
# support a claim on its own.
MIN_QUOTE_CHARS = 25


@dataclass
class Passage:
    paper_id: str
    section: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Verdict:
    """One reader answer about one paper."""
    paper_id: str
    qid: str
    answer: bool | None            # None = the model declined / unparseable
    quote: str = ""
    quote_verified: bool = False
    reasoning: str = ""
    # EXTRACTION mode only: the answer read out of the text. Empty in EXISTENCE
    # mode, where the question is "does this paper do X" and the quote IS the
    # whole answer.
    answer_text: str = ""
    section: str = ""
    escalated: bool = False        # reached only after the section pass found nothing
    raw: str = ""

    @property
    def supported(self) -> bool:
        """A yes counts only when its citation is real."""
        return bool(self.answer) and self.quote_verified

    @property
    def errored(self) -> bool:
        """The call itself failed -- no opinion was ever obtained.

        Distinct from an unparseable reply, and BOTH are distinct from a "no".
        The first run reported `failed: 0` while 55% of its calls were returning
        400s, because `failed` only counted reads where EVERY sample died. A
        systematic context-overflow looked like a clean set of negatives.
        """
        return self.answer is None and self.raw.startswith("ERROR:")


@dataclass
class Consensus:
    """Repeats collapsed into a per-paper result."""
    paper_id: str
    qid: str
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def votes(self) -> Counter:
        return Counter(v.supported for v in self.verdicts)

    @property
    def usable(self) -> list[Verdict]:
        """Verdicts the model actually produced. A call that errored or replied
        unparseably has no opinion, and must not be counted as one."""
        return [v for v in self.verdicts if v.answer is not None]

    @property
    def unanimous(self) -> bool:
        return len(set(v.supported for v in self.usable)) == 1

    @property
    def answer(self) -> bool | None:
        """Unanimous -> the answer. Split, or nothing usable -> None.

        None means "a human must look", and it covers two different situations
        that must not be flattened into False:

          the samples disagreed        -> genuinely uncertain
          every call failed            -> we never asked successfully

        An earlier version derived this from `supported`, which is False for an
        errored verdict as much as for a real negative. A dead endpoint then
        reported "this paper does not do X" for all 60 papers — evidence of
        absence manufactured from an outage.
        """
        usable = self.usable
        return usable[0].supported if usable and self.unanimous else None

    @property
    def best_quote(self) -> str:
        for v in self.verdicts:
            if v.supported:
                return v.quote
        return ""


# --------------------------------------------------------------------------
# Reading the source text -- and NOTHING else
# --------------------------------------------------------------------------

def passages(conn, paper_id: str, section_patterns: Optional[Iterable[str]] = None
             ) -> list[Passage]:
    """Harvested text for one paper, optionally restricted to matching sections.

    Reads `source_block` only. If this function ever grows a join to `entity` or
    `assertion`, the reader has stopped being an independent check.
    """
    where = ["ss.paper_id = ?", f"sb.kind NOT IN ({','.join('?' * len(_DROP_KINDS))})"]
    params: list = [paper_id, *_DROP_KINDS]
    for pattern in _DROP_SECTIONS:
        where.append("COALESCE(sb.section_title,'') NOT LIKE ?")
        params.append(pattern)

    if section_patterns:
        ors = " OR ".join("LOWER(COALESCE(sb.section_title,'')) LIKE ?"
                          for _ in section_patterns)
        where.append(f"({ors})")
        params.extend(f"%{p.lower()}%" for p in section_patterns)

    rows = conn.execute(
        "SELECT COALESCE(sb.section_title,'(untitled)') AS section, sb.text AS text"
        "  FROM source_block sb JOIN source_snapshot ss USING(source_hash)"
        f" WHERE {' AND '.join(where)}"
        " ORDER BY sb.block_order",
        params,
    ).fetchall()
    return [Passage(paper_id, r["section"], r["text"]) for r in rows if r["text"].strip()]


def chunks(items: list[Passage], limit: int = CHUNK_CHARS) -> list[Passage]:
    """Pack passages into reading windows.

    The rule is: never SPLIT a section across two windows, but freely COMBINE
    whole sections into one. What costs a model context is finding half a
    section's argument in one window and half in the next; several complete
    small sections sharing a window costs it nothing.

    An earlier version flushed on every section change, which is the same rule
    stated carelessly: 2001.06899 came out as 16 windows averaging 3.5k tokens
    against a 32k context -- five times the calls for no benefit.

    A section longer than the window on its own is split with overlap, because
    there is no alternative.
    """
    out: list[Passage] = []
    buf: list[str] = []
    sections: list[str] = []
    size = 0
    paper = items[0].paper_id if items else ""

    def flush() -> None:
        nonlocal buf, sections, size
        if buf:
            out.append(Passage(paper, " + ".join(sections), "\n\n".join(buf)))
        buf, sections, size = [], [], 0

    # Group consecutive blocks by section, so a section is placed as one unit.
    for section, blocks in _by_section(items):
        text = "\n\n".join(b.text for b in blocks)
        if len(text) > limit:            # too big to place whole: split it alone
            flush()
            step = limit - CHUNK_OVERLAP
            for i in range(0, len(text), step):
                out.append(Passage(paper, section, text[i:i + limit]))
            continue
        if size + len(text) > limit:
            flush()
        buf.append(text)
        if section not in sections:
            sections.append(section)
        size += len(text)
    flush()
    return out


def _by_section(items: list[Passage]) -> list[tuple[str, list[Passage]]]:
    """Consecutive passages grouped by section title, order preserved."""
    grouped: list[tuple[str, list[Passage]]] = []
    for p in items:
        if grouped and grouped[-1][0] == p.section:
            grouped[-1][1].append(p)
        else:
            grouped.append((p.section, [p]))
    return grouped


# --------------------------------------------------------------------------
# The check that makes a claim evidence
# --------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """Words only: lowercase alphanumeric runs, markup and punctuation discarded.

    Substring matching on the raw text does not work, and the failure runs in the
    dangerous direction -- it rejects TRUE quotes, which would silently suppress
    correct answers and make the gold falsely conservative. A model asked for a
    verbatim quote reliably reproduces the words and unreliably reproduces the
    markup around them:

        source  "...efficiency of ${\\approx}80\\%$ with the..."
        model   "...efficiency of approximately 80%"

    Same sentence, and no amount of whitespace folding makes those equal. Words
    are the level at which the two genuinely agree.
    """
    return re.findall(r"[a-z0-9]+", clean_latex(text or "").lower())


# A run this long is what counts as a citation. Long enough that ordinary
# phrasing ("the analysis uses a") cannot reach it by accident, short enough to
# survive a model dropping or rewording the LaTeX at either end of a sentence.
MIN_MATCH_TOKENS = 8


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def verify_quote(conn, paper_id: str, quote: str) -> bool:
    """Is this sentence actually in this paper?

    Verified when a run of at least MIN_MATCH_TOKENS consecutive words from the
    quote appears consecutively somewhere in the paper's harvested text.

    A hard check, not a judgement: an invented sentence cannot produce an
    8-word run that happens to sit in the paper. What it does NOT catch is a
    real quote that has been MISREAD -- correctly copied, wrongly interpreted.
    That is why the output of this module is called corroborated gold and not
    truth.
    """
    if not quote or len(quote.strip()) < MIN_QUOTE_CHARS:
        return False
    needle = _tokens(quote)
    if len(needle) < MIN_MATCH_TOKENS:
        return False

    wanted = _ngrams(needle, MIN_MATCH_TOKENS)
    for row in conn.execute(
        "SELECT sb.text AS text FROM source_block sb"
        "  JOIN source_snapshot ss USING(source_hash) WHERE ss.paper_id = ?",
        (paper_id,),
    ):
        haystack = _tokens(row["text"])
        if len(haystack) < MIN_MATCH_TOKENS:
            continue
        if wanted & _ngrams(haystack, MIN_MATCH_TOKENS):
            return True
    return False


# --------------------------------------------------------------------------
# Where to look first
# --------------------------------------------------------------------------

# Question keyword -> section-title fragments to try before the whole paper.
# Coverage measured over the 60 papers: detector/data 53, selection 49,
# systematics 47, background 44, results 38, statistics 31. So roughly a quarter
# of papers will fall through to the full read on any given question.
SECTION_ROUTES: list[tuple[str, tuple[str, ...]]] = [
    (r"abcd|background estimat|control region|normalis|normaliz|fake|data.driven",
     ("background", "estimation", "control region")),
    (r"unfold|histfitter|statistic|likelihood|fit\b|limit|cls\b",
     ("statistic", "fit", "interpretation", "limit", "result")),
    (r"systematic|uncertaint",
     ("systematic", "uncertaint")),
    (r"b.jet|b.tag|missing transverse|met\b|lepton|electron|muon|photon|jet|"
     r"candidate|object|select|reconstruct",
     ("object", "selection", "reconstruction", "event", "detector")),
    (r"yield|observed|expected|excluded|limit on|mass limit|compare",
     ("result", "summary", "conclusion", "interpretation")),
]


def route(question_text: str) -> tuple[str, ...]:
    """Section fragments worth trying first. Empty means go straight to the whole paper."""
    text = question_text.lower()
    hits: list[str] = []
    for pattern, sections in SECTION_ROUTES:
        if re.search(pattern, text):
            hits.extend(s for s in sections if s not in hits)
    return tuple(hits)


READER_PROMPT = """\
You are reading one part of a published high-energy-physics paper to answer a \
factual question about THIS analysis.

Answer ONLY from the text below. You know a great deal of particle physics and \
none of it is admissible here: the question is about what this paper did, not \
about what is usually done.

QUESTION
{question}

Answer "yes" only if the text shows that THIS analysis does the thing asked \
about. A paper mentioning a technique, citing it, or comparing against someone \
else who used it is NOT the same as this analysis using it.

If yes, you must quote the sentence that shows it, copied EXACTLY from the text \
below - not paraphrased, not reconstructed from memory. A quote that does not \
appear verbatim in the text will be discarded and your answer will not count.

If the text below does not show it, answer "no". "no" here means "not shown in \
this passage", which is the useful answer - do not guess to be helpful.

Reply as JSON, nothing else:
{{"answer": "yes"|"no", "quote": "<exact sentence, or empty>", "why": "<one short sentence>"}}

PAPER TEXT ({paper_id}, section: {section})
---
{text}
---
"""


def _read_prompt(question: str, passage: Passage, mode: str = EXISTENCE) -> str:
    template = EXTRACT_PROMPT if mode == EXTRACTION else READER_PROMPT
    return template.format(
        question=question, paper_id=passage.paper_id,
        section=passage.section or "(untitled)", text=passage.text,
    )


async def _ask(client, model: str, prompt: str, temperature: float) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


async def read_passage(conn, client, model, qid: str, question: str, passage: Passage,
                       temperature: float, escalated: bool = False,
                       mode: str = EXISTENCE) -> Verdict:
    """One question against one window. Verifies the quote before returning."""
    try:
        raw = await _ask(client, model, _read_prompt(question, passage, mode), temperature)
    except Exception as exc:                       # noqa: BLE001 - recorded, not raised
        logger.warning("reader call failed on %s/%s: %s", passage.paper_id, qid, exc)
        return Verdict(passage.paper_id, qid, None, section=passage.section,
                       escalated=escalated, raw=f"ERROR: {type(exc).__name__}: {exc}")

    answer, quote, extra = parse_reply(raw, mode)
    verified = bool(answer) and verify_quote(conn, passage.paper_id, quote)
    return Verdict(
        paper_id=passage.paper_id, qid=qid, answer=answer, quote=quote,
        quote_verified=verified,
        reasoning="" if mode == EXTRACTION else extra,
        answer_text=extra if mode == EXTRACTION else "",
        section=passage.section, escalated=escalated, raw=raw,
    )


async def read_paper(conn, client, model, qid: str, question: str, paper_id: str,
                     *, repeats: int = 3, temperature: float = 0.3,
                     cascade: bool = True, mode: str = EXISTENCE) -> Consensus:
    """One question against one paper, section-first then whole paper.

    The cascade is sound because these are EXISTENCE questions: a yes with a
    verified quote is final whatever the rest of the paper says, so only a NO
    escalates. Applied to a "how many" question it would be wrong -- a partial
    answer from one section would look complete.

    Repeats run at non-zero temperature so that agreement means something. At
    temperature 0 a model repeats itself, and three identical samples would look
    like consensus while measuring nothing.
    """
    verdicts: list[Verdict] = []

    async def sweep_windows(patterns: tuple[str, ...], escalated: bool) -> bool:
        """Read every window matching `patterns`. True if anything was supported."""
        windows = chunks(passages(conn, paper_id, patterns or None))
        for window in windows:
            batch = [
                await read_passage(conn, client, model, qid, question, window,
                                   temperature, escalated=escalated, mode=mode)
                for _ in range(repeats)
            ]
            verdicts.extend(batch)
            if any(v.supported for v in batch):
                return True                         # found it; no need to read on
        return False

    # The routed pass, when there is somewhere to route to. A question the router
    # has no pattern for goes straight to the full read rather than reading
    # nothing -- an earlier version fell through both branches and returned an
    # EMPTY consensus whenever cascade was off, which is how the whole
    # single-paper run would have produced no data at all while "succeeding".
    sections = route(question) if cascade else ()
    found = await sweep_windows(sections, escalated=False) if sections else False

    # Escalate on a NO. Sound because a verified yes is final whatever the rest
    # of the paper says; for a "how many" question it would not be.
    if not found:
        await sweep_windows((), escalated=bool(sections))

    return Consensus(paper_id, qid, verdicts)


EXTRACT_PROMPT = """\
You are reading one part of a published high-energy-physics paper to answer a \
question about THIS analysis.

Answer ONLY from the text below. You know a great deal of particle physics and \
none of it is admissible here: the question is about what this paper says, not \
about what is usually true.

QUESTION
{question}

If the text below answers the question, give the answer and quote the sentence \
it comes from, copied EXACTLY from the text - not paraphrased, not reconstructed \
from memory. A quote that does not appear verbatim will be discarded and your \
answer will not count.

If the text below does not answer it, set "found" to false. That is the useful \
answer here, not a failure - do not guess to be helpful, and do not answer from \
your own knowledge of physics.

Reply as JSON, nothing else:
{{"found": true|false, "answer": "<the answer, or empty>", \
"quote": "<exact sentence, or empty>"}}

PAPER TEXT ({paper_id}, section: {section})
---
{text}
---
"""

def parse_reply(raw: str, mode: str = EXISTENCE) -> tuple[bool | None, str, str]:
    """(answer, quote, text) from a model reply. Unparseable -> (None, '', '').

    Returns None rather than defaulting to "no", because a parse failure and a
    genuine negative are different things and collapsing them would silently
    turn broken calls into evidence of absence.

    In EXTRACTION mode the third element is the extracted answer rather than a
    justification, and `found` carries the boolean.
    """
    if not raw:
        return None, "", ""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None, "", ""
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None, "", ""

    if mode == EXTRACTION:
        found = data.get("found")
        if isinstance(found, str):
            found = found.strip().lower() in ("true", "yes")
        if not isinstance(found, bool):
            return None, "", ""
        return found, str(data.get("quote") or ""), str(data.get("answer") or "")

    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        return None, "", ""
    return answer == "yes", str(data.get("quote") or ""), str(data.get("why") or "")


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------

def _client():
    """The aliases layer's client, reused. Same env vars, same retry policy."""
    from hepcoveragekg.aliases.adjudicate import _get_llm_client

    return _get_llm_client()


async def run(conn, questions: list[dict], papers_for, out_path: Path | str,
              *, repeats: int = 3, temperature: float = 0.3,
              concurrency: int | None = None, cascade: bool = True,
              limit_papers: int | None = None, mode: str = EXISTENCE) -> dict:
    """Read every (question, paper) pair and stream the result.

    `papers_for(question_record) -> list[paper_id]` decides scope. It is passed
    in rather than computed here so the caller owns the sweep/single-paper
    decision, which is made mechanically from the question text in
    eval/supervisor.py and must not be re-derived by guessing.

    Output is streamed one JSON record per (question, paper) as it completes.
    A run that dies at hour three keeps everything up to hour three -- the
    alternative was measured the hard way in D-049.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client, model = _client()
    gate = asyncio.Semaphore(concurrency or DEFAULT_CONCURRENCY)

    jobs: list[tuple[dict, str]] = []
    for q in questions:
        papers = papers_for(q)
        if limit_papers:
            papers = papers[:limit_papers]
        jobs.extend((q, p) for p in papers)

    logger.info("reader: %d question(s) x papers = %d reads, model=%s, repeats=%d",
                len(questions), len(jobs), model, repeats)

    consecutive_errors = 0
    stats: Counter = Counter()

    async def one(q: dict, paper_id: str) -> Consensus:
        async with gate:
            return await read_paper(conn, client, model, q["qid"], q["text"], paper_id,
                                    repeats=repeats, temperature=temperature,
                                    cascade=cascade, mode=mode)

    with out_path.open("w", encoding="utf-8") as handle:
        for coro in asyncio.as_completed([one(q, p) for q, p in jobs]):
            c: Consensus = await coro
            failed = all(v.answer is None for v in c.verdicts) and bool(c.verdicts)
            consecutive_errors = consecutive_errors + 1 if failed else 0
            stats["reads"] += 1
            stats["yes"] += c.answer is True
            stats["no"] += c.answer is False
            stats["split"] += c.answer is None and not failed
            stats["failed"] += failed
            stats["escalated"] += any(v.escalated for v in c.verdicts)
            # Per-CALL health, not per-read. A read that limps home on one usable
            # sample out of three is not a healthy read, and the old counters
            # could not say so.
            stats["calls"] += len(c.verdicts)
            stats["call_errors"] += sum(1 for v in c.verdicts if v.errored)
            stats["call_unparsed"] += sum(
                1 for v in c.verdicts if v.answer is None and not v.errored)

            handle.write(json.dumps({
                "qid": c.qid,
                "paper_id": c.paper_id,
                "answer": c.answer,
                "unanimous": c.unanimous,
                "quote": c.best_quote,
                # EXTRACTION only. Kept as every distinct wording rather than one
                # winner: three samples that agree on the VALUE but differ in
                # phrasing are agreement, and collapsing them early would hide
                # the case where they disagree on the number itself.
                "answers": sorted({v.answer_text for v in c.verdicts
                                   if v.supported and v.answer_text}),
                "votes": {str(k): v for k, v in c.votes.items()},
                "escalated": any(v.escalated for v in c.verdicts),
                "sections_read": sorted({v.section for v in c.verdicts if v.section}),
                "verdicts": [asdict(v) for v in c.verdicts],
            }, ensure_ascii=False) + "\n")
            handle.flush()

            if consecutive_errors >= ABORT_AFTER_CONSECUTIVE_ERRORS:
                logger.error("%d consecutive failed reads -- the endpoint is probably "
                             "gone. Stopping with %d records kept.",
                             consecutive_errors, stats["reads"])
                break

    # Written after the file is closed. Doing this inside the open block is what
    # corrupted the last circuit-breaker output (D-049).
    calls = int(stats["calls"]) or 1
    meta = {"model": model, "repeats": repeats, "temperature": temperature,
            "cascade": cascade, "mode": mode,
            **{k: int(v) for k, v in stats.items()},
            "call_error_rate": round(stats["call_errors"] / calls, 3),
            "call_usable_rate": round(
                (calls - stats["call_errors"] - stats["call_unparsed"]) / calls, 3)}
    if meta["call_error_rate"] > 0.05:
        logger.error("%.0f%% of calls FAILED -- this result is not trustworthy. "
                     "Check the window size against the served context length.",
                     100 * meta["call_error_rate"])
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta
