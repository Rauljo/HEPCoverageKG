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
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from hepcoveragekg.aliases.context import clean_latex

logger = logging.getLogger(__name__)

# A yes/no plus one quoted sentence. Generous enough for a long quote, tight
# enough that a model cannot wander into an essay.
#
# A REASONING model needs far more: QwQ and friends emit their chain of thought
# as ordinary output tokens before the answer, so 500 truncates the thinking and
# the reply never reaches its JSON. Raise it via READER_MAX_TOKENS when serving
# one -- the symptom otherwise is a 100% unparseable rate that looks like the
# model cannot follow the format.
MAX_COMPLETION_TOKENS = int(os.environ.get("READER_MAX_TOKENS", "500"))

# A reasoning model wraps its thinking in <think>...</think>. vLLM strips this
# only when served with --reasoning-parser; without it the tags arrive inline and
# the JSON hides behind them, so the extractor below tolerates both.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)

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

# ORDER MATTERS INSIDE THE JSON. These prompts originally put "answer" first
# and "why" last, so the model emitted its verdict before generating a single
# token of justification -- chain-of-thought backwards, and free to fix. The
# reasoning field now comes first and the verdict last, which is the whole
# mechanism by which CoT helps.
#
# Worth keeping honest about: a stated reason that FOLLOWS the token order is
# not thereby a faithful account of the computation (Turpin et al. 2023). This
# buys accuracy, not interpretability -- the `reasoning` field is evidence about
# the answer, not proof of how it was reached.

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
    # Which reading window this came from. Verdicts are pooled per window, never
    # across windows, so this must identify the window uniquely -- `section`
    # alone does not, because a long section is split into several windows.
    window: int = 0
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

    def _by_window(self) -> list[list[Verdict]]:
        """Usable verdicts grouped by the window they came from.

        THE distinction this class got wrong. Repeats within one window are
        samples of the SAME question and should agree -- disagreement there is
        genuine uncertainty. Verdicts from DIFFERENT windows answer different
        questions: window 5 holds the ABCD sentence, window 12 is the detector
        description, and "not in this passage" is the correct answer for 22 of
        23 windows. Pooling them makes unanimity impossible EXACTLY WHEN the
        answer is found, so the successful reads were the ones marked unusable:

            2004.01678  3 supported verdicts, verified quote  -> reported None
            2011.07812  3 supported verdicts, verified quote  -> reported None

        while papers where nothing was found agreed trivially and reported a
        confident False. The logic was inverted -- finding the answer is what
        made the result unusable.
        """
        groups: dict[tuple, list[Verdict]] = {}
        for v in self.usable:
            groups.setdefault((v.window, v.section), []).append(v)
        return list(groups.values())

    @property
    def unanimous(self) -> bool:
        """Every window that has an opinion is internally consistent."""
        return all(len(set(v.supported for v in g)) == 1 for g in self._by_window())

    @property
    def answer(self) -> bool | None:
        """Did any window find it?

        True   some window's samples agree that it is there, with a real quote.
        False  every window agreed it is not, and at least one call was usable.
        None   a window's samples disagreed, or nothing usable came back.

        None means "a human must look", and covers two situations that must not
        be flattened into False: the samples genuinely disagreed, or we never
        successfully asked. A dead endpoint reporting "this paper does not do X"
        for all 60 papers is evidence of absence manufactured from an outage.
        """
        groups = self._by_window()
        if not groups:
            return None
        found, split = False, False
        for g in groups:
            verdicts = set(v.supported for v in g)
            if verdicts == {True}:
                found = True
            elif len(verdicts) > 1:
                split = True
        # A confident find outranks an unrelated window's wobble: the evidence
        # exists and is quoted. Only report uncertainty when nothing was found.
        if found:
            return True
        return None if split else False

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

Reply as JSON, nothing else. Fill the fields IN ORDER -- the reasoning first,
the verdict last:
{{"reasoning": "<what the text does and does not establish, one or two sentences>", \
"quote": "<exact sentence, or empty>", "answer": "yes"|"no"}}

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


async def _ask(client, model: str, prompt: str, temperature: float,
               max_tokens: int | None = None) -> str:
    """One call. `max_tokens` per call, because servers differ.

    vLLM counts prompt + completion against the context limit, so a budget sized
    for a reasoning model's 16k window makes an 8k server reject the request
    outright: "requested 8471 tokens (4471 in the messages, 4000 in the
    completion)". The budget belongs to the endpoint, not to the module.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens or MAX_COMPLETION_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


async def read_passage(conn, client, model, qid: str, question: str, passage: Passage,
                       temperature: float, escalated: bool = False,
                       mode: str = EXISTENCE, window: int = 0) -> Verdict:
    """One question against one window. Verifies the quote before returning."""
    try:
        raw = await _ask(client, model, _read_prompt(question, passage, mode), temperature)
    except Exception as exc:                       # noqa: BLE001 - recorded, not raised
        logger.warning("reader call failed on %s/%s: %s", passage.paper_id, qid, exc)
        return Verdict(passage.paper_id, qid, None, section=passage.section,
                       window=window, escalated=escalated,
                       raw=f"ERROR: {type(exc).__name__}: {exc}")

    answer, quote, extra = parse_reply(raw, mode)
    verified = bool(answer) and verify_quote(conn, passage.paper_id, quote)
    return Verdict(
        paper_id=passage.paper_id, qid=qid, answer=answer, quote=quote,
        quote_verified=verified,
        reasoning="" if mode == EXTRACTION else extra,
        answer_text=extra if mode == EXTRACTION else "",
        section=passage.section, window=window, escalated=escalated, raw=raw,
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
        for index, window in enumerate(windows):
            batch = [
                await read_passage(conn, client, model, qid, question, window,
                                   temperature, escalated=escalated, mode=mode,
                                   window=index)
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

Reply as JSON, nothing else. Fill the fields IN ORDER -- the reasoning first,
the verdict last:
{{"reasoning": "<what the text does and does not say about this, one or two sentences>", \
"quote": "<exact sentence, or empty>", "answer": "<the answer, or empty>", "found": true|false}}

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
    raw = _THINK_RE.sub(" ", raw)
    # A reasoning model's chain of thought routinely contains braces and even
    # draft JSON. Take the LAST balanced object, which is the answer it settled
    # on, rather than the first thing that looks like one.
    match = None
    for match in re.finditer(r"\{[^{}]*\}", raw, re.S):
        pass
    if match is None:
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


def rescore(path: Path | str, out_path: Path | str | None = None) -> dict:
    """Recompute per-paper answers from a run's stored verdicts.

    Every consensus rule here is derived from `verdicts`, which are the raw
    model replies -- so a scoring change costs a file read, not 22,000 LLM
    calls. That distinction was learned the expensive way on the query harness,
    where a metric change meant re-running everything.

    Needed immediately: the first sweep pooled verdicts across windows, so any
    paper where the answer WAS found came out as "split" and was counted as not
    found. The reads were good; only the arithmetic over them was wrong.

    Records written before `window` existed fall back to grouping by section,
    which is the same thing except where one section spanned several windows.
    """
    path = Path(path)
    out_path = Path(out_path) if out_path else path.with_suffix(".rescored.jsonl")
    stats: Counter = Counter()
    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        verdicts = [
            Verdict(
                paper_id=v.get("paper_id", record["paper_id"]),
                qid=v.get("qid", record["qid"]),
                answer=v.get("answer"),
                quote=v.get("quote", ""),
                quote_verified=v.get("quote_verified", False),
                answer_text=v.get("answer_text", ""),
                section=v.get("section", ""),
                window=v.get("window", 0),
                escalated=v.get("escalated", False),
                raw=v.get("raw", ""),
            )
            for v in record.get("verdicts", [])
        ]
        c = Consensus(record["paper_id"], record["qid"], verdicts)
        was, now = record.get("answer"), c.answer
        stats["reads"] += 1
        stats["changed"] += was != now
        stats[{True: "yes", False: "no", None: "split"}[now]] += 1
        rows.append({**record, "answer": now, "unanimous": c.unanimous,
                     "quote": c.best_quote,
                     "answers": sorted({v.answer_text for v in verdicts
                                        if v.supported and v.answer_text}),
                     "previous_answer": was})

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"out": str(out_path), **{k: int(v) for k, v in stats.items()}}


# --------------------------------------------------------------------------
# Does the quote actually answer the question?
# --------------------------------------------------------------------------
#
# The gap this closes, measured rather than assumed. `verify_quote` proves a
# sentence is IN the paper; it cannot tell whether the sentence supports the
# claim. On gf-08 ("exactly 2 electrons OR exactly 2 muons"), three of six
# sampled yes-answers cited real sentences that answer a different question:
#
#   "exactly two muons ... no identified electrons"      -> muons only, not an OR
#   "Events are required to have at least one primary vertex"  -> irrelevant
#
# Precision on that question was about 50% while every quote verified. So the
# reader catches invention and misses misreading, exactly as its docstring
# says -- and this is the second pass that turns that caveat into a filter.
#
# Cheap by construction: only the supported yes-answers are re-checked, a few
# dozen per question rather than 22,000 calls. The judge sees ONLY the question
# and the quote -- never the paper, never the model's own reasoning -- so it
# cannot be talked into agreement by the surrounding argument.

SUPPORT_PROMPT = """\
Someone is checking whether ONE PARTICULAR PAPER does the thing described below. \
They cited this sentence from that paper as their evidence.

Judge ONE thing: does the sentence show that this paper does it?

THE THING BEING LOOKED FOR
{question}

CITED SENTENCE FROM THE PAPER
{quote}

{claim}The question is phrased across many papers ("which analyses ..."), but you \
are judging ONE sentence from ONE paper. So do not ask whether the sentence names \
which papers - it cannot. Ask only whether it shows that THIS paper does the thing.

Say "yes" if the sentence shows this paper does it, even in passing.
Say "no" if the sentence is merely on a related topic, describes something similar \
but not the same, or does not establish the thing at all. A sentence about muons \
alone does not establish a CHOICE between electrons and muons. A sentence about an \
unrelated selection cut establishes nothing.

Reply as JSON, nothing else. Fill the fields IN ORDER -- the reasoning first,
the verdict last:
{{"why": "<what the sentence establishes, and whether that is the thing asked for>", \
"supports": true|false}}
"""


async def check_support(client, model, question: str, quote: str,
                        claimed: str = "", temperature: float = 0.0,
                        max_tokens: int | None = None) -> tuple[bool | None, str]:
    """(supports, why). None when the call or the reply failed.

    Temperature 0: this is an adjudication, not a sample. Disagreement between
    runs here would be noise, not signal -- the uncertainty worth measuring was
    already measured upstream by the repeats.
    """
    claim = f'The reader answered: "{claimed}"\n\n' if claimed else ""
    prompt = SUPPORT_PROMPT.format(question=question, quote=quote, claim=claim)
    try:
        raw = await _ask(client, model, prompt, temperature, max_tokens=max_tokens)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("support check failed: %s", exc)
        return None, f"ERROR: {type(exc).__name__}"
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None, ""
    value = data.get("supports")
    if isinstance(value, str):
        value = value.strip().lower() in ("true", "yes")
    if not isinstance(value, bool):
        return None, ""
    return value, str(data.get("why") or "")


async def verify_supports(rescored_path: Path | str, questions: list[dict],
                          out_path: Path | str | None = None,
                          *, concurrency: int | None = None,
                          client=None, model: str | None = None,
                          max_tokens: int | None = None) -> dict:
    """Re-check every YES in a rescored run, and write the surviving answers.

    A yes that fails here is downgraded to `false_support` rather than deleted:
    the read happened, the quote is real, and the record of the model reading it
    wrongly is itself the measurement.
    """
    rescored_path = Path(rescored_path)
    out_path = Path(out_path) if out_path else rescored_path.with_suffix(".supported.jsonl")
    text = {q["qid"]: q["text"] for q in questions}
    # A different endpoint from the reader when one is supplied: a model marking
    # its own homework is the self-preference problem (MT-Bench), and avoiding it
    # here costs nothing because both servers are already up.
    if client is None:
        client, model = _client()
    gate = asyncio.Semaphore(concurrency or DEFAULT_CONCURRENCY)

    rows = [json.loads(l) for l in rescored_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    targets = [r for r in rows if r.get("answer") is True and r.get("quote")]

    async def one(row: dict) -> None:
        async with gate:
            answers = row.get("answers") or []
            ok, why = await check_support(
                client, model, text.get(row["qid"], row["qid"]), row["quote"],
                answers[0] if answers else "", max_tokens=max_tokens)
        row["quote_supports"] = ok
        row["support_why"] = why
        row["judged_by"] = model
        if ok is False:
            row["answer"] = False
            row["downgraded"] = "quote does not answer the question"

    await asyncio.gather(*(one(r) for r in targets))

    stats = Counter()
    for r in rows:
        stats["reads"] += 1
        if r.get("quote_supports") is True:
            stats["upheld"] += 1
        elif r.get("quote_supports") is False:
            stats["downgraded"] += 1
        elif "quote_supports" in r:
            stats["unchecked"] += 1
    with out_path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    checked = stats["upheld"] + stats["downgraded"]
    return {"out": str(out_path), "checked": checked,
            **{k: int(v) for k, v in stats.items()},
            "precision": round(stats["upheld"] / checked, 3) if checked else None}


# --------------------------------------------------------------------------
# The cascade: cheap model for coverage, strong model for its misses, judge for
# everything either of them claims.
# --------------------------------------------------------------------------
#
# Each stage is aimed at a MEASURED failure of the stage before it.
#
#   1. a fast literal model reads everything. It under-finds: it read
#      "A jet pair is tagged as a Higgs boson candidate if the NN score..."
#      and answered no.
#   2. a reasoning model re-reads only what stage 1 rejected -- that is exactly
#      where its misses are, and it is 3-10x slower, so pointing it at the whole
#      corpus would cost 8-17 hours to re-derive answers already in hand.
#   3. a judge checks every YES from either model.
#
# Stage 3 is not optional, and the reason is arithmetic rather than distrust.
# Aggregation is OR-over-windows: one window with a verified quote flips the
# paper. 2504.13081 has 14 windows; QwQ answered "no" on 13 and yes on one,
# citing "the Higgs boson transverse mass must be greater than 60 GeV" -- a cut
# on a Higgs quantity, not a reconstructed Higgs object. At a per-window false
# positive rate p over N windows the paper flips with probability 1-(1-p)^N,
# which for p=0.07 and N=14 is 64%. A better reader RAISES this, because it
# finds more per window. The judge is what makes OR-over-windows safe.
#
# The judge should not be the model that produced the answer. Self-preference in
# LLM judging is well documented (MT-Bench), and it is free to avoid here.

def _pending(rows: list[dict]) -> list[dict]:
    """Reads a second opinion could still change: negatives and splits.

    A confident YES is not re-read -- stage 3 handles those. A NO is where a
    literal reader's misses live.
    """
    return [r for r in rows if r.get("answer") is not True]


async def recheck(conn, rescored_path: Path | str, questions: list[dict],
                  out_path: Path | str | None = None, *,
                  repeats: int = 1, temperature: float = 0.0,
                  concurrency: int | None = None,
                  max_tokens: int | None = None) -> dict:
    """Stage 2: re-read only the papers stage 1 did not find, with this model.

    `repeats=1` by default: this is a recall pass, and the uncertainty that
    repeats measure is handled by the judge downstream. Spending three samples
    per window on a reasoning model to re-derive a negative is not worth it.
    """
    rescored_path = Path(rescored_path)
    out_path = Path(out_path) if out_path else rescored_path.with_suffix(".rechecked.jsonl")
    text = {q["qid"]: (q.get("per_paper") or q["text"]) for q in questions}
    client, model = _client()
    gate = asyncio.Semaphore(concurrency or DEFAULT_CONCURRENCY)

    rows = [json.loads(l) for l in rescored_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = _pending(rows)
    logger.info("recheck: %d of %d reads pending, model=%s", len(todo), len(rows), model)

    async def one(row: dict) -> None:
        async with gate:
            c = await read_paper(conn, client, model, row["qid"],
                                 text.get(row["qid"], row["qid"]), row["paper_id"],
                                 repeats=repeats, temperature=temperature,
                                 cascade=False, mode=EXISTENCE)
        row["recheck_answer"] = c.answer
        row["recheck_model"] = model
        if c.answer is True:
            # Recovered by the stronger reader. Marked, never silently merged --
            # which model found a fact is part of the finding.
            row["answer"] = True
            row["quote"] = c.best_quote
            row["recovered_by"] = model

    await asyncio.gather(*(one(r) for r in todo))

    stats = Counter(recovered=sum(1 for r in todo if r.get("recovered_by")),
                    rechecked=len(todo), untouched=len(rows) - len(todo))
    with out_path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"out": str(out_path), "model": model, **{k: int(v) for k, v in stats.items()}}
