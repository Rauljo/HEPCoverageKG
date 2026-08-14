"""
HEPCoverageKG query layer: the candidate critic.

The relevance step between retrieval and use (system.md S-69, D-060). `search`
returns candidates, this marks which of them bear on **the question actually
asked**, and the planner proceeds with both the full set and the marked one.

WHY THERE HAS TO BE ONE. Nothing currently sits between "search found things"
and "the answer counts them": `search` takes the top `SEARCH_BREADTH` hits and
saves *all* of them as a set, and `count` treats every member as part of the
concept. That is the measured Tier B failure -- 0.058, sixty near-neighbours
counted as one thing.

AND WHY IT CANNOT BE A SCORE CUTOFF. Relative scores behave completely
differently by concept, which was measured on this corpus:

    "Pythia"      rank 20 scores 50% of the best hit and is still a real Pythia
    "top squark"  rank 20 scores 71% and is already "single top" -- a different
                  particle entirely

Any threshold that keeps the Pythias throws away nothing at rank 20 for top
squark, and any threshold that cuts single-top cuts half the Pythias. Only
something that reads the question can separate those.

THREE RUNGS, NOT A BINARY (D-060). `exact` / `broader` / `unrelated`, because
the two failures the design has to catch point in opposite directions: our Tier
B errs by answering too COARSE, the supervisor's Q5 errs by matching too FINE. A
keep/drop flag records that the critic was wrong without recording which way, and
the direction is the diagnosis. Collapsing to binary for scoring stays available;
recovering the direction afterwards does not.

`broader` is usually "matched the SEARCH TEXT but not the QUESTION", which is
why both are shown. Asked *"how many analyses use a Pythia 8 shower?"* the
planner rightly searches the wider "Pythia" -- the graph holds one entity per
version and tune -- so `Pythia 6.428` is a legitimate hit on the search and a
wrong answer to the question.

FLAG, NEVER FILTER (D-016, D-018, D-060). Nothing here removes anything. A wrong
discard is silent: nothing in the trace shows what was taken away. The caller
keeps every candidate and receives the verdicts alongside.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# The three rungs, ordered from tightest to loosest.
EXACT = "exact"
BROADER = "broader"
UNRELATED = "unrelated"
RUNGS = (EXACT, BROADER, UNRELATED)

# What survives when the caller wants a set rather than a diagnosis. `broader`
# is kept: the whole S-68 point is that a wider rung is the RIGHT answer to some
# questions, and dropping it here would hard-code one rung as correct in the
# place least able to tell.
KEPT_RUNGS = (EXACT, BROADER)

# Candidates per call.
#
# Not one call for all sixty. "Lost in the middle" is a long-context effect
# (Liu et al. 2023): a model attends well to the start and end of a long list
# and poorly to the middle, so a genuinely relevant candidate at position 30 of
# 60 is the likeliest thing to be dropped -- silently, which is the failure mode
# this project keeps being bitten by.
#
# Not one call per candidate either: a template query is 0.29 ms and an LLM call
# is seconds, so sixty calls per search would dominate everything the planner
# does.
#
# Fifteen is short enough that no position is buried and long enough that the
# per-call overhead is amortised. It is a considered default, not a measured
# one, and `chunk` is a parameter so 15-vs-60 is a sweep rather than a belief.
CHUNK = 15

# Candidates are judged in RETRIEVAL ORDER by default, not shuffled (D-060).
#
# Shuffling makes position bias measurable -- with order carrying no information,
# any position effect in the verdicts is pure bias. But it also RELOCATES the
# lost-in-the-middle risk onto the best candidate: ranked, the middle holds
# mid-relevance items where a miss costs least; shuffled, the middle can hold
# rank 1. That is a real cost in answer quality paid for measurement
# convenience.
#
# So ranked is the default and `seed` exists for the measurement arms:
#   shuffle A vs shuffle B   pure position bias (verdict flip rate)
#   ranked   vs shuffled     how far the critic merely restates the retriever
# The second is the one that matters. If verdicts barely move between them, this
# module is an expensive re-statement of BM25 and should be reported as such.
SEED_RANKED = None

_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    """One candidate, judged."""

    entity_id: str
    rung: str
    reason: str = ""
    # True when no verdict came back for this candidate and the default applied.
    # Counted rather than hidden: a critic that silently skips a third of its
    # input looks identical to one that kept everything.
    defaulted: bool = False

    @property
    def kept(self) -> bool:
        return self.rung in KEPT_RUNGS


@dataclass
class Review:
    """Every candidate for one search, with the verdicts and what they cost."""

    question: str
    search_text: str
    verdicts: list[Verdict] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0

    @property
    def kept_ids(self) -> list[str]:
        return [v.entity_id for v in self.verdicts if v.kept]

    @property
    def tally(self) -> dict[str, int]:
        counts = {rung: 0 for rung in RUNGS}
        for v in self.verdicts:
            counts[v.rung] = counts.get(v.rung, 0) + 1
        return counts

    @property
    def defaulted(self) -> int:
        return sum(1 for v in self.verdicts if v.defaulted)

    def tail_keep_rate(self, chunk: int = CHUNK) -> Optional[float]:
        """Keep-rate over the LAST chunk, in retrieval order.

        The stopping signal for widening a search (D-060). The question is
        whether the good candidates have RUN OUT by the end of the list: if they
        have not, the list was too short and rank 61 probably holds more.

        Deliberately not the overall keep-rate, which points the wrong way on
        the clearest case. A search where all 60 hits are relevant and the list
        came back full is the most truncated case there is -- "widen when some
        were irrelevant" would stop exactly there, and would widen on
        `top squark`, which is junk all the way down.
        """
        if not self.verdicts:
            return None
        tail = self.verdicts[-chunk:]
        return sum(1 for v in tail if v.kept) / len(tail)


PROMPT = """\
You are filtering search results for a knowledge graph of high-energy-physics \
papers.

A search was run and returned candidate entities. Your job is to say, for each \
one INDEPENDENTLY, how it relates to the question that was asked. You are not \
ranking them against each other and you are not choosing a best few -- judge \
each on its own, and it is entirely normal for all of them to be relevant or for \
none of them to be.

THE QUESTION ASKED:
{question}

THE SEARCH THAT PRODUCED THESE CANDIDATES:
{search_text}

These are different, and the difference is the point. The search is usually \
WIDER than the question on purpose, because the graph stores one entity per \
version, tune and spelling. A candidate can match the search perfectly and still \
be a wrong answer to the question.

Judge each candidate against THE QUESTION, not against the search. A candidate the search found perfectly well can still be unrelated to what was asked.

  "exact"      -- it is the thing the question is about
  "broader"    -- THE SAME KIND OF THING as the question asks about, but wider or
                  less specific. "Pythia" for a question about Pythia 8. Not for
                  a question about something else entirely.
  "unrelated"  -- not what was asked about. Use this whenever the candidate is a
                  DIFFERENT KIND OF THING from the question (a generator when a
                  systematic uncertainty was asked about), or a different member
                  of the same family (electron energy scale when the question is
                  about jet energy scale).

Be decisive. If your "why" says the candidate is not what was asked, then the rung is "unrelated" -- not "broader". "broader" is only for a genuine generalisation of the thing asked about.

CANDIDATES:
{candidates}

Reply with JSON only, one entry per candidate, using the index numbers shown:

{{"verdicts": [{{"i": 1, "why": "<a few words>", "rung": "exact"}}, ...]}}

Give a verdict for every index from 1 to {n}. Put "why" before "rung": state \
what the candidate is, then decide.\
"""


def _render_candidates(hits: Sequence[Any]) -> str:
    """The candidate block, one line each.

    Facet tags are included because they are the coarse rung made visible: a
    critic that can see two candidates share a facet key can say "same family,
    wider" instead of guessing.
    """
    lines = []
    for i, hit in enumerate(hits, 1):
        label = str(getattr(hit, "label", "") or "")
        kind = str(getattr(hit, "kind", "") or "")
        facets = list(getattr(hit, "facets", None) or [])
        line = f"{i}. {label}   [{kind}]"
        if facets:
            line += f"   facets: {', '.join(str(f) for f in facets)}"
        lines.append(line)
    return "\n".join(lines)


def _parse(raw: str, n: int) -> dict[int, tuple[str, str]]:
    """Index -> (rung, why), for whatever came back that is usable.

    Tolerant on purpose. A reply that is unparseable, or that returns 47
    verdicts for 60 candidates, must not become 13 silent drops -- the caller
    defaults the missing ones to KEPT. Failing loudly here and quietly there
    would be the wrong way round.
    """
    span = _JSON_SPAN.search(raw or "")
    if not span:
        return {}
    try:
        parsed = json.loads(span.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[int, tuple[str, str]] = {}
    for item in (parsed.get("verdicts") or []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= n:
            continue
        rung = str(item.get("rung", "")).strip().lower()
        if rung not in RUNGS:
            continue
        out[index] = (rung, str(item.get("why", ""))[:200])
    return out


def _chunks(hits: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(hits), size):
        yield hits[start:start + size]


def judge_candidates(
    chat: Callable,
    question: str,
    search_text: str,
    hits: Sequence[Any],
    *,
    chunk: int = CHUNK,
    seed: Optional[int] = SEED_RANKED,
) -> Review:
    """Judge every candidate, and return them in RETRIEVAL order.

    `chat(messages)` returns an object shaped like an OpenAI completion. Passed
    in rather than constructed so the planner's client, a stub, or a
    different-family model (S-37, an ablation) all work unchanged.

    `seed` shuffles the candidates before showing them and restores retrieval
    order afterwards, so a caller can compare two orderings over identical input.
    The returned order never depends on it -- only what the model saw does.
    """
    import time

    review = Review(question=question, search_text=search_text)
    if not hits:
        return review

    order = list(range(len(hits)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    shown = [hits[i] for i in order]

    # Index in the prompt -> position in `hits`. Kept explicit so a shuffled run
    # and a ranked run produce verdicts that are directly comparable.
    verdicts: dict[int, Verdict] = {}
    started = time.perf_counter()
    offset = 0
    for group in _chunks(shown, chunk):
        n = len(group)
        prompt = PROMPT.format(question=question, search_text=search_text,
                               candidates=_render_candidates(group), n=n)
        try:
            reply = chat([{"role": "user", "content": prompt}])
            review.calls += 1
            usage = getattr(reply, "usage", None)
            if usage:
                review.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                review.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            content = reply.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 -- one bad chunk must not lose the rest
            logger.warning("critic chunk failed: %s", exc)
            review.errors += 1
            content = ""

        marks = _parse(content, n)
        for i in range(1, n + 1):
            position = order[offset + i - 1]
            hit = hits[position]
            entity_id = str(getattr(hit, "entity_id", ""))
            if i in marks:
                rung, why = marks[i]
                verdicts[position] = Verdict(entity_id, rung, why)
            else:
                # THE ASYMMETRY THAT MATTERS. A missing verdict defaults to the
                # loosest kept rung, never to a drop. Flagging is recoverable --
                # the candidate is still in the set and still in the trace --
                # while a drop the model never actually made is invisible.
                verdicts[position] = Verdict(
                    entity_id, BROADER, "no verdict returned", defaulted=True)
        offset += n

    review.verdicts = [verdicts[i] for i in range(len(hits))]
    review.seconds = time.perf_counter() - started
    if review.defaulted:
        logger.info("critic defaulted %d/%d candidates to kept",
                    review.defaulted, len(review.verdicts))
    return review


def should_widen(review: Review, hits_returned: int, limit: int,
                 *, threshold: float = 0.5, chunk: int = CHUNK) -> bool:
    """Is there good reason to think rank `limit + 1` holds more of the same?

    Two conditions, and the first is not optional: the hit list must have come
    back EXACTLY full. A search that found 43 of a possible 60 has no rank 61 to
    fetch, so relevance at its tail says nothing.

    Measured before any of this was built: 297 of 458 Tier B searches (64.8%)
    came back exactly full, so the ceiling binds on two questions in three and
    this is load-bearing rather than a refinement.

    `threshold` is generous by design (D-060). Flagging is reversible -- a
    wrongly-flagged candidate is still in the set and still in the trace -- but
    NOT widening is not: a candidate never retrieved is gone, and nothing
    downstream can recover it. Strict flagging and generous widening are the
    right way round for those two different costs.
    """
    if hits_returned < limit:
        return False
    rate = review.tail_keep_rate(chunk)
    return rate is not None and rate >= threshold


# ---------------------------------------------------------------------------
# The facets site: a label reader, NOT a filter.
# ---------------------------------------------------------------------------
#
# A facet tag is a FAMILY and the label is what the paper actually did. Six
# papers carry `background_methods = ABCD` and describe six different methods:
#
#     Modified ABCD estimate
#     ABCD data-driven background estimation method
#     ABCD-style ratio method using eight non-overlapping regions (A-H)
#     Two-dimensional ABCD sideband method using control regions B, C, D
#     ABCD (matrix) data-driven estimation using control regions in data
#     Multidimensional ABCD reweighting technique
#
# The question decides which answer is right, and BOTH answers come from those
# same six rows: "how many use a data-driven estimate?" is 6, "how many use the
# standard four-region ABCD?" is not 6.
#
# WHY THIS SITE NEVER REMOVES ANYTHING, unlike the search site. At `search`,
# `unrelated` means the candidate is not the thing -- Herwig in a Pythia set --
# and dropping it loses nothing. Here every match is genuine: the tag is right,
# only the variant differs. And for a coverage map the variants ARE the finding.
# "There are at least five distinct ways this literature does ABCD" is the sort
# of answer this project exists to produce, and narrowing to the two that match
# the question as asked would destroy it.
#
# So a `broader` verdict here is a LEAD, not a demotion: it names how the paper
# differs, which is directly usable as the next query. `facet_entities` is the
# tool that follows it.

FACET_PROMPT = """\
A question was answered by looking up a closed-vocabulary tag. The tag is a \
FAMILY; each paper's own words say what it actually did, and those differ.

THE QUESTION ASKED:
{question}

THE TAG THAT MATCHED:
{field} = {values}

WHAT EACH PAPER ACTUALLY DID:
{papers}

For each paper say how its own description relates to the question:
  "exact"      -- it is what the question asked for
  "broader"    -- same family, but a different or wider variant than asked. Say
                  HOW it differs, specifically ("eight regions, not four"),
                  because that difference is the useful part.
  "unrelated"  -- the tag is a mis-classification for this paper

Nothing is being discarded. This is a reading of what is there.

Reply with JSON only:

{{"verdicts": [{{"i": 1, "why": "<how it differs, specifically>", \
"rung": "exact"}}, ...]}}

Give a verdict for every index from 1 to {n}.\
"""


def _render_papers(rows: Sequence[dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        labels = row.get("evidence_labels") or row.get("matched") or ""
        lines.append(f"{i}. {row.get('paper_id', '?')}   {str(labels)[:300]}")
    return "\n".join(lines)


def read_facet_labels(chat: Callable, question: str, field: str,
                      values: Sequence[str], rows: Sequence[dict],
                      *, chunk: int = CHUNK) -> Review:
    """How each tagged paper's own words relate to the question.

    Same three rungs as `judge_candidates` and the same missing-verdict
    asymmetry, so the two sites report comparably. What differs is that the
    caller must not use `kept_ids` to narrow anything here -- see above.
    """
    import time

    review = Review(question=question, search_text=f"{field}={list(values)}")
    if not rows:
        return review

    verdicts: dict[int, Verdict] = {}
    started = time.perf_counter()
    offset = 0
    for group in _chunks(list(rows), chunk):
        n = len(group)
        prompt = FACET_PROMPT.format(question=question, field=field,
                                     values=list(values),
                                     papers=_render_papers(group), n=n)
        try:
            reply = chat([{"role": "user", "content": prompt}])
            review.calls += 1
            usage = getattr(reply, "usage", None)
            if usage:
                review.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                review.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            content = reply.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("facet reader chunk failed: %s", exc)
            review.errors += 1
            content = ""

        marks = _parse(content, n)
        for i in range(1, n + 1):
            row = group[i - 1]
            paper = str(row.get("paper_id", ""))
            if i in marks:
                rung, why = marks[i]
                verdicts[offset + i - 1] = Verdict(paper, rung, why)
            else:
                verdicts[offset + i - 1] = Verdict(
                    paper, BROADER, "no verdict returned", defaulted=True)
        offset += n

    review.verdicts = [verdicts[i] for i in range(len(rows))]
    review.seconds = time.perf_counter() - started
    return review


def facet_summary(review: Review) -> str:
    """What the planner is told: the count as asked, and how the rest differ.

    Both halves matter and neither replaces the other. The narrow count answers
    the question; the differences are the coverage finding, and they name where
    to look next.
    """
    if not review.verdicts:
        return ""
    exact = [v for v in review.verdicts if v.rung == EXACT]
    others = [v for v in review.verdicts if v.rung == BROADER]
    wrong = [v for v in review.verdicts if v.rung == UNRELATED]

    parts = [f"{len(review.verdicts)} papers carry this tag; "
             f"{len(exact)} match the question as asked."]
    if others:
        differ = "; ".join(f"{v.entity_id} ({v.reason})" for v in others[:6])
        parts.append(f"{len(others)} are the same family but differ: {differ}."
                     " Those differences are themselves coverage information --"
                     " use facet_entities or contents_of to follow one.")
    if wrong:
        parts.append(f"{len(wrong)} look mis-tagged: "
                     + ", ".join(v.entity_id for v in wrong[:6]) + ".")
    parts.append("No paper has been removed from the result.")
    return " ".join(parts)
