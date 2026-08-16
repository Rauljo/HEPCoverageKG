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
        """Keep-rate over the LAST chunk, in retrieval order."""
        if not self.verdicts:
            return None
        tail = self.verdicts[-chunk:]
        return sum(1 for v in tail if v.kept) / len(tail)

    def head_keep_rate(self, chunk: int = CHUNK) -> Optional[float]:
        """Keep-rate over the FIRST chunk, in retrieval order."""
        if not self.verdicts:
            return None
        head = self.verdicts[:chunk]
        return sum(1 for v in head if v.kept) / len(head)

    def decay(self, chunk: int = CHUNK) -> Optional[float]:
        """Tail keep-rate as a fraction of head keep-rate. The widen signal.

        The question is whether the good candidates have RUN OUT by the end of
        the list: if they have not, the list was too short and rank 61 probably
        holds more.

        **Measured as a ratio, not an absolute, because an absolute threshold
        makes the widen decision hostage to how strict the critic is overall.**
        On 2026-08-15 that is exactly what happened: the critic marked 75% of all
        candidates `unrelated`, so a tail keep-rate of 0.5 was unreachable and
        the widen loop fired ZERO times across 1,074 searches -- leaving the
        measurement that motivated it (64.8% of Tier B searches truncated at the
        ceiling) untested. D-060 decision 9 says flagging may be strict while
        widening must be generous; feeding both from one absolute number broke
        that on the first run.

        A ratio is immune to it. A critic keeping 40% at the head and 35% at the
        tail has not run out, whether the 40% is 40% or 8%. One keeping 40% then
        5% has.

        Deliberately not the overall keep-rate, which points the wrong way on the
        clearest case: a search where every hit is relevant and the list came
        back full is the MOST truncated case there is, and "widen when some were
        irrelevant" would stop exactly there.
        """
        head = self.head_keep_rate(chunk)
        tail = self.tail_keep_rate(chunk)
        if head is None or tail is None:
            return None
        if head == 0:
            # Nothing was kept anywhere. There is no relevance to have run out
            # of, so this is not evidence of truncation.
            return 0.0
        return tail / head


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

  "exact"      -- counting it would answer the question. It is the thing asked
                  about, OR it is direct evidence of it.
  "broader"    -- it is a wider or less specific version of the thing asked
                  about: "Pythia" for a question about Pythia 8.
  "unrelated"  -- counting it would NOT help answer the question: a different
                  thing entirely, or a different member of the same family
                  (electron energy scale when jet energy scale was asked about).

The bracketed word after each candidate is its record type in the database. IGNORE IT when deciding. A record of a different type can still be exact: a SAMPLE produced with Sherpa is evidence that an analysis used Sherpa, and a SYSTEMATIC derived from varying Pythia is evidence that an analysis used Pythia. Ask whether the candidate bears on the question, never what type of record it is.

Be decisive. If your "why" says the candidate does not bear on the question, then the rung is "unrelated" -- not "broader". "broader" is only for a genuine generalisation of the thing asked about.

CANDIDATES:
{candidates}

Reply with JSON only, one entry per candidate, using the index numbers shown:

{{"verdicts": [{{"i": 1, "why": "<a few words>", "rung": "exact"}}, ...]}}

Give a verdict for every index from 1 to {n}. Put "why" before "rung": state \
what the candidate is, then decide.\
"""


# A prompt for a SMALL judge.
#
# The 8B failed the gate on 2026-08-16 in a specific way: every reason it gave
# was a description of the candidate -- "jet energy scale systematic" -- with no
# mention of the question at all. Asked whether `jet-energy-scale` bore on a
# PYTHIA question it answered "this is a jet energy scale systematic" and kept
# it. It recognised the family and stopped, which is the same failure recorded
# for the 8B on the aliases task, where it answered "are these related?" instead
# of "are these the same?" and merged 8 distinct SMEFT Wilson coefficients.
#
# So the comparison is made STRUCTURAL rather than requested. The model must
# write what the candidate is AND what the question wants, as separate fields,
# before choosing. Ignoring the question stops being possible: there is a slot
# for it that has to be filled.
#
# Two worked examples, because a small model takes far more from a demonstration
# than from a definition -- and both examples are DROPS, since keeping is the
# failure mode being corrected.
SMALL_PROMPT = """\
Decide whether each candidate helps answer one question.

THE QUESTION: {question}

For every candidate, fill three fields:
  "is"    - what the candidate is, in a few words
  "asks"  - what THIS question is about, in a few words (the same every time)
  "rung"  - "exact" if counting the candidate helps answer the question,
            "broader" if it is a wider version of what the question asks about,
            "unrelated" if it does not help answer THIS question

Compare "is" against "asks" before choosing. A candidate can be a perfectly good
thing and still be unrelated to what was asked.

EXAMPLE, question "Which analyses use the Pythia generator?":
  candidate "jet energy scale uncertainty"
  {{"i": 1, "is": "a jet energy scale systematic", "asks": "the Pythia generator",
   "rung": "unrelated"}}

EXAMPLE, question "Which analyses apply a jet energy scale uncertainty?":
  candidate "PYTHIA 8.230"
  {{"i": 2, "is": "the Pythia generator, version 8.230",
   "asks": "a jet energy scale uncertainty", "rung": "unrelated"}}

CANDIDATES:
{candidates}

Reply with JSON only, one entry per candidate, indexes 1 to {n}:
{{"verdicts": [{{"i": 1, "is": "...", "asks": "...", "rung": "exact"}}, ...]}}\
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
        why = item.get("why")
        if why is None and ("is" in item or "asks" in item):
            why = f"{item.get('is','')} | asked: {item.get('asks','')}"
        out[index] = (rung, str(why or "")[:200])
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
    prompt: Optional[str] = None,
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
        template = prompt or PROMPT
        body = (template.format(question=question, candidates=_render_candidates(group), n=n)
                if "{search_text}" not in template
                else template.format(question=question, search_text=search_text,
                                     candidates=_render_candidates(group), n=n))
        try:
            reply = chat([{"role": "user", "content": body}])
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
                 *, threshold: float = 0.6, chunk: int = CHUNK) -> bool:
    """Is there good reason to think rank `limit + 1` holds more of the same?

    Two conditions, and the first is not optional: the hit list must have come
    back EXACTLY full. A search that found 43 of a possible 60 has no rank 61 to
    fetch, so relevance at its tail says nothing.

    Measured before any of this was built: 297 of 458 Tier B searches (64.8%)
    came back exactly full, so the ceiling binds on two questions in three and
    this is load-bearing rather than a refinement.

    `threshold` is a RATIO of tail keep-rate to head keep-rate (see
    `Review.decay`), so a globally strict critic cannot suppress widening the
    way it did on 2026-08-15 -- zero widened searches out of 1,074, because a
    75%-unrelated critic can never reach an absolute 0.5.

    Generous by design (D-060). Flagging is reversible -- a wrongly-flagged
    candidate is still in the set and still in the trace -- but NOT widening is
    not: a candidate never retrieved is gone, and nothing downstream can recover
    it. Strict flagging and generous widening are the right way round for those
    two different costs.
    """
    if hits_returned < limit:
        return False
    ratio = review.decay(chunk)
    return ratio is not None and ratio >= threshold


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
