"""
Evaluation harness: the scorers.

Each scorer is a pure function of (Question, Answer) returning a dict of named
numbers -- or **nothing at all**.

That last part is the design rule of this file, and it is worth stating plainly
because getting it wrong is silent: **a scorer that cannot apply returns no
value, never zero.** Plain RAG has no tool calls, so `plan_validity` does not
apply to it; scoring it 0 would invent a difference that does not exist and
would drag its average down for a reason invisible in the table. An unlabelled
question is not a failed question.

The report then states, per metric, how many records it covered -- so a metric
computed over 12 of 200 records cannot masquerade as an overall result.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from .questions import Question
from .systems import Answer

Scorer = Callable[[Question, Answer], Optional[dict]]


# --------------------------------------------------------------------------
# label-free -- these apply to every system, from day one
# --------------------------------------------------------------------------

def cost(q: Question, a: Answer) -> dict:
    """Always available, and the only metric that never abstains."""
    return {
        "seconds": a.seconds,
        "llm_calls": float(a.llm_calls),
        "tokens": float(a.prompt_tokens + a.completion_tokens),
        "rounds": float(a.rounds),
        "errored": 1.0 if a.error else 0.0,
    }


# Phrases a refusal is written in. Needed because `answerable` is the planner's
# own flag and it is not reliable: on 2026-08-02 hw-0010 came back with
# answerable=True and the text "the graph does not contain information about...".
# Rather than redefine `answered` -- which would hide the disagreement -- the two
# are reported side by side, and the gap between them IS the finding.
_REFUSALS = (
    "does not contain", "does not record", "no information", "not in the graph",
    "cannot answer", "does not have any", "no data", "unable to",
)


def responded(q: Question, a: Answer) -> dict:
    """Did it produce an answer at all, did it decline, and does the prose agree?

    Declining is sometimes right: for a `not_in_graph` question, abstention IS
    the correct answer (S-13).
    """
    text = (a.text or "").lower()
    refused_in_text = any(p in text for p in _REFUSALS)
    out = {
        "answered": 1.0 if (a.answered and a.text.strip()) else 0.0,
        "abstained": 0.0 if a.answered else 1.0,
        "refused_in_text": 1.0 if refused_in_text else 0.0,
        # The flag says one thing and the prose says another. Worth its own
        # number: it is the difference between "it knows it failed" and "it
        # does not".
        "refusal_flag_disagrees": 1.0 if (refused_in_text and a.answered) else 0.0,
    }
    if "not_in_graph" in q.needs:
        # The one place abstention is scored as success. The prose counts too --
        # a refusal written out is a refusal, whatever the flag says.
        correct = (not a.answered) or refused_in_text
        out["abstention_correct"] = 1.0 if correct else 0.0
    return out


def faithfulness(q: Question, a: Answer) -> Optional[dict]:
    """S-14, already computed by verify.py during the run.

    Abstains for systems that cannot supply it -- which is most baselines, and
    is exactly the point: mechanical faithfulness is a capability this graph has
    and a chunk-retrieval baseline does not.
    """
    if a.verification_score is None:
        return None
    return {
        "faithfulness": float(a.verification_score),
        "unsupported_claims": float(len(a.unsupported_claims)),
    }


# --------------------------------------------------------------------------
# labelled -- these need q.truth
# --------------------------------------------------------------------------

def _numbers_in(text: str) -> list[str]:
    import re
    return re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text or "")


def exact_count(q: Question, a: Answer) -> Optional[dict]:
    """For counting questions: is the true number in the answer?

    Deliberately generous about *where* -- the answer is prose, and requiring a
    bare number would measure formatting. Deliberately strict about the value:
    "about 60" does not count as 58. Both halves are what S-33 labels are for.
    """
    if q.truth.kind != "count" or q.truth.value is None:
        return None
    if a.value is not None:
        hit = float(a.value) == float(q.truth.value)
    else:
        wanted = {str(q.truth.value), str(int(q.truth.value))
                  if float(q.truth.value).is_integer() else str(q.truth.value)}
        found = {n.replace(",", "") for n in _numbers_in(a.text)}
        hit = bool(wanted & found)
    return {"count_correct": 1.0 if hit else 0.0}


def set_f1(q: Question, a: Answer) -> Optional[dict]:
    """For set questions: precision/recall/F1 over paper ids.

    Papers are compared, not entities, because papers are what the truth from
    SQL is expressed in and what a physicist means by "which analyses".

    **Only for set-shaped questions.** A counting question also carries a paper
    list, but its answer is a number and legitimately names no papers -- scoring
    it here dragged the 2026-08-02 headline from 0.60 to 0.12 and made a correct
    system look broken. Naming papers when asked for a count is not the task.
    """
    if q.shape != "set" or q.truth.kind == "subset" or not q.truth.papers:
        return None
    truth = set(q.truth.papers)
    got = set(a.papers) or set(_arxiv_ids_in(a.text))
    if not got:
        return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}
    tp = len(truth & got)
    precision = tp / len(got)
    recall = tp / len(truth) if truth else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"set_precision": precision, "set_recall": recall, "set_f1": f1}


# Above this many papers, no prose answer can list the full set, so the
# "did it mention it?" question becomes unfair by construction. Measured: the
# planner writes out 11 papers at the median and 15 at most, so a concept in 58
# papers gives a randomly-chosen known positive roughly a 1-in-4 chance of being
# mentioned however good retrieval was.
PROSE_LISTABLE = 10


def known_positive_recall(q: Question, a: Answer) -> Optional[dict]:
    """Two different questions, kept apart because they diverge.

      RETRIEVED -- did the system FIND a paper we know qualifies? Read from the
          entities it actually pulled out of the graph. This is the retrieval
          measurement, and it is the one that is fair on every question.

      MENTIONED -- did the ANSWER say so? Read only from the prose. This measures
          whether the model reports what it found, which is a real and separate
          capability.

    The gap between them is the cost of prose truncation -- how much a physicist
    reading the answer never gets to see.

    `mentioned_paper_recall` **abstains for common concepts** (see
    PROSE_LISTABLE): asking a model to list 58 papers and scoring it on whether
    one particular paper survived the summary measures nothing about the system.
    Retrieval recall still applies there.

    `returned_papers` guards against the obvious game: recall of a subset is
    trivially perfect if you return everything, so the set size is always
    reported beside it.
    """
    if q.truth.kind != "subset" or not q.truth.papers:
        return None
    must = set(q.truth.papers)
    retrieved = set(a.papers)
    mentioned = set(_arxiv_ids_in(a.text))

    out = {
        "retrieved_paper_recall": len(must & retrieved) / len(must),
        "returned_papers": float(len(retrieved or mentioned)),
        "mentioned_papers": float(len(mentioned)),
    }
    concept_size = q.provenance.get("concept_papers")
    if concept_size is None or concept_size <= PROSE_LISTABLE:
        out["mentioned_paper_recall"] = len(must & mentioned) / len(must)
        # Only meaningful where both apply: how much was found and not said.
        out["truncation_loss"] = out["retrieved_paper_recall"] - out["mentioned_paper_recall"]
    return out


def label_recall(q: Question, a: Answer) -> Optional[dict]:
    """Tier A set questions: did it find, and did it name, this paper's entities?

    Split for the same reason as `known_positive_recall`: the system routinely
    finds far more than it writes down, and scoring only the prose measures its
    writing rather than its searching.

      retrieved  truth entity ids present in what the planner pulled from the
                 graph. Exact -- ids, not text.
      mentioned  the readable labels appearing in the answer. Deliberately
                 lenient (normalised substring), because a label like
                 "Simultaneous binned maximum-likelihood fit to SR m_bb
                 distributions..." will never be reproduced verbatim, and
                 demanding that would measure transcription.
    """
    if q.truth.kind != "labels" or not q.truth.items:
        return None
    must = set(q.truth.items)
    found = set(a.entity_ids)
    out = {
        "retrieved_label_recall": len(must & found) / len(must),
        "retrieved_entities": float(len(found)),
    }

    labels = q.provenance.get("labels") or []
    if labels:
        text = _normalise_text(a.text)
        # A short label can match by accident inside a longer word, so require a
        # few characters before trusting containment.
        hits = sum(1 for lbl in labels
                   if len(_normalise_text(lbl)) >= 4 and _normalise_text(lbl) in text)
        out["mentioned_label_recall"] = hits / len(labels)
        out["truncation_loss_labels"] = (
            out["retrieved_label_recall"] - out["mentioned_label_recall"])
    return out


def _normalise_text(text: str) -> str:
    import re

    from ..aliases.context import clean_latex

    return re.sub(r"[^a-z0-9]+", " ", clean_latex(text or "").lower()).strip()


def entity_retrieved(q: Question, a: Answer) -> Optional[dict]:
    """Did the target entity come back? The one measurement ambiguity cannot spoil.

    A count depends on where the concept boundary is drawn; "was this entity in
    what the planner retrieved" does not. So this applies to the 81% of concepts
    Tier B has to discard as ambiguous, which is most of the graph.
    """
    if q.truth.kind != "entity" or not q.truth.items:
        return None
    found = set(a.entity_ids)
    hit = len(set(q.truth.items) & found) / len(q.truth.items)
    return {"entity_retrieved": hit, "entities_seen": float(len(found))}


def _arxiv_ids_in(text: str) -> list[str]:
    import re
    return re.findall(r"\b\d{4}\.\d{4,5}\b", text or "")


# --------------------------------------------------------------------------
# per-layer -- these need a trace, so they abstain for flat baselines
# --------------------------------------------------------------------------

# Tools that read the graph. `answer` is excluded: it is the stopping move.
_RETRIEVING_TOOLS = {"search", "describe", "subjects_of", "papers_of", "contents_of",
                     "count", "list", "compare", "crosstab", "quotes"}


def tool_use(q: Question, a: Answer) -> Optional[dict]:
    """Shape of the plan: how many calls, how many failed, did it retrieve.

    Not tool *selection* accuracy (S-49) -- that needs an expected tool on the
    question and arrives with the tool-choice question set.
    """
    if not a.has_trace:
        return None
    failed = sum(1 for s in a.steps if s.get("error"))
    # Any tool that reads the graph counts, not just `search`. Anchoring on
    # `search` alone dropped this from 1.00 to 0.18 the moment `contents_of`
    # existed -- the planner was retrieving perfectly well by a different route,
    # and the metric was measuring one code path.
    searched = any(s.get("tool") in _RETRIEVING_TOOLS for s in a.steps)
    return {
        "tool_calls": float(len(a.steps)),
        "tool_errors": float(failed),
        "tool_error_rate": failed / len(a.steps) if a.steps else 0.0,
        "retrieved_first": 1.0 if searched else 0.0,
        "entities_retrieved": float(len(a.entity_ids)),
        "evidence_quotes": float(len(a.evidence_ids)),
    }


def default_scorers() -> list[Scorer]:
    return [cost, responded, faithfulness, exact_count, set_f1,
            known_positive_recall, label_recall, entity_retrieved, tool_use]


def score_all(q: Question, a: Answer, scorers: Iterable[Scorer]) -> dict:
    """Run every scorer; skip the ones that abstain."""
    out: dict = {}
    for scorer in scorers:
        try:
            result = scorer(q, a)
        except Exception as exc:  # noqa: BLE001 -- a broken scorer must not kill a run
            out[f"_error_{getattr(scorer, '__name__', 'scorer')}"] = str(exc)
            continue
        if result:
            out.update(result)
    return out
