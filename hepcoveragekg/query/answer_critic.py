"""Judge each candidate PAPER against the question, before the answer names any.

THE GAP THIS SITS ON. Measured 2026-09-05 over eight arms on 164 questions:
retrieval reaches 93-98% of the right papers and the answer then scores
0.18-0.40. The gap is 0.57-0.76 for every arm, and no mechanism addresses it.
`resolve_citations` recorded the same thing in August -- "the planner named 11
arXiv ids at the median, 100% genuinely retrieved, and hit only 22.8% of the
gold papers while retrieval had reached 91.7%. **Selection, not hallucination**".

WHY THE EXISTING CRITIC DOES NOT COVER IT. `critic.judge_candidates` asks
"is this ENTITY related to the SEARCH TEXT" -- 'b-tagged jet' against 'BJet'.
That is a retrieval filter. It cannot answer "does 2103.06956 actually USE
b-tagged jets in its event selection", because it never sees a paper and never
sees the question's condition. Different unit, different question:

    critic         entity  vs  search term     keeps the candidate list sane
    answer_critic  paper   vs  THE QUESTION    decides what the answer asserts

THE TASK IS DELIBERATELY GABRIEL'S TASK. He is shown a question, a paper, and
the sentence we retrieved from it, and answers yes/no -- 253 such verdicts so
far. This asks a model to do the same thing on the same evidence, which means
his verdicts measure it directly rather than by proxy. No other component in
this project can be checked that way.

DEFAULTS TO KEEP, AND SAYS SO LOUDLY. Same asymmetry as `critic.py`: a missing
verdict keeps the paper, because a drop the model never made is invisible in the
output. That asymmetry is also exactly what hid a dead critic for weeks (D-105:
33,836 candidates, 100% defaulted, zero drops, because a reasoning judge spent
its whole budget thinking and returned empty content). So `defaulted` is
counted, returned, and logged at WARNING above `DEFAULT_ALARM` -- a filter that
silently keeps everything must be loud, not merely recorded.

EVIDENCE, NOT LABELS. Each paper is shown with the retrieved entity labels AND
the verbatim quotes backing them, because "does this paper use b-tagged jets"
cannot be answered from an entity name alone -- gf-01-condition turns on
`veto = use`, which lives in the sentence and not in the label.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: Papers judged per call. The critic chunks at 15 for "lost in the middle";
#: papers carry more text each (labels plus quotes), so they chunk smaller.
CHUNK = 8

#: Above this share of missing verdicts the filter is not filtering. D-105 ran
#: at 100% for weeks without anyone noticing, so this is a WARNING, not a note.
DEFAULT_ALARM = 0.25

#: Quote characters shown per paper. Enough to judge on, short enough that eight
#: papers still fit a prompt a small judge can hold.
QUOTE_CHARS = 320

_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PaperVerdict:
    paper_id: str
    keep: bool
    why: str = ""
    defaulted: bool = False


@dataclass
class AnswerReview:
    question: str
    verdicts: list[PaperVerdict] = field(default_factory=list)
    calls: int = 0
    errors: int = 0
    seconds: float = 0.0

    @property
    def kept(self) -> list[str]:
        return [v.paper_id for v in self.verdicts if v.keep]

    @property
    def dropped(self) -> list[str]:
        return [v.paper_id for v in self.verdicts if not v.keep]

    @property
    def defaulted(self) -> int:
        return sum(1 for v in self.verdicts if v.defaulted)

    def to_dict(self) -> dict:
        return {"question": self.question[:200], "candidates": len(self.verdicts),
                "kept": len(self.kept), "dropped": len(self.dropped),
                "defaulted": self.defaulted, "calls": self.calls,
                "errors": self.errors, "seconds": round(self.seconds, 1)}


PROMPT = """\
You decide which PAPERS answer a question. For each paper you are given the \
material retrieved from it: entity labels, and verbatim sentences from the paper.

Answer for each: does THIS paper satisfy what the question asks?

Judge the CONDITION, not the topic. A paper about b-tagging is not automatically \
a paper whose event selection uses b-tagged jets. A paper that VETOES b-jets does \
use them -- a veto is a selection on the object. A paper that merely mentions a \
technique in passing does not use it.

Decide only from the material shown. If it does not establish that the paper \
satisfies the condition, answer no.

Reply with JSON and nothing else:
{"verdicts": [{"paper": "<arxiv id>", "keep": true|false, "why": "<8 words>"}]}"""


def _render(paper_id: str, labels: Iterable[str], quotes: Iterable[str]) -> str:
    lab = ", ".join(sorted({str(l) for l in labels if l})[:8]) or "(none)"
    qs = [str(q).strip().replace("\n", " ") for q in quotes if str(q or "").strip()]
    out = [f"PAPER {paper_id}", f"  retrieved: {lab}"]
    for q in qs[:3]:
        out.append(f"  quote: {q[:QUOTE_CHARS]}")
    if not qs:
        out.append("  quote: (no verbatim sentence retrieved)")
    return "\n".join(out)


def _parse(raw: str, expected: set) -> dict:
    """paper_id -> (keep, why). Unparseable returns {} so callers default."""
    span = _JSON_SPAN.search(raw or "")
    if not span:
        return {}
    try:
        parsed = json.loads(span.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for item in (parsed.get("verdicts") or []):
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paper") or "").strip()
        if pid not in expected:
            continue
        keep = item.get("keep")
        if isinstance(keep, str):
            keep = keep.strip().lower() in ("true", "yes", "y", "1")
        if not isinstance(keep, bool):
            continue
        out[pid] = (keep, str(item.get("why") or "")[:120])
    return out


def evidence_by_paper(conn, entity_ids: Iterable[str],
                      papers: Optional[Iterable[str]] = None) -> dict:
    """{paper_id: (labels, quotes)} for the RETRIEVED entities only.

    Restricted to entities the run actually retrieved, so the judge sees what
    the system found rather than everything the paper contains -- judging a
    paper on material the system never surfaced would measure the corpus, not
    the system.
    """
    ids = [str(e) for e in entity_ids if e]
    if not conn or not ids:
        return {}
    marks = ",".join("?" * len(ids))
    want = {str(p) for p in papers} if papers else None
    out: dict = {}
    rows = conn.execute(
        f"""SELECT eo.paper_id AS pid, eo.label AS label, ev.quote AS quote
            FROM entity_occurrence eo
            LEFT JOIN assertion a
                   ON a.paper_id = eo.paper_id
                  AND (a.subject_id = eo.entity_id OR a.object_id = eo.entity_id)
            LEFT JOIN assertion_evidence ae ON ae.assertion_id = a.assertion_id
            LEFT JOIN evidence ev ON ev.evidence_id = ae.evidence_id
            WHERE eo.entity_id IN ({marks})""",
        tuple(ids),
    ).fetchall()
    for r in rows:
        pid = r["pid"] if not isinstance(r, tuple) else r[0]
        if not pid or (want is not None and pid not in want):
            continue
        label = r["label"] if not isinstance(r, tuple) else r[1]
        quote = r["quote"] if not isinstance(r, tuple) else r[2]
        labels, quotes = out.setdefault(pid, (set(), []))
        if label:
            labels.add(label)
        if quote and quote not in quotes:
            quotes.append(quote)
    return out


def judge_papers(chat: Callable, question: str, evidence: dict,
                 chunk: int = CHUNK) -> AnswerReview:
    """Ask, per paper, whether it satisfies the question. Never raises."""
    review = AnswerReview(question=question)
    started = time.perf_counter()
    papers = sorted(evidence)
    if not papers:
        return review

    verdicts: dict = {}
    for i in range(0, len(papers), chunk):
        batch = papers[i:i + chunk]
        blocks = "\n\n".join(_render(p, *evidence[p]) for p in batch)
        user = f"QUESTION: {question}\n\n{blocks}"
        try:
            response = chat([{"role": "system", "content": PROMPT},
                             {"role": "user", "content": user}])
            review.calls += 1
            raw = (response.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 -- a judge must not kill a run
            logger.warning("answer-critic call failed, keeping batch: %s", exc)
            review.errors += 1
            raw = ""
        got = _parse(raw, set(batch))
        for p in batch:
            if p in got:
                keep, why = got[p]
                verdicts[p] = PaperVerdict(p, keep, why)
            else:
                verdicts[p] = PaperVerdict(p, True, "no verdict returned",
                                           defaulted=True)

    review.verdicts = [verdicts[p] for p in papers]
    review.seconds = time.perf_counter() - started
    share = review.defaulted / len(review.verdicts) if review.verdicts else 0.0
    if share >= DEFAULT_ALARM:
        logger.warning(
            "answer-critic defaulted %d/%d papers to KEEP (%.0f%%) -- it is not "
            "filtering; check the judge's completion budget and that thinking is "
            "off (D-105)", review.defaulted, len(review.verdicts), 100 * share)
    return review


# --------------------------------------------------------------------------
# RANKING, which is what this judge is actually good at (D-113)
# --------------------------------------------------------------------------
#
# Measured on Gabriel's 253 verdicts, same judge and same evidence, the only
# difference being what is done with the output:
#
#                             prec   recall      F1
#     keep everything        0.419    1.000    0.591
#     drop by verdict        0.723    0.321    0.445
#     rank, top-16           0.560    0.788    0.654
#
# WHY. The grades are monotone in truth but the bottom of the scale is weak:
#
#     grade 3  P(gold) 0.79        grade 1  P(gold) 0.36
#     grade 2  P(gold) 0.52        grade 0  P(gold) 0.31
#
# The judge recognises good evidence and cannot rule things out. Dropping
# discards the half it judges badly; ranking uses only the half it judges well.
#
# AND NOTHING IS CUT HERE. A set question's answer IS the set and a count over
# a truncated set is simply a wrong number. The ranking exists because the
# answerer already truncates on its own -- it writes about sixteen papers
# whatever the gold holds -- so the only decision available is WHICH sixteen
# come first. Imposing a cut on top of that would cap recall by construction,
# which is the mistake `drop by verdict` above already makes.

#: Grades, high to low. Four rungs rather than a 0-100 score: the judge has to
#: separate "shown" from "likely" from "topical", and a finer scale would be
#: precision the labels cannot support.
GRADES = (3, 2, 1, 0)

#: Below this share of papers in the MIDDLE grades, the ranking is not usable.
#: llama-3.1-8b put 14 of 253 papers in grades 1-2 -- a binary classifier in a
#: grader's prompt -- and scored 0.575, WORSE than keeping everything (0.591).
#: Checkable at runtime with no gold, which is what `defaulted` could never do:
#: it catches a judge that answers nothing, not one that answers with one grade.
MIN_MIDDLE_SHARE = 0.12

RANK_PROMPT = """\
You rank PAPERS by how well each one satisfies a question. For each paper you \
are given the material retrieved from it: entity labels, and verbatim sentences.

Grade each paper 0-3:
  3  the retrieved text SHOWS the paper satisfies the condition asked about
  2  the paper very likely satisfies it, but the text is suggestive not explicit
  1  related to the topic, but does not satisfy the condition asked
  0  does not bear on the question at all

Judge the CONDITION, not the topic. A paper about b-tagging is not automatically \
a paper whose event selection uses b-tagged jets. A paper that VETOES b-jets does \
use them -- a veto is a selection on the object. A paper that merely mentions a \
technique in passing does not use it.

Use the whole scale. If everything looks the same grade, you are not \
discriminating and the ranking is useless.

Reply with JSON and nothing else:
{"grades": [{"paper": "<arxiv id>", "grade": 0|1|2|3, "why": "<8 words>"}]}"""


@dataclass
class Ranking:
    """Papers ordered best-first, and whether the ordering is worth having."""
    question: str
    order: list = field(default_factory=list)          # paper ids, best first
    grades: dict = field(default_factory=dict)         # paper -> grade
    ungraded: list = field(default_factory=list)
    calls: int = 0
    errors: int = 0
    seconds: float = 0.0

    @property
    def spread(self) -> dict:
        counts = {g: 0 for g in GRADES}
        for g in self.grades.values():
            counts[int(g)] = counts.get(int(g), 0) + 1
        return counts

    @property
    def usable(self) -> bool:
        """False when the judge refused the middle of the scale (D-113)."""
        if not self.grades:
            return False
        middle = self.spread.get(2, 0) + self.spread.get(1, 0)
        return middle / len(self.grades) >= MIN_MIDDLE_SHARE

    def to_dict(self) -> dict:
        return {"candidates": len(self.grades) + len(self.ungraded),
                "graded": len(self.grades), "ungraded": len(self.ungraded),
                "spread": {str(k): v for k, v in self.spread.items()},
                "usable": self.usable, "calls": self.calls,
                "errors": self.errors, "seconds": round(self.seconds, 2)}


def _parse_grades(raw: str, expected: set) -> dict:
    """paper_id -> grade. Unparseable returns {} so the caller keeps its order."""
    span = _JSON_SPAN.search(raw or "")
    if not span:
        return {}
    try:
        parsed = json.loads(span.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for item in (parsed.get("grades") or []):
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paper") or "").strip()
        grade = item.get("grade")
        if pid not in expected or not isinstance(grade, (int, float)):
            continue
        if isinstance(grade, bool):
            continue
        out[pid] = max(0, min(3, int(grade)))
    return out


def rank_papers(chat: Callable, question: str, evidence: dict,
                chunk: int = CHUNK) -> Ranking:
    """Order the papers best-first. Never drops one. Never raises.

    An ungraded paper keeps its place at the BACK rather than being removed:
    this ranks, and a judge that failed to answer must not be able to delete a
    candidate through silence -- the asymmetry that hid a dead critic for weeks
    (D-105).
    """
    ranking = Ranking(question=question)
    started = time.perf_counter()
    papers = sorted(evidence)
    if not papers:
        return ranking

    for i in range(0, len(papers), chunk):
        batch = papers[i:i + chunk]
        blocks = "\n\n".join(_render(p, *evidence[p]) for p in batch)
        try:
            response = chat([{"role": "system", "content": RANK_PROMPT},
                             {"role": "user",
                              "content": f"QUESTION: {question}\n\n{blocks}"}])
            ranking.calls += 1
            raw = (response.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 -- a ranker must not kill a run
            logger.warning("rerank call failed, order unchanged: %s", exc)
            ranking.errors += 1
            raw = ""
        ranking.grades.update(_parse_grades(raw, set(batch)))

    ranking.ungraded = [p for p in papers if p not in ranking.grades]
    # Stable within a grade: the incoming order is retrieval order, which is
    # already a weak ranking, so ties should not be shuffled.
    pos = {p: n for n, p in enumerate(papers)}
    ranking.order = sorted(ranking.grades,
                           key=lambda p: (-ranking.grades[p], pos[p]))
    ranking.order += ranking.ungraded
    ranking.seconds = time.perf_counter() - started
    if not ranking.usable:
        logger.warning(
            "rerank unusable: %d of %d papers in the middle grades (need %.0f%%) "
            "-- this judge is answering yes/no in a grader's prompt, and a "
            "binary judge measured WORSE than not ranking at all (D-113). "
            "spread %s", ranking.spread.get(2, 0) + ranking.spread.get(1, 0),
            len(ranking.grades), 100 * MIN_MIDDLE_SHARE, ranking.spread)
    return ranking


def ranked_candidates(rankings, top_n: int = 15) -> list:
    """This run's ranked papers, best first, merged across `papers_of` calls.

    THE ANSWER NEVER SAW THE RANKING (D-128). `rank_papers` reorders tool rows
    at execute time; by the time the model writes its answer, rounds later, it
    composes from memory with no ranked list and no instruction that one exists.
    gf-05: reach 1.00, 19 papers named, 13 wrong. This is the list the answer
    step is handed, so the ranking finally acts where the answer is decided.

    Only USABLE rankings count (a judge that refused the middle grades is not a
    ranking, D-113). A paper keeps its best grade across calls. Returns
    [(paper_id, grade)], best first, at most `top_n`.
    """
    best: dict = {}
    for r in rankings or []:
        if not getattr(r, "usable", False):
            continue
        for pid in r.order:
            g = r.grades.get(pid)
            if g is None:
                continue
            if pid not in best or g > best[pid]:
                best[pid] = g
    out = sorted(best.items(), key=lambda kv: -kv[1])
    return out[:top_n]


RANKED_MESSAGE = (
    "Before that answer stands: a judge graded the candidate papers this run "
    "retrieved against the question (3 = the retrieved text shows it satisfies "
    "the condition, 2 = very likely, 1 = related but does not satisfy it). Best "
    "first:\n{listing}\n\nYour answer names {named} of the {top} papers graded "
    "3 or 2. Rewrite it, naming every paper that satisfies the question -- keep "
    "any you already named that belong, and write the arXiv ids into the text."
)
