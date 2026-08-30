"""Combining the typed planner and the free-SQL control, which fail differently.

D-079 measured it on the supervisor's gold: three questions tied, two clear wins
each way, one mutual failure. They are not two attempts at the same thing -- they
are two instruments with different blind spots, and the interesting question is
whether the union of what they find beats either alone.

THE PATTERN IS ALREADY IN THIS CODEBASE, in `reader.merge_gathered`:

    Recall is a union, not a vote. The 24B and QwQ fail differently, so a
    sentence either of them surfaces is a sentence the judge should see.
    Requiring both to agree would discard exactly the evidence the second model
    was added to recover. The judge, downstream, is what stops the union from
    being credulous.

The same reasoning applies here, so the same shape is used: union to gather,
judge to filter. What changes is the unit -- papers rather than sentences.

THREE MODES, and reporting all three is the point rather than a hedge:

  union      every paper either side named. Maximum recall, and the ceiling on
             what any merge can achieve.
  intersect  only papers both named. Maximum precision, and the floor.
  judged     the union, with each paper checked against the question by a
             model. This is the one that could beat both inputs; union and
             intersect can only ever bracket them.

MERGED POST HOC, FROM STORED RUNS. Both systems have already answered the same
questions, so merging their files costs nothing and is reproducible. A live
merged system would run them concurrently and cost the sum of both -- worth
building only if the post-hoc numbers say the merge is worth having.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from .questions import Question
from .systems import Answer

UNION = "union"
INTERSECT = "intersect"
JUDGED = "judged"


def _papers(answer: dict) -> set:
    """What an answer ASSERTED, by the D-080 rule.

    Ids written in prose count; a cited set counts; a raw retrieval footprint
    does not. Merging footprints would union two piles of everything the two
    systems happened to touch, which is not an answer from either of them.
    """
    from .scoring import _arxiv_ids_in

    named = set(_arxiv_ids_in(answer.get("text") or ""))
    if named:
        return named
    if answer.get("cited"):
        return set(answer.get("papers") or [])
    return set()


def load_run(path: Path | str) -> dict:
    """{(qid, repeat): record} for one run file."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "_meta" in rec:
            continue
        out[(rec["qid"], rec.get("repeat", 0))] = rec
    return out


def merge_answers(a: dict, b: dict, mode: str = UNION,
                  judge: Optional[Callable] = None,
                  question: str = "") -> Answer:
    """One merged Answer from two records covering the same (question, repeat)."""
    pa, pb = _papers(a.get("answer") or {}), _papers(b.get("answer") or {})
    if mode == INTERSECT:
        papers = pa & pb
    else:
        papers = pa | pb

    if mode == JUDGED and judge is not None and papers:
        papers = set(judge(question, sorted(papers)))

    # Both sides' costs, because a merge that doubles the bill has to show it.
    aa, ab = a.get("answer") or {}, b.get("answer") or {}
    return Answer(
        text=" ".join(sorted(papers)),
        answered=bool(papers),
        papers=sorted(papers),
        cited=f"merge:{mode}",
        seconds=(aa.get("seconds") or 0) + (ab.get("seconds") or 0),
        llm_calls=(aa.get("llm_calls") or 0) + (ab.get("llm_calls") or 0),
        prompt_tokens=(aa.get("prompt_tokens") or 0) + (ab.get("prompt_tokens") or 0),
        completion_tokens=(aa.get("completion_tokens") or 0) + (ab.get("completion_tokens") or 0),
        rounds=max(aa.get("rounds") or 0, ab.get("rounds") or 0),
    )


def merge_runs(path_a: Path | str, path_b: Path | str, questions: dict,
               mode: str = UNION, judge: Optional[Callable] = None) -> list[dict]:
    """Score the merge of two runs. Returns one scored record per shared key."""
    from . import scoring

    run_a, run_b = load_run(path_a), load_run(path_b)
    shared = sorted(set(run_a) & set(run_b))
    scorers = scoring.default_scorers()

    out = []
    for key in shared:
        qid = key[0]
        q = questions.get(qid)
        if q is None:
            continue
        merged = merge_answers(run_a[key], run_b[key], mode, judge, q.text)
        out.append({"qid": qid, "repeat": key[1], "shape": q.shape,
                    "answer": merged.to_dict() if hasattr(merged, "to_dict")
                              else merged.__dict__,
                    "scores": scoring.score_all(q, merged, scorers)})
    return out


def summarise(records: Iterable[dict], metrics=("judged_f1", "judged_precision",
                                                "judged_recall", "set_f1",
                                                "count_correct", "seconds")) -> dict:
    import statistics
    got = defaultdict(list)
    for r in records:
        for k, v in (r.get("scores") or {}).items():
            if isinstance(v, (int, float)):
                got[k].append(v)
    return {m: statistics.mean(got[m]) for m in metrics if got.get(m)}
