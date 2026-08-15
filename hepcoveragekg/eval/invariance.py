"""
Does the system answer the same question the same way? Measured without gold.

Tier B's truth was computed by querying the graph, so a disagreement between the
system and the gold is not automatically the system's mistake -- it can be the
system being righter than the pipeline that wrote the gold. That makes every
gold-scored number arguable in exactly the place it matters most.

**Two wordings of one question have the same true answer, whatever that answer
is.** So comparing the system's two answers *to each other* tests it without
appealing to the gold at all, and nothing about how the gold was built can
contest the result. A system tracking the concept gives both wordings the same
count; one keying off surface words does not.

The pairs are already in the question set: each record carries `group` and
`relation: paraphrase_of:<qid>`, from the metamorphic generation (S-34, S-67).

WHAT THIS CAN AND CANNOT SHOW. High invariance is necessary, not sufficient --
a system answering "0" to everything is perfectly invariant and useless. So
`agreement` is always reported beside the answer RATE, and a pair where both
sides abstained is excluded rather than counted as agreement, which would reward
silence. Read it as a ceiling on how much of the disagreement with gold could be
the system's inconsistency rather than the gold's.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

# The number a counting answer asserts. Same shape as the scorer's, kept here so
# this module can read a run file with nothing else loaded.
_CLAIMED = re.compile(r"\b(\d[\d,]*)\s+(?:distinct\s+|different\s+|unique\s+)?"
                      r"(?:papers?|analyses|analysis|studies)\b", re.I)
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


def paraphrase_pairs(questions_path: str | Path) -> list[tuple[str, str]]:
    """(qid, qid) for every pair of wordings of one question.

    Read from `relation: paraphrase_of:<qid>` rather than from `group`, because
    a group can hold more than two wordings and a pair is what a comparison
    needs. Pairs are normalised and de-duplicated, so A->B and B->A count once.
    """
    seen: set[tuple[str, str]] = set()
    for line in Path(questions_path).read_text().splitlines():
        try:
            q = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel = str(q.get("relation") or "")
        if not rel.startswith("paraphrase_of:"):
            continue
        other = rel.split(":", 1)[1]
        pair = tuple(sorted((q["qid"], other)))
        seen.add(pair)  # type: ignore[arg-type]
    return sorted(seen)


def _answer_of(record: dict) -> tuple[Optional[float], frozenset, bool]:
    """(claimed count, papers named, abstained) for one record."""
    answer = record.get("answer") or {}
    text = answer.get("text") or ""
    abstained = bool(record.get("scores", {}).get("abstained")) or bool(answer.get("error"))
    value = answer.get("value")
    if value is None:
        match = _CLAIMED.search(text)
        value = float(match.group(1).replace(",", "")) if match else None
    return (float(value) if value is not None else None,
            frozenset(_ARXIV.findall(text)), abstained)


def measure(records: Iterable[dict], pairs: Sequence[tuple[str, str]],
            shapes: Optional[dict] = None) -> dict:
    """How often the two wordings of a question get the same answer.

    `count_agree` is exact equality of the asserted number. `set_jaccard` is the
    overlap of the papers named, which is the right shape for a list: two
    answers naming four of the same five papers agree far more than a
    same-or-not flag can express.
    """
    by_qid = {r["qid"]: r for r in records if r.get("qid")}
    shapes = shapes or {}

    count_same, count_seen = 0, 0
    jaccards: list[float] = []
    both_abstained, one_abstained, usable = 0, 0, 0
    deltas: list[float] = []

    for a_id, b_id in pairs:
        a, b = by_qid.get(a_id), by_qid.get(b_id)
        if a is None or b is None:
            continue
        va, pa, aa = _answer_of(a)
        vb, pb, ab = _answer_of(b)

        if aa and ab:
            both_abstained += 1          # excluded: silence is not agreement
            continue
        if aa or ab:
            one_abstained += 1           # itself a disagreement about answerability
            continue
        usable += 1

        shape = shapes.get(a_id) or shapes.get(b_id) or "count"
        if shape == "count" and va is not None and vb is not None:
            count_seen += 1
            count_same += int(va == vb)
            deltas.append(abs(va - vb))
        elif shape == "set" and (pa or pb):
            union = pa | pb
            jaccards.append(len(pa & pb) / len(union) if union else 1.0)

    return {
        "pairs": len(pairs),
        "usable": usable,
        "both_abstained": both_abstained,
        "one_abstained": one_abstained,
        "count_pairs": count_seen,
        "count_agree": (count_same / count_seen) if count_seen else None,
        "count_median_gap": statistics.median(deltas) if deltas else None,
        "set_pairs": len(jaccards),
        "set_jaccard": (statistics.mean(jaccards) if jaccards else None),
    }


def compare(arms: dict[str, list[dict]], pairs: Sequence[tuple[str, str]],
            shapes: Optional[dict] = None) -> str:
    """One line per arm. No gold is consulted anywhere in this table."""
    lines = ["paraphrase invariance -- no gold consulted", "=" * 64,
             f"{'arm':10s} {'usable':>7} {'count agree':>12} {'median gap':>11} "
             f"{'set jaccard':>12} {'1 abstained':>12}"]
    for name, records in arms.items():
        m = measure(records, pairs, shapes)
        fmt = lambda v, s="{:.3f}": "--" if v is None else s.format(v)  # noqa: E731
        lines.append(f"{name:10s} {m['usable']:7d} {fmt(m['count_agree']):>12} "
                     f"{fmt(m['count_median_gap'], '{:.0f}'):>11} "
                     f"{fmt(m['set_jaccard']):>12} {m['one_abstained']:12d}")
    lines.append("")
    lines.append("higher agreement is necessary but not sufficient -- a system that "
                 "answers nothing is perfectly invariant, which is why pairs where "
                 "both sides abstained are excluded rather than counted as agreeing.")
    return "\n".join(lines)
