"""A smaller question set that still measures the same thing.

WHY SUBSET AT ALL. Six arms over 436 concept questions x 3 repeats is 7,848
question-runs against one 72B server. The power analysis (D-063) says the answer
arrives at 200 questions: a subsample that size reproduced the full-data verdict
95%+ of the time. Running the other 236 buys precision nobody reads and costs a
night of cluster time per arm.

WHOLE GROUPS, NEVER LOOSE QUESTIONS. Tier B is built as metamorphic groups --
one concept asked four ways -- and paraphrase invariance is measured WITHIN a
group. Splitting a group across the subset boundary silently destroys that
measure for every group it splits, and the file still looks fine. So the unit of
sampling is the group, exactly as the unit of the power analysis is the question
and never the repeat.

STRATIFIED BY SHAPE. Tier B is 75% count and 25% set, and the two behave
differently under the critic -- it roughly doubled set F1 while barely moving
counting (D-062 addendum). A subset that drifted toward one shape would move the
headline number for a reason that has nothing to do with the arm. Groups are
drawn per shape in proportion, so the mix survives.

DETERMINISTIC. A fixed seed and sorted inputs, so the same subset comes back on
every machine. An ablation whose question set cannot be reproduced is an
ablation whose result cannot be checked.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def group_of(record: dict) -> str:
    """The metamorphic group, falling back to the qid so an ungrouped question
    is its own group rather than silently joining a bucket called '?'."""
    return record.get("group") or record["qid"]


def shape_of(group: list[dict]) -> str:
    """A group's shape. Groups are single-shape by construction; if one ever is
    not, the majority decides and the group still travels whole."""
    shapes = [r.get("shape", "count") for r in group]
    return max(set(shapes), key=shapes.count)


def sample(records: Iterable[dict], target: int = 200, *, seed: int = 20260828
           ) -> list[dict]:
    """`target` questions or as near as whole groups allow, mix preserved."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[group_of(record)].append(record)

    by_shape: dict[str, list[str]] = defaultdict(list)
    for name, members in groups.items():
        by_shape[shape_of(members)].append(name)

    total = sum(len(m) for m in groups.values())
    if total <= target:
        return [r for name in sorted(groups) for r in groups[name]]

    rng = random.Random(seed)
    chosen: list[dict] = []
    for shape in sorted(by_shape):
        names = sorted(by_shape[shape])
        rng.shuffle(names)
        # This shape's fair share of the target, in questions not groups.
        share = target * sum(len(groups[n]) for n in names) / total
        taken = 0
        for name in names:
            if taken >= share:
                break
            chosen.extend(groups[name])
            taken += len(groups[name])
    return sorted(chosen, key=lambda r: r["qid"])


def write(src: Path | str, dest: Path | str, target: int = 200,
          *, seed: int = 20260828) -> tuple[Path, dict]:
    records = [json.loads(line) for line in
               Path(src).read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = sample(records, target, seed=seed)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rs: list[dict]) -> dict:
        out: dict = defaultdict(int)
        for r in rs:
            out[r.get("shape", "count")] += 1
        return dict(out)

    return dest, {
        "source_questions": len(records),
        "source_groups": len({group_of(r) for r in records}),
        "questions": len(picked),
        "groups": len({group_of(r) for r in picked}),
        "source_mix": mix(records),
        "subset_mix": mix(picked),
        "seed": seed,
    }
