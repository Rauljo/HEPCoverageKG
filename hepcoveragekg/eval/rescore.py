"""Re-score stored runs with today's scorers, without re-running anything.

Scores are computed at RUN time and frozen into the file, which is right --
re-deriving them from a moving scorer would make old results silently change.
It is also why a scorer BUG contaminates every result taken before it was found,
and the only honest response is to re-score and say by how much.

The occasion: `judged_set_f1` credited the retrieval footprint when an answer
named no papers (D-080). Every typed-planner number reported this week was
inflated by it, and the correction inverted the comparison against the control.

The original scores are never overwritten. A `rescored` block sits beside them,
so a file always says what it was judged as at the time AND what it is judged as
now, and the two can be compared rather than one quietly replacing the other.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from . import questions as Q
from . import scoring
from .systems import Answer


def _questions(paths: Iterable[str]) -> dict:
    out = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    q = Q.parse(json.loads(line))
                except Exception:          # a set we cannot parse is skipped, not fatal
                    continue
                out[q.qid] = q
    return out


def rescore_file(path: Path | str, questions: dict,
                 scorers: Optional[list] = None) -> dict:
    """Returns {metric: (old_mean, new_mean, n)} and rewrites the file in place."""
    path = Path(path)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    scorers = scorers or scoring.default_scorers()

    old_sums: dict = {}
    new_sums: dict = {}
    counts: dict = {}
    changed = 0
    for rec in lines:
        if "_meta" in rec:
            continue
        q = questions.get(rec["qid"])
        if q is None:
            continue
        answer = Answer(**{k: v for k, v in (rec.get("answer") or {}).items()
                           if k in Answer.__dataclass_fields__})
        fresh = scoring.score_all(q, answer, scorers)
        old = rec.get("scores") or {}
        if fresh != old:
            changed += 1
        rec["rescored"] = fresh
        for k, v in fresh.items():
            if not isinstance(v, (int, float)):
                continue
            new_sums[k] = new_sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
            if isinstance(old.get(k), (int, float)):
                old_sums[k] = old_sums.get(k, 0.0) + old[k]

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines),
                    encoding="utf-8")
    return {k: (old_sums.get(k, 0.0) / counts[k], new_sums[k] / counts[k], counts[k])
            for k in sorted(new_sums)}, changed
