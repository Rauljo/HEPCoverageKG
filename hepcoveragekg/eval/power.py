"""
How many questions does a conclusion actually need?

Every run so far has used the whole question set because that was the cautious
default, not because anyone measured what was required. At 436 questions x 3
repeats and ~90s each with a critic, that caution costs entire days -- and it
buys nothing on an effect that was decided by the first eighty questions.

The method is a bootstrap over the runs we already have, so it costs no GPU:
draw `n` questions with replacement, recompute the paired difference, and ask
how often a subsample of that size reproduces what the full data says. The
answer is a **reproduction rate** per size, and the useful number is the
smallest `n` that reproduces reliably.

TWO THINGS THIS DELIBERATELY DOES NOT DO.

It does not report a p-value against a null of zero. The question is not "is
there an effect" -- we have the full data and we know. It is "would a smaller
run have told us the same thing", which is a question about *reproduction*, and
answering it directly is less to misread than a significance test used as a
proxy for it.

And it never bootstraps over repeats. Repeats of one question are not
independent observations of the effect; resampling them would shrink the
apparent spread and overstate how small a set can be. Questions are the unit,
and a question's repeats travel together.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# The between-run noise floor measured by replicating one configuration on one
# commit (D-063): two runs of the same system differed by 0.0008 on
# count_correct. A subsample "reproduces" the full result only if it clears
# this as well as agreeing in sign -- a difference smaller than the floor is not
# a finding at any sample size.
NOISE_FLOOR = 0.0008

DEFAULT_SIZES = (25, 50, 100, 200, 400, 800)


@dataclass
class Point:
    n: int
    reproduced: float      # fraction of subsamples agreeing with the full data
    mean_delta: float
    spread: float          # stdev of the subsample deltas

    @property
    def reliable(self) -> bool:
        return self.reproduced >= 0.95


def by_question(records_a: dict, records_b: dict, metric: str) -> dict[str, tuple]:
    """Per QUESTION, the mean of its repeats in each arm.

    Collapsing repeats before resampling is the point: three readings of one
    question are one observation of the effect, not three, and treating them as
    three is how a power analysis talks itself into a smaller number than the
    data supports.
    """
    per: dict[str, list] = defaultdict(lambda: [[], []])
    for side, records in ((0, records_a), (1, records_b)):
        for (qid, _repeat), record in records.items():
            value = (record.get("scores") or {}).get(metric)
            if value is not None:
                per[qid][side].append(float(value))
    return {qid: (statistics.mean(a), statistics.mean(b))
            for qid, (a, b) in per.items() if a and b}


def power_curve(paired: dict[str, tuple], sizes: Sequence[int] = DEFAULT_SIZES,
                bootstrap: int = 500, seed: int = 0,
                floor: float = NOISE_FLOOR) -> list[Point]:
    """For each size, how often a subsample reproduces the full-data verdict."""
    qids = sorted(paired)
    if not qids:
        return []
    full = statistics.mean(b - a for a, b in paired.values())
    sign = 1 if full > 0 else -1
    rng = random.Random(seed)

    out: list[Point] = []
    for n in sizes:
        if n > len(qids):
            break
        deltas = []
        agreed = 0
        for _ in range(bootstrap):
            sample = [paired[rng.choice(qids)] for _ in range(n)]
            delta = statistics.mean(b - a for a, b in sample)
            deltas.append(delta)
            if delta * sign > 0 and abs(delta) > floor:
                agreed += 1
        out.append(Point(n=n, reproduced=agreed / bootstrap,
                         mean_delta=statistics.mean(deltas),
                         spread=statistics.stdev(deltas) if len(deltas) > 1 else 0.0))
    return out


def smallest_reliable(curve: Sequence[Point]) -> Optional[int]:
    """The smallest tested size that reproduces at least 95% of the time."""
    for point in curve:
        if point.reliable:
            return point.n
    return None


def render(name: str, full_delta: float, curve: Sequence[Point]) -> str:
    lines = [f"{name}   full-data delta {full_delta:+.4f}",
             f"    {'n':>6} {'reproduced':>11} {'mean delta':>12} {'spread':>9}"]
    for p in curve:
        mark = "" if p.reliable else "   <- unreliable"
        lines.append(f"    {p.n:6d} {100 * p.reproduced:10.0f}% {p.mean_delta:+12.4f} "
                     f"{p.spread:9.4f}{mark}")
    smallest = smallest_reliable(curve)
    lines.append(f"    smallest reliable size: "
                 f"{smallest if smallest else 'none of those tested'}")
    return "\n".join(lines)
