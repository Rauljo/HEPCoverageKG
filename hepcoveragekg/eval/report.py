"""
Evaluation harness: turning run records into something readable.

Three rules, each of which exists because the obvious alternative is misleading.

**Never a bare number.** Every metric prints mean and spread across repeats
(S-52). A single figure invites reading a 3% difference as an improvement when
identical runs swing 5%, and that is how a week disappears.

**Always the coverage.** A metric computed over 12 of 200 records must say so,
or an abstaining scorer (scoring.py) silently turns into an overall result.

**Comparison states whether the difference is readable.** `compare` puts the
gap next to the noise floor and says plainly when the gap is inside it. A table
that shows only the gap is a table that will be over-read -- by us first.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .runner import Record, load_records


@dataclass
class Metric:
    name: str
    mean: float
    spread: float      # stdev across repeats of the per-repeat means; 0 if repeats == 1
    n: int             # records the metric applied to
    total: int         # records in the run

    @property
    def coverage(self) -> float:
        return self.n / self.total if self.total else 0.0

    def format(self) -> str:
        value = f"{self.mean:.3g}"
        if self.spread:
            value += f" ± {self.spread:.2g}"
        if self.coverage < 1.0:
            value += f"  (n={self.n}/{self.total})"
        return value


def summarise(records: Iterable[Record], metric_names: Optional[list[str]] = None
              ) -> dict[str, Metric]:
    """Aggregate scores across records, keeping the repeat structure.

    The spread is the standard deviation of the *per-repeat means*, not of all
    the individual values. That is the quantity that matters: it answers "if I
    ran this again, how different would the reported number be?", which is what
    an ablation comparison is implicitly asking.
    """
    records = list(records)
    total = len(records)
    if not total:
        return {}

    by_repeat: dict[int, list[Record]] = {}
    for r in records:
        by_repeat.setdefault(r.repeat, []).append(r)

    names = metric_names or sorted(
        {k for r in records for k in r.scores if not k.startswith("_")}
    )

    out: dict[str, Metric] = {}
    for name in names:
        values = [r.scores[name] for r in records
                  if name in r.scores and isinstance(r.scores[name], (int, float))]
        if not values:
            continue
        repeat_means = []
        for group in by_repeat.values():
            vals = [g.scores[name] for g in group
                    if name in g.scores and isinstance(g.scores[name], (int, float))]
            if vals:
                repeat_means.append(statistics.fmean(vals))
        spread = statistics.stdev(repeat_means) if len(repeat_means) > 1 else 0.0
        out[name] = Metric(name=name, mean=statistics.fmean(values), spread=spread,
                           n=len(values), total=total)
    return out


def by_tag(records: Iterable[Record], tag: str = "needs") -> dict[str, dict[str, Metric]]:
    """Break the metrics down by question tag.

    This is what makes a low score attributable (S-10): 'shape' says which
    template struggles, and 'needs' says whether the query layer was wrong or
    the data was never there -- `signatures` scoring badly is an M3 problem, not
    ours (S-57).
    """
    groups: dict[str, list[Record]] = {}
    for r in records:
        keys = r.needs if tag == "needs" else [getattr(r, tag, "?")]
        for key in keys:
            groups.setdefault(str(key), []).append(r)
    return {k: summarise(v) for k, v in sorted(groups.items())}


def render(meta: dict, records: list[Record], *, tag: str = "shape") -> str:
    """The human-readable report for one run."""
    lines: list[str] = []
    lines.append(f"run      {meta.get('run_id', '?')}")
    lines.append(f"system   {meta.get('system', '?')}  "
                 f"[config {meta.get('config_hash', '?')}]  [git {meta.get('git_sha', '?')}]")
    lines.append(f"model    {meta.get('model') or '(none)'}")
    lines.append(f"questions {meta.get('n_questions', '?')} "
                 f"x {meta.get('repeats', 1)} repeats "
                 f"= {len(records)} records   [set {meta.get('questions_hash', '?')}]")
    if str(meta.get("git_sha", "")).endswith("-dirty"):
        lines.append("WARNING  the working tree was dirty -- the sha does not describe what ran")
    lines.append("")

    metrics = summarise(records)
    if not metrics:
        return "\n".join(lines + ["(no scores)"])

    width = max(len(n) for n in metrics)
    lines.append("overall")
    for name, m in metrics.items():
        lines.append(f"  {name:<{width}}  {m.format()}")

    if meta.get("repeats", 1) == 1:
        lines.append("")
        lines.append("NOTE  one repeat, so no spread is known. Run with --repeats 3+ before")
        lines.append("      comparing anything -- an unmeasured noise floor makes any gap unreadable.")

    lines.append("")
    lines.append(f"by {tag}")
    for key, group in by_tag(records, tag).items():
        parts = [f"{n}={m.mean:.3g}" for n, m in group.items()
                 if n in ("count_correct", "set_f1", "faithfulness", "answered", "seconds")]
        n_records = next(iter(group.values())).total if group else 0
        lines.append(f"  {key:<18} ({n_records:>3})  " + "  ".join(parts))

    errors = [r for r in records if r.answer.get("error")]
    if errors:
        lines.append("")
        lines.append(f"errors ({len(errors)})")
        for r in errors[:5]:
            lines.append(f"  {r.qid}: {r.answer['error'][:100]}")
    return "\n".join(lines)


def compare(a_path: str | Path, b_path: str | Path) -> str:
    """Two runs, side by side, with the noise floor stated.

    The whole point is the last column. A gap smaller than the combined spread
    is not a result, and saying so here is what stops it becoming one in a
    dissertation table.
    """
    meta_a, rec_a = load_records(a_path)
    meta_b, rec_b = load_records(b_path)
    m_a, m_b = summarise(rec_a), summarise(rec_b)

    lines = [
        f"A  {meta_a.get('system')} [{meta_a.get('config_hash')}] {meta_a.get('run_id')}",
        f"B  {meta_b.get('system')} [{meta_b.get('config_hash')}] {meta_b.get('run_id')}",
        "",
    ]
    if meta_a.get("questions_hash") != meta_b.get("questions_hash"):
        lines.append("WARNING  different question sets -- these numbers are not comparable.")
        lines.append("")

    names = sorted(set(m_a) | set(m_b))
    width = max((len(n) for n in names), default=10)
    lines.append(f"  {'metric':<{width}}  {'A':>12}  {'B':>12}  {'B-A':>10}   readable?")
    for name in names:
        x, y = m_a.get(name), m_b.get(name)
        if x is None or y is None:
            here = "A only" if y is None else "B only"
            shown = (x or y)
            lines.append(f"  {name:<{width}}  {shown.format():>12}  {here:>25}")
            continue
        delta = y.mean - x.mean
        noise = x.spread + y.spread
        if noise == 0:
            verdict = "no spread measured"
        elif abs(delta) <= noise:
            verdict = f"NO -- inside noise (±{noise:.2g})"
        else:
            verdict = f"yes (noise ±{noise:.2g})"
        lines.append(f"  {name:<{width}}  {x.mean:>12.3g}  {y.mean:>12.3g}  "
                     f"{delta:>+10.3g}   {verdict}")
    return "\n".join(lines)


def render_path(path: str | Path, *, tag: str = "shape") -> str:
    meta, records = load_records(path)
    return render(meta, records, tag=tag)
