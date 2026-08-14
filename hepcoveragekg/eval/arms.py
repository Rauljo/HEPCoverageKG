"""
Comparing two ablation arms that were each run as several shards.

`report.compare` already does the honest part -- it prints the gap next to the
noise floor and says when the gap is inside it. What it cannot do is take
eighteen files, and the gap between "eighteen files" and "two arms" is where a
comparison quietly stops being one.

TWO GUARDS, both earned today.

**Align on question ids, never on counts.** A shard that died leaves its arm
short, and comparing the survivors against a complete arm compares two different
question sets while reporting a single clean delta. So the arms are intersected
on `(qid, repeat)` and the dropped ids are counted out loud. A comparison over
94% of one arm is reportable; one that does not say so is not.

**Check they are the same system apart from the thing under test.** Every record
carries `config_hash` and `git_sha`. Two arms differing by more than the flag --
because code moved between them, or because one ran on a different graph -- is
the failure this project has hit repeatedly (D-059, and twice more on 2026-08-14).
It is cheap to detect and impossible to see afterwards in a table of means.

The critic-specific columns live here too, because they are what makes a score
difference explicable: how many candidates were judged, how many flagged down,
how often a search widened, and how many tool calls only survived because the
serving-side parser was patched around.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .runner import Record, load_records


def merge(paths: Sequence[str | Path]) -> tuple[dict, list[Record]]:
    """Every record from a set of shards, with one representative meta.

    Shards of one arm share a config; they deliberately do NOT share a
    `questions_hash`, because each was given a different slice of the question
    file. That is why the caller must align on qids rather than trusting the
    hash that `report.compare` checks.
    """
    records: list[Record] = []
    metas: list[dict] = []
    for path in paths:
        meta, recs = load_records(path)
        metas.append(meta)
        records.extend(recs)

    merged = dict(metas[0]) if metas else {}
    merged["shards"] = len(metas)
    merged["shard_run_ids"] = [m.get("run_id") for m in metas]
    merged["n_records"] = len(records)
    return merged, records


def identity(records: Iterable[Record]) -> dict[str, set]:
    """What system produced these records, as sets that should each be size 1."""
    seen: dict[str, set] = {"config_hash": set(), "git_sha": set(), "system": set()}
    for r in records:
        seen["config_hash"].add(r.config_hash)
        seen["git_sha"].add(r.git_sha)
        seen["system"].add(r.system)
    return seen


def align(a: Sequence[Record], b: Sequence[Record]) -> tuple[list[Record], list[Record], dict]:
    """The two arms restricted to the questions both actually answered."""
    key = lambda r: (r.qid, r.repeat)          # noqa: E731
    index_a = {key(r): r for r in a}
    index_b = {key(r): r for r in b}
    shared = sorted(set(index_a) & set(index_b))
    stats = {
        "shared": len(shared),
        "a_only": len(set(index_a) - set(index_b)),
        "b_only": len(set(index_b) - set(index_a)),
        "a_total": len(index_a),
        "b_total": len(index_b),
    }
    return [index_a[k] for k in shared], [index_b[k] for k in shared], stats


def critic_activity(records: Iterable[Record]) -> dict:
    """What the critic did, aggregated. Empty-ish when the arm ran without one."""
    totals = Counter()
    rungs = Counter()
    searches = 0
    widened = 0
    for r in records:
        answer = r.answer if isinstance(r.answer, dict) else {}
        totals["recovered_calls"] += answer.get("recovered_calls", 0) or 0
        for review in answer.get("reviews") or []:
            searches += 1
            totals["candidates"] += review.get("candidates", 0)
            totals["kept"] += review.get("kept", 0)
            totals["defaulted"] += review.get("defaulted", 0)
            totals["critic_calls"] += review.get("calls", 0)
            totals["critic_errors"] += review.get("errors", 0)
            for rung, n in (review.get("tally") or {}).items():
                rungs[rung] += n
            if review.get("candidates", 0) > 60:
                widened += 1
        for step in answer.get("steps") or []:
            if step.get("redirected_from"):
                totals["redirects"] += 1
    return {"searches_judged": searches, "widened": widened,
            "rungs": dict(rungs), **dict(totals)}


def compare_arms(a_paths: Sequence[str | Path], b_paths: Sequence[str | Path],
                 *, a_name: str = "control", b_name: str = "critic") -> str:
    """The full comparison: identity check, alignment, metrics, critic activity."""
    from . import report

    meta_a, rec_a = merge(a_paths)
    meta_b, rec_b = merge(b_paths)

    lines = [
        f"{a_name}: {len(rec_a)} records from {meta_a.get('shards')} shards",
        f"{b_name}: {len(rec_b)} records from {meta_b.get('shards')} shards",
        "",
    ]

    # -- is this one system, twice, differing by the thing under test? ------
    id_a, id_b = identity(rec_a), identity(rec_b)
    for field in ("git_sha", "system"):
        if id_a[field] != id_b[field]:
            lines.append(f"WARNING  {field} differs between arms: "
                         f"{sorted(id_a[field])} vs {sorted(id_b[field])}. "
                         f"These arms differ by more than the flag under test.")
        if len(id_a[field] | id_b[field]) > 1 and id_a[field] == id_b[field]:
            lines.append(f"WARNING  {field} is not constant WITHIN an arm: "
                         f"{sorted(id_a[field])}")
    if id_a["config_hash"] == id_b["config_hash"]:
        lines.append("WARNING  both arms share a config_hash -- the flag under "
                     "test did not reach the system, or these are the same run.")
    if any("dirty" in s for s in id_a["git_sha"] | id_b["git_sha"]):
        lines.append("NOTE  a git sha is marked -dirty, so it does not identify "
                     "the code that produced these records.")
    lines.append("")

    # -- same questions, or say so -----------------------------------------
    aligned_a, aligned_b, overlap = align(rec_a, rec_b)
    lines.append(f"aligned on {overlap['shared']} shared (qid, repeat) pairs "
                 f"[{a_name} had {overlap['a_total']}, {b_name} had {overlap['b_total']}]")
    if overlap["a_only"] or overlap["b_only"]:
        lines.append(f"  dropped: {overlap['a_only']} answered only by {a_name}, "
                     f"{overlap['b_only']} only by {b_name}")
    if not overlap["shared"]:
        lines.append("nothing to compare.")
        return "\n".join(lines)
    lines.append("")

    # -- the numbers, with the noise floor beside them ---------------------
    m_a, m_b = report.summarise(aligned_a), report.summarise(aligned_b)
    names = sorted(set(m_a) | set(m_b))
    width = max((len(n) for n in names), default=10)
    lines.append(f"  {'metric':<{width}}  {a_name:>14}  {b_name:>14}  {'delta':>10}")
    for name in names:
        x, y = m_a.get(name), m_b.get(name)
        if x is None or y is None:
            continue
        delta = y.mean - x.mean
        floor = x.spread + y.spread
        mark = "" if not floor else ("" if abs(delta) > floor else "  (within noise)")
        lines.append(f"  {name:<{width}}  {x.format():>14}  {y.format():>14}  "
                     f"{delta:>+10.3g}{mark}")

    # -- what the critic actually did --------------------------------------
    act_a, act_b = critic_activity(aligned_a), critic_activity(aligned_b)
    lines += ["", "critic activity", "-" * 40]
    for key in sorted(set(act_a) | set(act_b)):
        if key == "rungs":
            continue
        lines.append(f"  {key:<20} {str(act_a.get(key, 0)):>12} {str(act_b.get(key, 0)):>12}")
    if act_b.get("rungs"):
        lines.append(f"  rungs                {'':>12} {json.dumps(act_b['rungs'])}")
    if not act_b.get("searches_judged"):
        lines.append("  WARNING  the critic arm judged nothing -- the flag did not take effect.")
    return "\n".join(lines)
