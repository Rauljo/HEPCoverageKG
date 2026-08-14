"""Batch 2: the reads our own pipeline could not settle.

Strictly additive. It does not touch the sheet already sent -- different row
range (from 1001), different browser-storage key, different files. Batch 1's
answers are keyed on row number, so anything that renumbers it would silently
reattach a verdict to a different paper, which is the one corruption nobody
could detect afterwards.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from hepcoveragekg.eval import review as RV
from hepcoveragekg.eval import review_app as RA
from hepcoveragekg.eval import supervisor as S

NOTE = (
    "<div class='flag'><p style='margin:0'><b>These are the ones we could not "
    "settle ourselves.</b> Each passage was read three times, and the three "
    "readings disagreed with each other — so rather than take a majority vote "
    "and hide it, they were set aside for you. They are separate from the first "
    "batch and numbered from 1001, so nothing here clashes with answers you have "
    "already given.</p></div>"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out-dir", default="eval/review/batch2")
    ap.add_argument("--start-row", type=int, default=1001)
    args = ap.parse_args()

    rows = [json.loads(l) for l in io.open(args.sweep, encoding="utf-8") if l.strip()]
    questions = S.build_records()
    items = RV.split_items(rows, questions, start_row=args.start_row)
    if not items:
        print("no splits found — nothing to build")
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    RV.write_sheet(items, out / "gabriel-splits.tsv")
    RV.write_key(items, out / "ANSWER_KEY-splits.jsonl")
    RA.write_app(items, out / "gabriel-splits-app.html",
                 "The ones we could not settle",
                 version="splits-v1", note=NOTE)

    from collections import Counter
    per_q = Counter(i["qid"] for i in items)
    print(f"  {len(items)} items, rows {items[0]['row']}–{items[-1]['row']}")
    for qid, n in sorted(per_q.items()):
        print(f"    {qid:8} {n:3}")
    print(f"\n  -> {out}/gabriel-splits-app.html")
    print(f"  -> {out}/gabriel-splits.tsv")
    print(f"  -> {out}/ANSWER_KEY-splits.jsonl  (kept back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
