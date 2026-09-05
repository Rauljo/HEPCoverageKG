"""Build batch 2 of the review sheet: the questions our own machinery could not settle.

Three sources, three reasons a human is needed:

  splits      three reads of ONE passage disagreed with each other. A claim the
              judge upheld is usually easy; a split is a real question, which is
              what an adjudicator is for.
  gf-01       the three-part conjunction, asked ONE CONDITION AT A TIME from a
              decomposed run. `split_items` would hand him a single sentence for
              a three-part claim -- the D-058 failure, repeated at him.
  value       the questions asking for a number or a list. Batch 1 showed a
              quote and a yes/no box, and he wrote "Not a yes/no question. What
              to do here?" on all four he attempted, then stopped. Here we show
              OUR answer and ask whether it is right, which is a yes/no.

Numbered from 1001 so batch 1's answers -- keyed on row number, in a spreadsheet
and in browser storage -- cannot silently reattach to a different paper.
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
from pathlib import Path

from hepcoveragekg.eval import gabriel_gold as G
from hepcoveragekg.eval import review as RV
from hepcoveragekg.eval import review_app as RA
from hepcoveragekg.eval import supervisor as S
from hepcoveragekg.kg import store

SWEEP = "eval/reader/20260809T202844-sweep-48330.reparsed.supported.jsonl"
GF01 = "eval/reader/20260812T150442-gf01-rerun-48386.supported.jsonl"
VALUE = "eval/reader/single-valuejudged-final.jsonl"


def _rows(path):
    p = Path(path)
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--gf01", default=GF01)
    ap.add_argument("--value", default=VALUE)
    ap.add_argument("--out-dir", default="eval/review/batch2")
    ap.add_argument("--db", default=str(store.DEFAULT_DB_PATH))
    args = ap.parse_args()

    questions = S.build_records()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    splits = RV.split_items(_rows(args.sweep), questions)

    # A yes he has already given cannot be changed by a second quote, so asking
    # again is pure cost to him. Papers he answered no or unsure are kept: there,
    # new evidence genuinely can move the gold.
    verdicts = G.read_verdicts()
    settled = {(q, p) for q, ps in verdicts.items() for p, g in ps.items()
               if G.resolve(g) == "yes"}
    kept = [i for i in splits if (i["qid"], i["paper_id"]) not in settled]
    gf01_papers = {i["paper_id"] for i in kept if i["qid"] == "gf-01"}
    kept = [i for i in kept if i["qid"] != "gf-01"]

    conditions = RV.decomposed_condition_items(_rows(args.gf01), gf01_papers, conn=conn)
    conn = sqlite3.connect("file:data/processed/hepkg.db?mode=ro", uri=True)
    values = RV.value_items(_rows(args.value), questions, conn=conn)

    # ORDERED BY WHAT A PARTIAL RETURN IS WORTH, for the same reason the probe
    # set puts Gabriel's questions first: these get answered in file order and
    # often not to the end.
    #
    #   conditions  decide whether decomposition ships (D-069). Nothing else in
    #               the batch settles a pending decision.
    #   splits      three reads of one passage that disagreed -- our machinery
    #               could not resolve them, which is what an adjudicator is for.
    #   values      he already told us the old framing was unanswerable; these
    #               are the reframed seven, and the cheapest rows to lose.
    items = conditions + kept + values
    for n, item in enumerate(items, 1001):
        item["row"] = n

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    RV.write_sheet(items, out / "gabriel-splits.tsv")
    RV.write_key(items, out / "ANSWER_KEY-splits.jsonl")
    RA.write_app(items, out / "gabriel-splits-app.html",
                 "The ones we could not settle", version="v2")

    print(f"  splits (not gf-01)  {len(kept):4}")
    print(f"  gf-01 per-condition {len(conditions):4}  from {len(gf01_papers)} papers")
    print(f"  value questions     {len(values):4}")
    print(f"  ---------------------------")
    print(f"  total               {len(items):4}   rows {items[0]['row']}-{items[-1]['row']}")
    print(f"  -> {out}/gabriel-splits-app.html   (the clickable one)")
    print(f"  -> {out}/ANSWER_KEY-splits.jsonl   (NEVER send this)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
