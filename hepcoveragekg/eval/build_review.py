"""Build the review sheet Gabriel actually marks.

A driver, not a notebook. The first sheet was assembled by hand from whichever
run files happened to be current, which meant the provenance of any given row
was whatever I had typed that afternoon -- and the sheet is the instrument that
produces our human ground truth, so it is the last place that should be
irreproducible.

Three sources, three shapes, one sheet:

  sweep         corpus-wide existence questions, 60 papers each. Every claimed
                yes, plus the disagreements with his gold, plus a blind sample
                of negatives carrying retrieved candidate sentences so a "no"
                is answerable at all.
  gf-01         the three-part conjunction, one item per paper, showing every
                condition's quote -- a reviewer handed one sentence for a
                three-part claim would repeat the judge's own D-058 failure.
  single-paper  the questions that name their own paper, showing the whole
                gathered union and, where the value judge overruled us, the
                answer IT says the evidence supports.

Nothing here reveals a verdict before the reviewer has given one: our answers
sit behind a <details> in the HTML, and the answer key is a separate file.
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
from pathlib import Path

from hepcoveragekg.eval import review as RV
from hepcoveragekg.eval import supervisor as S
from hepcoveragekg.kg import store


def _rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def disagreements(rows: list[dict], questions: list[dict]) -> dict:
    """Papers his gold names that we did not answer yes to.

    Worth more to a reviewer than any random negative: each one is either our
    miss or his gold being generous, and only he can say which.
    """
    found: dict[str, set] = {}
    for r in rows:
        if r.get("answer") is True:
            found.setdefault(r["qid"], set()).add(r["paper_id"])
    out = {}
    for q in questions:
        gold = (q["provenance"].get("gabriel_gold") or {}).get("papers")
        if not gold:
            continue
        missed = set(gold) - found.get(q["qid"], set())
        if missed:
            out[q["qid"]] = missed
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--gf01", default=None)
    ap.add_argument("--single", default=None)
    ap.add_argument("--out-dir", default="eval/review")
    ap.add_argument("--db", default=str(store.DEFAULT_DB_PATH))
    args = ap.parse_args()

    questions = S.build_records()
    # Read-only: the sheet is built FROM the papers, and a builder that can write
    # to the graph is one bad query away from contaminating the thing it measures.
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    sweep = _rows(args.sweep)
    items = RV.build(sweep, questions, conn=conn,
                     disagreements=disagreements(sweep, questions))
    print(f"  sweep        {len(items):4} items from {len(sweep)} reads")

    if args.gf01:
        gf01 = _rows(args.gf01)
        gold = set(next(q for q in questions
                        if q["qid"] == "gf-01")["provenance"]["gabriel_gold"]["papers"])
        extra = RV.condition_items(gf01, gold, conn=conn)
        print(f"  gf-01        {len(extra):4} items")
        items.extend(extra)

    if args.single:
        single = _rows(args.single)
        extra = RV.single_paper_items(single, questions, conn=conn)
        print(f"  single-paper {len(extra):4} items")
        items.extend(extra)

    # Renumber across the whole sheet: the row number is what he writes his marks
    # against, so it has to be unique over the file he is actually given.
    for n, item in enumerate(items, 1):
        item["row"] = n

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    RV.write_sheet(items, out / "gabriel-review.tsv")
    RV.write_key(items, out / "ANSWER_KEY.jsonl")
    RV.write_html(items, out / "gabriel-review.html",
                  "HEPCoverageKG — does the paper say this?")
    machine = sum(1 for i in items if i["_machine"])
    print(f"\n  {len(items)} items: {machine} we claimed, {len(items) - machine} we did not")
    print(f"  -> {out}/gabriel-review.html")
    print(f"  -> {out}/gabriel-review.tsv")
    print(f"  -> {out}/ANSWER_KEY.jsonl  (kept OUT of the sheet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
