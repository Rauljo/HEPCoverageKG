"""
Turn a reader run into a review sheet a physicist can actually label.

Why this exists. Everything measured so far is a model judging a model: the
reader claims a fact, a judge decides whether the quote supports it, and the
"56% precision" is one model's opinion of another's. That number cannot settle
anything on its own, and it is the last place a human is genuinely needed.

Two design choices that decide whether the result is worth having.

**The machine verdict is withheld.** The reviewer sees the question, the paper
and the quote -- never what the judge said. Showing it would anchor them, and the
agreement rate we are trying to measure is exactly what anchoring destroys. Our
verdicts go to a separate answer key, compared only afterwards.

**Negatives are included.** Labelling only the YES answers measures precision and
says nothing about what the reader MISSED. A sample of negatives is the only way
to estimate recall, and recall is what decides whether a disagreement with the
supervisor's gold is a finding or a failure. They are unmarked and shuffled in
with the rest, so the reviewer cannot tell which is which.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Iterable, Optional

# Enough negatives to estimate recall without doubling the reviewer's work. At
# ~30 across 7 questions a recall estimate is rough but honest; the alternative
# -- labelling all 285 -- is not a favour anyone will do twice.
DEFAULT_NEGATIVE_SAMPLE = 30

# Stop words for scoring candidate sentences. Deliberately tiny -- the goal is to
# drop the words every physics sentence contains, not to build a linguistics.
_STOP = set("""the a an of in on for to and or is are was were be been this that
these those it its as by with from at we our using used use does do it's which
than then so such can may might will would analysis paper study
rather whose merely simply also each their they there when what how any not""".split())


def candidate_sentences(conn, paper_id: str, question: str, top: int = 3,
                        embed: bool = True) -> list[str]:
    """The sentences most worth showing for a paper where NOTHING was found.

    A miss cannot be reviewed without something to look at. Asking "does this
    paper do X?" with no sentence attached means reading the whole paper -- ten
    minutes an item instead of twenty seconds, which is the difference between a
    review that happens and one that does not.

    Uses the SAME hybrid retrieval as the query layer: BM25 for the words, dense
    embeddings for the meaning, fused by reciprocal rank. The first version of
    this scored raw word overlap, which is precisely the failure the dense half
    exists to fix -- measured elsewhere in this project at 0.893 on named
    concepts against 0.335 on described ones. A question phrased as "correct them
    back to particle level" shares almost no vocabulary with "the data are
    unfolded", and word counting cannot bridge that.

    Reuses retrieve.Index rather than reimplementing it, with sentences standing
    in for surface forms. `conn` is not passed to `search`, because canonical
    cluster deduplication is meaningless here -- these are sentences, not
    entities.
    """
    import re as _re

    from hepcoveragekg.eval.reader import passages
    from hepcoveragekg.query import retrieve

    sentences: list[str] = []
    for p in passages(conn, paper_id):
        for sentence in _re.split(r"(?<=[.!?])\s+", p.text):
            sentence = " ".join(sentence.split())
            if 60 <= len(sentence) <= 400:
                sentences.append(sentence)
    if not sentences:
        return []

    index = retrieve.Index()
    index.entity_ids = [str(i) for i in range(len(sentences))]
    index.texts = sentences
    index.kinds = [""] * len(sentences)
    index._tokens = [retrieve.tokenize(t) for t in sentences]
    index._prepare_sparse()
    if embed:
        try:
            from hepcoveragekg.aliases import semantics
            index.embeddings = semantics.embed(sentences)
        except Exception:            # no model here -> BM25 alone, still useful
            index.embeddings = None

    hits = retrieve.search(index, question, conn=None, limit=top)
    return [h.label for h in hits]


def build(rows: list[dict], questions: list[dict], *,
          negative_sample: int = DEFAULT_NEGATIVE_SAMPLE,
          seed: int = 20260812, conn=None,
          disagreements: Optional[dict] = None) -> list[dict]:
    """Review items: every claimed YES, plus a blind sample of negatives."""
    text = {q["qid"]: (q.get("per_paper") or q["text"]) for q in questions}

    # Everything the reader CLAIMED, including claims the judge later overturned.
    # Selecting on the final answer would drop all 61 downgrades -- and those are
    # the most informative items in the set, because they are exactly where the
    # judge overruled the reader. Without them we could measure the reader and
    # never the judge, which is the thing a human is here to settle.
    claimed = [r for r in rows
               if r.get("quote") and (r.get("answer") is True or "quote_supports" in r)]
    # A negative has no quote to show -- there is nothing the reader pointed at.
    # The reviewer is instead asked the question about the paper directly, which
    # is the only way to find a miss.
    negatives = [r for r in rows
                 if r.get("answer") is False and "quote_supports" not in r]
    rng = random.Random(seed)
    by_key = {(r["qid"], r["paper_id"]): r for r in rows}

    # Prefer the DISAGREEMENTS to a random sample. A random negative almost
    # always comes back "no, correctly" and teaches nothing; a paper the
    # supervisor's gold contains and we did not find is, by definition, either
    # our miss or his gold being generous -- and there are only ~25 of them.
    sampled: list[dict] = []
    if disagreements:
        for qid, papers in sorted(disagreements.items()):
            for pid in sorted(papers):
                row = by_key.get((qid, pid))
                if row is not None and row.get("answer") is not True:
                    sampled.append(row)

    # Top up with a random sample as WELL as the disagreements, not instead of.
    # After the cascade only 2 of his listed gold papers are still missing, which
    # is good news and leaves recall unmeasured -- and the count-only questions
    # (gf-01 "18 papers", gf-04 "10 papers") name no papers at all, so they can
    # contribute no disagreements even in principle. Sampled negatives are now
    # answerable, because they carry candidate sentences.
    seen_keys = {(r["qid"], r["paper_id"]) for r in sampled}
    if negatives and negative_sample:
        per_q = defaultdict(list)
        for r in negatives:
            per_q[r["qid"]].append(r)
        take = max(1, negative_sample // max(len(per_q), 1))
        for qid in sorted(per_q):
            pool = [r for r in per_q[qid] if (qid, r["paper_id"]) not in seen_keys]
            sampled.extend(rng.sample(pool, min(take, len(pool))))

    items = []
    for r in claimed:
        items.append({
            "qid": r["qid"], "question": text.get(r["qid"], r["qid"]),
            "paper_id": r["paper_id"], "quote": r["quote"],
            "_machine": True, "_judge": r.get("quote_supports"),
            "_why": r.get("support_why", ""),
            "_by": r.get("recovered_by") or "stage-1",
        })
    for r in sampled:
        # Show what we found and rejected, so the item is answerable at all.
        cands = (candidate_sentences(conn, r["paper_id"], text.get(r["qid"], ""))
                 if conn is not None else [])
        items.append({
            "qid": r["qid"], "question": text.get(r["qid"], r["qid"]),
            "paper_id": r["paper_id"], "quote": "",
            "candidates": cands,
            "_machine": False, "_judge": None, "_why": "",
            "_by": "none",
        })

    # Grouped by question so the reviewer holds one concept in mind at a time,
    # but shuffled WITHIN a question so claimed and missed are indistinguishable.
    items.sort(key=lambda i: i["qid"])
    out: list[dict] = []
    for qid in sorted({i["qid"] for i in items}):
        group = [i for i in items if i["qid"] == qid]
        rng.shuffle(group)
        out.extend(group)
    for n, item in enumerate(out, 1):
        item["row"] = n
    return out


def write_sheet(items: list[dict], path: Path | str) -> Path:
    """The TSV the reviewer fills in. Two columns for them, the rest context."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["row", "question_id", "paper", "question",
                    "sentence_cited_from_the_paper",
                    "YOUR_VERDICT_yes_no_unsure", "YOUR_NOTES"])
        for i in items:
            if i["quote"]:
                shown = i["quote"]
            elif i.get("candidates"):
                # We found nothing. Show what we DID find and rejected, so the
                # item can be answered without reading the paper end to end.
                shown = ("WE FOUND NO EVIDENCE. The closest sentences were:  "
                         + "   ||   ".join(i["candidates"]))
            else:
                shown = "WE FOUND NO EVIDENCE, and nothing in the paper looked close."
            w.writerow([i["row"], i["qid"], i["paper_id"], i["question"], shown, "", ""])
    return path


def write_key(items: list[dict], path: Path | str) -> Path:
    """Our verdicts, held back until the reviewer's are in."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for i in items:
            handle.write(json.dumps({
                "row": i["row"], "qid": i["qid"], "paper_id": i["paper_id"],
                "machine_said_yes": i["_machine"], "judge_upheld": i["_judge"],
                "found_by": i["_by"],
            }) + "\n")
    return path


def write_html(items: list[dict], path: Path | str, title: str) -> Path:
    """A readable version. Same row numbers as the sheet, so a reviewer can read
    here and type there."""
    path = Path(path)
    by_q = defaultdict(list)
    for i in items:
        by_q[i["qid"]].append(i)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        "<style>body{font:16px/1.55 -apple-system,Segoe UI,sans-serif;max-width:52em;"
        "margin:2em auto;padding:0 1em;color:#1a1a1a}"
        "h2{margin-top:2.2em;border-bottom:2px solid #333;padding-bottom:.3em}"
        ".q{background:#f4f4f4;padding:.8em 1em;border-left:4px solid #333;margin:1em 0}"
        ".item{margin:1.1em 0;padding:.7em 0;border-bottom:1px solid #e5e5e5}"
        ".row{font-weight:700;color:#666}.paper{font-family:ui-monospace,monospace;color:#555}"
        ".quote{margin:.4em 0 0 0;padding:.5em .8em;background:#fafafa;border-left:3px solid #bbb}"
        ".none{color:#888;font-style:italic}"
        "details{margin-top:.5em}"
        "details summary{cursor:pointer;color:#777;font-size:.88em;user-select:none}"
        "details summary:hover{color:#333}"
        "details[open] summary{color:#333;font-weight:600}"
        ".verdict{margin-top:.45em;padding:.55em .8em;background:#f7f7f9;"
        "border-left:3px solid #999;font-size:.92em}"
        ".v-yes{border-left-color:#2e7d32}.v-no{border-left-color:#c62828}"
        "@media (prefers-color-scheme:dark){body{background:#151517;color:#e8e8e8}"
        ".q{background:#232326;border-left-color:#888}"
        ".quote{background:#1d1d20;border-left-color:#555}"
        ".verdict{background:#1d1d20}.item{border-bottom-color:#333}"
        "h2{border-bottom-color:#555}}</style>",
        f"<h1>{escape(title)}</h1>",
        "<p>For each item: <b>does the sentence show that this paper does the thing "
        "described?</b> Answer yes / no / unsure in the spreadsheet, against the row "
        "number.</p>",
        "<p>Each item has a collapsed line at the bottom: <b>open it only after you "
        "have written your own answer</b>. It shows what our model concluded and why, "
        "so you can see where we disagree — but reading it first would turn your "
        "answer into a copy of ours.</p>",
        "<p>Some items say <b>we found no evidence</b>. For those we list the closest "
        "sentences we did find — if none of them shows it, answer no; if one plainly "
        "does, we missed it, which is just as useful to know.</p>",
    ]
    for qid in sorted(by_q):
        group = by_q[qid]
        parts.append(f"<h2>{escape(qid)} — {len(group)} items</h2>")
        parts.append(f"<div class='q'>{escape(group[0]['question'])}</div>")
        for i in group:
            if i["quote"]:
                quote = f"<div class='quote'>{escape(i['quote'])}</div>"
            elif i.get("candidates"):
                inner = "".join(f"<div class='quote'>{escape(c)}</div>"
                                for c in i["candidates"])
                quote = ("<div class='none'>We found no evidence. The closest "
                         "sentences in this paper were:</div>" + inner)
            else:
                quote = ("<div class='quote none'>We found no evidence, and nothing "
                         "in this paper looked close.</div>")
            # Behind a <details> so it cannot be read by accident. Opening it
            # before deciding would make the reviewer's verdict a copy of ours,
            # which is the one thing this exercise cannot afford.
            if i["_machine"]:
                verdict = "supports the claim" if i["_judge"] else "does NOT support the claim"
                klass = "v-yes" if i["_judge"] else "v-no"
                why = escape(i["_why"]) if i["_why"] else "(no reason recorded)"
                reveal = (
                    "<details><summary>open only after you have decided — "
                    "what our model said</summary>"
                    f"<div class='verdict {klass}'><b>{verdict}</b><br>{why}"
                    f"<br><span style='color:#888'>found by: {escape(i['_by'])}</span>"
                    "</div></details>")
            else:
                reveal = ("<details><summary>open only after you have decided — "
                          "what our model said</summary>"
                          "<div class='verdict v-no'><b>found no evidence in this paper"
                          "</b><br>The sentences above are the closest matches we could "
                          "retrieve, not something the model chose.</div></details>")
            parts.append(
                f"<div class='item'><span class='row'>#{i['row']}</span> "
                f"<span class='paper'>{escape(i['paper_id'])}</span>{quote}{reveal}</div>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def condition_items(rows: list[dict], gold: set, conn=None,
                    limit: int = 16) -> list[dict]:
    """Review items at the CONDITION level, for a multi-condition question.

    gf-01 asks three things at once and scores 4/18. Splitting it helped barely
    (2/18 -> 4/18), and the per-condition tally says why:

        "is this a SEARCH?"      20 yes / 31 no /  9 split
        "uses b-tagged jets?"    25 yes / 24 no / 11 split
        "requires MET?"          16 yes / 39 no /  5 split

    Every condition fails on most papers, so the conjunction was a symptom, not
    the cause. Only 16 of 60 papers judged to require missing transverse momentum
    is not credible for a corpus of LHC searches.

    Asking a reviewer "is this a search AND b-jets AND MET?" cannot separate
    those. Asking "does this paper require missing transverse momentum?" can, and
    it is a question a physicist answers in seconds. So for a paper in the gold
    that we missed, we surface the ONE condition that failed.
    """
    items = []
    for r in rows:
        if r["paper_id"] not in gold or r.get("answer") is True:
            continue
        for cond in r.get("missing", []):
            cands = (candidate_sentences(conn, r["paper_id"], cond)
                     if conn is not None else [])
            items.append({
                "qid": f"{r['qid']}-condition", "question": cond,
                "paper_id": r["paper_id"], "quote": "",
                "candidates": cands,
                "_machine": False, "_judge": None, "_why": "",
                "_by": "none",
            })
            break                     # one condition per paper: the first failure
        if len(items) >= limit:
            break
    return items
