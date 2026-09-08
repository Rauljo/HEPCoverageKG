"""Re-score the 36 per-paper questions with the fuzzy label match (caveat 5 of
analysis/controls-and-gabriel.md). Usage: rescore_labels.py LABEL=RUN.jsonl ..."""
import json, sys, statistics
from hepcoveragekg.eval.scoring import _label_fuzzy_hit, _normalise_text
Q = {q["qid"]: q for q in (json.loads(l) for l in open("eval/questions/discriminating-2026-09-03.jsonl"))}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    recs = [json.loads(l) for l in open(path).read().splitlines()[1:]]
    strict, fuzzy, retrieved = [], [], []
    for r in recs:
        q = Q.get(r["qid"])
        if not q or q["truth"]["kind"] != "labels": continue
        labels = (q.get("provenance") or {}).get("labels") or []
        if not labels: continue
        text = _normalise_text(r["answer"].get("text") or "")
        strict.append(r["scores"].get("mentioned_label_recall") or 0.0)
        fuzzy.append(sum(1 for l in labels if _label_fuzzy_hit(l, text)) / len(labels))
        retrieved.append(r["scores"].get("retrieved_label_recall") or 0.0)
    print(f"{name:22s} n={len(fuzzy):2d}  retrieved {statistics.mean(retrieved):.3f}  mentioned strict {statistics.mean(strict):.3f}  mentioned FUZZY {statistics.mean(fuzzy):.3f}")
