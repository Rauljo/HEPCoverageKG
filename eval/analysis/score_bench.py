"""Small-truth bench (D-144): arms on the same questions x repeats, set_f1 read
the D-142 way -- named-only with the footprint-fallback rate -- plus how many
ids each arm names and what the grade strike removed.
Usage: score_bench.py LABEL=RUN.jsonl [LABEL=RUN.jsonl ...]"""
import json, sys, statistics, re
ARX = re.compile(r"\b\d{4}\.\d{4,5}\b")
def load(f): return [json.loads(l) for l in open(f).read().splitlines()[1:]]
arms = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
def m(v): return statistics.mean(v) if v else float("nan")
common = set.intersection(*(set(r["qid"] for r in recs) for _, recs in arms))
print(f"paired questions {len(common)}")
print(f"{'arm':14s} {'n':>3s} {'set_f1':>7s} {'named-only':>10s} {'prec':>6s} {'rec':>6s} {'fallback':>9s} {'essay':>6s} {'ids':>5s} {'struck':>6s} {'reach':>6s} {'errs':>4s}")
for name, recs in arms:
    rs = [r for r in recs if r["qid"] in common and r["scores"].get("set_f1") is not None]
    named = [r for r in rs if r["scores"].get("set_named_none") != 1.0]
    ids = [len(ARX.findall(r["answer"].get("text") or "")) for r in rs]
    struck = [len(r["answer"].get("grade_struck") or []) for r in rs]
    errs = sum(1 for r in recs if r["qid"] in common and r["answer"].get("error"))
    essay = m([r["scores"]["set_f1"] if r["scores"].get("set_named_none") != 1.0 else 0.0 for r in rs])
    reach = m([r["scores"]["retrieval_reach"] for r in rs if r["scores"].get("retrieval_reach") is not None])
    print(f"{name:14s} {len(rs):3d} {m([r['scores']['set_f1'] for r in rs]):7.3f} {m([r['scores']['set_f1'] for r in named]):10.3f} {m([r['scores']['set_precision'] for r in named]):6.3f} {m([r['scores']['set_recall'] for r in named]):6.3f} {len(rs)-len(named):4d}/{len(rs):<4d} {essay:6.3f} {m(ids):5.1f} {sum(struck):6d} {reach:6.3f} {errs:4d}")
if len(arms) >= 2:
    base = {(r["qid"], r.get("repeat", 0)): r for r in arms[0][1]}
    for name, recs in arms[1:]:
        d = [r["scores"]["set_f1"] - base[(r["qid"], r.get("repeat", 0))]["scores"]["set_f1"] for r in recs
             if (r["qid"], r.get("repeat", 0)) in base and r["scores"].get("set_f1") is not None and base[(r["qid"], r.get("repeat", 0))]["scores"].get("set_f1") is not None]
        if d: print(f"paired set_f1 delta {name} - {arms[0][0]}: {m(d):+.3f}  sd {statistics.pstdev(d):.3f}  n {len(d)}  se {statistics.pstdev(d)/len(d)**0.5:.3f}")
