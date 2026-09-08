"""The 2x2: control / gate / answer-critic / both, on the 164-question set."""
import json, glob, statistics, collections, sys

ARMS = [("control", 54232), ("gate", 54233), ("acritic", 54234), ("both", 54235)]
D = {}
for name, j in ARMS:
    f = glob.glob(f"eval/runs/*{j}.jsonl")[0]
    d = {}
    for line in open(f):
        if line.startswith('{"_meta"'):
            continue
        r = json.loads(line)
        d[r["qid"]] = r
    D[name] = d
ks = sorted(set.intersection(*[set(v) for v in D.values()]))
print(f"paired questions: {len(ks)}\n")

def m(vals):
    return statistics.mean(vals) if vals else float("nan")

def col(name, key):
    return [D[name][q]["scores"][key] for q in ks
            if D[name][q]["scores"].get(key) is not None]

print("%-10s%8s%8s%8s%6s%9s%8s%8s%9s" % (
    "arm", "judged", "set_f1", "errors", "n", "printed%", "gate", "critic", "dropped%"))
print("-" * 74)
for name, _ in ARMS:
    jf = col(name, "judged_f1")
    sf = col(name, "set_f1")
    er = m(col(name, "errored"))
    printed = []
    gate = crit = cand = drop = 0
    for q in ks:
        s = D[name][q]["scores"]
        a = D[name][q].get("answer") or {}
        v = s.get("answer_names_papers")
        if v is not None:
            printed.append(v)
        if a.get("gate_retried"):
            gate += 1
        rv = a.get("answer_review") or {}
        if rv:
            crit += 1
            cand += rv.get("candidates", 0)
            drop += rv.get("dropped", 0)
    print("%-10s%8.3f%8.3f%8.3f%6d%9.0f%8d%8d%9.0f" % (
        name, m(jf), m(sf), er, len(jf), 100 * m(printed) if printed else 0,
        gate, crit, 100 * drop / max(cand, 1)))
print("-" * 74)

# the decomposition D-107 asked for
print("\nSCORE = (prints ids) x (quality when it does)\n")
print("%-10s%10s%12s%14s" % ("arm", "score", "print-rate", "when-printed"))
print("-" * 46)
for name, _ in ARMS:
    sc, pr, wp = [], [], []
    for q in ks:
        s = D[name][q]["scores"]
        v = s.get("judged_f1", s.get("set_f1"))
        nn = s.get("answer_names_papers")
        if v is None or nn is None:
            continue
        sc.append(v); pr.append(nn)
        if nn == 1.0:
            wp.append(v)
    print("%-10s%10.3f%12.2f%14.3f" % (name, m(sc), m(pr), m(wp)))

# per pool
print("\nBY QUESTION TYPE\n")
pool = {}
for f, p in (("gabriel-gold-2026-09-03-full", "gabriel"),
             ("retrieval-converted-2026-09-03", "retrieval")):
    for l in open(f"eval/questions/{f}.jsonl"):
        pool[json.loads(l)["qid"]] = p
for p in ("overnight-2026-09-02-verified", "overnight2-2026-09-02-verified"):
    for l in open(f"eval/questions/{p}.jsonl"):
        q = json.loads(l)
        pool.setdefault(q["qid"], q.get("pool"))
by = collections.defaultdict(list)
for q in ks:
    by[pool.get(q, "?")].append(q)
print("%-14s%5s" % ("pool", "n") + "".join("%11s" % n for n, _ in ARMS))
print("-" * (19 + 11 * len(ARMS)))
for p, qs in sorted(by.items(), key=lambda x: -len(x[1])):
    row = "%-14s%5d" % (p, len(qs))
    for name, _ in ARMS:
        vals = []
        for q in qs:
            s = D[name][q]["scores"]
            v = s.get("judged_f1", s.get("set_f1"))
            if v is None:
                v = s.get("mentioned_label_recall", s.get("count_closeness"))
            if v is not None:
                vals.append(v)
        row += "%11.3f" % m(vals)
    print(row)
