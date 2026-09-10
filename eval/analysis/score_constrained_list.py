"""Score constrained-selection arms on the SELECTED LIST alone (D-161).

The scorers read every arXiv id in the answer text, which is the draft the
planner wrote plus the `Papers:` line the constrained step appended. Under
constrained selection the appended list is the system's declared paper set;
the draft's ids are incidental (D-158 addendum 4: the search-critic arm's
draft alone added ten wrong ids per answer). This scores each record on
`constrained_ids` minus whatever the answer critic struck.

Usage: python eval/analysis/score_constrained_list.py LABEL=JOB [LABEL=JOB ...]
Jobs are looked up in eval/runs/dias/*-JOB.jsonl. Retrieval questions are
scored against the exact truth set; Gabriel's questions against his verdicts
(precision on the judged set, recall over gold). Pass BASE=LABEL to get paired
deltas against that arm.
"""
import glob, json, math, re, statistics as st, sys

ARX = re.compile(r"\b\d{4}\.\d{4,5}\b")
QS = {q["qid"]: q for q in (json.loads(l) for l in open("eval/questions/discriminating-2026-09-03.jsonl"))}
G = {json.loads(l)["qid"]: json.loads(l) for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")}


def load(job):
    f = sorted(glob.glob(f"eval/runs/dias/*-{job}.jsonl"))[-1]
    return [json.loads(l) for l in open(f).read().splitlines()[1:]]


def listset(a):
    s = set(map(str, a.get("constrained_ids") or []))
    return s - {str(d["id"]) for d in (a.get("answer_review") or {}).get("dropped_papers") or []}


def prf(named, gold, universe=None):
    ju = named & universe if universe else named
    tp = len(named & gold)
    p = tp / len(ju) if ju else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(argv):
    base = None
    arms = []
    for a in argv:
        k, v = a.split("=", 1)
        if k == "BASE":
            base = v
        else:
            arms.append((k, v))
    m = lambda v: st.mean(v) if v else float("nan")
    per = {}
    print(f"{'arm':22s} {'F1':>6s} {'P':>5s} {'R':>5s} {'size':>5s} {'right':>6s} {'wrong':>6s} {'n':>4s}")
    for label, job in arms:
        R = load(job)
        f1 = []; P = []; Rc = []; sz = []; ri = []; wr = []
        per[label] = {}
        for r in R:
            a = r["answer"]; L = listset(a)
            if r["qid"] in G:
                q = G[r["qid"]]; g = set(q["truth"]["papers"]); u = set(q["truth"].get("universe") or []) | g
                p, rc, f = prf(L, g, u); wrong = len((L & u) - g)
            elif r["qid"] in QS and QS[r["qid"]]["truth"].get("papers") is not None:
                g = set(QS[r["qid"]]["truth"]["papers"]); p, rc, f = prf(L, g); wrong = len(L - g)
            else:
                continue
            f1.append(f); sz.append(len(L)); ri.append(len(L & g)); wr.append(wrong); per[label][(r["qid"], r["repeat"])] = f
            if L:
                P.append(p); Rc.append(rc)
        print(f"{label:22s} {m(f1):6.3f} {m(P):5.2f} {m(Rc):5.2f} {m(sz):5.1f} {m(ri):6.1f} {m(wr):6.1f} {len(f1):4d}")
    if base and base in per:
        for label in per:
            if label == base:
                continue
            d = [per[label][k] - per[base][k] for k in per[label] if k in per[base]]
            if len(d) > 1:
                print(f"  {label:20s} vs {base}: {st.mean(d):+.3f} (se {st.stdev(d) / math.sqrt(len(d)):.3f}, n={len(d)})")


if __name__ == "__main__":
    main(sys.argv[1:])
