"""Score constrained-selection arms on the SELECTED LIST alone (D-161).

The scorers read every arXiv id in the answer text, which is the draft the
planner wrote plus the `Papers:` line the constrained step appended. Under
constrained selection the appended list is the system's declared paper set;
the draft's ids are incidental (D-158 addendum 4: the search-critic arm's
draft alone added ten wrong ids per answer). This scores each record on
`constrained_ids` minus whatever the answer critic struck.

Usage: python eval/analysis/score_constrained_list.py LABEL=JOB [LABEL=JOB ...]
       [BASE=LABEL] [FILTER=<qid prefix>]
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
    """Records for one job, or for several pooled with `+`.

    Pooling matters: the wave-2 control is two jobs of one repeat each, and
    scoring only one of them against a two-repeat arm compares 84 answers with
    168 and moves the control by 0.03 (0.118 vs 0.146 on the two repeats).
    """
    out = []
    for i, part in enumerate(str(job).split("+")):
        hits = sorted(glob.glob(f"eval/runs/dias/*-{part.strip()}.jsonl"))
        if not hits:
            raise SystemExit(f"no run file for job {part.strip()}")
        for line in open(hits[-1]).read().splitlines()[1:]:
            rec = json.loads(line)
            # Two one-repeat jobs both call their records repeat 0, so a
            # (question, repeat) key collides and half the pairs vanish from
            # the paired comparison. Offset by the job's position.
            rec["repeat"] = rec.get("repeat", 0) + i * 1000
            out.append(rec)
    return out


def listset(a):
    s = set(map(str, a.get("constrained_ids") or []))
    return s - {str(d["id"]) for d in (a.get("answer_review") or {}).get("dropped_papers") or []}


def scores_on_text(records) -> bool:
    """True when this run has no selected list anywhere, so its answers must be
    read from the text instead.

    A control that predates constrained selection carries no `constrained_ids`
    at all, and scoring it by the list rule gave F1 0.000 with an undefined
    precision -- a whole column of zeros that looked like a result. The test is
    run-wide on purpose: inside a constrained run, a record whose selector
    chose nothing scored zero and must keep scoring zero.
    """
    return not any((r.get("answer") or {}).get("constrained_ids") for r in records)


def answer_set(a, on_text: bool) -> set:
    return set(ARX.findall(a.get("text") or "")) if on_text else listset(a)


def prf(named, gold, universe=None):
    ju = named & universe if universe else named
    tp = len(named & gold)
    p = tp / len(ju) if ju else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(argv):
    base = None
    prefix = ""
    arms = []
    for a in argv:
        k, v = a.split("=", 1)
        if k == "BASE":
            base = v
        elif k == "FILTER":
            # A run file can hold several question types (the 164-question set
            # mixes retrieval, per-paper, counts and the supervisor's nine).
            # Comparing it with an 84-question run without this filter scores
            # different populations and reads as a difference between arms.
            prefix = v
        else:
            arms.append((k, v))
    m = lambda v: st.mean(v) if v else float("nan")
    per = {}
    print(f"{'arm':22s} {'F1':>6s} {'P':>5s} {'R':>5s} {'size':>5s} {'right':>6s} {'wrong':>6s} {'n':>4s}")
    for label, job in arms:
        R = [r for r in load(job) if r["qid"].startswith(prefix)] if prefix else load(job)
        on_text = scores_on_text(R)
        f1 = []; P = []; Rc = []; sz = []; ri = []; wr = []
        per[label] = {}
        for r in R:
            a = r["answer"]; L = answer_set(a, on_text)
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
        tag = " (text)" if on_text else ""
        print(f"{label + tag:22s} {m(f1):6.3f} {m(P):5.2f} {m(Rc):5.2f} {m(sz):5.1f} {m(ri):6.1f} {m(wr):6.1f} {len(f1):4d}")
    if base and base in per:
        for label in per:
            if label == base:
                continue
            shared = [k for k in per[label] if k in per[base]]
            how = "by record"
            if len(shared) < min(len(per[label]), len(per[base])):
                # The two arms do not share repeat labels (a pooled control of
                # two one-repeat jobs against a two-repeat arm). Pairing by
                # record would silently drop half the data, so pair the
                # per-question means instead and say so.
                how = "by question"
                qa, qb = {}, {}
                for (qid, _), v in per[label].items():
                    qa.setdefault(qid, []).append(v)
                for (qid, _), v in per[base].items():
                    qb.setdefault(qid, []).append(v)
                d = [st.mean(qa[q]) - st.mean(qb[q]) for q in qa if q in qb]
            else:
                d = [per[label][k] - per[base][k] for k in shared]
            if len(d) > 1:
                print(f"  {label:20s} vs {base}: {st.mean(d):+.3f} "
                      f"(se {st.stdev(d) / math.sqrt(len(d)):.3f}, n={len(d)}, {how})")


if __name__ == "__main__":
    main(sys.argv[1:])
