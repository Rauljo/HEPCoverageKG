"""Score the seven value questions of Gabriel's batch 2 by the FACTS STATED.

The label-recall metric ignores numbers (a "60% b-tagging efficiency" label
is matched by any sentence about b-tagging efficiency), so for questions
whose answer IS a number it would score a wrong value as right. Each
question in eval/questions/gabriel-value7.jsonl carries `provenance.facts`:
a list of facts, each a list of alternative patterns; a fact counts as
stated when any alternative occurs in the normalised answer text.

Usage: python eval/analysis/score_value7.py LABEL=JOB [LABEL=JOB ...] [BASE=LABEL]
"""
import glob, json, math, statistics as st, sys
from hepcoveragekg.eval.scoring import _normalise_text

QS = {q["qid"]: q for q in (json.loads(l) for l in open("eval/questions/gabriel-value7.jsonl"))}


def load(job):
    out = []
    for i, part in enumerate(str(job).split("+")):
        hits = sorted(glob.glob(f"eval/runs/dias/*-{part.strip()}.jsonl")) or sorted(glob.glob(f"eval/runs/*-{part.strip()}.jsonl"))
        if not hits:
            raise SystemExit(f"no run file for job {part.strip()}")
        for line in open(hits[-1]).read().splitlines()[1:]:
            rec = json.loads(line); rec["repeat"] = rec.get("repeat", 0) + i * 1000; out.append(rec)
    return [r for r in out if r["qid"] in QS]


def facts_stated(text: str, facts) -> float:
    t = " " + _normalise_text(text) + " "
    return sum(1 for f in facts if any((" " + alt.strip() + " ") in t or alt.strip() in t for alt in f)) / len(facts)


def main(argv):
    base = None; arms = []
    for a in argv:
        k, v = a.split("=", 1)
        if k == "BASE": base = v
        else: arms.append((k, v))
    per = {}
    print(f"{'arm':20s} {'facts':>6s} {'silent':>6s} {'rounds':>6s} {'calls':>5s} {'n':>3s}")
    for label, job in arms:
        R = load(job); f = []; silent = 0; rd = []; calls = []
        for r in R:
            a = r["answer"]; q = QS[r["qid"]]
            s = facts_stated(a.get("text") or "", q["provenance"]["facts"])
            f.append(s); silent += (not (a.get("text") or "").strip()); rd.append(a.get("rounds", 0)); calls.append(a.get("llm_calls", 0))
            per.setdefault(label, {})[(r["qid"], r["repeat"])] = s
        m = lambda v: st.mean(v) if v else float("nan")
        print(f"{label:20s} {m(f):6.2f} {silent:6d} {m(rd):6.2f} {m(calls):5.1f} {len(R):3d}")
    if base in per:
        for label in per:
            if label == base: continue
            d = [per[label][k] - per[base][k] for k in per[label] if k in per[base]]
            if len(d) > 1:
                print(f"  {label:18s} vs {base}: {st.mean(d):+.3f} (se {st.stdev(d)/math.sqrt(len(d)):.3f}, n={len(d)})")
    print("\nper question (facts stated, mean over repeats):")
    qids = sorted(QS)
    print(f"{'q':14s}" + "".join(f"{l[:10]:>12s}" for l, _ in arms))
    for q in qids:
        row = []
        for label, _ in arms:
            v = [s for (qq, _), s in per.get(label, {}).items() if qq == q]
            row.append(f"{(st.mean(v) if v else float('nan')):12.2f}")
        print(f"{q[8:]:14s}" + "".join(row))


if __name__ == "__main__":
    main(sys.argv[1:])
