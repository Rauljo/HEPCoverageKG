"""Wave 3 on the 164: paired columns, each with its n. set_f1 exists on 97 set
records, count_correct on 20, judged_f1 (Gabriel) on 10 (D-141)."""
import json, glob, sys, statistics
D=sys.argv[1]
def load(j):
    f=glob.glob(f"{D}/*{j}.jsonl") or glob.glob(f"{D}/*/*{j}.jsonl")
    return {r["qid"]:r for r in (json.loads(l) for l in open(f[0]).read().splitlines()[1:])} if f else {}
ARMS=[("ctrl-a w1 (old code)",54251),("ctrl-b w1 (old code)",54252),("nocritic-a w2",54257),("nocritic-b w2",54258),("ctrl-c w3 (merged)",54261),("stack w3 (54268)",54268)]
R={n:load(j) for n,j in ARMS}
def m(v): return statistics.mean(v) if v else float("nan")
def col(n,k,qs): return [R[n][q]["scores"][k] for q in qs if q in R[n] and R[n][q]["scores"].get(k) is not None]
def row(n,qs):
    e=sum(1 for q in qs if q in R[n] and R[n][q]["answer"].get("error"))
    parts=[f"{n:22s} n={len([q for q in qs if q in R[n]]):3d}"]
    for k,lab in (("set_f1","set_f1"),("count_correct","count"),("judged_f1","judged10"),("retrieval_reach","reach"),("answer_names_papers","prints")):
        v=col(n,k,qs); parts.append(f"{lab}={m(v):.3f}(n{len(v):3d})")
    # D-142: set_f1 falls back to the retrieval footprint when the text names
    # no paper. Report the named-only score, its precision/recall, and how
    # often the fallback fired, so the two answers are not one number.
    setq=[q for q in qs if q in R[n] and R[n][q]["scores"].get("set_f1") is not None]
    named=[q for q in setq if R[n][q]["scores"].get("set_named_none")!=1.0]
    if setq:
        fb=len(setq)-len(named)
        parts.append(f"named-only f1={m([R[n][q]['scores']['set_f1'] for q in named]):.3f} p={m([R[n][q]['scores']['set_precision'] for q in named]):.3f} r={m([R[n][q]['scores']['set_recall'] for q in named]):.3f} (n{len(named):3d}) fallback={fb}/{len(setq)} essay-only={m([R[n][q]['scores']['set_f1'] if q in named else 0.0 for q in setq]):.3f}")
    parts.append(f"errs={e}")
    print("  ".join(parts))
ctrl=set.intersection(*(set(R[n]) for n,_ in ARMS[:5] if R[n]))
print(f"CONTROLS, paired on {len(ctrl)} questions")
for n,_ in ARMS[:5]:
    if R[n]: row(n,ctrl)
st="stack w3 (54268)"; cc="ctrl-c w3 (merged)"
if R[st]:
    qs=set(R[st])&set(R[cc])
    print(f"\nSTACK vs ctrl-c, paired on the stack's {len(qs)} questions")
    row(cc,qs); row(st,qs)
    for n in ("ctrl-a w1 (old code)","nocritic-a w2"):
        row(n,qs&set(R[n]))
    d=[(R[st][q]["scores"]["set_f1"]-R[cc][q]["scores"]["set_f1"]) for q in qs if R[st][q]["scores"].get("set_f1") is not None and R[cc][q]["scores"].get("set_f1") is not None]
    if d: print(f"  paired set_f1 delta stack-ctrl: mean {m(d):+.3f}, sd {statistics.pstdev(d):.3f}, n {len(d)}, se {statistics.pstdev(d)/len(d)**0.5:.3f}")
