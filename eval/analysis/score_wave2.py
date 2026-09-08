"""Wave 2: critic ON (wave-1 ctrl-a/b) vs critic OFF (nocritic-a/b); free-SQL a/b floor.
usage: score_wave2.py <dir with wave1 and wave2 jsonl>"""
import json, glob, sys, statistics
D=sys.argv[1]
def load(j):
    f=glob.glob(f"{D}/*{j}.jsonl") or glob.glob(f"{D}/*/*{j}.jsonl")
    d={}
    for l in open(f[0]):
        if l.startswith('{"_meta"'): continue
        r=json.loads(l); d[r['qid']]=r
    return d
ARMS=[("ctrl-a (critic)",54251),("ctrl-b (critic)",54252),("nocritic-a",54257),("nocritic-b",54258),("freesql-a",54259),("freesql-b",54260)]
R={n:load(j) for n,j in ARMS}
ks=sorted(set.intersection(*[set(v) for v in R.values()]))
m=lambda v: statistics.mean(v) if v else float('nan')
def col(n,k): return [R[n][q]['scores'][k] for q in ks if R[n][q]['scores'].get(k) is not None]
print(f"paired questions {len(ks)}\n")
print("%-18s%9s%9s%9s%8s%7s" % ("arm","judged","set_f1","reach","print%","errs"))
for n,_ in ARMS:
    pr=[R[n][q]['scores'].get('answer_names_papers') for q in ks]; pr=[x for x in pr if x is not None]
    print("%-18s%9.3f%9.3f%9.3f%7.0f%%%7d" % (n,m(col(n,'judged_f1')),m(col(n,'set_f1')),m(col(n,'retrieval_reach')),100*m(pr) if pr else 0,sum(1 for q in ks if R[n][q]['answer'].get('error'))))
f=lambda a,b,k: abs(m(col(a,k))-m(col(b,k)))
print(f"\nfloors  typed+critic judged {f('ctrl-a (critic)','ctrl-b (critic)','judged_f1'):.3f} set_f1 {f('ctrl-a (critic)','ctrl-b (critic)','set_f1'):.3f}")
print(f"        typed-nocritic judged {f('nocritic-a','nocritic-b','judged_f1'):.3f} set_f1 {f('nocritic-a','nocritic-b','set_f1'):.3f}")
print(f"        free-sql       judged {f('freesql-a','freesql-b','judged_f1'):.3f} set_f1 {f('freesql-a','freesql-b','set_f1'):.3f}")
on=(m(col('ctrl-a (critic)','judged_f1'))+m(col('ctrl-b (critic)','judged_f1')))/2; off=(m(col('nocritic-a','judged_f1'))+m(col('nocritic-b','judged_f1')))/2
print(f"\nCRITIC EFFECT (typed, judged_f1): on {on:.3f}  off {off:.3f}  delta {on-off:+.3f}")
on=(m(col('ctrl-a (critic)','set_f1'))+m(col('ctrl-b (critic)','set_f1')))/2; off=(m(col('nocritic-a','set_f1'))+m(col('nocritic-b','set_f1')))/2
print(f"CRITIC EFFECT (typed, set_f1):    on {on:.3f}  off {off:.3f}  delta {on-off:+.3f}")
