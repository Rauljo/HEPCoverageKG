"""Paired per-question comparison of two OpenRouter runs on Gabriel's 9.
usage: compare_arms.py <control.jsonl> <arm.jsonl> [label]"""
import json, re, sys, statistics, collections
GOLD={json.loads(l)["qid"]:{str(p) for p in json.loads(l)["truth"]["papers"]} for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")}
ARX=re.compile(r"\b\d{4}\.\d{4,5}\b")
def load(p):
    by=collections.defaultdict(list)
    for l in open(p):
        if l.startswith('{"_meta"'): continue
        r=json.loads(l); a=r['answer']; q=r['qid']
        if a.get('error'): continue
        named=set(ARX.findall(a.get('text') or ''))
        sets={k:len(v) for k,v in (a.get('sets') or {}).items() if not k.endswith('_kept')}
        by[q].append(dict(f1=r['scores'].get('judged_f1'), reach=r['scores'].get('retrieval_reach'),
                          ng=len(named&GOLD[q]), n=len(named), set1=sets.get('set_1',0)))
    return by
C,A=load(sys.argv[1]),load(sys.argv[2]); label=sys.argv[3] if len(sys.argv)>3 else "arm"
m=lambda vs,k: statistics.mean(v[k] for v in vs if v[k] is not None) if any(v[k] is not None for v in vs) else float('nan')
sd=lambda vs,k: statistics.pstdev([v[k] for v in vs if v[k] is not None]) if sum(1 for v in vs if v[k] is not None)>1 else 0.0
print("%-16s%4s%4s │%7s%7s%7s │%7s%7s%7s │%6s%6s │%6s%6s" % ("qid","nC","nA","f1 C","f1 A","Δ","reachC","reachA","Δ","ngC","ngA","set1C","set1A"))
print("─"*100)
allC=[];allA=[]
for q in sorted(GOLD):
    c=C.get(q,[]); a=A.get(q,[])
    if not c or not a: print("%-16s%4d%4d   (missing)" % (q.replace("gabriel-",""),len(c),len(a))); continue
    allC+=c; allA+=a
    print("%-16s%4d%4d │%7.2f%7.2f%+7.2f │%7.2f%7.2f%+7.2f │%6.1f%6.1f │%6.0f%6.0f" % (q.replace("gabriel-",""),len(c),len(a),
          m(c,'f1'),m(a,'f1'),m(a,'f1')-m(c,'f1'), m(c,'reach'),m(a,'reach'),m(a,'reach')-m(c,'reach'), m(c,'ng'),m(a,'ng'), m(c,'set1'),m(a,'set1')))
print("─"*100)
print("%-16s%4d%4d │%7.3f%7.3f%+7.3f │%7.3f%7.3f%+7.3f │%6.1f%6.1f" % ("ALL",len(allC),len(allA),m(allC,'f1'),m(allA,'f1'),m(allA,'f1')-m(allC,'f1'),m(allC,'reach'),m(allA,'reach'),m(allA,'reach')-m(allC,'reach'),m(allC,'ng'),m(allA,'ng')))
print(f"\ncontrol spread (pstdev over records): f1 {sd(allC,'f1'):.3f}  reach {sd(allC,'reach'):.3f}   <- a Δ inside this is noise")
