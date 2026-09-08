"""Class D on the offline bench: does indexing quotes / values make the
quote-only misses reachable? Concept queries, limit 60, no LLM. 2x2 over
{values, quotes}."""
import json, sys, collections
from hepcoveragekg.query import retrieve, templates
SP=sys.argv[1]
conn=templates.read_only(f"{SP}/wave1/hepkg-dias.db")
GOLD={json.loads(l)["qid"]:{str(p) for p in json.loads(l)["truth"]["papers"]} for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")}
Q={"gabriel-gf-01":["b-tagged jet","missing transverse momentum"],"gabriel-gf-01-condition":["b-tagged jet","b-jet","b-hadron"],
   "gabriel-gf-01-met":["missing transverse momentum","ETmiss","pTmiss"],"gabriel-gf-02":["ABCD method","sideband","matrix method"],
   "gabriel-gf-03":["HistFitter"],"gabriel-gf-04":["unfolding","particle level","TUnfold"],
   "gabriel-gf-05":["Higgs boson candidate","Higgs candidate","H->bb candidate","diphoton candidate"],
   "gabriel-gf-07":["ttZ background","ttZ control region","ttbar+Z"],"gabriel-gf-08":["exactly two electrons","exactly two muons","dielectron","dimuon"]}
QUOTE_ONLY = json.load(open(f"{SP}/wave1/gabriel_replay.json"))
qo=set()
for k,d in QUOTE_ONLY.items():
    if k.startswith("ctrl-a|"):
        for p in d["detail"]["in_graph_not_reached"]: qo.add((k.split("|")[1],p))
def papers(ids): return {r[0] for r in conn.execute(f"SELECT DISTINCT paper_id FROM entity_occurrence WHERE entity_id IN ({','.join('?'*len(ids))})", tuple(ids))} if ids else set()
print("%-22s%8s%10s%10s%12s   %s" % ("index","forms","reach","of","D-118 misses","(gold papers never reached live, ctrl-a)"))
for vals,quotes in ((False,False),(True,False),(False,True),(True,True)):
    idx=retrieve.build(conn, cache=f"{SP}/wave1/index-dias-v{int(vals)}q{int(quotes)}.npz", include_values=vals, include_quotes=quotes)
    tot=hit=0; rec=0
    for q,qs in Q.items():
        ids=set()
        for s in qs: ids|={h.entity_id for h in retrieve.search(idx, s, conn=conn, limit=60)}
        r=papers(ids); g=GOLD[q]; tot+=len(g); hit+=len(g&r)
        rec+=sum(1 for (qq,p) in qo if qq==q and p in r)
    n=len(getattr(idx,"forms",[]) or getattr(idx,"labels",[]) or [])
    print("%-22s%8s%10d%10d%12d/%d" % (f"values={int(vals)} quotes={int(quotes)}", n or "?", hit, tot, rec, len(qo)))
