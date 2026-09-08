"""Does search alone reach Gabriel's gold papers? No LLM. Two query sets per
question: the question's own concept words, and what the models actually typed."""
import json, sys, sqlite3, collections
from hepcoveragekg.query import retrieve, templates
SP=sys.argv[1]; LIMIT=int(sys.argv[2]) if len(sys.argv)>2 else 60
conn=templates.read_only(f"{SP}/wave1/hepkg-dias.db")
index=retrieve.build(conn, cache=f"{SP}/wave1/index-dias.npz")
GOLD={json.loads(l)["qid"]:json.loads(l) for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")}
# (concept queries from the question text) , (what the models typed in the controls)
Q={
 "gabriel-gf-01":(["b-tagged jet","missing transverse momentum"],["b-tagged jets","missing transverse momentum"]),
 "gabriel-gf-01-condition":(["b-tagged jet","b-jet","b-hadron"],["b-tagged jets"]),
 "gabriel-gf-01-met":(["missing transverse momentum","ETmiss","pTmiss"],["missing transverse momentum"]),
 "gabriel-gf-02":(["ABCD method","sideband","matrix method"],["ABCD"]),
 "gabriel-gf-03":(["HistFitter"],["HistFitter"]),
 "gabriel-gf-04":(["unfolding","particle level","TUnfold"],["unfolding"]),
 "gabriel-gf-05":(["Higgs boson candidate","Higgs candidate","H->bb candidate","diphoton candidate"],["Higgs"]),
 "gabriel-gf-07":(["ttZ background","ttZ control region","ttbar+Z"],["ttZ"]),
 "gabriel-gf-08":(["exactly two electrons","exactly two muons","dielectron","dimuon"],["exactly two electrons","exactly two muons"]),
}
def papers(ids):
    if not ids: return set()
    m=",".join("?"*len(ids))
    return {r[0] for r in conn.execute(f"SELECT DISTINCT paper_id FROM entity_occurrence WHERE entity_id IN ({m})", tuple(ids))}
print(f"limit={LIMIT}   recall = gold papers reachable through ANY hit's entity\n")
print("%-18s%5s%10s%10s   %s" % ("qid","gold","concept","as-typed","missing after CONCEPT queries (first 4)"))
print("-"*100)
tot=collections.Counter(); miss_all={}
for q,(concept,typed) in Q.items():
    gold={str(p) for p in GOLD[q]["truth"]["papers"]}
    def run(qs):
        ids=set()
        for s in qs: ids|={h.entity_id for h in retrieve.search(index, s, conn=conn, limit=LIMIT)}
        return papers(ids)
    rc=run(concept); rt=run(typed)
    miss=sorted(gold-rc); miss_all[q]=miss
    print("%-18s%5d%9d/%-2d%8d/%-2d   %s" % (q.replace("gabriel-",""),len(gold),len(gold&rc),len(gold),len(gold&rt),len(gold),miss[:4]))
    tot["gold"]+=len(gold); tot["concept"]+=len(gold&rc); tot["typed"]+=len(gold&rt)
print("-"*100)
print("TOTAL  gold=%d   concept-queries reach %d (%.0f%%)   as-typed reach %d (%.0f%%)" % (tot["gold"],tot["concept"],100*tot["concept"]/tot["gold"],tot["typed"],100*tot["typed"]/tot["gold"]))
json.dump(miss_all, open(f"{SP}/wave1/bench_missing.json","w"), indent=1)
