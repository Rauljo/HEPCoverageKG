import json, sqlite3, glob, collections, pickle, sys, os
from hepcoveragekg.eval import judge_gold as JG
from hepcoveragekg.query import answer_critic as AC
# Directory of screened run files; override with SCREEN_DIR.
S = os.environ.get("SCREEN_DIR", "eval/analysis/screen")
qs=[json.loads(l) for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")]
pool=collections.defaultdict(set)
for f in sorted(glob.glob(f"{S}/*.jsonl")):
    for line in open(f, encoding="utf-8"):
        if line.startswith('{"_meta"'): continue
        try: r=json.loads(line)
        except Exception: continue
        if not r.get("qid","").startswith("gabriel-"): continue
        pool[r["qid"]].update((r.get("answer") or {}).get("entity_ids") or [])
conn=sqlite3.connect("file:data/processed/hepkg.db?mode=ro", uri=True)
conn.row_factory=sqlite3.Row
def evidence_for(qid, papers):
    return AC.evidence_by_paper(conn, sorted(pool.get(qid, [])), papers)
pairs=JG.pairs_from(qs, evidence_for)
with open(sys.argv[1],"wb") as fh: pickle.dump(pairs, fh)
print("pairs:", len(pairs))
