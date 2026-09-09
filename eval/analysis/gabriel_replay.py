"""Replay each Gabriel question's tool calls against the DIAS DB and decompose
the loss on every gold paper into: named / reached-not-named / in-graph-not-
reached / not-in-graph. Previews were dropped at serialisation, so this is the
only way to know what the tools actually returned."""
import json, glob, re, sys, collections
from hepcoveragekg.query import retrieve, templates, planner
from hepcoveragekg.eval.systems import _papers_of

# Usage: gabriel_replay.py DB.db LABEL=RUN.jsonl [LABEL=RUN.jsonl ...]
DB = sys.argv[1]
RUNS = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[2:]}
GOLD = {json.loads(l)["qid"]: json.loads(l)
        for l in open("eval/questions/gabriel-gold-2026-09-03-full.jsonl")}
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")

conn = templates.read_only(DB)
index = retrieve.build(conn, cache=DB.rsplit("/",1)[0]+"/index-dias.npz")
in_db = {r[0] for r in conn.execute("SELECT arxiv_id FROM paper")}

def papers_from(result):
    """Papers a tool result touched: direct paper_id columns, else via entities."""
    direct = {str(r["paper_id"]) for r in result.rows
              if isinstance(r, dict) and r.get("paper_id")}
    ents = planner._collect_ids(result.rows, result.note)
    via_ent = set(_papers_of(conn, sorted(ents))) if ents else set()
    return direct, via_ent, ents

KW = {"gabriel-gf-01": ["b-tag","bjet","b-jet","b jet","missing transverse","etmiss","ptmiss","met"],
      "gabriel-gf-01-condition": ["b-tag","bjet","b-jet","b jet","b-hadron"],
      "gabriel-gf-01-met": ["missing transverse","etmiss","ptmiss"," met"],
      "gabriel-gf-02": ["abcd","sideband","matrix method"],
      "gabriel-gf-03": ["histfitter"],
      "gabriel-gf-04": ["unfold"],
      "gabriel-gf-05": ["higgs candidate","h candidate","higgs boson candidate","h->bb","h→bb","diphoton candidate","higgs"],
      "gabriel-gf-07": ["ttz","tt+z","ttbar+z","tt z"],
      "gabriel-gf-08": ["two electrons","two muons","dielectron","dimuon","ee channel","mumu","μμ","same-flavour"]}
def why_missed(q, paper):
    kws = KW.get(q, [])
    lab = conn.execute("SELECT label FROM entity_occurrence WHERE paper_id=?", (paper,)).fetchall()
    labels = [str(r[0]).lower() for r in lab if r[0]]
    if any(k in l for l in labels for k in kws): return "entity-label-present"
    qs = conn.execute("""SELECT ev.quote FROM evidence ev JOIN assertion_evidence ae ON ae.evidence_id=ev.evidence_id
                         JOIN assertion a ON a.assertion_id=ae.assertion_id WHERE a.paper_id=?""", (paper,)).fetchall()
    if any(k in str(r[0]).lower() for r in qs if r[0] for k in kws): return "quote-only"
    return "concept-absent"
out = {}
for run, path in RUNS.items():
    for line in open(path):
        if line.startswith('{"_meta"'): continue
        r = json.loads(line); q = r["qid"]
        if q not in GOLD: continue
        a = r["answer"]; g = GOLD[q]["truth"]
        if a.get("error"):
            print(f"  skipping {run} {q}: errored -> {a['error'][:60]}"); continue
        gold = {str(p) for p in g["papers"]}; uni = {str(p) for p in g.get("universe") or []}
        sets = {}
        ex = planner.build_executor(conn, index, sets, critic=None)
        reached = set(); instrumented_ents = set(); per_tool = collections.defaultdict(set); errs = []
        for s in a.get("steps") or []:
            args = {k: (re.sub(r"_kept$", "", v) if isinstance(v, str) and v.endswith("_kept") else v)
                    for k, v in (s["args"] or {}).items()}
            try:
                res = ex(s["tool"], args)
            except Exception as e:
                errs.append(f'{s["tool"]}: {type(e).__name__}: {str(e)[:60]}'); continue
            direct, via_ent, ents = papers_from(res)
            per_tool[s["tool"]] |= direct | via_ent
            reached |= direct | via_ent
            instrumented_ents |= ents
        footprint = set(_papers_of(conn, sorted(instrumented_ents)))   # what the scorer calls a.papers
        text = a.get("text") or ""
        named = set(ARXIV.findall(text)) or (set(a.get("papers") or []) if a.get("cited") else set())
        named_in_uni = named & uni
        dec = {
            "named_correct":        sorted(gold & named),
            "reached_not_named":    sorted((gold & reached) - named),
            "in_graph_not_reached": sorted((gold & in_db) - reached),
            "not_in_graph":         sorted(gold - in_db),
            "false_positives":      sorted(named_in_uni - gold),
            "named_outside_universe": sorted(named - uni),
        }
        why = collections.Counter(why_missed(q, p) for p in dec["in_graph_not_reached"])
        out[(run, q)] = {"why_missed": dict(why), "gold": len(gold), "universe": len(uni),
                         "reached_true": len(gold & reached), "reached_instrumented": len(gold & footprint),
                         "per_tool": {t: len(gold & ps) for t, ps in per_tool.items()},
                         "exit": a.get("stopped_because"), "judged_f1": r["scores"].get("judged_f1"),
                         "errors": errs, **{k: len(v) for k, v in dec.items()}, "detail": dec}

# ---- print ----
print("GOLD-PAPER LOSS DECOMPOSITION, replayed against the DIAS DB\n")
hdr = "%-8s%-18s%5s%6s%7s%8s%8s%8s%8s%8s   %s"
print(hdr % ("run","qid","gold","named","reach","reach*","r-not-n","miss","noDB","FP","exit"))
print("-"*112)
for (run,q),d in sorted(out.items()):
    print(hdr % (run, q.replace("gabriel-",""), d["gold"], d["named_correct"], d["reached_true"],
                 d["reached_instrumented"], d["reached_not_named"], d["in_graph_not_reached"],
                 d["not_in_graph"], d["false_positives"], (d["exit"] or "")[:30]))
print("\nreach  = gold papers ANY replayed tool touched      reach* = via entity_ids only (what retrieval_reach sees)")
print("r-not-n = reached but not named (answer-stage loss)   miss = in the DB, never reached (retrieval loss)")
print("noDB   = gold paper absent from the graph             FP   = named, in universe, Gabriel said no\n")
tot = collections.Counter()
for d in out.values():
    for k in ("gold","named_correct","reached_not_named","in_graph_not_reached","not_in_graph","false_positives"): tot[k]+=d[k]
print("TOTAL over 18 (9 questions x 2 controls):", dict(tot))
print("\nper-tool reach of gold (which tool actually found them):")
pt = collections.Counter()
for d in out.values():
    for t,n in d["per_tool"].items(): pt[t]+=n
print("  ", dict(pt))
print("\nWHY THE MISSED GOLD PAPERS WERE MISSED (in the DB, never reached by any tool):")
print("  entity-label-present = an entity with the concept exists on that paper -> search/alias/hop miss")
print("  quote-only           = the concept is in a verbatim quote but no entity carries it -> extraction gap")
print("  concept-absent       = nothing on the paper mentions it -> not in the graph as that concept\n")
wt = collections.Counter()
for (run,q),d in sorted(out.items()):
    if d["in_graph_not_reached"]:
        print("  %-8s%-18s miss=%2d  %s" % (run, q.replace("gabriel-",""), d["in_graph_not_reached"], d["why_missed"]))
        wt.update(d["why_missed"])
print("  TOTAL:", dict(wt))
errs = [(k, d["errors"]) for k,d in out.items() if d["errors"]]
if errs:
    print("\nreplay errors:"); [print("  ", k, e) for k,e in errs]
json.dump({f"{r}|{q}": d for (r,q),d in out.items()}, open(DB.rsplit("/",1)[0]+"/gabriel_replay-"+"-".join(RUNS)+".json","w"), indent=1)
