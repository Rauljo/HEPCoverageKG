"""Rank-aware scoring for the precision-first arm (free-SQL + reranking).

WHY THIS EXISTS. Every metric in the project so far is set-valued: set_f1,
judged_f1, label recall. They ask whether the right papers are in the answer
and do not care where. The routing use case asks something different -- are the
FIRST two or three papers right -- and a set metric cannot see it.

WHAT IT FOUND IMMEDIATELY. free-SQL's `papers` field is sorted by arXiv id
(1703.02649, 1703.02650, 1703.02651, ...), not by relevance. So P@1 on the
list as it ships measures the alphabet. That is the finding that motivates the
arm rather than a flaw in it: the candidates exist, the ordering does not, and
the reranking is the whole experiment.

Usage:
  PYTHONPATH=. python eval/analysis/score_precision_at_k.py LABEL=JOB [...] \
      [FIELD=papers|constrained_ids] [RERANK=none|lexical|dense]
"""
import glob, io, json, math, os, statistics as st, sys

QFILE = os.environ.get("QFILE", "eval/questions/retrieval84.jsonl")
QS = {q["qid"]: q for q in (json.loads(l) for l in io.open(QFILE, encoding="utf-8"))}
KS = (1, 2, 3, 5)


def truth_papers(qid):
    t = (QS.get(qid) or {}).get("truth") or {}
    return set(t.get("papers") or [])


def candidates(answer, field):
    """The ordered list this record declares, longest available first."""
    for key in ([field] if field else []) + ["constrained_ids", "papers", "named_ids"]:
        v = answer.get(key)
        if v:
            return [str(x) for x in v], key
    return [], "none"


def rerank(question, papers, how, conn):
    """Reorder candidates most-relevant-first. `none` keeps the shipped order."""
    if how == "none" or len(papers) < 2:
        return papers
    from hepcoveragekg.query import answer_critic as AC
    # evidence_by_paper() is keyed on ENTITY ids and returns {} for paper ids --
    # the reranker silently ranked nothing and reproduced the shipped order.
    # evidence_by_paper_wide() is the paper-keyed one (D-165/166), built for
    # free-SQL precisely because it retrieves no entities.
    ev = AC.evidence_by_paper_wide(conn, papers)
    words = {w for w in question.lower().split() if len(w) > 3}

    def lexical(pid):
        labels, quotes, aliases = ev.get(pid, (set(), [], set()))
        text = " ".join(list(labels) + [str(q) for q in quotes] + list(aliases)).lower()
        return sum(1 for w in words if w in text)

    if how == "lexical":
        return sorted(papers, key=lambda p: (-lexical(p), p))
    if how == "dense":
        from hepcoveragekg.aliases import semantics
        import numpy as np
        blobs = []
        for pid in papers:
            labels, quotes, aliases = ev.get(pid, (set(), [], set()))
            # QUOTES, NOT JUST LABELS. free-SQL papers come back with 60-200
            # quotes and zero labels, so a blob built from labels alone was
            # "(nothing)" for every paper -- identical embeddings, no reorder.
            parts = sorted(labels) + sorted(aliases) + [str(q) for q in quotes[:40]]
            blobs.append(" ; ".join(parts or ["(nothing)"])[:2000])
        vecs = semantics.embed([question] + blobs)
        q, rest = vecs[0], vecs[1:]
        sims = (rest @ q) / (np.linalg.norm(rest, axis=1) * np.linalg.norm(q) + 1e-9)
        order = sorted(range(len(papers)), key=lambda i: (-float(sims[i]), papers[i]))
        return [papers[i] for i in order]
    raise SystemExit("RERANK must be none, lexical or dense")


def main(argv):
    field = next((a.split("=", 1)[1] for a in argv if a.startswith("FIELD=")), "")
    how = next((a.split("=", 1)[1] for a in argv if a.startswith("RERANK=")), "none")
    arms = [a.split("=", 1) for a in argv if "=" in a and a.split("=", 1)[0] not in
            ("FIELD", "RERANK", "QFILE")]
    conn = None
    if how != "none":
        from hepcoveragekg.query import templates
        conn = templates.read_only(os.environ.get("HEPKG_DB", "data/processed/hepkg.db"))

    print("question file: %s   field: %s   rerank: %s\n"
          % (QFILE, field or "auto", how))
    head = "%-18s" % "arm" + "".join("%8s" % ("P@%d" % k) for k in KS) \
           + "%8s%8s%8s%8s%9s" % ("hit@3", "MRR", "listlen", "empty", "records")
    print(head); print("-" * len(head))
    for name, job in arms:
        f = sorted(glob.glob("eval/runs/dias/*-%s.jsonl" % job)
                   or glob.glob("eval/runs/*-%s.jsonl" % job))[-1]
        pk = {k: [] for k in KS}; hit3 = []; rr = []; lens = []; empty = 0; n = 0
        for ln in io.open(f, encoding="utf-8").read().splitlines()[1:]:
            r = json.loads(ln)
            if r["qid"] not in QS or r["answer"].get("error"):
                continue
            n += 1
            gold = truth_papers(r["qid"])
            if not gold:
                continue
            cand, _ = candidates(r["answer"], field)
            if not cand:
                empty += 1
                for k in KS:
                    pk[k].append(0.0)
                hit3.append(0.0); rr.append(0.0); continue
            cand = rerank(r["question"], cand, how, conn)
            lens.append(len(cand))
            for k in KS:
                top = cand[:k]
                pk[k].append(sum(1 for p in top if p in gold) / max(len(top), 1))
            hit3.append(1.0 if any(p in gold for p in cand[:3]) else 0.0)
            rr.append(next((1.0 / (i + 1) for i, p in enumerate(cand) if p in gold), 0.0))
        print("%-18s" % name + "".join("%8.3f" % st.mean(pk[k]) for k in KS)
              + "%8.3f%8.3f%8.1f%8d%9d" % (st.mean(hit3), st.mean(rr),
                                           st.mean(lens) if lens else 0.0, empty, n))


if __name__ == "__main__":
    main(sys.argv[1:])
