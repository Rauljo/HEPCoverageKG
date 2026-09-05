"""Trim SQL-derived gold to papers a stored quote actually backs.

WHY THIS EXISTS. Tier B's truth is `concept_papers` -- a SQL count of papers
whose assertions mention a given entity. Checked against the evidence table on
2026-09-02: 65% of (paper, concept) pairs in Tier B set questions carry a
supporting quote, 35% carry NONE. In three spot-checked questions, the pattern
was the same each time -- a "2 paper" gold turned out to be one paper with a
quote and one paper with nothing behind it at all.

WHAT THIS DOES NOT PROVE. A quote's presence is necessary, not sufficient: it
shows an assertion exists and was extracted from real text, not that the text
genuinely supports the specific claim in the label. A structural spot-check on
2026-09-03 found the sampled quotes DID connect correctly (one appeared wrong on
a naive keyword match and turned out to be the same physics quantity in garbled
LaTeX). So this filter catches the cheap, common failure -- a claim manufactured
with no textual backing at all -- and is not a substitute for a human reading
the paper. Treat a filtered question as "worth spending an arm on", not as
"verified truth" on Gabriel's level.

THE OTHER HALF, RECALL, IS NOT WHAT THIS CHECKS. A phrase-level search on
2026-09-03 found only 1 of 31 distinctive concept labels had a paper mentioning
the full phrase that our gold had missed, and that one was a generic phrase
("photon energy scale and...") plausibly shared by unrelated systematics. So
under-coverage looks rare here, but this module does not measure it -- it only
removes papers whose claim has NOTHING behind it, never adds a paper back in.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def supported_papers(conn: sqlite3.Connection, entity_id: str,
                     candidates: list[str]) -> set[str]:
    """Of `candidates`, the ones with >=1 evidence quote linking them to `entity_id`."""
    if not candidates:
        return set()
    placeholders = ",".join("?" * len(candidates))
    rows = conn.execute(
        f"""SELECT DISTINCT a.paper_id FROM assertion a
           JOIN assertion_evidence ae ON ae.assertion_id = a.assertion_id
           WHERE a.paper_id IN ({placeholders})
             AND (a.subject_id = ? OR a.object_id = ?)""",
        (*candidates, entity_id, entity_id),
    ).fetchall()
    return {r[0] for r in rows}


def backfill_retrieval_papers(conn: sqlite3.Connection, q: dict,
                             min_papers: int = 3) -> Optional[dict]:
    """Fill in `truth.papers` for a retrieval-bank question from the graph.

    THE BUG THIS FIXES. Retrieval-bank questions were built to test entity
    lookup -- "does the string 'Pythia 8.212' resolve to the right id" -- and
    their truth reflects that: `kind="entity"`, `items=[entity_id]`,
    `papers=[]`. The question TEXT asks "which analyses used...", which reads
    as a paper-set question, but every paper-based scorer (`set_f1`,
    `judged_set_f1`) requires `papers` and finds it empty, so none of them ever
    score. Checked 2026-09-02: true of all 720 in the bank. They were never
    broken for `entity_retrieved` (which reads `items`, not `papers`) -- only
    for anything that grades a paper set.

    ONLY WORTH DOING FOR MULTI-PAPER CONCEPTS. 447 of 720 (62%) resolve to
    exactly one paper; a one-paper gold is all-or-nothing and, per the D-096
    screen, contributes almost no discriminating power. `min_papers=3` keeps
    the ~128 where a system can be partially right.

    Uses evidence-backed papers only (same query as `supported_papers`), so a
    backfilled question starts already evidence-filtered rather than needing a
    second pass.
    """
    provenance = q.get("provenance") or {}
    entity_id = provenance.get("entity_id")
    if not entity_id:
        return None
    rows = conn.execute(
        """SELECT DISTINCT a.paper_id FROM assertion a
          JOIN assertion_evidence ae ON ae.assertion_id = a.assertion_id
          WHERE a.subject_id = ? OR a.object_id = ?""",
        (entity_id, entity_id),
    ).fetchall()
    papers = sorted({r[0] for r in rows})
    if len(papers) < min_papers:
        return None
    out = json.loads(json.dumps(q))
    out["shape"] = "set"
    out["truth"] = {"kind": "set", "papers": papers, "value": len(papers)}
    out["provenance"] = dict(provenance, backfilled_from="verify_truth.backfill_retrieval_papers",
                            evidence_filter="already evidence-backed by construction")
    return out


def trim_question(conn: sqlite3.Connection, q: dict) -> Optional[dict]:
    """Return `q` with its gold trimmed to evidence-backed papers, or None if
    that empties it. Only touches questions that name an `entity_id` and carry
    a paper-shaped truth -- anything else (Gabriel's human gold, counts) passes
    through untouched, because this filter has nothing to check there.
    """
    truth = q.get("truth") or {}
    provenance = q.get("provenance") or {}
    entity_id = provenance.get("entity_id")
    papers = truth.get("papers")
    if not entity_id or not papers:
        return q
    if q.get("truth_source") == "gabriel":
        return q  # human-verified already; this filter would be a downgrade

    keep = supported_papers(conn, entity_id, papers)
    dropped = [p for p in papers if p not in keep]
    if not keep:
        return None

    out = json.loads(json.dumps(q))  # deep copy
    out["truth"]["papers"] = sorted(keep)
    if out["truth"].get("universe"):
        # universe papers that were never supported are not evidence of a real
        # NEGATIVE either -- drop them from the universe too, don't silently
        # convert "no evidence found" into "confirmed absent".
        uni = set(out["truth"]["universe"])
        out["truth"]["universe"] = sorted(uni & (keep | (uni - set(papers))))
    if out["truth"].get("value") is not None:
        out["truth"]["value"] = len(keep)
    out["provenance"] = dict(provenance, evidence_trimmed=dropped,
                            evidence_filter="verify_truth.trim_question")
    return out


def trim_file(conn: sqlite3.Connection, in_path: str | Path,
             out_path: str | Path) -> dict:
    """Trim every question in `in_path`, write survivors to `out_path`.

    Returns counts: kept unchanged, trimmed (still has >=1 paper), dropped
    (emptied entirely). A question with NO entity_id/papers (already-clean
    banks, count-only questions) passes through under "unchanged" -- this
    function is a no-op for anything it cannot check.
    """
    kept_unchanged = trimmed = dropped = 0
    out_lines = []
    for line in Path(in_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        result = trim_question(conn, q)
        if result is None:
            dropped += 1
            continue
        if result.get("provenance", {}).get("evidence_trimmed"):
            trimmed += 1
        else:
            kept_unchanged += 1
        out_lines.append(json.dumps(result, ensure_ascii=False))
    Path(out_path).write_text("\n".join(out_lines) + ("\n" if out_lines else ""),
                              encoding="utf-8")
    logger.info(f"{in_path}: {kept_unchanged} unchanged, {trimmed} trimmed, "
               f"{dropped} dropped (no evidence-backed paper left)")
    return {"unchanged": kept_unchanged, "trimmed": trimmed, "dropped": dropped}
