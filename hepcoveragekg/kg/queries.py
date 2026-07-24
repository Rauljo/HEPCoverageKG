# =============================================================================
# HEPCoverageKG: read-side queries over the store
#
# Minimal for now -- just what the acceptance suite (the 7 contract pass
# conditions) needs: import_counts for the count checks, and trace_assertion for
# the evidence-trace condition. The trace is the seed of the milestone-2 query
# layer and gets extended there. See vault/ideas/bundle-importer-design.md.
# =============================================================================
from __future__ import annotations

import json
from typing import Any, Optional

# status that counts as "accepted knowledge" -- the accepted_view filter.
ACCEPTED_STATUS = "expert_accepted"


def import_counts(conn) -> dict[str, int]:
    """Record/view counts, keyed to match examples/integration/expected-import-counts.json."""

    def one(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        "assertions": one("SELECT COUNT(*) FROM assertion"),
        "accepted_assertions": one("SELECT COUNT(*) FROM accepted_view"),
        "superseded_assertions": one("SELECT COUNT(*) FROM assertion WHERE status = 'superseded'"),
        "entities": one("SELECT COUNT(*) FROM entity"),
        "evidence": one("SELECT COUNT(*) FROM evidence"),
        "expert_decisions": one("SELECT COUNT(*) FROM expert_decision"),
        "source_blocks": one("SELECT COUNT(*) FROM source_block"),
        "source_snapshots": one("SELECT COUNT(*) FROM source_snapshot"),
    }


def status_counts(conn) -> dict[str, int]:
    """Assertions grouped by lifecycle status (the milestone-1 count target)."""
    return dict(conn.execute("SELECT status, COUNT(*) FROM assertion GROUP BY status"))


def trace_assertion(conn, assertion_id: str) -> Optional[dict[str, Any]]:
    """Follow an assertion to its source: subject, predicate, object, and every
    backing quote with its block, section, and paper. Returns None if unknown.

    This is contract condition 5 ("evidence trace returns exact source/block/quote").
    """
    row = conn.execute(
        "SELECT assertion_id, paper_id, subject_id, predicate, family, object_id,"
        " object_value, signature, status, support FROM assertion WHERE assertion_id = ?",
        (assertion_id,),
    ).fetchone()
    if row is None:
        return None

    evidence = conn.execute(
        "SELECT e.evidence_id, e.quote, e.quote_hash, e.char_start, e.char_end,"
        " e.section_title, e.source_hash, e.block_id, b.kind AS block_kind, b.text AS block_text"
        " FROM assertion_evidence ae"
        " JOIN evidence e ON e.evidence_id = ae.evidence_id"
        " JOIN source_block b ON b.source_hash = e.source_hash AND b.block_id = e.block_id"
        " WHERE ae.assertion_id = ?"
        " ORDER BY e.evidence_id",
        (assertion_id,),
    ).fetchall()

    return {
        "assertion_id": row["assertion_id"],
        "paper_id": row["paper_id"],
        "subject_id": row["subject_id"],
        "predicate": row["predicate"],
        "family": row["family"],
        "object_id": row["object_id"],
        "object_value": json.loads(row["object_value"]) if row["object_value"] else None,
        "signature": json.loads(row["signature"]) if row["signature"] else None,
        "status": row["status"],
        "support": row["support"],
        "evidence": [dict(e) for e in evidence],
    }
