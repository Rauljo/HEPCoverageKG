# =============================================================================
# HEPCoverageKG: single-bundle transactional upsert (pipeline stage 5)
#
# Takes a validated bundle and writes every record into the store, in ONE
# transaction (rollback on any error). Records are inserted parent-first to
# satisfy the 21 enforced foreign keys.
#
# Entities MERGE rather than duplicate: each bundle contributes an
# entity_occurrence row (its own verbatim view), and the single canonical
# `entity` row is RECOMPUTED from the full set of occurrences -- so the merged
# row is a function of the SET of occurrences, not of import order. That
# order-independence is what protects the contract's deterministic export.
# Merged attributes are consensus-only (a key is kept only if every occurrence
# that states it agrees); contested detail stays, per-paper, in the occurrences.
#
# This step does NOT do idempotency or conflict detection -- that is step 6,
# which wraps a guard AROUND this writer (it decides whether the write happens
# at all; it never has to undo it). See vault/ideas/bundle-importer-design.md.
# =============================================================================
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from hepcoveragekg.ingest.canonical import (
    assertion_identity_hash,
    bundle_fingerprint,
    stable_json,
)
from hepcoveragekg.ingest.errors import BundleConflictError
from hepcoveragekg.ingest.reader import load_bundle
from hepcoveragekg.ingest.validate import validate_semantics
from hepcoveragekg.kg import store


@dataclass
class ImportResult:
    """A receipt describing what one import did (not the data itself)."""

    bundle_id: str
    paper_id: str
    inserted: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False  # step 6 sets this True on a byte-identical re-import


# --------------------------------------------------------------------------- #
# Entity merge (pure, order-independent)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MergedEntity:
    kind: str
    label: str
    aliases: list[str]
    attributes: dict[str, Any]
    external_ids: dict[str, Any]


def _mode(values: Sequence[str]) -> str:
    """Most frequent value; ties broken lexicographically (deterministic)."""
    counts = Counter(values)
    top = max(counts.values())
    return sorted(key for key, n in counts.items() if n == top)[0]


def _consensus(dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Keep a key only if every occurrence that states it agrees on the value.

    Contested keys (two papers, two values) are dropped from the merged row --
    they remain, per-paper, in entity_occurrence. Single-paper keys are kept
    (a lone statement does not contradict anyone).
    """
    values_by_key: dict[str, set[str]] = {}
    for mapping in dicts:
        for key, value in mapping.items():
            values_by_key.setdefault(key, set()).add(stable_json(value))
    return {
        key: json.loads(next(iter(values)))
        for key, values in values_by_key.items()
        if len(values) == 1
    }


def merge_entity(occurrences: Sequence[dict[str, Any]]) -> MergedEntity:
    """Canonical entity row from all its occurrences. Pure and order-independent."""
    return MergedEntity(
        kind=_mode([o["kind"] for o in occurrences]),
        label=_mode([o["label"] for o in occurrences]),
        aliases=sorted({a for o in occurrences for a in (o.get("aliases") or [])}),
        attributes=_consensus([o.get("attributes") or {} for o in occurrences]),
        external_ids=_consensus([o.get("external_ids") or {} for o in occurrences]),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _json(value: Any) -> Optional[str]:
    """Canonical JSON, or NULL for None (so the column is NULL, not the text 'null')."""
    return None if value is None else stable_json(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Per-table inserts (parent-first)
# --------------------------------------------------------------------------- #


def _insert_paper(conn, bundle) -> None:
    p = bundle["paper"]
    conn.execute(
        "INSERT INTO paper (arxiv_id, title, category, experiments, doi, inspire_id,"
        " cds_id, latest_bundle_id, source_hash, paper_map)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            p["arxiv_id"], p["title"], p.get("category"), _json(p.get("experiments") or []),
            p.get("doi"), p.get("inspire_id"), p.get("cds_id"),
            bundle["bundle_id"], bundle["source"]["source_hash"], _json(bundle.get("paper_map")),
        ),
    )


def _insert_source_snapshot(conn, bundle) -> None:
    s = bundle["source"]
    conn.execute(
        "INSERT INTO source_snapshot (source_hash, paper_id, normalization_version, title,"
        " experiments, problems, source_path, conversion_qa) VALUES (?,?,?,?,?,?,?,?)",
        (
            s["source_hash"], s["paper_id"], s.get("normalization_version"), s["title"],
            _json(s.get("experiments") or []), _json(s.get("problems") or []),
            s["source_path"], _json(s.get("conversion_qa")),
        ),
    )


def _insert_source_blocks(conn, bundle) -> int:
    source_hash = bundle["source"]["source_hash"]
    rows = [
        (
            source_hash, b["block_id"], b["kind"], b["order"], b.get("section_id"),
            b.get("section_title"), b.get("dom_id"), b["text"], b["text_hash"],
            _json(b.get("table_rows") or []), _json(b.get("attributes") or {}),
        )
        for b in bundle["source"].get("blocks") or []
    ]
    conn.executemany(
        "INSERT INTO source_block (source_hash, block_id, kind, block_order, section_id,"
        " section_title, dom_id, text, text_hash, table_rows, attributes)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_bundle_import(conn, bundle, imported_at) -> None:
    conn.execute(
        "INSERT INTO bundle_import (bundle_id, paper_arxiv_id, source_hash,"
        " bundle_content_hash, schema_version, imported_at, usage, warnings)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            bundle["bundle_id"], bundle["paper"]["arxiv_id"], bundle["source"]["source_hash"],
            bundle_fingerprint(bundle), bundle["schema_version"], imported_at,
            _json(bundle.get("usage") or []), _json(bundle.get("warnings") or []),
        ),
    )


def _upsert_entities(conn, bundle) -> int:
    """Insert this bundle's occurrences, then recompute each merged entity row
    from the full set of occurrences (order-independent)."""
    bundle_id = bundle["bundle_id"]
    paper_id = bundle["paper"]["arxiv_id"]
    entities = bundle.get("entities") or []

    for e in entities:
        # Ensure the canonical row exists (FK target for the occurrence). Provisional
        # values; overwritten by the recompute below. OR IGNORE leaves a shared
        # entity from a prior bundle untouched for now.
        conn.execute(
            "INSERT OR IGNORE INTO entity (entity_id, kind, label, aliases, attributes,"
            " external_ids) VALUES (?,?,?,?,?,?)",
            (
                e["entity_id"], e["kind"], e["label"], _json(e.get("aliases") or []),
                _json(e.get("attributes") or {}), _json(e.get("external_ids") or {}),
            ),
        )
        conn.execute(
            "INSERT INTO entity_occurrence (bundle_id, entity_id, paper_id, kind, label,"
            " aliases, attributes, external_ids) VALUES (?,?,?,?,?,?,?,?)",
            (
                bundle_id, e["entity_id"], paper_id, e["kind"], e["label"],
                _json(e.get("aliases") or []), _json(e.get("attributes") or {}),
                _json(e.get("external_ids") or {}),
            ),
        )

    for e in entities:
        rows = conn.execute(
            "SELECT kind, label, aliases, attributes, external_ids"
            " FROM entity_occurrence WHERE entity_id = ?",
            (e["entity_id"],),
        ).fetchall()
        occurrences = [
            {
                "kind": r["kind"], "label": r["label"],
                "aliases": json.loads(r["aliases"]), "attributes": json.loads(r["attributes"]),
                "external_ids": json.loads(r["external_ids"]),
            }
            for r in rows
        ]
        merged = merge_entity(occurrences)
        conn.execute(
            "UPDATE entity SET kind=?, label=?, aliases=?, attributes=?, external_ids=?"
            " WHERE entity_id=?",
            (
                merged.kind, merged.label, stable_json(merged.aliases),
                stable_json(merged.attributes), stable_json(merged.external_ids),
                e["entity_id"],
            ),
        )
    return len(entities)


def _insert_activities(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            a["activity_id"], bundle_id, a["kind"], a.get("model"), a.get("provider"),
            a.get("prompt_id"), a["software_version"], _json(a.get("input_hashes") or []),
            _json(a.get("output_hashes") or []), _json(a.get("metadata") or {}),
        )
        for a in bundle.get("activities") or []
    ]
    conn.executemany(
        "INSERT INTO activity (activity_id, bundle_id, kind, model, provider, prompt_id,"
        " software_version, input_hashes, output_hashes, metadata) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_assertions(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    paper_id = bundle["paper"]["arxiv_id"]
    rows = [
        (
            a["assertion_id"], bundle_id, paper_id, a["subject_id"], a["predicate"],
            a["family"], a.get("object_id"), _json(a.get("object_value")),
            _json(a.get("signature")), a["status"], a.get("support"),
            a.get("extraction_method"), a.get("activity_id"), a.get("revision_of"),
            a.get("notes"), _json(a.get("qualifiers") or {}), assertion_identity_hash(a),
        )
        for a in bundle.get("assertions") or []
    ]
    conn.executemany(
        "INSERT INTO assertion (assertion_id, bundle_id, paper_id, subject_id, predicate,"
        " family, object_id, object_value, signature, status, support, extraction_method,"
        " activity_id, revision_of, notes, qualifiers, identity_hash)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_evidence(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            e["evidence_id"], bundle_id, e["source_hash"], e["block_id"], e.get("dom_id"),
            e["quote"], e["quote_hash"], e["char_start"], e["char_end"],
            e["section_title"], e.get("normalization_version"),
        )
        for e in bundle.get("evidence") or []
    ]
    conn.executemany(
        "INSERT INTO evidence (evidence_id, bundle_id, source_hash, block_id, dom_id, quote,"
        " quote_hash, char_start, char_end, section_title, normalization_version)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_assertion_evidence(conn, bundle) -> int:
    rows = [
        (a["assertion_id"], evidence_id)
        for a in bundle.get("assertions") or []
        for evidence_id in a.get("evidence_ids") or []
    ]
    conn.executemany(
        "INSERT INTO assertion_evidence (assertion_id, evidence_id) VALUES (?,?)", rows
    )
    return len(rows)


def _insert_artifacts(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            a["artifact_id"], bundle_id, a["kind"], a["url"], a["source"],
            a.get("label"), _json(a.get("metadata") or {}),
        )
        for a in bundle.get("artifacts") or []
    ]
    conn.executemany(
        "INSERT INTO artifact (artifact_id, bundle_id, kind, url, source, label, metadata)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_qa_findings(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            q["finding_id"], bundle_id, q["check"], q["severity"], q["verdict"], q["message"],
            _json(q.get("assertion_ids") or []), _json(q.get("evidence_ids") or []),
        )
        for q in bundle.get("qa_findings") or []
    ]
    conn.executemany(
        "INSERT INTO qa_finding (finding_id, bundle_id, check_name, severity, verdict, message,"
        " assertion_ids, evidence_ids) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_completeness_findings(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            c["finding_id"], bundle_id, c["rule_id"], c["rule_version"],
            1 if c["trigger_fired"] else 0, c["verdict"], c.get("note"),
            _json(c.get("trigger_evidence_ids") or []),
            _json(c.get("expectation_evidence_ids") or []),
        )
        for c in bundle.get("completeness_findings") or []
    ]
    conn.executemany(
        "INSERT INTO completeness_finding (finding_id, bundle_id, rule_id, rule_version,"
        " trigger_fired, verdict, note, trigger_evidence_ids, expectation_evidence_ids)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _insert_expert_decisions(conn, bundle) -> int:
    bundle_id = bundle["bundle_id"]
    rows = [
        (
            d["decision_id"], bundle_id, d["action"], d["assertion_id"],
            d.get("corrected_assertion_id"), d["reviewer"], d["rationale"], d["decided_at"],
        )
        for d in bundle.get("expert_decisions") or []
    ]
    conn.executemany(
        "INSERT INTO expert_decision (decision_id, bundle_id, action, assertion_id,"
        " corrected_assertion_id, reviewer, rationale, decided_at) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


# --------------------------------------------------------------------------- #
# Re-import guard (idempotency + conflict) -- runs BEFORE any write
# --------------------------------------------------------------------------- #


def _classify_reimport(conn, bundle) -> tuple[list[str], list[str], list[str]]:
    """Compare an incoming bundle's assertions to what is already stored.

    Returns (conflicts, status_changes, new_assertions) by assertion_id.
    A conflict = same assertion_id but a changed identity_hash, i.e. a non-status
    change the acquisition pipeline cannot legitimately produce (D-027).
    """
    conflicts: list[str] = []
    status_changes: list[str] = []
    new_assertions: list[str] = []
    for a in bundle.get("assertions") or []:
        row = conn.execute(
            "SELECT status, identity_hash FROM assertion WHERE assertion_id = ?",
            (a["assertion_id"],),
        ).fetchone()
        if row is None:
            new_assertions.append(a["assertion_id"])
        elif row["identity_hash"] != assertion_identity_hash(a):
            conflicts.append(a["assertion_id"])
        elif row["status"] != a.get("status"):
            status_changes.append(a["assertion_id"])
    return conflicts, status_changes, new_assertions


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def import_bundle(
    conn, bundle: dict[str, Any], *, path: Optional[Path] = None, imported_at: Optional[str] = None
) -> ImportResult:
    """Import one already-validated bundle. Idempotent by bundle_id + content hash;
    a non-status change under an existing assertion_id is a conflict (aborts before
    any write). Applying legitimate re-import updates is milestone 4 -- for now
    those are detected and skipped with a warning."""
    imported_at = imported_at or _now()
    result = ImportResult(bundle_id=bundle["bundle_id"], paper_id=bundle["paper"]["arxiv_id"])

    existing = conn.execute(
        "SELECT bundle_content_hash FROM bundle_import WHERE bundle_id = ?",
        (bundle["bundle_id"],),
    ).fetchone()
    if existing is not None:
        # Same bundle_id -> a re-import. Decide before touching the database.
        if existing["bundle_content_hash"] == bundle_fingerprint(bundle):
            result.skipped = True  # byte-identical -> no-op
            return result
        conflicts, status_changes, new_assertions = _classify_reimport(conn, bundle)
        if conflicts:
            raise BundleConflictError(
                f"{len(conflicts)} assertion(s) changed under an existing id "
                f"(e.g. {conflicts[0]}); a non-status change is a conflict, not an update",
                path=path,
            )
        # Legitimate changes we do not yet apply (milestone 4). Skip, but say what
        # we saw so nothing is silently dropped.
        result.skipped = True
        result.warnings.append(
            f"re-import of an existing bundle_id with changes "
            f"({len(status_changes)} status change(s), {len(new_assertions)} new claim(s)); "
            "applying re-import updates is milestone 4 -- skipped"
        )
        return result

    with store.transaction(conn):
        _insert_paper(conn, bundle)
        _insert_source_snapshot(conn, bundle)
        result.inserted["source_block"] = _insert_source_blocks(conn, bundle)
        _insert_bundle_import(conn, bundle, imported_at)
        result.inserted["entity"] = _upsert_entities(conn, bundle)
        result.inserted["activity"] = _insert_activities(conn, bundle)
        result.inserted["assertion"] = _insert_assertions(conn, bundle)
        result.inserted["evidence"] = _insert_evidence(conn, bundle)
        result.inserted["assertion_evidence"] = _insert_assertion_evidence(conn, bundle)
        result.inserted["artifact"] = _insert_artifacts(conn, bundle)
        result.inserted["qa_finding"] = _insert_qa_findings(conn, bundle)
        result.inserted["completeness_finding"] = _insert_completeness_findings(conn, bundle)
        result.inserted["expert_decision"] = _insert_expert_decisions(conn, bundle)

    return result


def import_bundle_file(
    conn, path: Union[str, Path], *, imported_at: Optional[str] = None
) -> ImportResult:
    """Read, validate (shape + semantics), then import. The CLI entry point."""
    path = Path(path)
    bundle = load_bundle(path)
    warnings = validate_semantics(bundle, path=path)
    result = import_bundle(conn, bundle, path=path, imported_at=imported_at)
    result.warnings = warnings
    return result
