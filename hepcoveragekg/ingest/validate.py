# =============================================================================
# HEPCoverageKG: bundle semantic gate (pipeline stage 2)
#
# The shape gate (reader.py) proves a bundle matches the supervisor's schema.
# This proves what the schema CANNOT express: contract rules and internal
# consistency. Nothing here touches the database.
#
#   ERRORS reject the bundle -- contract violations, plus anything the store
#   itself would refuse (FK / PK / CHECK). Pre-checking those here turns a
#   cryptic mid-transaction SQLite failure into a message naming the record.
#
#   WARNINGS are returned but never block -- metadata inconsistencies and
#   authenticity signals that cannot corrupt the scientific graph.
#
# Note that `revision_of` is deliberately a soft pointer in schema.sql (no FK,
# to avoid insert-order fragility), so the check here is the ONLY thing
# guarding it.
#
# Every rule below was verified against all 60 pilot bundles with zero
# violations, so none can block the milestone-1 count target.
# See vault/ideas/bundle-importer-design.md.
# =============================================================================
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from hepcoveragekg.ingest.canonical import bundle_id_matches_content
from hepcoveragekg.ingest.errors import BundleSemanticError

# (collection key, id field) for every record type that becomes a primary key.
# Ids must be unique WITHIN a bundle; across bundles entity_ids repeat by
# design (shared canonical nodes) and are merged, not rejected.
ID_FIELDS = (
    ("entities", "entity_id"),
    ("assertions", "assertion_id"),
    ("evidence", "evidence_id"),
    ("activities", "activity_id"),
    ("artifacts", "artifact_id"),
    ("qa_findings", "finding_id"),
    ("completeness_findings", "finding_id"),
    ("expert_decisions", "decision_id"),
)

OBJECT_SHAPE_FIELDS = ("object_id", "object_value", "signature")

MAX_REPORTED_ERRORS = 5


def _duplicates(records: Iterable[dict], id_field: str) -> list[str]:
    counts = Counter(r[id_field] for r in records if id_field in r)
    return sorted(key for key, n in counts.items() if n > 1)


def validate_semantics(
    bundle: dict[str, Any], *, path: Optional[Path] = None
) -> list[str]:
    """Stage 2. Raises BundleSemanticError listing every violation; returns warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    paper = bundle.get("paper") or {}
    source = bundle.get("source") or {}
    entities = bundle.get("entities") or []
    assertions = bundle.get("assertions") or []
    evidence = bundle.get("evidence") or []
    activities = bundle.get("activities") or []

    entity_ids = {e["entity_id"] for e in entities}
    assertion_ids = {a["assertion_id"] for a in assertions}
    evidence_ids = {e["evidence_id"] for e in evidence}
    activity_ids = {a["activity_id"] for a in activities}
    source_hash = source.get("source_hash")
    block_ids = {b["block_id"] for b in source.get("blocks") or []}

    # --- ids unique within this bundle (they become primary keys) ------------
    for collection, id_field in ID_FIELDS:
        for duplicate in _duplicates(bundle.get(collection) or [], id_field):
            errors.append(f"{collection}: duplicate {id_field} {duplicate!r}")

    # --- paper / source coherence -------------------------------------------
    if paper.get("arxiv_id") != source.get("paper_id"):
        errors.append(
            f"paper.arxiv_id {paper.get('arxiv_id')!r} != "
            f"source.paper_id {source.get('paper_id')!r}"
        )

    # --- evidence ------------------------------------------------------------
    for item in evidence:
        eid = item["evidence_id"]
        if item.get("source_hash") != source_hash:
            errors.append(
                f"evidence {eid}: source_hash {item.get('source_hash')!r} does not "
                f"match the bundle source {source_hash!r}"
            )
        if item.get("block_id") not in block_ids:
            errors.append(
                f"evidence {eid}: block_id {item.get('block_id')!r} is not a block "
                "in this bundle's source snapshot"
            )

    # --- assertions ----------------------------------------------------------
    for assertion in assertions:
        aid = assertion["assertion_id"]
        status = assertion.get("status")
        cited = assertion.get("evidence_ids") or []

        shapes = sum(assertion.get(field) is not None for field in OBJECT_SHAPE_FIELDS)
        if shapes != 1:
            present = [f for f in OBJECT_SHAPE_FIELDS if assertion.get(f) is not None]
            errors.append(
                f"assertion {aid}: {shapes} object shapes present {present} "
                "(exactly one required)"
            )

        if status == "expert_accepted" and not cited:
            errors.append(f"assertion {aid}: expert_accepted but cites no evidence")
        elif not cited and status != "quarantined":
            # Observed invariant in the pilot: every evidence-less assertion is
            # quarantined (the pipeline quarantines unlocatable quotes). Anything
            # else is anomalous but not fatal.
            warnings.append(f"assertion {aid}: status {status!r} but cites no evidence")

        if assertion.get("subject_id") not in entity_ids:
            errors.append(
                f"assertion {aid}: subject_id {assertion.get('subject_id')!r} "
                "is not an entity in this bundle"
            )
        object_id = assertion.get("object_id")
        if object_id is not None and object_id not in entity_ids:
            errors.append(
                f"assertion {aid}: object_id {object_id!r} is not an entity in this bundle"
            )
        for cited_id in cited:
            if cited_id not in evidence_ids:
                errors.append(
                    f"assertion {aid}: evidence_id {cited_id!r} is not in this bundle"
                )
        revision_of = assertion.get("revision_of")
        if revision_of is not None and revision_of not in assertion_ids:
            errors.append(
                f"assertion {aid}: revision_of {revision_of!r} is not an assertion "
                "in this bundle"
            )
        activity_id = assertion.get("activity_id")
        if activity_id is not None and activity_id not in activity_ids:
            errors.append(
                f"assertion {aid}: activity_id {activity_id!r} is not an activity "
                "in this bundle"
            )

    # --- expert decisions ----------------------------------------------------
    for decision in bundle.get("expert_decisions") or []:
        did = decision["decision_id"]
        if decision.get("assertion_id") not in assertion_ids:
            errors.append(
                f"expert_decision {did}: assertion_id "
                f"{decision.get('assertion_id')!r} is not in this bundle"
            )
        corrected = decision.get("corrected_assertion_id")
        if corrected is not None and corrected not in assertion_ids:
            errors.append(
                f"expert_decision {did}: corrected_assertion_id {corrected!r} "
                "is not in this bundle"
            )

    # --- warnings: metadata references (never corrupt the graph) -------------
    for collection, id_field in (
        ("qa_findings", "finding_id"),
        ("completeness_findings", "finding_id"),
    ):
        for finding in bundle.get(collection) or []:
            for ref in finding.get("assertion_ids") or []:
                if ref not in assertion_ids:
                    warnings.append(
                        f"{collection} {finding[id_field]}: references unknown "
                        f"assertion {ref!r}"
                    )
            for key in ("evidence_ids", "trigger_evidence_ids", "expectation_evidence_ids"):
                for ref in finding.get(key) or []:
                    if ref not in evidence_ids:
                        warnings.append(
                            f"{collection} {finding[id_field]}: references unknown "
                            f"evidence {ref!r}"
                        )

    # --- warning: is the declared bundle_id authentic? -----------------------
    if not bundle_id_matches_content(bundle):
        warnings.append(
            f"bundle_id {bundle.get('bundle_id')!r} does not match its content "
            "(mislabelled bundle, or the supervisor changed his id algorithm)"
        )

    if errors:
        shown = "; ".join(errors[:MAX_REPORTED_ERRORS])
        omitted = len(errors) - MAX_REPORTED_ERRORS
        suffix = f" (+{omitted} more)" if omitted > 0 else ""
        raise BundleSemanticError(
            f"{len(errors)} semantic violation(s) -- {shown}{suffix}", path=path
        )
    return warnings
