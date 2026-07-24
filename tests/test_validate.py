# =============================================================================
# HEPCoverageKG: tests for the semantic gate (step 4)
#
# One targeted test per rule on minimal inline bundles, plus the real fixtures.
# The acceptance pair: invalid_accepted_without_evidence must be REJECTED here
# (it passed the shape gate), while conflicting_bundle must PASS -- its problem
# is a conflict, which belongs to step 6, not to this gate.
# =============================================================================
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from hepcoveragekg.ingest import canonical
from hepcoveragekg.ingest.errors import BundleSemanticError
from hepcoveragekg.ingest.validate import validate_semantics

ACQUISITION_DIR = Path(
    os.environ.get(
        "HEPKG_ACQUISITION_DIR",
        Path(__file__).resolve().parents[2] / "HEPKG_promopt_tests",
    )
)
requires_acquisition = pytest.mark.skipif(
    not ACQUISITION_DIR.is_dir(), reason=f"acquisition repo not found at {ACQUISITION_DIR}"
)

SOURCE_HASH = "sha256:" + "a" * 64
QUOTE_HASH = "sha256:" + "b" * 64


def _assertion(**overrides):
    assertion = {
        "assertion_id": "a1",
        "subject_id": "e-subject",
        "predicate": "paper_reports_result",
        "family": "publication_context",
        "object_id": "e-object",
        "object_value": None,
        "signature": None,
        "status": "machine_verified",
        "support": "explicit",
        "extraction_method": "llm",
        "evidence_ids": ["ev1"],
        "qualifiers": {},
        "notes": None,
        "revision_of": None,
        "activity_id": None,
    }
    assertion.update(overrides)
    return assertion


def _bundle(**overrides):
    """A semantically clean bundle: every reference resolves, no warnings."""
    bundle = {
        "schema_version": "hepkg-acquisition-v0.2",
        "paper": {"arxiv_id": "9999.00001", "title": "t", "experiments": ["ATLAS"]},
        "source": {
            "source_hash": SOURCE_HASH,
            "paper_id": "9999.00001",
            "title": "t",
            "source_path": "/x/index.html",
            "experiments": ["ATLAS"],
            "normalization_version": "latexml-blocks-v1",
            "blocks": [
                {"block_id": "title", "kind": "title", "order": 0, "text": "t",
                 "text_hash": QUOTE_HASH},
            ],
        },
        "entities": [
            {"entity_id": "e-subject", "kind": "paper", "label": "subject"},
            {"entity_id": "e-object", "kind": "result", "label": "object"},
        ],
        "evidence": [
            {"evidence_id": "ev1", "source_hash": SOURCE_HASH, "block_id": "title",
             "quote": "t", "quote_hash": QUOTE_HASH, "char_start": 0, "char_end": 1,
             "section_title": ""},
        ],
        "assertions": [_assertion()],
        "activities": [],
        "artifacts": [],
        "qa_findings": [],
        "completeness_findings": [],
        "expert_decisions": [],
    }
    bundle.update(overrides)
    # derive bundle_id from the content so the authenticity check stays quiet
    bundle["bundle_id"] = canonical.expected_bundle_id(bundle)
    return bundle


def _fails_with(bundle, fragment):
    with pytest.raises(BundleSemanticError) as excinfo:
        validate_semantics(bundle)
    assert fragment in str(excinfo.value), str(excinfo.value)


# --- the happy path ----------------------------------------------------------


def test_accepts_a_clean_bundle():
    assert validate_semantics(_bundle()) == []


# --- contract rules ----------------------------------------------------------


def test_rejects_accepted_without_evidence():
    """The rule the schema cannot express -- the invalid fixture's defect."""
    bundle = _bundle(assertions=[_assertion(status="expert_accepted", evidence_ids=[])])
    _fails_with(bundle, "expert_accepted but cites no evidence")


def test_rejects_two_object_shapes():
    bundle = _bundle(assertions=[_assertion(object_value='"x"')])
    _fails_with(bundle, "2 object shapes present")


def test_rejects_zero_object_shapes():
    bundle = _bundle(assertions=[_assertion(object_id=None)])
    _fails_with(bundle, "0 object shapes present")


# --- in-bundle referential integrity -----------------------------------------


def test_rejects_dangling_subject():
    _fails_with(_bundle(assertions=[_assertion(subject_id="ghost")]), "subject_id 'ghost'")


def test_rejects_dangling_object():
    _fails_with(_bundle(assertions=[_assertion(object_id="ghost")]), "object_id 'ghost'")


def test_rejects_dangling_evidence_reference():
    _fails_with(_bundle(assertions=[_assertion(evidence_ids=["ghost"])]), "evidence_id 'ghost'")


def test_rejects_dangling_revision_of():
    """revision_of is a soft pointer in schema.sql (no FK), so this gate is the
    only thing guarding it."""
    _fails_with(_bundle(assertions=[_assertion(revision_of="ghost")]), "revision_of 'ghost'")


def test_rejects_dangling_activity():
    _fails_with(_bundle(assertions=[_assertion(activity_id="ghost")]), "activity_id 'ghost'")


def test_rejects_dangling_decision_assertion():
    decision = {"decision_id": "d1", "assertion_id": "ghost", "action": "accept",
                "reviewer": "r", "rationale": "x", "decided_at": "2026-01-01T00:00:00Z",
                "corrected_assertion_id": None}
    _fails_with(_bundle(expert_decisions=[decision]), "assertion_id 'ghost'")


def test_rejects_dangling_corrected_assertion():
    decision = {"decision_id": "d1", "assertion_id": "a1", "action": "correct",
                "reviewer": "r", "rationale": "x", "decided_at": "2026-01-01T00:00:00Z",
                "corrected_assertion_id": "ghost"}
    _fails_with(_bundle(expert_decisions=[decision]), "corrected_assertion_id 'ghost'")


def test_rejects_evidence_pointing_at_unknown_block():
    evidence = [{"evidence_id": "ev1", "source_hash": SOURCE_HASH, "block_id": "ghost",
                 "quote": "t", "quote_hash": QUOTE_HASH, "char_start": 0, "char_end": 1,
                 "section_title": ""}]
    _fails_with(_bundle(evidence=evidence), "block_id 'ghost'")


def test_rejects_evidence_from_a_different_snapshot():
    evidence = [{"evidence_id": "ev1", "source_hash": "sha256:" + "c" * 64,
                 "block_id": "title", "quote": "t", "quote_hash": QUOTE_HASH,
                 "char_start": 0, "char_end": 1, "section_title": ""}]
    _fails_with(_bundle(evidence=evidence), "does not match the bundle source")


# --- uniqueness and coherence ------------------------------------------------


def test_rejects_duplicate_entity_ids():
    entities = [
        {"entity_id": "e-subject", "kind": "paper", "label": "a"},
        {"entity_id": "e-subject", "kind": "paper", "label": "b"},
        {"entity_id": "e-object", "kind": "result", "label": "o"},
    ]
    _fails_with(_bundle(entities=entities), "duplicate entity_id 'e-subject'")


def test_rejects_duplicate_assertion_ids():
    _fails_with(_bundle(assertions=[_assertion(), _assertion()]), "duplicate assertion_id 'a1'")


def test_rejects_paper_id_mismatch():
    source = dict(_bundle()["source"], paper_id="0000.00000")
    _fails_with(_bundle(source=source), "!= source.paper_id")


def test_reports_multiple_violations_together():
    bundle = _bundle(assertions=[_assertion(subject_id="ghost1", object_id="ghost2")])
    with pytest.raises(BundleSemanticError) as excinfo:
        validate_semantics(bundle)
    assert "2 semantic violation(s)" in str(excinfo.value)


# --- warnings (recorded, never blocking) -------------------------------------


def test_warns_on_inauthentic_bundle_id():
    bundle = _bundle()
    bundle["bundle_id"] = "hepkg:bundle:deadbeefdeadbeefdeadbeef"
    warnings = validate_semantics(bundle)
    assert any("does not match its content" in w for w in warnings)


def test_warns_on_unquarantined_assertion_without_evidence():
    warnings = validate_semantics(
        _bundle(assertions=[_assertion(status="machine_verified", evidence_ids=[])])
    )
    assert any("cites no evidence" in w for w in warnings)


def test_quarantined_assertion_without_evidence_is_silent():
    """The pilot's 450 evidence-less assertions are all quarantined -- normal."""
    assert validate_semantics(
        _bundle(assertions=[_assertion(status="quarantined", evidence_ids=[])])
    ) == []


def test_warns_on_dangling_qa_finding_reference():
    finding = {"finding_id": "q1", "check": "x", "severity": "warning", "verdict": "warn",
               "message": "m", "assertion_ids": ["ghost"], "evidence_ids": []}
    warnings = validate_semantics(_bundle(qa_findings=[finding]))
    assert any("references unknown assertion" in w for w in warnings)


# --- against the real data ---------------------------------------------------


@requires_acquisition
@pytest.mark.parametrize("name", ["accepted_bundle", "corrected_bundle"])
def test_valid_fixtures_pass(name):
    bundle = json.loads((ACQUISITION_DIR / "examples" / "integration" / f"{name}.json").read_text())
    assert validate_semantics(bundle) == []


@requires_acquisition
def test_invalid_fixture_is_rejected_here():
    """It passed the shape gate; this is the gate that must catch it."""
    path = ACQUISITION_DIR / "examples" / "integration" / "invalid_accepted_without_evidence.json"
    _fails_with(json.loads(path.read_text()), "expert_accepted but cites no evidence")


@requires_acquisition
def test_conflicting_fixture_passes_the_semantic_gate():
    """Boundary marker: conflicting_bundle is internally consistent. Its defect
    is a CONFLICT against already-stored data, which is step 6's job."""
    path = ACQUISITION_DIR / "examples" / "integration" / "conflicting_bundle.json"
    assert validate_semantics(json.loads(path.read_text())) == []


@requires_acquisition
def test_all_pilot_bundles_pass():
    paths = sorted((ACQUISITION_DIR / "pilot" / "bundles").glob("*.json.gz"))
    assert len(paths) == 60
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            validate_semantics(json.load(handle), path=path)
