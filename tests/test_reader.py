# =============================================================================
# HEPCoverageKG: tests for the bundle reader + shape gate (step 3)
#
# Mostly self-contained: a minimal valid bundle is small, so most shape checks
# need no external files. Tests that read the acquisition repo are skipped when
# it is absent, so the suite still runs anywhere.
#
# The boundary test is test_invalid_fixture_passes_the_shape_gate: it asserts
# the known-bad fixture PASSES here, documenting that accepted-without-evidence
# is the semantic layer's job (step 4), not the schema's.
# =============================================================================
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from hepcoveragekg.ingest import reader
from hepcoveragekg.ingest.errors import BundleReadError, BundleShapeError

ACQUISITION_DIR = Path(
    os.environ.get(
        "HEPKG_ACQUISITION_DIR",
        Path(__file__).resolve().parents[2] / "HEPKG_promopt_tests",
    )
)
requires_acquisition = pytest.mark.skipif(
    not ACQUISITION_DIR.is_dir(), reason=f"acquisition repo not found at {ACQUISITION_DIR}"
)


def _minimal_bundle(**overrides):
    """Smallest bundle the schema accepts: root needs bundle_id, paper, source."""
    bundle = {
        "schema_version": "hepkg-acquisition-v0.2",
        "bundle_id": "hepkg:bundle:0123456789abcdef01234567",
        "paper": {"arxiv_id": "9999.00001", "title": "t", "experiments": ["ATLAS"]},
        "source": {
            "source_hash": "sha256:" + "0" * 64,
            "paper_id": "9999.00001",
            "title": "t",
            "source_path": "/x/index.html",
            "experiments": ["ATLAS"],
            "blocks": [],
        },
    }
    bundle.update(overrides)
    return bundle


# --- shape gate --------------------------------------------------------------


def test_accepts_minimal_valid_bundle():
    reader.validate_shape(_minimal_bundle())  # must not raise


def test_rejects_missing_schema_version():
    """The schema declares schema_version `const` but does not require it, so a
    bundle omitting it would otherwise pass. This is our explicit check."""
    bundle = _minimal_bundle()
    del bundle["schema_version"]
    with pytest.raises(BundleShapeError, match="missing schema_version"):
        reader.validate_shape(bundle)


def test_rejects_unsupported_schema_version():
    with pytest.raises(BundleShapeError, match="unsupported schema_version"):
        reader.validate_shape(_minimal_bundle(schema_version="hepkg-acquisition-v0.3"))


def test_rejects_missing_required_field():
    bundle = _minimal_bundle()
    del bundle["paper"]
    with pytest.raises(BundleShapeError, match="schema violation"):
        reader.validate_shape(bundle)


def test_rejects_wrong_type():
    with pytest.raises(BundleShapeError, match="schema violation"):
        reader.validate_shape(_minimal_bundle(assertions="not a list"))


def test_rejects_unknown_top_level_field():
    """The schema sets additionalProperties: false at the root."""
    with pytest.raises(BundleShapeError, match="schema violation"):
        reader.validate_shape(_minimal_bundle(surprise="unexpected"))


def test_rejects_non_object():
    with pytest.raises(BundleShapeError, match="expected a JSON object"):
        reader.validate_shape(["not", "a", "bundle"])


def test_error_reports_the_offending_path():
    bundle = _minimal_bundle(entities=[{"kind": "paper"}])  # missing entity_id + label
    with pytest.raises(BundleShapeError) as excinfo:
        reader.validate_shape(bundle)
    assert "entities/0" in str(excinfo.value)


# --- reading files -----------------------------------------------------------


def test_reads_plain_json(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(_minimal_bundle()), encoding="utf-8")
    assert reader.read_bundle(path)["bundle_id"].startswith("hepkg:bundle:")


def test_reads_gzipped_json(tmp_path):
    path = tmp_path / "b.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_minimal_bundle(), handle)
    assert reader.read_bundle(path)["schema_version"] == "hepkg-acquisition-v0.2"


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleReadError, match="not valid JSON"):
        reader.read_bundle(path)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(BundleReadError, match="cannot read file"):
        reader.read_bundle(tmp_path / "nope.json")


def test_rejects_json_that_is_not_an_object(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(BundleReadError, match="expected a JSON object"):
        reader.read_bundle(path)


def test_load_bundle_reads_and_validates(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(_minimal_bundle()), encoding="utf-8")
    assert reader.load_bundle(path)["paper"]["arxiv_id"] == "9999.00001"


def test_load_bundle_reports_the_file_path(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_minimal_bundle(schema_version="v9")), encoding="utf-8")
    with pytest.raises(BundleShapeError) as excinfo:
        reader.load_bundle(path)
    assert "bad.json" in str(excinfo.value)


# --- the vendored schema -----------------------------------------------------


def test_vendored_schema_matches_its_pinned_hash():
    """Catches an accidental local edit to the vendored copy."""
    digest = hashlib.sha256(reader.SCHEMA_PATH.read_bytes()).hexdigest()
    assert digest == reader.VENDORED_SCHEMA_SHA256


@requires_acquisition
def test_vendored_schema_matches_upstream():
    """Catches the supervisor changing the schema under us. Skipped if his repo
    is not present, so the suite stays runnable anywhere."""
    upstream = ACQUISITION_DIR / "schemas" / "hepkg-acquisition-v0.2.schema.json"
    assert reader.SCHEMA_PATH.read_bytes() == upstream.read_bytes(), (
        "vendored schema has drifted from upstream -- re-vendor and update "
        "VENDORED_SCHEMA_SHA256"
    )


# --- against the real data ---------------------------------------------------


@requires_acquisition
def test_real_pilot_bundle_passes(tmp_path):
    bundle = reader.load_bundle(ACQUISITION_DIR / "pilot" / "bundles" / "2001.06899.json.gz")
    assert bundle["paper"]["arxiv_id"] == "2001.06899"
    assert len(bundle["assertions"]) == 125


@requires_acquisition
def test_invalid_fixture_passes_the_shape_gate():
    """Boundary marker: this fixture is KNOWN-BAD (an expert_accepted assertion
    with no evidence) yet is shape-valid. Proving it passes here is what
    justifies having a separate semantic layer in step 4."""
    path = ACQUISITION_DIR / "examples" / "integration" / "invalid_accepted_without_evidence.json"
    bundle = reader.load_bundle(path)
    assertion = bundle["assertions"][0]
    assert assertion["status"] == "expert_accepted"
    assert assertion["evidence_ids"] == []
