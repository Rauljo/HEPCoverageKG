# =============================================================================
# HEPCoverageKG: tests for canonical serialization and fingerprints (step 2)
#
# The load-bearing test here is test_matches_supervisor_bundle_id: it pins our
# canonicalization to the supervisor's byte-for-byte using a real bundle_id.
# Self-contained -- no reads from the acquisition repo (that happens in step 7).
# =============================================================================
from __future__ import annotations

from hepcoveragekg.ingest import canonical

# Real identity fields and declared bundle_id of pilot bundle 2001.06899.
REAL_SOURCE_HASH = "sha256:00828c47d7433fcba9e3e2e6c5abece600339ebac2918c7b42333252e6eca867"
REAL_BUNDLE_ID = "hepkg:bundle:b614e6c6a72f4c83e789320b"


def _bundle(**overrides):
    bundle = {
        "schema_version": "hepkg-acquisition-v0.2",
        "bundle_id": REAL_BUNDLE_ID,
        "paper": {"arxiv_id": "2001.06899"},
        "source": {
            "source_hash": REAL_SOURCE_HASH,
            "normalization_version": "latexml-blocks-v1",
        },
        "assertions": [],
    }
    bundle.update(overrides)
    return bundle


def _assertion(**overrides):
    assertion = {
        "assertion_id": "hepkg:assertion:c8985f2cf200e0bb537e67e7",
        "subject_id": "hepkg:paper:arxiv:9999.00001",
        "predicate": "paper_reports_result",
        "family": "publication_context",
        "object_id": "hepkg:result:31660eccdbac026e79c5547f",
        "object_value": None,
        "signature": None,
        "qualifiers": {},
        "evidence_ids": ["hepkg:evidence:3c3fc6e56e477d8438408f37"],
        "notes": None,
        "support": "explicit",
        "status": "expert_accepted",
    }
    assertion.update(overrides)
    return assertion


# --- stable_json -------------------------------------------------------------


def test_stable_json_is_key_order_independent():
    assert canonical.stable_json({"b": 1, "a": 2}) == canonical.stable_json({"a": 2, "b": 1})


def test_stable_json_is_compact():
    assert canonical.stable_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_stable_json_preserves_unicode():
    # real bundles carry things like "muon channel (μμ+jets)"
    assert "μμ" in canonical.stable_json({"channel": "μμ+jets"})


# --- content_hash ------------------------------------------------------------


def test_content_hash_format():
    digest = canonical.content_hash({"a": 1})
    prefix, _, hexpart = digest.partition(":")
    assert prefix == "sha256"
    assert len(hexpart) == 64
    assert set(hexpart) <= set("0123456789abcdef")


def test_content_hash_is_deterministic_and_order_independent():
    assert canonical.content_hash({"b": 1, "a": 2}) == canonical.content_hash({"a": 2, "b": 1})


def test_content_hash_changes_with_content():
    assert canonical.content_hash({"a": 1}) != canonical.content_hash({"a": 2})


# --- compatibility with the supervisor's ids ---------------------------------


def test_matches_supervisor_bundle_id():
    """Our canonicalization must be byte-identical to the supervisor's.

    These are the real identity fields of pilot bundle 2001.06899, and the
    expected value is the bundle_id that bundle actually declares. If
    stable_json ever drifts (a space, key order, ascii escaping), this fails.
    """
    recomputed = canonical.content_id(
        "bundle",
        {
            "schema": "hepkg-acquisition-v0.2",
            "paper": "2001.06899",
            "source": REAL_SOURCE_HASH,
            "normalization": "latexml-blocks-v1",
        },
    )
    assert recomputed == REAL_BUNDLE_ID


# --- bundle_fingerprint ------------------------------------------------------


def test_bundle_fingerprint_matches_for_identical_bundles():
    assert canonical.bundle_fingerprint(_bundle()) == canonical.bundle_fingerprint(_bundle())


def test_bundle_fingerprint_detects_any_change():
    """Even a status-only change makes the bundle non-identical (that is the point:
    it only answers 'byte-identical?', and anything else falls through to the
    per-assertion check)."""
    base = _bundle(assertions=[_assertion()])
    changed = _bundle(assertions=[_assertion(status="superseded")])
    assert canonical.bundle_fingerprint(base) != canonical.bundle_fingerprint(changed)


# --- expected_bundle_id ------------------------------------------------------


def test_expected_bundle_id_reproduces_the_declared_id():
    assert canonical.expected_bundle_id(_bundle()) == REAL_BUNDLE_ID


def test_expected_bundle_id_ignores_assertions_and_statuses():
    """The correction that reshaped the conflict rule: bundle_id depends only on
    schema + paper + source + normalization, so a re-extraction with different
    assertions keeps the SAME bundle_id."""
    with_assertions = _bundle(assertions=[_assertion(), _assertion(status="superseded")])
    assert canonical.expected_bundle_id(with_assertions) == canonical.expected_bundle_id(_bundle())


def test_bundle_id_matches_content_accepts_authentic_bundle():
    assert canonical.bundle_id_matches_content(_bundle()) is True


def test_bundle_id_matches_content_detects_mislabelled_bundle():
    assert canonical.bundle_id_matches_content(_bundle(bundle_id="hepkg:bundle:deadbeef")) is False


# --- assertion_identity_hash (the conflict primitive) ------------------------


def test_assertion_identity_ignores_status_change():
    """corrected_bundle: c898 goes expert_accepted -> superseded. Allowed."""
    before = _assertion(status="expert_accepted")
    after = _assertion(status="superseded")
    assert canonical.assertion_identity_hash(before) == canonical.assertion_identity_hash(after)


def test_assertion_identity_detects_notes_change():
    """conflicting_bundle: c898 comes back with different notes. Must conflict."""
    before = _assertion(notes=None)
    after = _assertion(notes="Different content under the same assertion ID: reject.")
    assert canonical.assertion_identity_hash(before) != canonical.assertion_identity_hash(after)


def test_assertion_identity_detects_claim_change():
    before = _assertion()
    after = _assertion(object_id="hepkg:result:something_else")
    assert canonical.assertion_identity_hash(before) != canonical.assertion_identity_hash(after)
