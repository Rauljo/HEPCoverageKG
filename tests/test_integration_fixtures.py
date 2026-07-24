# =============================================================================
# HEPCoverageKG: the acceptance suite -- the 7 contract pass conditions (step 7)
#
# One test per condition from docs/hepkg-integration.md, all against the
# supervisor's own examples/integration/ fixtures. Counts are compared to HIS
# expected-import-counts.json, never hardcoded here -- if he changes a fixture,
# these follow. Skipped wholesale when the acquisition repo is absent.
# =============================================================================
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hepcoveragekg.ingest import importer
from hepcoveragekg.ingest.errors import BundleConflictError
from hepcoveragekg.kg import queries, store

ACQUISITION_DIR = Path(
    os.environ.get(
        "HEPKG_ACQUISITION_DIR",
        Path(__file__).resolve().parents[2] / "HEPKG_promopt_tests",
    )
)
pytestmark = pytest.mark.skipif(
    not ACQUISITION_DIR.is_dir(), reason=f"acquisition repo not found at {ACQUISITION_DIR}"
)
FIXTURES = ACQUISITION_DIR / "examples" / "integration"
BUNDLES = ACQUISITION_DIR / "pilot" / "bundles"


def _db():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    return conn


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _expected_counts():
    return json.loads((FIXTURES / "expected-import-counts.json").read_text())


def _all_counts(conn):
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


# --- Condition 1: initial import matches expected counts ---------------------


@pytest.mark.parametrize("scenario", ["accepted_bundle", "corrected_bundle"])
def test_c1_initial_import_matches_expected_counts(scenario):
    conn = _db()
    importer.import_bundle(conn, _fixture(scenario))
    counts = queries.import_counts(conn)
    expected = _expected_counts()[scenario]
    # compare only the keys the supervisor's fixture specifies
    for key, want in expected.items():
        assert counts[key] == want, f"{scenario}.{key}: got {counts[key]}, want {want}"


# --- Condition 2: identical reimport changes no counts -----------------------


def test_c2_identical_reimport_changes_nothing():
    conn = _db()
    importer.import_bundle(conn, _fixture("accepted_bundle"))
    before = _all_counts(conn)
    result = importer.import_bundle(conn, _fixture("accepted_bundle"))
    assert result.skipped is True
    assert _all_counts(conn) == before


# --- Condition 3: conflicting content aborts transactionally -----------------


def test_c3_conflicting_content_aborts():
    conn = _db()
    importer.import_bundle(conn, _fixture("accepted_bundle"))
    before = _all_counts(conn)
    with pytest.raises(BundleConflictError):
        importer.import_bundle(conn, _fixture("conflicting_bundle"))
    assert _all_counts(conn) == before  # transactional: nothing left behind


# --- Condition 4: accepted view EXCLUDES proposed/quarantined/rejected/superseded


def test_c4_accepted_view_excludes_superseded():
    conn = _db()
    importer.import_bundle(conn, _fixture("corrected_bundle"))
    # 2 assertions, but only the correction is accepted -- the superseded proposal is excluded
    assert conn.execute("SELECT COUNT(*) FROM assertion").fetchone()[0] == 2
    accepted = conn.execute(
        "SELECT status FROM accepted_view"
    ).fetchall()
    assert len(accepted) == 1
    assert accepted[0]["status"] == "expert_accepted"


def test_c4_accepted_view_excludes_machine_and_quarantined_and_rejected():
    """A real pilot bundle has only machine_verified/quarantined/rejected -- none
    accepted -- so the accepted view must be empty."""
    import gzip
    conn = _db()
    with gzip.open(BUNDLES / "2001.06899.json.gz", "rt", encoding="utf-8") as handle:
        importer.import_bundle(conn, json.load(handle))
    statuses = queries.status_counts(conn)
    assert set(statuses) <= {"machine_verified", "quarantined", "rejected"}
    assert conn.execute("SELECT COUNT(*) FROM accepted_view").fetchone()[0] == 0


# --- Condition 5: evidence trace returns exact source/block/quote -------------


def test_c5_evidence_trace_returns_exact_quote():
    conn = _db()
    bundle = _fixture("accepted_bundle")
    importer.import_bundle(conn, bundle)
    assertion_id = bundle["assertions"][0]["assertion_id"]

    trace = queries.trace_assertion(conn, assertion_id)
    assert trace is not None
    assert trace["paper_id"] == "9999.00001"
    assert trace["predicate"] == "paper_reports_result"
    assert len(trace["evidence"]) == 1
    ev = trace["evidence"][0]
    assert ev["quote"] == "Search for X in two-lepton events"
    assert ev["block_id"] == "title"
    # the block the quote points at really contains that text
    assert ev["quote"] in ev["block_text"]


def test_c5_trace_of_unknown_assertion_is_none():
    conn = _db()
    assert queries.trace_assertion(conn, "hepkg:assertion:doesnotexist") is None


# --- Condition 6: corrected assertion links to its proposal ------------------


def test_c6_corrected_assertion_links_to_its_proposal():
    conn = _db()
    importer.import_bundle(conn, _fixture("corrected_bundle"))

    superseded = conn.execute(
        "SELECT assertion_id FROM assertion WHERE status = 'superseded'"
    ).fetchone()["assertion_id"]
    correction = conn.execute(
        "SELECT assertion_id, revision_of FROM assertion WHERE status = 'expert_accepted'"
    ).fetchone()
    # the correction points back at the proposal it supersedes
    assert correction["revision_of"] == superseded
    # and the expert_decision records the same link
    decision = conn.execute(
        "SELECT assertion_id, corrected_assertion_id FROM expert_decision WHERE action = 'correct'"
    ).fetchone()
    assert decision["assertion_id"] == superseded
    assert decision["corrected_assertion_id"] == correction["assertion_id"]


# --- Condition 7: completeness findings never appear as domain edges ----------


def test_c7_completeness_findings_are_not_referenced_by_the_graph():
    conn = _db()
    schemas = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    assert not any(
        "REFERENCES completeness_finding" in (row["sql"] or "") for row in schemas
    )
