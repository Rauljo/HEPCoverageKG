# =============================================================================
# HEPCoverageKG: tests for the single-bundle importer (step 5)
#
# The acceptance test is test_accepted_fixture_counts: importing accepted_bundle
# into a fresh DB must reproduce the counts the acquisition side expects, read
# back FROM the database (not from the receipt). Merge and rollback behaviour
# are unit-tested; the real pilot bundle is checked when the repo is present.
# =============================================================================
from __future__ import annotations

import gzip
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hepcoveragekg.ingest import importer
from hepcoveragekg.ingest.importer import ImportResult, merge_entity
from hepcoveragekg.kg import store

ACQUISITION_DIR = Path(
    os.environ.get(
        "HEPKG_ACQUISITION_DIR",
        Path(__file__).resolve().parents[2] / "HEPKG_promopt_tests",
    )
)
requires_acquisition = pytest.mark.skipif(
    not ACQUISITION_DIR.is_dir(), reason=f"acquisition repo not found at {ACQUISITION_DIR}"
)
FIXTURES = ACQUISITION_DIR / "examples" / "integration"

SOURCE_HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _db():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    return conn


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --- merge_entity (pure, no DB) ----------------------------------------------


def test_merge_unions_aliases():
    merged = merge_entity(
        [
            {"kind": "detector_object", "label": "b-jet", "aliases": ["b jet"], "attributes": {}},
            {"kind": "detector_object", "label": "b-jet", "aliases": ["bottom jet"], "attributes": {}},
        ]
    )
    assert merged.aliases == ["b jet", "bottom jet"]


def test_merge_keeps_agreeing_attributes_drops_contested():
    merged = merge_entity(
        [
            {"kind": "detector_object", "label": "b-jet",
             "attributes": {"jet_flavor": "bottom", "tagger": "CSV v2"}},
            {"kind": "detector_object", "label": "b-jet",
             "attributes": {"jet_flavor": "bottom", "tagger": "DeepCSV"}},
        ]
    )
    assert merged.attributes == {"jet_flavor": "bottom"}  # agreed kept, contested dropped


def test_merge_is_order_independent():
    a = {"kind": "detector_object", "label": "A", "aliases": ["x"], "attributes": {"k": 1}}
    b = {"kind": "object_definition", "label": "B", "aliases": ["y"], "attributes": {"k": 2}}
    assert merge_entity([a, b]) == merge_entity([b, a])


def test_merge_breaks_kind_ties_lexicographically():
    merged = merge_entity(
        [
            {"kind": "object_definition", "label": "x", "attributes": {}},
            {"kind": "detector_object", "label": "x", "attributes": {}},
        ]
    )
    assert merged.kind == "detector_object"  # tie -> lexicographically first


# --- the acceptance fixture --------------------------------------------------


@requires_acquisition
def test_accepted_fixture_counts():
    conn = _db()
    bundle = json.loads((FIXTURES / "accepted_bundle.json").read_text())
    importer.import_bundle(conn, bundle)

    assert _count(conn, "assertion") == 1
    assert _count(conn, "entity") == 2
    assert _count(conn, "evidence") == 1
    assert _count(conn, "expert_decision") == 1
    assert _count(conn, "source_block") == 8
    assert _count(conn, "source_snapshot") == 1
    assert conn.execute("SELECT COUNT(*) FROM accepted_view").fetchone()[0] == 1


@requires_acquisition
def test_corrected_fixture_preserves_revision_chain():
    conn = _db()
    bundle = json.loads((FIXTURES / "corrected_bundle.json").read_text())
    importer.import_bundle(conn, bundle)

    assert _count(conn, "assertion") == 2
    statuses = dict(conn.execute("SELECT status, COUNT(*) FROM assertion GROUP BY status"))
    assert statuses == {"superseded": 1, "expert_accepted": 1}
    corrected = conn.execute(
        "SELECT revision_of FROM assertion WHERE status = 'expert_accepted'"
    ).fetchone()["revision_of"]
    superseded = conn.execute(
        "SELECT assertion_id FROM assertion WHERE status = 'superseded'"
    ).fetchone()["assertion_id"]
    assert corrected == superseded  # the chain points back correctly


@requires_acquisition
def test_import_via_file_entry_point():
    conn = _db()
    result = importer.import_bundle_file(conn, FIXTURES / "accepted_bundle.json")
    assert isinstance(result, ImportResult)
    assert result.warnings == []
    assert result.inserted["assertion"] == 1


# --- a real pilot bundle -----------------------------------------------------


@requires_acquisition
def test_real_bundle_counts():
    conn = _db()
    path = ACQUISITION_DIR / "pilot" / "bundles" / "2001.06899.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    importer.import_bundle(conn, bundle)

    assert _count(conn, "assertion") == 125
    assert _count(conn, "entity") == 59
    assert _count(conn, "evidence") == 79
    assert _count(conn, "source_block") == 363
    assert _count(conn, "activity") == 15
    assert _count(conn, "qa_finding") == 46


@requires_acquisition
def test_evidence_is_many_to_many():
    conn = _db()
    path = ACQUISITION_DIR / "pilot" / "bundles" / "2001.06899.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    importer.import_bundle(conn, bundle)

    # an assertion citing 2 quotes should own 2 junction rows
    multi = [a for a in bundle["assertions"] if len(a["evidence_ids"]) == 2][0]
    n = conn.execute(
        "SELECT COUNT(*) FROM assertion_evidence WHERE assertion_id = ?", (multi["assertion_id"],)
    ).fetchone()[0]
    assert n == 2


# --- entity merge across two bundles -----------------------------------------


@requires_acquisition
def test_shared_entity_merges_across_bundles():
    """b_jet appears in several papers; it must end up as ONE entity row with
    multiple occurrences, and order must not change the merged row."""
    ids = ["2001.06899", "2004.04545"]  # both define hepkg:object:b_jet

    def build(order):
        conn = _db()
        for arxiv in order:
            path = ACQUISITION_DIR / "pilot" / "bundles" / f"{arxiv}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                importer.import_bundle(conn, json.load(handle))
        return conn

    conn = build(ids)
    entity_rows = conn.execute(
        "SELECT COUNT(*) FROM entity WHERE entity_id = 'hepkg:object:b_jet'"
    ).fetchone()[0]
    occ_rows = conn.execute(
        "SELECT COUNT(*) FROM entity_occurrence WHERE entity_id = 'hepkg:object:b_jet'"
    ).fetchone()[0]
    assert entity_rows == 1
    assert occ_rows == 2

    merged_forward = build(ids).execute(
        "SELECT aliases, attributes FROM entity WHERE entity_id = 'hepkg:object:b_jet'"
    ).fetchone()
    merged_reverse = build(list(reversed(ids))).execute(
        "SELECT aliases, attributes FROM entity WHERE entity_id = 'hepkg:object:b_jet'"
    ).fetchone()
    assert merged_forward["aliases"] == merged_reverse["aliases"]
    assert merged_forward["attributes"] == merged_reverse["attributes"]


# --- transaction safety ------------------------------------------------------


def test_rollback_leaves_db_untouched():
    """A mid-import failure must leave zero rows behind."""
    conn = _db()
    # entity subject_id points nowhere -> FK failure partway through the transaction
    bundle = {
        "schema_version": "hepkg-acquisition-v0.2",
        "bundle_id": "hepkg:bundle:0123456789abcdef01234567",
        "paper": {"arxiv_id": "9999.00001", "title": "t", "experiments": ["ATLAS"]},
        "source": {"source_hash": SOURCE_HASH, "paper_id": "9999.00001", "title": "t",
                   "source_path": "/x", "experiments": ["ATLAS"], "blocks": []},
        "entities": [{"entity_id": "e1", "kind": "paper", "label": "s"}],
        "evidence": [],
        "assertions": [{"assertion_id": "a1", "bundle_id": "x", "subject_id": "GHOST",
                        "predicate": "p", "family": "f", "object_id": "e1",
                        "object_value": None, "signature": None, "status": "machine_verified",
                        "evidence_ids": [], "qualifiers": {}}],
        "activities": [], "artifacts": [], "qa_findings": [],
        "completeness_findings": [], "expert_decisions": [],
    }
    with pytest.raises(sqlite3.IntegrityError):
        importer.import_bundle(conn, bundle)

    for table in ("paper", "source_snapshot", "bundle_import", "entity", "assertion"):
        assert _count(conn, table) == 0, f"{table} not rolled back"


@requires_acquisition
def test_shared_qa_finding_id_survives_across_bundles():
    """qa_finding.finding_id is content-derived and recurs across papers (e.g.
    'met outside vocabulary'). The (bundle_id, finding_id) key keeps one row per
    bundle rather than colliding."""
    conn = _db()
    # 2001.06899 and 2012.08600 share qa finding hepkg:qa:ac8e39f712e41407d279dda4
    for arxiv in ("2001.06899", "2012.08600"):
        path = ACQUISITION_DIR / "pilot" / "bundles" / f"{arxiv}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            importer.import_bundle(conn, json.load(handle))
    rows = conn.execute(
        "SELECT COUNT(*) FROM qa_finding WHERE finding_id = 'hepkg:qa:ac8e39f712e41407d279dda4'"
    ).fetchone()[0]
    assert rows == 2  # one per bundle, not a collision


@requires_acquisition
def test_completeness_findings_are_isolated_from_the_graph():
    """Contract condition 7: completeness findings are never scientific edges.
    Structurally guaranteed -- the table is joined to nothing in the graph."""
    conn = _db()
    path = ACQUISITION_DIR / "pilot" / "bundles" / "2001.06899.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        importer.import_bundle(conn, json.load(handle))
    # no foreign key anywhere points INTO completeness_finding
    fks = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    assert not any("REFERENCES completeness_finding" in (row["sql"] or "") for row in fks)
