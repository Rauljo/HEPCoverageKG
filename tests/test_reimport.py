# =============================================================================
# HEPCoverageKG: re-import guard tests (step 6)
#
# Milestone-1 contract conditions 2 (identical reimport changes nothing) and 3
# (conflicting content aborts). Plus: a status-only re-import is NOT a conflict
# (it warns), proving the guard keys on identity, not whole-assertion equality.
# =============================================================================
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path

import pytest

from hepcoveragekg.ingest import importer
from hepcoveragekg.ingest.errors import BundleConflictError
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
BUNDLES = ACQUISITION_DIR / "pilot" / "bundles"


def _db():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    return conn


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _all_counts(conn):
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


# --- condition 2: idempotency ------------------------------------------------


@requires_acquisition
def test_identical_reimport_is_a_noop():
    conn = _db()
    bundle = _fixture("accepted_bundle")
    importer.import_bundle(conn, bundle)
    before = _all_counts(conn)

    result = importer.import_bundle(conn, copy.deepcopy(bundle))
    assert result.skipped is True
    assert _all_counts(conn) == before  # nothing changed anywhere


@requires_acquisition
def test_reimport_all_60_changes_no_counts():
    conn = _db()
    paths = sorted(BUNDLES.glob("*.json.gz"))
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            importer.import_bundle(conn, json.load(handle))
    after_first = _all_counts(conn)
    assert after_first["assertion"] == 14188  # sanity: the full load landed

    skipped = 0
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            if importer.import_bundle(conn, json.load(handle)).skipped:
                skipped += 1
    assert skipped == 60
    assert _all_counts(conn) == after_first


# --- condition 3: conflict aborts --------------------------------------------


@requires_acquisition
def test_conflicting_content_aborts_and_leaves_db_untouched():
    conn = _db()
    importer.import_bundle(conn, _fixture("accepted_bundle"))
    before = _all_counts(conn)

    with pytest.raises(BundleConflictError, match="non-status change is a conflict"):
        importer.import_bundle(conn, _fixture("conflicting_bundle"))
    assert _all_counts(conn) == before  # aborted before any write


# --- the guard keys on identity, not equality --------------------------------


@requires_acquisition
def test_status_only_reimport_is_not_a_conflict_but_warns():
    """Same bundle, one assertion's status flipped -> a legitimate (M4) change,
    NOT a conflict. Must warn and skip, not raise."""
    conn = _db()
    bundle = _fixture("accepted_bundle")
    importer.import_bundle(conn, bundle)
    before = _all_counts(conn)

    promoted = copy.deepcopy(bundle)
    promoted["assertions"][0]["status"] = "superseded"  # status only
    result = importer.import_bundle(conn, promoted)

    assert result.skipped is True
    assert any("status change" in w for w in result.warnings)
    assert _all_counts(conn) == before  # not applied yet (milestone 4)


@requires_acquisition
def test_new_bundle_id_still_imports_normally():
    """A different paper (new bundle_id) is unaffected by the guard."""
    conn = _db()
    importer.import_bundle(conn, _fixture("accepted_bundle"))
    with gzip.open(BUNDLES / "2001.06899.json.gz", "rt", encoding="utf-8") as handle:
        result = importer.import_bundle(conn, json.load(handle))
    assert result.skipped is False
    assert result.inserted["assertion"] == 125
