# =============================================================================
# HEPCoverageKG: the milestone-1 count gate (step 8)
#
# The official finish line: all 60 pilot bundles reproduce the exact status
# counts, a second full pass is a no-op, and import order does not change the
# result. Also drives the CLI entry point end-to-end. Skipped without the repo.
# =============================================================================
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from hepcoveragekg import cli
from hepcoveragekg.ingest import importer
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
BUNDLES = ACQUISITION_DIR / "pilot" / "bundles"

TARGET = {"machine_verified": 11309, "quarantined": 2555, "rejected": 324}


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _fresh_db():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    return conn


def _import_all(conn, paths):
    for path in paths:
        importer.import_bundle(conn, _load(path))


def test_sixty_bundles_reproduce_the_status_target():
    conn = _fresh_db()
    paths = sorted(BUNDLES.glob("*.json.gz"))
    assert len(paths) == 60
    _import_all(conn, paths)

    assert queries.status_counts(conn) == TARGET
    assert sum(TARGET.values()) == 14188  # the headline number


def test_record_counts_match_the_totals():
    conn = _fresh_db()
    _import_all(conn, sorted(BUNDLES.glob("*.json.gz")))
    c = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert c("entity") == 5114
    assert c("entity_occurrence") == 6047
    assert c("evidence") == 8482
    assert c("activity") == 617
    assert c("assertion") == 14188


def test_second_full_pass_is_a_noop():
    conn = _fresh_db()
    paths = sorted(BUNDLES.glob("*.json.gz"))
    _import_all(conn, paths)
    before = queries.status_counts(conn)

    skipped = sum(importer.import_bundle(conn, _load(p)).skipped for p in paths)
    assert skipped == 60
    assert queries.status_counts(conn) == before


def test_import_order_does_not_change_counts():
    paths = sorted(BUNDLES.glob("*.json.gz"))
    forward = _fresh_db()
    _import_all(forward, paths)
    reverse = _fresh_db()
    _import_all(reverse, list(reversed(paths)))
    assert queries.status_counts(forward) == queries.status_counts(reverse)


# --- the CLI end-to-end ------------------------------------------------------


def test_cli_import_then_verify(tmp_path, capsys):
    db = str(tmp_path / "kg.db")

    rc = cli.main(["--db", db, "import", str(BUNDLES)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "60 imported, 0 skipped, 0 failed" in out
    assert "machine_verified 11309" in out

    rc = cli.main(["--db", db, "verify-counts"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "milestone-1 target met" in out


def test_cli_reimport_reports_all_skipped(tmp_path, capsys):
    db = str(tmp_path / "kg.db")
    cli.main(["--db", db, "import", str(BUNDLES)])
    capsys.readouterr()  # discard first run

    rc = cli.main(["--db", db, "import", str(BUNDLES)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 imported, 60 skipped, 0 failed" in out


def test_cli_verify_fails_on_empty_db(tmp_path, capsys):
    rc = cli.main(["--db", str(tmp_path / "empty.db"), "verify-counts"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT met" in out
