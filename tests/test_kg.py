# =============================================================================
# HEPCoverageKG: smoke tests for the SQLite store (build step 1)
#
# Proves schema.sql builds, foreign keys are actually enforced, and the
# exactly-one-object CHECK on assertion rejects malformed rows. Uses in-memory
# databases so every test starts clean and leaves nothing behind.
# See vault/ideas/bundle-importer-design.md (build order, step 1).
# =============================================================================
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.kg import store

EXPECTED_TABLES = {
    "paper",
    "source_snapshot",
    "source_block",
    "bundle_import",
    "entity",
    "entity_occurrence",
    "activity",
    "assertion",
    "evidence",
    "assertion_evidence",
    "artifact",
    "qa_finding",
    "completeness_finding",
    "expert_decision",
    "assertion_status_history",
}
EXPECTED_VIEWS = {"accepted_view"}


def _fresh_db() -> sqlite3.Connection:
    conn = store.connect(":memory:")
    store.init_schema(conn)
    return conn


def _seed_minimal(conn: sqlite3.Connection) -> None:
    """Insert the parent rows an assertion needs, so its FKs are satisfied."""
    conn.execute("INSERT INTO paper (arxiv_id, title, experiments) VALUES ('p1', 't', '[]')")
    conn.execute(
        "INSERT INTO source_snapshot (source_hash, paper_id, title, source_path, experiments)"
        " VALUES ('sh1', 'p1', 't', '/x', '[]')"
    )
    conn.execute(
        "INSERT INTO bundle_import"
        " (bundle_id, paper_arxiv_id, source_hash, bundle_content_hash, schema_version, imported_at)"
        " VALUES ('b1', 'p1', 'sh1', 'h', 'hepkg-acquisition-v0.2', '2026-07-23T00:00:00Z')"
    )
    conn.execute("INSERT INTO entity (entity_id, kind, label) VALUES ('e1', 'paper', 'subject')")
    conn.execute("INSERT INTO entity (entity_id, kind, label) VALUES ('e2', 'result', 'object')")


def _insert_assertion(conn, object_id, object_value, signature) -> None:
    conn.execute(
        "INSERT INTO assertion"
        " (assertion_id, bundle_id, subject_id, predicate, family, status,"
        "  object_id, object_value, signature)"
        " VALUES ('a1', 'b1', 'e1', 'paper_reports_result', 'publication_context',"
        "         'machine_verified', ?, ?, ?)",
        (object_id, object_value, signature),
    )


def test_schema_creates_all_tables_and_views():
    conn = _fresh_db()
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }
    assert EXPECTED_TABLES <= names, f"missing tables: {EXPECTED_TABLES - names}"
    assert EXPECTED_VIEWS <= names, f"missing views: {EXPECTED_VIEWS - names}"


def test_init_schema_is_idempotent():
    conn = _fresh_db()
    store.init_schema(conn)  # second run must not raise or duplicate anything
    count = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'assertion'"
    ).fetchone()[0]
    assert count == 1


def test_foreign_keys_enabled():
    conn = _fresh_db()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_accepts_exactly_one_object_shape():
    conn = _fresh_db()
    _seed_minimal(conn)
    _insert_assertion(conn, object_id="e2", object_value=None, signature=None)
    assert conn.execute("SELECT COUNT(*) FROM assertion").fetchone()[0] == 1


def test_rejects_two_object_shapes():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_assertion(conn, object_id="e2", object_value='"x"', signature=None)


def test_rejects_zero_object_shapes():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_assertion(conn, object_id=None, object_value=None, signature=None)


def test_rejects_dangling_foreign_key():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_assertion(conn, object_id="does-not-exist", object_value=None, signature=None)


def test_rejects_malformed_json():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_assertion(conn, object_id=None, object_value="{not json", signature=None)


def test_accepted_view_filters_to_expert_accepted():
    conn = _fresh_db()
    _seed_minimal(conn)
    _insert_assertion(conn, object_id="e2", object_value=None, signature=None)
    assert conn.execute("SELECT COUNT(*) FROM accepted_view").fetchone()[0] == 0
    conn.execute("UPDATE assertion SET status = 'expert_accepted' WHERE assertion_id = 'a1'")
    assert conn.execute("SELECT COUNT(*) FROM accepted_view").fetchone()[0] == 1


def test_status_history_preserves_the_prior_status():
    """A promotion must be visible in the accepted view AND still recoverable."""
    conn = _fresh_db()
    _seed_minimal(conn)
    _insert_assertion(conn, object_id="e2", object_value=None, signature=None)

    # simulate what step 6 will do on a status-only change
    conn.execute(
        "INSERT INTO assertion_status_history"
        " (assertion_id, old_status, new_status, bundle_id, observed_at)"
        " VALUES ('a1', 'machine_verified', 'expert_accepted', 'b1', '2026-07-24T00:00:00Z')"
    )
    conn.execute("UPDATE assertion SET status = 'expert_accepted' WHERE assertion_id = 'a1'")

    assert conn.execute("SELECT COUNT(*) FROM accepted_view").fetchone()[0] == 1
    row = conn.execute(
        "SELECT old_status, new_status FROM assertion_status_history WHERE assertion_id = 'a1'"
    ).fetchone()
    assert row["old_status"] == "machine_verified"
    assert row["new_status"] == "expert_accepted"


def test_status_history_rejects_unknown_assertion():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO assertion_status_history"
            " (assertion_id, old_status, new_status, bundle_id, observed_at)"
            " VALUES ('ghost', 'machine_verified', 'expert_accepted', 'b1', '2026-07-24T00:00:00Z')"
        )


def test_transaction_rolls_back_on_error():
    conn = _fresh_db()
    _seed_minimal(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction(conn):
            _insert_assertion(conn, object_id="e2", object_value=None, signature=None)
            _insert_assertion(conn, object_id=None, object_value=None, signature=None)  # bad
    assert conn.execute("SELECT COUNT(*) FROM assertion").fetchone()[0] == 0
