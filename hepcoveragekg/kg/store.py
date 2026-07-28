# =============================================================================
# HEPCoverageKG: SQLite store — connection, schema init, transactions
#
# The system of record for the bundle-import KG (milestone 1). See
# vault/ideas/bundle-importer-design.md (D-022, D-024). The graph model lives in
# schema.sql (STRICT tables, JSON-as-TEXT guarded by json_valid, an
# exactly-one-object CHECK on assertion, and the accepted_view). Any graph
# engine (NetworkX / Neo4j) is a projection built FROM this store, never the
# foundation.
# =============================================================================
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Persistent on-disk graph. Local to this machine, gitignored, rebuildable from
# the bundles. Callers pass their own path; tests use ":memory:".
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "hepkg.db"


def connect(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if absent) the SQLite store with FK enforcement on.

    isolation_level=None keeps us in autocommit mode so transaction() can drive
    BEGIN/COMMIT/ROLLBACK explicitly. Pass ":memory:" for a throwaway DB (tests).
    """
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# STRICT tables landed in SQLite 3.37.0 (2021-11). The HPC cluster ships 3.36,
# where the keyword is a syntax error -- and it is a parse error even for
# CREATE TABLE IF NOT EXISTS on a table that already exists, so it cannot be
# ignored. Rather than keep a hand-edited copy of schema.sql on the cluster
# (which silently drifts from this one), degrade at runtime.
_STRICT_MIN_VERSION = (3, 37, 0)


def supports_strict() -> bool:
    return sqlite3.sqlite_version_info >= _STRICT_MIN_VERSION


def _schema_sql() -> str:
    """The schema, with STRICT dropped when the local SQLite is too old.

    Losing STRICT loses per-column type enforcement, nothing else: every CHECK
    constraint, foreign key and index still applies. The importer's own
    validation gates do not rely on it.
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    if supports_strict():
        return sql
    return re.sub(r"\)\s*STRICT\s*;", ");", sql)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables/views/indexes if absent. Safe to call on an existing DB."""
    conn.executescript(_schema_sql())


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """All-or-nothing unit of work: commit on success, roll back on any error."""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
