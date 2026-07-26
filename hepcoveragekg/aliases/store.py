# =============================================================================
# HEPCoverageKG aliases: DB access for the derived quality layer
#
# Opens the SAME SQLite file as the import store, ensures the aliases tables,
# and reads entities / writes proposals / materializes canonical clusters. It
# never modifies entity / entity_occurrence / assertion.
#
# Timeline of DB effects (see D-025):
#   build()       -> writes same_as rows (status 'proposed') only
#   report        -> read-only
#   confirm()     -> flips proposed -> confirmed/auto
#   materialize() -> (re)builds entity_canonical from confirmed+auto edges
# =============================================================================
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from hepcoveragekg.aliases import cluster
from hepcoveragekg.aliases.normalize import Proposal
from hepcoveragekg.kg import store as kg_store

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

RESOLVING_STATUSES = ("confirmed", "auto")  # which same_as edges actually resolve


def connect(db_path: Union[str, Path] = kg_store.DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = kg_store.connect(db_path)
    kg_store.init_schema(conn)  # import store (idempotent) — aliases tables FK into entity
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def entities_with_kind(conn) -> list[tuple[str, str]]:
    return [(r["entity_id"], r["kind"]) for r in conn.execute("SELECT entity_id, kind FROM entity")]


def paper_counts(conn) -> dict[str, int]:
    """entity_id -> number of distinct papers it occurs in (for the canonical pick)."""
    return {
        r["entity_id"]: r["n"]
        for r in conn.execute(
            "SELECT entity_id, COUNT(DISTINCT paper_id) AS n FROM entity_occurrence GROUP BY entity_id"
        )
    }


def write_proposals(conn, proposals: Iterable[Proposal], *, now: Optional[str] = None) -> int:
    """Insert proposals as status 'proposed'. Idempotent (INSERT OR IGNORE on the
    (a, b, method) key), so re-running build adds nothing new."""
    now = now or datetime.now(timezone.utc).isoformat()
    rows = [(p.entity_id_a, p.entity_id_b, p.method, p.score, "proposed", now) for p in proposals]
    before = conn.execute("SELECT COUNT(*) FROM same_as").fetchone()[0]
    with kg_store.transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO same_as"
            " (entity_id_a, entity_id_b, method, score, status, created_at)"
            " VALUES (?,?,?,?,?,?)",
            rows,
        )
    return conn.execute("SELECT COUNT(*) FROM same_as").fetchone()[0] - before


def edges(conn, statuses: Sequence[str]) -> list[tuple[str, str]]:
    placeholders = ",".join("?" * len(statuses))
    return [
        (r["entity_id_a"], r["entity_id_b"])
        for r in conn.execute(
            f"SELECT entity_id_a, entity_id_b FROM same_as WHERE status IN ({placeholders})",
            tuple(statuses),
        )
    ]


def clusters(conn, statuses: Sequence[str]) -> list[set[str]]:
    """Connected-component clusters (size >= 2) over same_as edges of the given statuses."""
    return cluster.connected_components(edges(conn, statuses))


def _overrides(conn) -> dict[str, str]:
    return {r["cluster_member"]: r["canonical_id"] for r in conn.execute(
        "SELECT cluster_member, canonical_id FROM canonical_override")}


def confirm(conn, *, method: Optional[str] = None, to_status: str = "auto") -> int:
    """Promote proposed edges to a resolving status (default 'auto'). Optionally
    restrict to one method. This is the step a human runs after reviewing the list."""
    sql = "UPDATE same_as SET status = ? WHERE status = 'proposed'"
    args: list = [to_status]
    if method:
        sql += " AND method = ?"
        args.append(method)
    with kg_store.transaction(conn):
        cur = conn.execute(sql, args)
    return cur.rowcount


def materialize_canonical(conn) -> int:
    """Rebuild entity_canonical from confirmed+auto edges. Rows for singletons are
    omitted (they resolve to themselves). Returns the number of mapped entities."""
    counts = paper_counts(conn)
    overrides = _overrides(conn)
    with kg_store.transaction(conn):
        conn.execute("DELETE FROM entity_canonical")
        for members in clusters(conn, RESOLVING_STATUSES):
            canonical = cluster.pick_canonical(members, counts, overrides)
            source = "override" if any(m in overrides for m in members) else "auto"
            conn.executemany(
                "INSERT INTO entity_canonical (entity_id, canonical_id, cluster_size, canonical_source)"
                " VALUES (?,?,?,?)",
                [(m, canonical, len(members), source) for m in members],
            )
    return conn.execute("SELECT COUNT(*) FROM entity_canonical").fetchone()[0]
