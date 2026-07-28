from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

def _write_csv(conn: sqlite3.Connection, out_path: Path, query: str) -> None:
    """Executes a query and writes the result to a CSV file."""
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Get column names from the cursor description
    columns = [desc[0] for desc in cursor.description]
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in cursor:
            writer.writerow(row)

def export_csvs(conn: sqlite3.Connection, out_dir: Path) -> None:
    """Exports SQLite tables into CSV files for Neo4j import."""
    
    # ---------------------------------------------------------
    # NODES
    # ---------------------------------------------------------
    
    # 1. Papers
    _write_csv(conn, out_dir / "nodes_paper.csv", """
        SELECT 
            arxiv_id AS "arxiv_id:ID", 
            title, 
            category, 
            doi, 
            inspire_id, 
            cds_id, 
            'Paper' AS ":LABEL" 
        FROM paper
    """)

    # 2. Occurrences
    _write_csv(conn, out_dir / "nodes_occurrence.csv", """
        SELECT 
            bundle_id || ':' || entity_id AS "id:ID", 
            bundle_id, 
            entity_id, 
            kind, 
            label, 
            'Occurrence' AS ":LABEL" 
        FROM entity_occurrence
    """)

    # 3. Canonicals
    # Derived from entity_canonical + singletons that have no aliases
    _write_csv(conn, out_dir / "nodes_canonical.csv", """
        SELECT 
            c.canonical_id AS "id:ID",
            e.kind,
            e.label,
            'Canonical' AS ":LABEL"
        FROM (
            SELECT DISTINCT canonical_id FROM entity_canonical
            UNION
            SELECT DISTINCT eo.entity_id 
            FROM entity_occurrence eo
            LEFT JOIN entity_canonical ec ON eo.entity_id = ec.entity_id
            WHERE ec.entity_id IS NULL
        ) c
        JOIN entity e ON c.canonical_id = e.entity_id
    """)

    # 4. Literal Values (synthetic nodes for assertions without an object_id)
    _write_csv(conn, out_dir / "nodes_literal.csv", """
        SELECT DISTINCT
            'val:' || object_value AS "id:ID",
            object_value AS value,
            'object_value' AS type,
            'LiteralValue' AS ":LABEL"
        FROM assertion 
        WHERE object_value IS NOT NULL
        UNION
        SELECT DISTINCT
            'sig:' || signature AS "id:ID",
            signature AS value,
            'signature' AS type,
            'LiteralValue' AS ":LABEL"
        FROM assertion 
        WHERE signature IS NOT NULL
    """)

    # ---------------------------------------------------------
    # EDGES
    # ---------------------------------------------------------

    # 1. HAS_OCCURRENCE (Paper -> Occurrence)
    _write_csv(conn, out_dir / "edges_has_occurrence.csv", """
        SELECT 
            paper_id AS ":START_ID", 
            bundle_id || ':' || entity_id AS ":END_ID", 
            'HAS_OCCURRENCE' AS ":TYPE" 
        FROM entity_occurrence
    """)

    # 2. RESOLVES_TO (Occurrence -> Canonical)
    _write_csv(conn, out_dir / "edges_resolves_to.csv", """
        SELECT 
            eo.bundle_id || ':' || eo.entity_id AS ":START_ID", 
            COALESCE(ec.canonical_id, eo.entity_id) AS ":END_ID", 
            'RESOLVES_TO' AS ":TYPE"
        FROM entity_occurrence eo
        LEFT JOIN entity_canonical ec ON eo.entity_id = ec.entity_id
    """)

    # 3. ASSERTIONS (Occurrence -> Occurrence | LiteralValue)
    _write_csv(conn, out_dir / "edges_assertion.csv", """
        SELECT 
            bundle_id || ':' || subject_id AS ":START_ID",
            CASE 
                WHEN object_id IS NOT NULL THEN bundle_id || ':' || object_id
                WHEN object_value IS NOT NULL THEN 'val:' || object_value
                WHEN signature IS NOT NULL THEN 'sig:' || signature
            END AS ":END_ID",
            predicate AS ":TYPE",
            assertion_id,
            family,
            status,
            support,
            qualifiers
        FROM assertion
    """)

