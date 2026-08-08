# =============================================================================
# HEPCoverageKG facets: derivation and storage
#
# Reads the graph, computes the two derived layers, replaces its own rows. Both
# derivations are idempotent (delete-then-insert scoped to one vocabulary
# version) and neither touches entity / entity_occurrence / assertion.
#
# Both report COVERAGE, because both are partial and silently so:
#
#   facets   -- a label matching no pattern gets no tag. Measured on the pilot:
#       65% of systematic uncertainties and 50% of detector objects match. The
#       misses split into genuine vocabulary gaps ("OS-SS subtraction") and
#       mis-kinded entities (signal regions filed as detector_object).
#   signatures -- 734 leaves from 2,692 candidates; the OR rule's `subchannel`
#       flag appears on 35 assertions in the whole graph.
#
# A miss is silent and safe: the entity keeps its label and every fact, it is
# just invisible to facet navigation. But a summary that does not say what it
# omitted invites overconfidence, so coverage travels with the answer rather
# than living only in a report. See D-052.
# =============================================================================
from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Union

from hepcoveragekg.facets import CARD_FIELDS, RENAMED_KINDS, signatures
from hepcoveragekg.facets.vocabulary import (
    FACET_VOCABULARY_VERSION,
    canonicalize_entity,
    facet_tags,
)
from hepcoveragekg.kg import store as kg_store

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Union[str, Path] = kg_store.DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = kg_store.connect(db_path)
    kg_store.init_schema(conn)          # facets tables FK into entity / assertion
    conn.executescript(kg_store.read_schema(_SCHEMA_PATH))
    return conn


# --------------------------------------------------------------------------
# Facets
# --------------------------------------------------------------------------

def facet_rows(conn, vocabulary: str = FACET_VOCABULARY_VERSION) -> tuple[list[tuple], Counter, Counter]:
    """(rows to insert, per-kind occurrence totals, per-kind matched).

    Reads `entity_occurrence`, NOT `entity`. A facet is a property of the label,
    and `entity.label` is a rollup that the schema itself flags as "convenience
    ONLY, not identity". 312 entity_ids carry different labels in different
    papers, and the difference decides the tag — 2107.12553 wrote "b-tagged
    small-R jet", which the anchored object pattern rejects, while the rollup
    "b-tagged jet" matches BJet. Deriving from the rollup credits that paper
    with an object it never named. Card parity caught this (D-052).
    """
    rows: list[tuple] = []
    total: Counter = Counter()
    matched: Counter = Counter()

    seen: set[tuple[str, str]] = set()
    for row in conn.execute(
        "SELECT DISTINCT paper_id, entity_id, kind, label FROM entity_occurrence"
    ):
        kind, label = row["kind"], row["label"]
        field = CARD_FIELDS.get(kind)
        if field is None:
            continue

        # One bundle per paper here, but count each (paper, entity) once so a
        # re-imported paper cannot inflate the coverage denominator.
        key = (row["paper_id"], row["entity_id"])
        first_time = key not in seen
        seen.add(key)
        if first_time:
            total[kind] += 1

        if kind in RENAMED_KINDS:
            canonical = canonicalize_entity(kind, label)
            if canonical:
                if first_time:
                    matched[kind] += 1
                rows.append((
                    row["paper_id"], row["entity_id"], field, canonical["canonical"],
                    canonical.get("version"), vocabulary,
                ))
        else:
            tags = facet_tags(kind, label)
            if tags and first_time:
                matched[kind] += 1
            for tag in tags:
                rows.append((
                    row["paper_id"], row["entity_id"], field, tag,
                    None, vocabulary,
                ))

    return rows, total, matched


def derive_facets(conn, *, vocabulary: str = FACET_VOCABULARY_VERSION) -> dict:
    """Replace this vocabulary's facet rows. Returns a coverage report."""
    rows, total, matched = facet_rows(conn, vocabulary)

    with kg_store.transaction(conn):
        conn.execute("DELETE FROM entity_facet WHERE vocabulary = ?", (vocabulary,))
        conn.executemany(
            "INSERT OR IGNORE INTO entity_facet"
            " (paper_id, entity_id, field, value, version, vocabulary) VALUES (?,?,?,?,?,?)",
            rows,
        )

    coverage = {
        kind: {
            "entities": total[kind],
            "matched": matched[kind],
            "missed": total[kind] - matched[kind],
            "hit_rate": matched[kind] / total[kind] if total[kind] else 0.0,
        }
        for kind in sorted(total)
    }
    return {
        "vocabulary": vocabulary,
        "rows": len(rows),
        "occurrences_tagged": len({(r[0], r[1]) for r in rows}),
        "entities_tagged": len({r[1] for r in rows}),
        "coverage": coverage,
    }


def unmatched_labels(conn, kind: str, limit: int = 50) -> list[str]:
    """Labels of `kind` that produced no facet — the vocabulary's blind spots.

    Kept as a first-class query rather than a debugging one-off: the gaps are a
    measurement about closed vocabularies, and the two causes (a real technique
    missing from the menu vs an entity filed under the wrong kind) are only
    separable by reading them.
    """
    rows = conn.execute(
        "SELECT DISTINCT eo.label FROM entity_occurrence eo"
        "  LEFT JOIN entity_facet f"
        "    ON f.paper_id = eo.paper_id AND f.entity_id = eo.entity_id"
        " WHERE eo.kind = ? AND f.entity_id IS NULL"
        " ORDER BY eo.label LIMIT ?",
        (kind, limit),
    )
    return [r["label"] for r in rows]


def card_coverage(conn, paper_id: str, vocabulary: str = FACET_VOCABULARY_VERSION) -> dict:
    """Per-kind (matched, total) for one paper's entities.

    What makes an analysis card honest. The card lists what matched; this says
    what it left out, so a partial summary cannot pass as a complete one.
    """
    rows = conn.execute(
        "SELECT eo.kind AS kind,"
        "       COUNT(DISTINCT eo.entity_id) AS total,"
        "       COUNT(DISTINCT f.entity_id) AS matched"
        "  FROM entity_occurrence eo"
        "  LEFT JOIN entity_facet f"
        "    ON f.paper_id = eo.paper_id AND f.entity_id = eo.entity_id"
        "   AND f.vocabulary = ?"
        " WHERE eo.paper_id = ? AND eo.kind IN (%s)"
        " GROUP BY eo.kind" % ",".join("?" * len(CARD_FIELDS)),
        (vocabulary, paper_id, *CARD_FIELDS),
    )
    return {
        r["kind"]: {"matched": r["matched"], "total": r["total"],
                    "missed": r["total"] - r["matched"]}
        for r in rows
    }


def card(conn, paper_id: str, vocabulary: str = FACET_VOCABULARY_VERSION) -> dict:
    """One paper's analysis card, in upstream's export shape.

    Rebuilt from the view rather than stored, and shaped to match
    `pilot/analysis_facets.jsonl` exactly so the parity test can compare them
    field for field.
    """
    paper = conn.execute(
        "SELECT arxiv_id, category, experiments FROM paper WHERE arxiv_id = ?",
        (paper_id,),
    ).fetchone()
    if paper is None:
        raise KeyError(f"no such paper: {paper_id}")

    values: dict[str, set[str]] = {field: set() for field in set(CARD_FIELDS.values())}
    for row in conn.execute(
        "SELECT field, value FROM analysis_card WHERE paper_id = ? AND vocabulary = ?",
        (paper_id, vocabulary),
    ):
        values.setdefault(row["field"], set()).add(row["value"])

    return {
        "paper_id": paper["arxiv_id"],
        "category": paper["category"],
        "experiments": sorted(json.loads(paper["experiments"] or "[]")),
        "vocabulary": vocabulary,
        **{field: sorted(vals) for field, vals in sorted(values.items())},
    }


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------

def derive_signatures(
    conn, *, vocabulary: str = signatures.SIGNATURE_SYNTHESIS_VERSION
) -> dict:
    """Replace this vocabulary's derived signatures. Returns a coverage report."""
    cuts = signatures.cuts_from_db(conn)
    derived = signatures.derive(cuts)

    rows = [
        (assertion_id, json.dumps(tree, sort_keys=True), int(is_or), vocabulary)
        for assertion_id, tree, is_or in derived
    ]
    with kg_store.transaction(conn):
        conn.execute(
            "DELETE FROM assertion_signature_derived WHERE vocabulary = ?", (vocabulary,)
        )
        conn.executemany(
            "INSERT OR REPLACE INTO assertion_signature_derived"
            " (assertion_id, signature, is_or_group, vocabulary) VALUES (?,?,?,?)",
            rows,
        )

    or_members = [a for a, _, is_or in derived if is_or]
    with_subchannel = sum(1 for c in cuts if signatures._subchannel(c) is not None)
    return {
        "vocabulary": vocabulary,
        "candidates": len(cuts),
        "derived": len(derived),
        "or_group_members": len(or_members),
        "with_subchannel_flag": with_subchannel,
        "native_signatures": sum(1 for c in cuts if c.has_native_signature),
    }
