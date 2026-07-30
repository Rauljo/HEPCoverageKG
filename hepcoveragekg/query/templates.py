"""
HEPCoverageKG query layer: the query templates.

Six parameterised shapes, filled with *retrieved entity ids* -- never with text a
model wrote (system.md S-04). The model chooses the shape and supplies the ids;
the SQL itself is fixed, tested, and lives here.

Free-form generation was rejected for this layer, and the pilot data makes the
case better than any benchmark: two traps in the schema are invisible unless you
have already been bitten by them, and both fail *silently*.

  join to papers through `entity_occurrence`, never `paper_reports_result`.
      That predicate is the semantically obvious choice -- it is literally named
      for the job -- and it links only the headline result of each paper: 60 of
      272. `entity_occurrence.paper_id` is written by the importer for every
      entity and reaches all 272. Using the predicate returns 22% of the truth
      with no error and no empty result.

  count facts, not rows.  One fact can be recorded as several assertions when a
      paper states it in more than one place, each with its own quote (219 of
      251 duplicate triples cite different evidence -- that is the graph working,
      not a defect). `paper_reports_result` doubles every one of its 60 facts.
      COUNT(*) therefore over-reports by up to 2x, and counting is the headline
      claim of the whole project.

  queries run over canonical clusters, not raw ids.  The aliases layer decided
      that `b-jet` / `bjet` / `b jet` are one thing; if the query layer ignored
      that, the deduplication work would have no effect on any answer, and the
      dedup-sensitive evaluation questions (S-11) could not discriminate between
      configurations.

Every template returns the evidence ids behind its rows, so the mechanical
faithfulness check (S-14) is a property of the result rather than a later bolt-on.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# One fact = one (subject, predicate, object) triple, with both ids resolved to
# their canonical cluster first.
#
# Two reasons for the resolution rather than raw ids:
#   it is what "the same fact" means.  `b-jet` and `bjet` are one entity after
#       deduplication, so two assertions differing only in which spelling they
#       used are one fact, not two.
#   it makes counting dedup-sensitive.  Improve the merging and fact counts fall,
#       because things collapse together. Without it, no evaluation question can
#       tell two deduplication configurations apart -- which is exactly what the
#       dedup-sensitive questions (S-11) are for.
#
# LEFT JOINs, so entities that were never canonicalised fall back to their own
# id and nothing needs a special case.
_FACT_KEY = (
    "COALESCE(cs.canonical_id, a.subject_id) || '|' || a.predicate || '|' ||"
    " COALESCE(co.canonical_id, a.object_id, a.object_value, '')"
)

# Joins that _FACT_KEY depends on. Kept beside it so the two cannot drift apart.
_FACT_JOINS = (
    " LEFT JOIN entity_canonical cs ON cs.entity_id = a.subject_id"
    " LEFT JOIN entity_canonical co ON co.entity_id = a.object_id"
)


@dataclass
class QueryResult:
    """Rows plus everything needed to check and cite them."""

    shape: str
    rows: list[dict] = field(default_factory=list)
    sql: str = ""
    params: tuple = ()
    evidence_ids: list[str] = field(default_factory=list)
    note: str = ""

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows


def read_only(db_path: Path | str) -> sqlite3.Connection:
    """A connection the query layer cannot write through.

    An agent-driven layer must not be able to modify the system of record, and
    the cheapest enforcement is at the connection rather than in review.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def expand_canonical(conn, entity_ids: str | list[str]) -> list[str]:
    """Every entity id in the same canonical cluster as any of the inputs.

    Accepts one id or many, because those cover two *different* gaps and a
    question usually needs both:

      canonical expansion closes what deduplication **merged** -- `b-jet`,
          `bjet` and `b jet` are one thing, so a question about any of them is a
          question about all of them.

      a multi-id call closes what deduplication **deliberately did not merge**.
          The graph holds 56 distinct Pythia entities (`PYTHIA 6`,
          `PYTHIA 8.226`, `PYTHIA v8.212`, ...). Merging them would be wrong --
          they are genuinely different generators and tunes -- but "how many
          analyses used Pythia?" is about all 56. Asking about one returns 1
          paper; the true answer is 58. Retrieval hands the whole set in.

    Returns the inputs unchanged when nothing was canonicalised, so callers need
    no special case for un-deduplicated entities.
    """
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    if not entity_ids:
        return []

    ids = set(entity_ids)
    marks = _placeholders(len(entity_ids))
    canonicals = {
        r["canonical_id"]
        for r in conn.execute(
            f"SELECT canonical_id FROM entity_canonical WHERE entity_id IN ({marks})",
            tuple(entity_ids),
        )
    }
    ids |= canonicals
    if canonicals:
        marks = _placeholders(len(canonicals))
        ids |= {
            r["entity_id"]
            for r in conn.execute(
                f"SELECT entity_id FROM entity_canonical WHERE canonical_id IN ({marks})",
                tuple(sorted(canonicals)),
            )
        }
    return sorted(ids)


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _evidence_for(conn, sql: str, params: tuple) -> list[str]:
    """Evidence ids behind whichever assertions a template matched.

    `sql` must select an `assertion_id` column; it is reused as a subquery so the
    evidence always corresponds to exactly the rows returned.
    """
    rows = conn.execute(
        "SELECT DISTINCT ae.evidence_id AS evidence_id FROM assertion_evidence ae"
        f" WHERE ae.assertion_id IN (SELECT assertion_id FROM ({sql}))",
        params,
    ).fetchall()
    return [r["evidence_id"] for r in rows]


# --------------------------------------------------------------------------
# The six shapes
# --------------------------------------------------------------------------

def count(conn, predicate: str, object_ids: str | list[str]) -> QueryResult:
    """How many papers are linked to this entity by this predicate.

    Counts distinct papers *and* distinct facts, because they answer different
    questions and conflating them is how "how many analyses" goes wrong.
    """
    ids = expand_canonical(conn, object_ids)
    sql = (
        "SELECT a.assertion_id AS assertion_id, eo.paper_id AS paper_id,"
        f"       {_FACT_KEY} AS fact"
        "  FROM assertion a"
        "  JOIN entity_occurrence eo"
        "    ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id"
        f"{_FACT_JOINS}"
        " WHERE a.predicate = ?"
        f"   AND a.object_id IN ({_placeholders(len(ids))})"
    )
    params = (predicate, *ids)
    rows = conn.execute(sql, params).fetchall()

    result = QueryResult(
        shape="count",
        rows=[{
            "papers": len({r["paper_id"] for r in rows}),
            "facts": len({r["fact"] for r in rows}),
            "assertions": len(rows),
        }],
        sql=sql,
        params=params,
        note=f"expanded to {len(ids)} canonical id(s)",
    )
    result.evidence_ids = _evidence_for(conn, sql, params)
    return result


def subjects_of(conn, predicate: str, object_ids: str | list[str]) -> QueryResult:
    """Which entities point AT these, through this predicate. The backward hop.

    `describe` walks subject -> object; this walks object -> subject. Both
    directions are needed or a chaining agent can only ever move one way through
    the graph, which rules out most real questions.

    Returns entities, not papers -- so the result can be fed straight into
    another hop. Use `papers_of` when the chain is finished.
    """
    ids = expand_canonical(conn, object_ids)
    sql = (
        "SELECT DISTINCT a.assertion_id AS assertion_id, a.subject_id AS entity_id,"
        "       es.label AS label, es.kind AS kind, a.predicate AS predicate"
        "  FROM assertion a"
        "  LEFT JOIN entity es ON es.entity_id = a.subject_id"
        " WHERE a.predicate = ?"
        f"   AND a.object_id IN ({_placeholders(len(ids))})"
        " ORDER BY es.label"
    )
    params = (predicate, *ids)
    result = QueryResult(
        shape="subjects_of",
        rows=[dict(r) for r in conn.execute(sql, params)],
        sql=sql,
        params=params,
    )
    result.evidence_ids = _evidence_for(conn, sql, params)
    return result


def papers_of(conn, entity_ids: str | list[str]) -> QueryResult:
    """Which papers these entities appear in. The terminal step of a chain.

    Reads `entity_occurrence`, which the importer fills for every entity, rather
    than following a predicate -- see the module docstring for why that
    distinction is load-bearing.
    """
    ids = expand_canonical(conn, entity_ids)
    if not ids:
        return QueryResult(shape="papers_of", note="no entities given")
    sql = (
        "SELECT DISTINCT eo.paper_id AS paper_id, eo.entity_id AS entity_id,"
        "       eo.label AS label"
        "  FROM entity_occurrence eo"
        f" WHERE eo.entity_id IN ({_placeholders(len(ids))})"
        " ORDER BY eo.paper_id"
    )
    params = tuple(ids)
    return QueryResult(
        shape="papers_of",
        rows=[dict(r) for r in conn.execute(sql, params)],
        sql=sql,
        params=params,
    )


def list_papers(conn, predicate: str, object_ids: str | list[str]) -> QueryResult:
    """Papers linked to these entities by this predicate.

    Convenience composition of `subjects_of` + `papers_of`, kept because it is by
    far the most common two-step chain and making an agent spend two rounds on it
    would be waste.
    """
    subjects = subjects_of(conn, predicate, object_ids)
    entity_ids = [r["entity_id"] for r in subjects.rows]
    papers = papers_of(conn, entity_ids) if entity_ids else QueryResult(shape="list")

    labels = {r["entity_id"]: r["label"] for r in subjects.rows}
    return QueryResult(
        shape="list",
        rows=[{"paper_id": r["paper_id"], "entity_id": r["entity_id"],
               "subject_label": labels.get(r["entity_id"], r["label"]),
               "predicate": predicate}
              for r in papers.rows],
        sql=papers.sql,
        params=papers.params,
        evidence_ids=subjects.evidence_ids,
        note="composed from subjects_of + papers_of",
    )


def describe(conn, subject_ids: str | list[str], predicate: Optional[str] = None) -> QueryResult:
    """Everything this entity connects to, optionally through one predicate.

    Objects may be nodes or literal values; both are returned in one `object`
    column so a caller never has to know which shape a predicate uses.
    """
    ids = expand_canonical(conn, subject_ids)
    where = f"a.subject_id IN ({_placeholders(len(ids))})"
    params: tuple = tuple(ids)
    if predicate:
        where += " AND a.predicate = ?"
        params = (*params, predicate)

    sql = (
        "SELECT a.assertion_id AS assertion_id, a.predicate AS predicate,"
        "       COALESCE(eo2.label, a.object_value) AS object,"
        "       a.object_id AS object_id, a.qualifiers AS qualifiers"
        "  FROM assertion a"
        "  LEFT JOIN entity eo2 ON eo2.entity_id = a.object_id"
        f" WHERE {where}"
        " ORDER BY a.predicate"
    )
    result = QueryResult(
        shape="describe",
        rows=[dict(r) for r in conn.execute(sql, params)],
        sql=sql,
        params=params,
    )
    result.evidence_ids = _evidence_for(conn, sql, params)
    return result


def canonical_of(conn, entity_id: str) -> str:
    """The cluster an entity belongs to, or itself when it was never merged."""
    row = conn.execute(
        "SELECT canonical_id FROM entity_canonical WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    return row["canonical_id"] if row else entity_id


def compare(conn, subject_a: str, subject_b: str, predicate: str) -> QueryResult:
    """What two entities share, and what only one of them has.

    Matching is on **canonical id**, with the label kept only for display.

    Labels are the tempting key and they are wrong in both directions. Two
    entities the aliases layer merged can still carry different labels
    (`PYTHIA 8.2` vs `Pythia v8.2`), so a label comparison reports nothing in
    common when everything is; and two genuinely different things can share a
    label across papers, so it also reports agreement that is not there. The
    canonical id is the identity the deduplication layer exists to produce --
    this is the query layer consuming it.

    Literal-valued objects have no id, so they fall back to matching on the value
    itself, which is all the graph knows about them.
    """
    def keyed(subject: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in describe(conn, subject, predicate).rows:
            if r["object"] is None:
                continue
            key = canonical_of(conn, r["object_id"]) if r["object_id"] else f"value:{r['object']}"
            out[key] = str(r["object"])
        return out

    a, b = keyed(subject_a), keyed(subject_b)
    shared_keys = a.keys() & b.keys()
    return QueryResult(
        shape="compare",
        rows=[{
            # display the A-side label for shared items; the B-side wording may
            # differ and showing both would suggest two things, not one
            "shared": sorted(a[k] for k in shared_keys),
            "only_a": sorted(a[k] for k in a.keys() - b.keys()),
            "only_b": sorted(b[k] for k in b.keys() - a.keys()),
        }],
        note=f"matched on canonical identity via '{predicate}'",
    )


def crosstab(conn, predicate_a: str, predicate_b: str) -> QueryResult:
    """Coverage grid: for each result, what it has under two predicates.

    The shape behind "which X has been studied for which Y" -- both sides are
    joined back to the same subject, so a row means one analysis genuinely did
    both, not that the two appear somewhere in the same paper.
    """
    sql = (
        "SELECT COALESCE(ea.label, a.object_value) AS a_value,"
        "       COALESCE(eb.label, b.object_value) AS b_value,"
        "       COUNT(DISTINCT eo.paper_id) AS papers"
        "  FROM assertion a"
        "  JOIN assertion b ON b.subject_id = a.subject_id AND b.bundle_id = a.bundle_id"
        "  JOIN entity_occurrence eo"
        "    ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id"
        "  LEFT JOIN entity ea ON ea.entity_id = a.object_id"
        "  LEFT JOIN entity eb ON eb.entity_id = b.object_id"
        " WHERE a.predicate = ? AND b.predicate = ?"
        " GROUP BY a_value, b_value"
        " ORDER BY papers DESC"
    )
    params = (predicate_a, predicate_b)
    return QueryResult(
        shape="crosstab",
        rows=[dict(r) for r in conn.execute(sql, params)],
        sql=sql,
        params=params,
        note="empty cells are combinations no analysis covered",
    )


def trace(conn, assertion_id: str) -> QueryResult:
    """Where one fact came from. Delegates to the M2 implementation."""
    from hepcoveragekg.kg import queries

    traced: Any = queries.trace_assertion(conn, assertion_id)
    return QueryResult(
        shape="trace",
        rows=[traced] if traced else [],
        params=(assertion_id,),
        evidence_ids=[e["evidence_id"] for e in (traced or {}).get("evidence", [])
                      if "evidence_id" in e],
    )


# The operations exposed to a planning agent as tools.
#
# Split into primitives (composable, one hop or one terminal step) and
# conveniences (compositions kept because they are common enough that making an
# agent spend several rounds on them would be pure waste). `search` is added by
# the retrieval module -- it is the only entry point that takes words rather
# than ids, so nothing else can start a chain without it.
PRIMITIVES = {
    "describe": describe,        # subject -> objects        (forward hop)
    "subjects_of": subjects_of,  # object  -> subjects       (backward hop)
    "papers_of": papers_of,      # entities -> papers        (terminal)
    "count": count,              # -> numbers                (terminal)
    "quotes": trace,             # fact -> evidence          (terminal)
}

CONVENIENCES = {
    "list": list_papers,         # subjects_of + papers_of
    "compare": compare,          # two describes + set ops
    "crosstab": crosstab,        # self-join on shared subject
}

SHAPES = {**PRIMITIVES, **CONVENIENCES, "trace": trace}
