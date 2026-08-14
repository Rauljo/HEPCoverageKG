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

    `check_same_thread=False` because the evaluation harness answers each
    question in a worker thread, so that one pathological question can be
    abandoned on a timeout rather than stalling an unattended run. That is safe
    *here specifically* and for two reasons that both have to hold: the
    connection is `mode=ro`, so no writer can interleave, and callers use it
    serially -- the harness runs a single worker at a time, and the Streamlit
    page is one request at a time. It is not a licence to share this connection
    across concurrent readers.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
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

    The check below is not defensive padding. Without an `assertion_id` in the
    subquery, SQLite resolves the bare name against the ENCLOSING scope --
    `ae.assertion_id` -- so the condition becomes `ae.assertion_id IN
    (ae.assertion_id ...)`, which is always true, and the template silently
    returns EVERY evidence row in the graph. `contents_of` hit exactly this on
    2026-08-03: 11 rows of output, 8,369 evidence ids, no error.
    """
    if "assertion_id" not in sql:
        raise ValueError(
            "_evidence_for needs an assertion_id column in the query; without it "
            "SQLite silently matches every row (see the note above)"
        )
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

    **`matched` says WHICH object the row came back for**, and without it the
    result is unreadable whenever the set holds more than one thing. Traced on a
    real question -- *"how many analyses estimate the t̄t+γ background?"*, true
    answer 1 -- a search returned 60 backgrounds (γ+jets, Z+jets, Wt, ttW, ...),
    the hop faithfully returned 142 rows for them, and **not one row named the
    background it was about**. 141 were wrong and nothing in the output could
    say which. `facets` already returns the labels that caused its match for
    exactly this reason; this is the same fix on the other hop.
    """
    ids = expand_canonical(conn, object_ids)
    sql = (
        "SELECT DISTINCT a.assertion_id AS assertion_id, a.subject_id AS entity_id,"
        "       es.label AS label, es.kind AS kind, a.predicate AS predicate,"
        "       eo.label AS matched"
        "  FROM assertion a"
        "  LEFT JOIN entity es ON es.entity_id = a.subject_id"
        "  LEFT JOIN entity eo ON eo.entity_id = a.object_id"
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


def contents_of(conn, paper_ids: str | list[str],
                predicate: Optional[str] = None) -> QueryResult:
    """What one paper actually says. The inverse of `papers_of`.

    Every other template starts from a CONCEPT and finds papers. Nothing went the
    other way, and on 2026-08-03 that turned out to matter: asked "which
    generators does analysis 2001.06899 use?", the planner searched for the arXiv
    id, found nothing usable, then passed the id straight into `describe` as if
    it were an entity id -- **60 `unknown_entity_id` errors across 45 questions**,
    and answers claiming a paper "is not found in the current graph" when it is
    plainly there.

    Not a tuning problem: the operation did not exist. And for a COVERAGE MAP the
    most natural question about one paper is "what does this analysis cover?", so
    the gap mattered to the product and not only to the evaluation.

    Joins `entity_occurrence` on **bundle_id AND entity_id**, the way `count`
    does. Joining on entity_id alone is the D-043 over-count: a subject entity
    shared across ten papers would drag in all ten regardless of which paper
    asserted the fact.
    """
    papers = [paper_ids] if isinstance(paper_ids, str) else list(paper_ids)
    if not papers:
        return QueryResult(shape="contents", note="no papers given")

    where = f"eo.paper_id IN ({_placeholders(len(papers))})"
    params: tuple = tuple(papers)
    if predicate:
        where += " AND a.predicate = ?"
        params = (*params, predicate)

    joins = (
        "  FROM assertion a"
        "  JOIN entity_occurrence eo"
        "    ON eo.bundle_id = a.bundle_id AND eo.entity_id = a.subject_id"
        "  LEFT JOIN entity o ON o.entity_id = a.object_id"
        f" WHERE {where}"
    )
    # Two queries on purpose. The rows are DISTINCT on (predicate, object) so the
    # model sees one line per fact rather than one per assertion; the evidence
    # lookup needs assertion_id, and adding it to the display query would undo
    # that grouping.
    sql = (
        "SELECT DISTINCT eo.paper_id AS paper_id, a.predicate AS predicate,"
        "       a.object_id AS object_id,"
        "       COALESCE(o.label, a.object_value) AS object,"
        "       o.kind AS object_kind"
        + joins + " ORDER BY a.predicate, object"
    )
    evidence_sql = "SELECT DISTINCT a.assertion_id AS assertion_id" + joins

    result = QueryResult(
        shape="contents",
        rows=[dict(r) for r in conn.execute(sql, params)],
        sql=sql,
        params=params,
        note=f"{len(papers)} paper(s)" + (f", predicate {predicate}" if predicate else ""),
    )
    result.evidence_ids = _evidence_for(conn, evidence_sql, params)
    return result


def facets(conn, field: str, values: str | list[str], mode: str = "all",
           category: Optional[str] = None, experiment: Optional[str] = None,
           vocabulary: str = "facets-v1") -> QueryResult:
    """Papers whose analysis card carries these closed-vocabulary values.

    The cheap rung of the ladder (S-68). No model call, no retrieval, no
    spelling to get right -- a set operation over enum keys. "Which searches
    select b-jets and missing transverse momentum?" is
    `facets("objects", ["BJet", "MET"], category="search")`.

    Returns THE ENTITY TRAIL, not just paper ids, and this is the point. The
    card is lossy on purpose: six papers carry the `ABCD` tag and all six do a
    different ABCD --

        2004.01678  Modified ABCD estimate
        2012.01581  Data-driven ABCD-style ratio method using eight regions
        2604.27044  Multidimensional ABCD reweighting technique (CR-to-SR)

    A reader shown only the six ids would conclude they share a method. Shown
    the labels, they can see six variants -- which for a coverage map is the
    more interesting answer. The tag says WHERE TO LOOK; the label says what is
    actually there.

    `mode` is "all" (every value present, an intersection) or "any" (a union).
    The distinction is the difference between "b-jets AND MET" and "b-jets OR
    MET", and getting it wrong silently changes the question.

    Coverage travels in `note`: a facet miss is invisible -- the entity keeps
    its label and every fact and simply never appears here -- so a result that
    did not say what it omitted would read as complete when it is not.
    """
    wanted = [values] if isinstance(values, str) else list(values)
    if not wanted:
        return QueryResult(shape="facets", note="no values given")
    if mode not in ("all", "any"):
        raise ValueError(f"mode must be 'all' or 'any', got {mode!r}")

    marks = _placeholders(len(wanted))
    where = ["f.field = ?", "f.vocabulary = ?", f"f.value IN ({marks})"]
    params: tuple = (field, vocabulary, *wanted)
    if category:
        where.append("p.category = ?")
        params = (*params, category)
    if experiment:
        # experiments is a JSON array on `paper`; EXISTS over json_each keeps
        # the match exact rather than substring-matching the serialised text.
        where.append("EXISTS (SELECT 1 FROM json_each(p.experiments) je"
                     " WHERE je.value = ?)")
        params = (*params, experiment)

    having = ""
    if mode == "all" and len(wanted) > 1:
        # Intersection: the paper must carry every requested value. Counted over
        # DISTINCT values, so one value tagged on three entities is still one.
        having = " HAVING COUNT(DISTINCT f.value) = ?"
        params = (*params, len(wanted))

    sql = (
        "SELECT p.arxiv_id AS paper_id, p.category AS category,"
        "       GROUP_CONCAT(DISTINCT f.value) AS matched"
        "  FROM entity_facet f"
        "  JOIN paper p ON p.arxiv_id = f.paper_id"
        f" WHERE {' AND '.join(where)}"
        " GROUP BY p.arxiv_id, p.category" + having + " ORDER BY p.arxiv_id"
    )
    rows = [dict(r) for r in conn.execute(sql, params)]
    paper_ids = [r["paper_id"] for r in rows]

    # The un-projected half: which label in each paper produced the match.
    trail: list[dict] = []
    if paper_ids:
        tmarks = _placeholders(len(paper_ids))
        trail = [dict(r) for r in conn.execute(
            "SELECT DISTINCT f.paper_id AS paper_id, f.value AS value,"
            "       f.entity_id AS entity_id, eo.label AS label, eo.kind AS kind"
            "  FROM entity_facet f"
            "  JOIN entity_occurrence eo"
            "    ON eo.paper_id = f.paper_id AND eo.entity_id = f.entity_id"
            f" WHERE f.field = ? AND f.vocabulary = ? AND f.value IN ({marks})"
            f"   AND f.paper_id IN ({tmarks})"
            " ORDER BY f.paper_id, f.value, eo.label",
            (field, vocabulary, *wanted, *paper_ids),
        )]
        for row in rows:
            row["evidence_labels"] = [
                t["label"] for t in trail if t["paper_id"] == row["paper_id"]
            ]

    result = QueryResult(shape="facets", rows=rows, sql=sql, params=params)
    notes = [f"{len(rows)} paper(s), {field} {mode} {wanted}"]

    # An empty result has two very different causes and looks identical either
    # way: no paper does this, or the key was not a vocabulary value. Saying
    # which is the difference between a correct "none" and a silent miss.
    unknown = _unknown_facet_values(conn, field, wanted, vocabulary)
    if unknown:
        notes.append(
            f"NOT IN THE VOCABULARY: {unknown} -- these match nothing by definition."
            " Use a key exactly as listed in the schema card, or drop to `search`"
            " if the concept has no facet"
        )
    notes.append(_facet_coverage_note(conn, field, vocabulary))
    notes.append(
        "facet tags are a closed vocabulary and miss what they have no pattern"
        " for -- treat this as a candidate set, and read the labels before counting"
    )
    result.note = "; ".join(notes)
    return result


def _unknown_facet_values(conn, field: str, wanted: list[str], vocabulary: str) -> list[str]:
    """Requested values that do not exist in this field at all."""
    marks = _placeholders(len(wanted))
    known = {
        r["value"] for r in conn.execute(
            f"SELECT DISTINCT value FROM entity_facet"
            f" WHERE field = ? AND vocabulary = ? AND value IN ({marks})",
            (field, vocabulary, *wanted),
        )
    }
    return [v for v in wanted if v not in known]


def facet_entities(conn, field: str, value: str, kind: Optional[str] = None,
                   vocabulary: str = "facets-v1") -> QueryResult:
    """The distinct THINGS carrying one facet tag, not the papers carrying it.

    `facets` answers "which papers use an ABCD estimate" -- 6. This answers
    "and what are the six of them", which for a coverage map is usually the
    more interesting question:

        ABCD data-driven background estimation method
        ABCD (matrix) data-driven background estimation using control regions
        Data-driven ABCD-style ratio method using eight non-overlapping regions
        Modified ABCD estimate
        Multidimensional ABCD reweighting technique (CR-to-SR, data-driven)
        Two-dimensional ABCD sideband method using control regions B, C, D

    One tag, six genuinely different methods. A tag is a family; this is what
    the family contains.

    Three counts, reported separately for the same reason `count` reports three:
    they answer different questions and conflating them is how "how many" goes
    wrong.
        papers    how many analyses mention anything with this tag
        entities  how many raw entity records carry it
        distinct  how many remain after alias merging -- the inventory answer

    `distinct` is the number that MOVES with deduplication. Paper counts do not:
    canonical expansion and search breadth already reach every spelling, so the
    dedup ablation measured on paper-counting questions would show nothing
    (D-051). Here the swing is real -- BJet is 10 raw entities and 4 after
    merging, JES 21 and 13 -- which is what makes this the question shape the
    ablation needs.
    """
    params: tuple = (field, vocabulary, value)
    kind_clause = ""
    if kind:
        kind_clause = " AND eo.kind = ?"
        params = (*params, kind)

    # Two queries rather than a GROUP_CONCAT of the labels. SQLite's
    # GROUP_CONCAT(DISTINCT x) cannot take a custom separator, so it joins on
    # commas -- and these labels contain commas ("b-tagged jet (MV2c10, 77%
    # efficiency)"), which silently split one label into two.
    sql = (
        "SELECT COALESCE(c.canonical_id, f.entity_id) AS canonical_id,"
        "       eo.kind AS kind,"
        "       COUNT(DISTINCT f.paper_id) AS papers,"
        "       COUNT(DISTINCT f.entity_id) AS records"
        "  FROM entity_facet f"
        "  JOIN entity_occurrence eo"
        "    ON eo.paper_id = f.paper_id AND eo.entity_id = f.entity_id"
        "  LEFT JOIN entity_canonical c ON c.entity_id = f.entity_id"
        " WHERE f.field = ? AND f.vocabulary = ? AND f.value = ?"
        f"{kind_clause}"
        " GROUP BY COALESCE(c.canonical_id, f.entity_id), eo.kind"
        " ORDER BY papers DESC, canonical_id"
    )
    rows = [dict(r) for r in conn.execute(sql, params)]

    labels: dict[str, list[str]] = {}
    if rows:
        for r in conn.execute(
            "SELECT DISTINCT COALESCE(c.canonical_id, f.entity_id) AS canonical_id,"
            "       eo.label AS label"
            "  FROM entity_facet f"
            "  JOIN entity_occurrence eo"
            "    ON eo.paper_id = f.paper_id AND eo.entity_id = f.entity_id"
            "  LEFT JOIN entity_canonical c ON c.entity_id = f.entity_id"
            " WHERE f.field = ? AND f.vocabulary = ? AND f.value = ?"
            f"{kind_clause}"
            " ORDER BY canonical_id, eo.label",
            params,
        ):
            labels.setdefault(r["canonical_id"], []).append(r["label"])
    for row in rows:
        row["labels"] = labels.get(row["canonical_id"], [])

    totals = conn.execute(
        "SELECT COUNT(DISTINCT f.paper_id) AS papers,"
        "       COUNT(DISTINCT f.entity_id) AS entities,"
        "       COUNT(DISTINCT COALESCE(c.canonical_id, f.entity_id)) AS distinct_things"
        "  FROM entity_facet f"
        "  LEFT JOIN entity_canonical c ON c.entity_id = f.entity_id"
        " WHERE f.field = ? AND f.vocabulary = ? AND f.value = ?",
        (field, vocabulary, value),
    ).fetchone()

    result = QueryResult(shape="facet_entities", rows=rows, sql=sql, params=params)
    notes = [
        f"'{value}' covers {totals['papers']} paper(s), {totals['entities']} entity record(s),"
        f" {totals['distinct_things']} distinct thing(s) after alias merging"
    ]
    if _unknown_facet_values(conn, field, [value], vocabulary):
        notes.append(
            f"NOT IN THE VOCABULARY: '{value}' -- it matches nothing by definition."
            " Take keys from search results rather than guessing")
    notes.append(
        "a tag is a FAMILY, not one method -- read the labels before treating"
        " these as the same thing")
    result.note = "; ".join(notes)
    return result


def _facet_coverage_note(conn, field: str, vocabulary: str) -> str:
    """How much of this field's raw material carries any tag at all.

    Kept beside `facets` rather than in a report: an uncaveated set of paper ids
    is exactly what invites a confident answer over a partial index.

    Kinds come from the declared mapping, NOT from whatever kinds happen to
    appear in `entity_facet`. Reading them back from the data joins on entity_id
    alone, and an entity_id that is `detector_object` in one paper and
    `object_definition` in another drags the second kind in -- inflating the
    denominator (515 rather than 486 for `objects`) and understating coverage.
    """
    from hepcoveragekg.facets import CARD_FIELDS

    kinds = sorted(k for k, f in CARD_FIELDS.items() if f == field)
    if not kinds:
        return f"unknown facet field '{field}'"

    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(tagged) AS matched FROM ("
        "  SELECT eo.paper_id, eo.entity_id,"
        "         MAX(CASE WHEN f.entity_id IS NULL THEN 0 ELSE 1 END) AS tagged"
        "    FROM entity_occurrence eo"
        "    LEFT JOIN entity_facet f"
        "      ON f.paper_id = eo.paper_id AND f.entity_id = eo.entity_id"
        "     AND f.field = ? AND f.vocabulary = ?"
        f"  WHERE eo.kind IN ({_placeholders(len(kinds))})"
        "   GROUP BY eo.paper_id, eo.entity_id)",
        (field, vocabulary, *kinds),
    ).fetchone()
    total, matched = row["total"] or 0, row["matched"] or 0
    if not total:
        return "coverage unknown"
    return f"vocabulary covers {matched}/{total} ({matched / total:.0%}) of '{field}' entities"


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


def quotes(conn, assertion_id: str) -> QueryResult:
    """Where one fact came from -- the verbatim sentence, its section, its paper.

    Called `quotes` everywhere in this layer. It wraps M2's `trace_assertion`,
    which keeps its own name in `kg/queries.py` because that is the contract's
    milestone -- but there is one name on this side of the boundary.
    """
    from hepcoveragekg.kg import queries

    traced: Any = queries.trace_assertion(conn, assertion_id)
    return QueryResult(
        shape="quotes",
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
    "contents_of": contents_of,  # paper -> its facts        (the inverse hop)
    "count": count,              # -> numbers                (terminal)
    "quotes": quotes,             # fact -> evidence          (terminal)
    "facets": facets,            # enum values -> papers     (the cheap rung)
    "facet_entities": facet_entities,  # one tag -> the things in it (inventory)
}

CONVENIENCES = {
    "list": list_papers,         # subjects_of + papers_of
    "compare": compare,          # two describes + set ops
    "crosstab": crosstab,        # self-join on shared subject
}

SHAPES = {**PRIMITIVES, **CONVENIENCES}
