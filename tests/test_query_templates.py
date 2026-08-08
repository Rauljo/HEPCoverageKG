"""Tests for the query-layer templates.

The tests that matter here are the ones guarding the two silent traps found in
the pilot data (see the module docstring of templates.py): joining to papers via
the wrong path, and counting rows instead of facts. Both return a plausible
wrong number rather than an error, so nothing else would catch them.
"""
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.query import templates as T


@pytest.fixture()
def conn(tmp_path):
    """A miniature graph reproducing the awkward shapes of the real one.

    - r1 IS linked to p1 by paper_reports_result; r2 is NOT (as in the pilot,
      where the predicate covers only the headline result of each paper)
    - the r1 -> genA fact is recorded TWICE with different evidence
    - genA1 and genA2 are merged into one canonical cluster
    - genB is a separate, deliberately unmerged generator
    """
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE paper (paper_id TEXT PRIMARY KEY);
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT, label TEXT);
        CREATE TABLE assertion (
            assertion_id TEXT PRIMARY KEY, bundle_id TEXT, paper_id TEXT,
            predicate TEXT, subject_id TEXT, object_id TEXT, object_value TEXT,
            qualifiers TEXT DEFAULT '{}');
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);

        INSERT INTO paper VALUES ('p1'), ('p2');
        INSERT INTO entity VALUES
            ('r1','result','Headline result'), ('r2','result','Sub result'),
            ('genA1','generator','PYTHIA 8.2'), ('genA2','generator','Pythia v8.2'),
            ('genB','generator','PYTHIA 6'),
            ('obj1','detector_object','Jet');
        INSERT INTO entity_occurrence VALUES
            ('b1','r1','p1','result','Headline result'),
            ('b1','r2','p1','result','Sub result'),
            ('b2','r3','p2','result','Other paper result');
        INSERT INTO entity_canonical VALUES ('genA1','genA1'), ('genA2','genA1');

        -- r1 is the only result the predicate links to a paper
        INSERT INTO assertion VALUES
            ('a-link','b1','p1','paper_reports_result','p1','r1',NULL,'{}');

        -- the SAME fact twice, different quotes (the real graph does this)
        INSERT INTO assertion VALUES
            ('a1','b1','p1','uses_generator','r1','genA1',NULL,'{}'),
            ('a2','b1','p1','uses_generator','r1','genA1',NULL,'{}');
        -- r2 uses the alias-merged twin; r3 (other paper) uses the unmerged one
        INSERT INTO assertion VALUES
            ('a3','b1','p1','uses_generator','r2','genA2',NULL,'{}'),
            ('a4','b2','p2','uses_generator','r3','genB',NULL,'{}');
        INSERT INTO assertion VALUES
            ('a5','b1','p1','defines_object','r1','obj1',NULL,'{}');

        INSERT INTO assertion_evidence VALUES
            ('a1','ev1'), ('a2','ev2'), ('a3','ev3'), ('a4','ev4');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


def test_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM assertion")


def test_counts_facts_not_rows(conn):
    """r1 -> genA1 is two assertions but ONE fact. Counting rows would report 2."""
    r = T.count(conn, "uses_generator", "genA1")
    assert r.rows[0]["assertions"] == 3, "raw rows: a1, a2 (dupes) + a3 (canonical twin)"
    assert r.rows[0]["facts"] == 2, "but only two distinct subject/predicate/object triples"


def test_canonical_expansion_pulls_in_merged_twin(conn):
    """Asking about genA1 must also answer for genA2 -- otherwise deduplication
    has no effect on any answer."""
    assert set(T.expand_canonical(conn, "genA1")) == {"genA1", "genA2"}
    assert set(T.expand_canonical(conn, "genA2")) == {"genA1", "genA2"}


def test_unmerged_entity_expands_to_itself(conn):
    assert T.expand_canonical(conn, "genB") == ["genB"]


def test_multiple_ids_are_unioned(conn):
    """The 56-Pythia case: entities deliberately NOT merged, asked about together."""
    one = T.count(conn, "uses_generator", "genA1").rows[0]
    many = T.count(conn, "uses_generator", ["genA1", "genB"]).rows[0]
    assert one["papers"] == 1
    assert many["papers"] == 2, "genB lives in p2, so the union must reach both papers"


def test_empty_id_list_is_safe(conn):
    assert T.expand_canonical(conn, []) == []


def test_papers_reached_via_occurrence_not_the_predicate(conn):
    """r2 has NO paper_reports_result row. It must still resolve to p1.

    This is the 60-of-272 trap: the obvious join loses every non-headline result.
    """
    linked = {r["object_id"] for r in conn.execute(
        "SELECT object_id FROM assertion WHERE predicate='paper_reports_result'")}
    assert "r2" not in linked, "fixture must reproduce the gap"

    papers = {r["paper_id"] for r in T.list_papers(conn, "uses_generator", "genA2").rows}
    assert papers == {"p1"}, "r2 reached p1 through entity_occurrence"


def test_results_carry_their_evidence(conn):
    """The faithfulness check (S-14) needs evidence attached to the rows, not
    fetched separately afterwards."""
    r = T.count(conn, "uses_generator", "genA1")
    assert set(r.evidence_ids) == {"ev1", "ev2", "ev3"}


def test_describe_returns_nodes_and_literals_in_one_column(conn):
    r = T.describe(conn, "r1")
    preds = {row["predicate"] for row in r.rows}
    assert {"uses_generator", "defines_object"} <= preds
    assert "Jet" in {row["object"] for row in r.rows}


def test_describe_can_filter_to_one_predicate(conn):
    r = T.describe(conn, "r1", "defines_object")
    assert {row["predicate"] for row in r.rows} == {"defines_object"}


def test_compare_matches_on_canonical_identity_not_labels(conn):
    """r1 uses genA1 ("PYTHIA 8.2"), r2 uses genA2 ("Pythia v8.2").

    Different ids AND different labels, but the aliases layer merged them, so
    they are one thing. Matching on labels reports nothing shared; matching on
    canonical identity gets it right. This is the query layer consuming the
    deduplication work.
    """
    r = T.compare(conn, "r1", "r2", "uses_generator")
    assert r.rows[0]["shared"], "merged entities must compare as shared"
    assert not r.rows[0]["only_b"], "r2 has nothing r1 lacks"


def test_compare_separates_genuinely_different_objects(conn):
    """r1 (genA1) vs r3 (genB): different generators, nothing shared."""
    r = T.compare(conn, "r1", "r3", "uses_generator")
    assert r.rows[0]["shared"] == []
    assert r.rows[0]["only_a"] and r.rows[0]["only_b"]


def test_primitives_cover_both_directions_and_a_terminal(conn):
    """A chaining agent needs to walk forward AND backward, and to end a chain.
    Missing any one of those makes most questions unreachable."""
    assert {"describe", "subjects_of", "papers_of", "count", "quotes"} <= set(T.PRIMITIVES)


def test_conveniences_are_compositions_not_new_capability(conn):
    assert set(T.CONVENIENCES) == {"list", "compare", "crosstab"}
    assert set(T.PRIMITIVES).isdisjoint(T.CONVENIENCES)


def test_subjects_of_walks_backward(conn):
    """genA1 is an object; the results using it must come back as entities."""
    r = T.subjects_of(conn, "uses_generator", "genA1")
    assert {row["entity_id"] for row in r.rows} == {"r1", "r2"}, "canonical twin included"
    assert all("kind" in row for row in r.rows), "entities, so a chain can continue"


def test_papers_of_ends_a_chain(conn):
    r = T.papers_of(conn, ["r1", "r2"])
    assert {row["paper_id"] for row in r.rows} == {"p1"}


def test_list_equals_its_two_primitives(conn):
    """The convenience must not diverge from the composition it stands in for."""
    composed = T.papers_of(conn, [x["entity_id"] for x in
                                  T.subjects_of(conn, "uses_generator", "genA1").rows])
    direct = T.list_papers(conn, "uses_generator", "genA1")
    assert {r["paper_id"] for r in direct.rows} == {r["paper_id"] for r in composed.rows}


def test_facts_collapse_across_merged_entities(conn):
    """r2 -> genA2 and r1 -> genA1 stay distinct (different subjects), but the
    fact key must resolve the OBJECT through its canonical cluster, so a merge
    changes the count. This is what makes counting dedup-sensitive."""
    r = T.count(conn, "uses_generator", "genA1")
    assert r.rows[0]["facts"] == 2
    assert r.rows[0]["assertions"] == 3

def test_contents_of_is_the_inverse_of_papers_of(conn):
    """Every other template starts from a concept and finds papers. Nothing went
    the other way, and on 2026-08-03 that produced 60 `unknown_entity_id` errors
    across 45 questions -- the planner passed an arXiv id into `describe` because
    no operation existed to read a paper."""
    rows = T.contents_of(conn, "p1").rows
    assert rows, "a paper must be readable"
    assert {r["paper_id"] for r in rows} == {"p1"}


def test_contents_of_can_filter_to_one_predicate(conn):
    all_facts = T.contents_of(conn, "p1").rows
    one = T.contents_of(conn, "p1", "sample_uses_generator").rows
    assert len(one) <= len(all_facts)
    assert {r["predicate"] for r in one} <= {"sample_uses_generator"}


def test_evidence_lookup_refuses_a_query_without_assertion_id():
    """Silent failure otherwise: SQLite resolves the bare column against the
    ENCLOSING scope, the condition becomes always-true, and the template returns
    every evidence row in the graph. `contents_of` hit this -- 11 rows of output,
    8,369 evidence ids, no error."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="assertion_id"):
        T._evidence_for(None, "SELECT paper_id FROM assertion", ())


# --- facets: the closed-vocabulary rung ---------------------------------------


@pytest.fixture()
def facet_conn(tmp_path):
    """A graph with a facet layer. Deliberately includes the two shapes that
    make facets easy to misread:

    - p1 and p2 both carry the ABCD tag from DIFFERENT labels (a family, not a
      shared method), so a result showing ids alone would imply they match
    - p3 has a detector_object entity that no pattern matched, so its coverage
      is partial while its card still looks complete
    """
    db = tmp_path / "f.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE paper (arxiv_id TEXT PRIMARY KEY, category TEXT, experiments TEXT);
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT, label TEXT);
        CREATE TABLE entity_facet (
            paper_id TEXT, entity_id TEXT, field TEXT, value TEXT,
            version TEXT, vocabulary TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);

        INSERT INTO paper VALUES
            ('p1','search','["ATLAS"]'), ('p2','search','["CMS"]'),
            ('p3','measurement','["ATLAS"]');
        INSERT INTO entity VALUES
            ('m1','background_method','Modified ABCD estimate'),
            ('m2','background_method','Two-dimensional ABCD sideband method'),
            ('o1','detector_object','b-tagged jet'),
            ('o1b','detector_object','b-tagged jet (MV2c10, 77% efficiency)'),
            ('o2','detector_object','Missing transverse momentum'),
            ('o3','detector_object','AFP forward proton candidate');
        INSERT INTO entity_occurrence VALUES
            ('b1','m1','p1','background_method','Modified ABCD estimate'),
            ('b1','o1','p1','detector_object','b-tagged jet'),
            ('b1','o2','p1','detector_object','Missing transverse momentum'),
            ('b2','m2','p2','background_method','Two-dimensional ABCD sideband method'),
            ('b2','o1','p2','detector_object','b-tagged jet'),
            -- a comma INSIDE a label: GROUP_CONCAT would split it in two
            ('b2','o1b','p2','detector_object','b-tagged jet (MV2c10, 77% efficiency)'),
            ('b3','o3','p3','detector_object','AFP forward proton candidate'),
            ('b3','o2','p3','detector_object','Missing transverse momentum');
        -- o1 and o1b are one thing after merging; o2 is unmerged
        INSERT INTO entity_canonical VALUES ('o1','o1'), ('o1b','o1');
        INSERT INTO entity_facet VALUES
            ('p1','m1','background_methods','ABCD',NULL,'facets-v1'),
            ('p2','m2','background_methods','ABCD',NULL,'facets-v1'),
            ('p1','o1','objects','BJet',NULL,'facets-v1'),
            ('p1','o2','objects','MET',NULL,'facets-v1'),
            ('p2','o1','objects','BJet',NULL,'facets-v1'),
            ('p2','o1b','objects','BJet',NULL,'facets-v1'),
            ('p3','o2','objects','MET',NULL,'facets-v1');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


def test_facets_all_is_an_intersection(facet_conn):
    """p1 has BJet AND MET; p2 has only BJet; p3 only MET."""
    r = T.facets(facet_conn, "objects", ["BJet", "MET"], mode="all")
    assert [row["paper_id"] for row in r.rows] == ["p1"]


def test_facets_any_is_a_union(facet_conn):
    r = T.facets(facet_conn, "objects", ["BJet", "MET"], mode="any")
    assert [row["paper_id"] for row in r.rows] == ["p1", "p2", "p3"]


def test_facets_single_value_needs_no_mode(facet_conn):
    r = T.facets(facet_conn, "objects", "BJet")
    assert [row["paper_id"] for row in r.rows] == ["p1", "p2"]


def test_facets_filters_by_category_and_experiment(facet_conn):
    assert [r["paper_id"] for r in
            T.facets(facet_conn, "objects", "MET", category="search").rows] == ["p1"]
    assert [r["paper_id"] for r in
            T.facets(facet_conn, "objects", "MET", experiment="ATLAS").rows] == ["p1", "p3"]
    # exact JSON membership, not a substring of the serialised array
    assert T.facets(facet_conn, "objects", "MET", experiment="ATLA").rows == []


def test_facets_returns_the_labels_that_caused_each_match(facet_conn):
    """The point of the tool. Two papers share the ABCD tag and do different
    things; ids alone would say they match."""
    r = T.facets(facet_conn, "background_methods", "ABCD")
    labels = {row["paper_id"]: row["evidence_labels"] for row in r.rows}
    assert labels["p1"] == ["Modified ABCD estimate"]
    assert labels["p2"] == ["Two-dimensional ABCD sideband method"]
    assert labels["p1"] != labels["p2"]


def test_facets_note_carries_coverage_and_a_caveat(facet_conn):
    """A facet miss is invisible, so an uncaveated result would read as complete.

    Six (paper, entity) detector_object pairs exist; five carry a tag. The odd
    one out is p3's forward proton, which no pattern matches -- a real
    vocabulary gap, and exactly what the coverage figure is there to expose.
    """
    r = T.facets(facet_conn, "objects", "BJet")
    assert "5/6" in r.note, r.note
    assert "candidate" in r.note.lower()


def test_facets_rejects_an_unknown_mode(facet_conn):
    with pytest.raises(ValueError):
        T.facets(facet_conn, "objects", ["BJet"], mode="either")


def test_facets_empty_values_is_safe(facet_conn):
    assert T.facets(facet_conn, "objects", []).rows == []


def test_facets_respects_the_vocabulary_version(facet_conn):
    """Two vocabularies can coexist so they can be diffed; a query must not mix
    them."""
    assert T.facets(facet_conn, "objects", "BJet", vocabulary="facets-v2").rows == []


def test_facets_says_when_a_value_is_not_in_the_vocabulary(facet_conn):
    """Zero rows has two causes that look identical: nobody does this, or the
    key was wrong. A model cannot tell them apart without being told."""
    r = T.facets(facet_conn, "objects", ["b-jet"])
    assert r.rows == []
    assert "NOT IN THE VOCABULARY" in r.note
    assert "b-jet" in r.note

    # a real key that simply matches nothing here must NOT be flagged as unknown
    ok = T.facets(facet_conn, "objects", ["BJet"], category="measurement")
    assert ok.rows == []
    assert "NOT IN THE VOCABULARY" not in ok.note


def test_facets_flags_only_the_unknown_half_of_a_mixed_call(facet_conn):
    r = T.facets(facet_conn, "objects", ["BJet", "bjet"], mode="any")
    assert "bjet" in r.note and "NOT IN THE VOCABULARY" in r.note
    assert [row["paper_id"] for row in r.rows] == ["p1", "p2"]


# --- facet_entities: the inventory shape --------------------------------------


def test_facet_entities_lists_the_things_not_the_papers(facet_conn):
    """Two papers share the ABCD tag and did different things. `facets` says
    '2 papers'; this says which two methods."""
    r = T.facet_entities(facet_conn, "background_methods", "ABCD")
    labels = sorted(lab for row in r.rows for lab in row["labels"])
    assert labels == ["Modified ABCD estimate",
                      "Two-dimensional ABCD sideband method"]


def test_facet_entities_collapses_merged_entities(facet_conn):
    """o1 and o1b are one thing after alias merging. Three raw BJet records
    across two papers, but ONE distinct thing -- this is the number that moves
    with deduplication, and the reason the dedup ablation needs this shape
    rather than paper counts (D-051)."""
    r = T.facet_entities(facet_conn, "objects", "BJet")
    assert len(r.rows) == 1
    assert r.rows[0]["papers"] == 2
    assert r.rows[0]["records"] == 2          # o1 and o1b
    assert "2 paper(s)" in r.note and "1 distinct thing(s)" in r.note


def test_facet_entities_keeps_labels_containing_commas(facet_conn):
    """SQLite's GROUP_CONCAT(DISTINCT x) cannot take a separator, so it joins on
    commas -- and these labels contain commas. Aggregating in SQL silently split
    'b-tagged jet (MV2c10, 77% efficiency)' into two labels."""
    r = T.facet_entities(facet_conn, "objects", "BJet")
    labels = r.rows[0]["labels"]
    assert "b-tagged jet (MV2c10, 77% efficiency)" in labels
    assert not any(lab.strip().startswith("77%") for lab in labels)
    assert len(labels) == 2


def test_facet_entities_unmerged_entity_stands_alone(facet_conn):
    r = T.facet_entities(facet_conn, "objects", "MET")
    assert len(r.rows) == 1
    assert r.rows[0]["papers"] == 2           # p1 and p3
    assert r.rows[0]["records"] == 1


def test_facet_entities_filters_by_kind(facet_conn):
    assert T.facet_entities(facet_conn, "objects", "BJet",
                            kind="background_method").rows == []
    assert T.facet_entities(facet_conn, "objects", "BJet",
                            kind="detector_object").rows != []


def test_facet_entities_flags_an_unknown_key(facet_conn):
    r = T.facet_entities(facet_conn, "objects", "bjet")
    assert r.rows == []
    assert "NOT IN THE VOCABULARY" in r.note


def test_facet_entities_warns_that_a_tag_is_a_family(facet_conn):
    r = T.facet_entities(facet_conn, "background_methods", "ABCD")
    assert "FAMILY" in r.note
