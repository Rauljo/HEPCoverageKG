"""The conjunctive multi-hop, as a removable arm.

gf-01 asks which analyses are searches whose event selection uses b-tagged jets
AND missing transverse momentum. Precision is 0.90 on single-condition questions
and 0.30 on that one; gf-01-condition is the same concept with ONE condition and
scores 0.89 against gf-01's 0.48. The typed planner walks one edge per round and
93% of runs are over by round four, so "both" was not expressible.
"""
import sqlite3

import pytest

from hepcoveragekg.query import planner, templates


@pytest.fixture
def graph():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE entity(entity_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE assertion(assertion_id TEXT, subject_id TEXT, predicate TEXT,
                               object_id TEXT, object_value TEXT);
        CREATE TABLE same_as(a TEXT, b TEXT);
        -- `expand_canonical` reads this: a constraint naming one spelling must
        -- match the papers that spell it differently (56 entities for Pythia).
        CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);
        INSERT INTO entity VALUES ('r1','SR A','event_region'),('r2','SR B','event_region'),
                                  ('r3','SR C','event_region'),
                                  ('bj','b-tagged jet','detector_object'),
                                  ('met','missing pT','detector_object');
        INSERT INTO entity_occurrence VALUES ('r1','2001.00001','SR A','event_region'),
                                             ('r2','2002.00002','SR B','event_region'),
                                             ('r3','2003.00003','SR C','event_region');
        -- r1 has BOTH, r2 only b-jets, r3 only MET
        INSERT INTO assertion VALUES ('a1','r1','region_requires_object','bj',NULL),
                                     ('a2','r1','region_requires_object','met',NULL),
                                     ('a3','r2','region_requires_object','bj',NULL),
                                     ('a4','r3','region_requires_object','met',NULL);
    """)
    return conn


def test_all_means_intersect_not_union(graph):
    """The whole point: 'both' is not 'either'. r2 and r3 satisfy one condition
    each and must not come back."""
    r = templates.path(graph, [
        {"predicate": "region_requires_object", "object_ids": ["bj"]},
        {"predicate": "region_requires_object", "object_ids": ["met"]}], mode="all")
    assert [x["entity_id"] for x in r.rows] == ["r1"]


def test_any_unions(graph):
    r = templates.path(graph, [
        {"predicate": "region_requires_object", "object_ids": ["bj"]},
        {"predicate": "region_requires_object", "object_ids": ["met"]}], mode="any")
    assert sorted(x["entity_id"] for x in r.rows) == ["r1", "r2", "r3"]


def test_projecting_to_papers_saves_the_round_that_kills_runs(graph):
    """93% of runs finish in four rounds; a separate papers_of hop after the
    conjunction is often the round they do not have."""
    r = templates.path(graph, [
        {"predicate": "region_requires_object", "object_ids": ["bj"]},
        {"predicate": "region_requires_object", "object_ids": ["met"]}],
        project="papers", mode="all")
    assert sorted({x["paper_id"] for x in r.rows}) == ["2001.00001"]


def test_a_malformed_constraint_is_reported_not_silently_dropped(graph):
    r = templates.path(graph, [{"predicate": "region_requires_object"}])
    assert "missing predicate or object_ids" in (r.note or "")
    assert r.rows == []


def test_the_note_shows_each_constraint_and_the_survivors(graph):
    """Without it, an empty intersection is indistinguishable from a bad
    predicate -- and the model cannot tell which condition killed it."""
    r = templates.path(graph, [
        {"predicate": "region_requires_object", "object_ids": ["bj"]},
        {"predicate": "region_requires_object", "object_ids": ["met"]}], mode="all")
    assert "2 subjects" in r.note and "1 satisfying all" in r.note


def test_the_tool_is_invisible_when_the_arm_is_off():
    """Adding a tool rewrites the prompt for EVERY question, so an unflagged
    `path` would move the baseline it is measured against."""
    off = [t["name"] for t in planner.tools_for()]
    on = [t["name"] for t in planner.tools_for(path_tool=True)]
    assert "path" not in off and "path" in on
    assert off == [t for t in on if t != "path"]
