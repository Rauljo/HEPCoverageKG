"""Tests for the retrieved-subgraph view.

Everything here is about what NOT to draw. The real graph offers ~395 nodes for
a single question, and a picture of 395 nodes is worse than no picture: it looks
like insight and conveys none.
"""
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.query import subgraph as SG
from hepcoveragekg.query import templates as T


@pytest.fixture()
def conn(tmp_path):
    """Two merged generators, one unmerged, and samples of varying connectivity."""
    db = tmp_path / "g.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE assertion (assertion_id TEXT PRIMARY KEY, predicate TEXT,
            subject_id TEXT, object_id TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);

        INSERT INTO entity VALUES
            ('gA1','generator','PYTHIA 8.2'), ('gA2','generator','Pythia v8.2'),
            ('gB','generator','Herwig 7'),
            ('s1','sample','ttbar sample'), ('s2','sample','Z+jets sample'),
            ('s3','sample','lonely sample'),
            ('lone','generator','Never used');
        -- gA1 and gA2 are one thing after deduplication
        INSERT INTO entity_canonical VALUES ('gA1','gA1'), ('gA2','gA1');

        -- s1 touches both generators; s2 touches one; s3 touches something else
        INSERT INTO assertion VALUES
            ('a1','uses_generator','s1','gA1'),
            ('a2','uses_generator','s1','gB'),
            ('a3','uses_generator','s2','gA2'),
            ('a4','uses_generator','s3','other');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


def test_merged_entities_collapse_to_one_node(conn):
    """gA1 and gA2 are one thing; drawing both would double-count the merge."""
    g = SG.build(conn, ["gA1", "gA2", "gB"])
    ids = {n.node_id for n in g.nodes}
    assert "gA2" not in ids, "the twin folded into its cluster"
    assert "gA1" in ids


def test_the_merge_is_shown_not_hidden(conn):
    """S-28: a merge that changes what you see should say so."""
    g = SG.build(conn, ["gA1", "gA2", "gB"])
    node = next(n for n in g.nodes if n.node_id == "gA1")
    assert node.members == 2
    assert "+1 spellings" in SG.to_dot(g)


def test_retrieved_nodes_are_marked(conn):
    g = SG.build(conn, ["gA1", "gA2"])
    assert any(n.retrieved for n in g.nodes)
    assert any(not n.retrieved for n in g.nodes), "neighbours are drawn too"


def test_edges_carry_the_predicate(conn):
    """Without it the graph says two things are related, which is the least
    interesting fact available about them."""
    g = SG.build(conn, ["gA1", "gB"])
    assert all(e[1] == "uses_generator" for e in g.edges)
    assert "uses_generator" in SG.to_dot(g)


def test_isolated_nodes_are_dropped_and_counted(conn):
    """A node with no surviving edge tells the reader nothing and crowds out
    what does. An earlier version kept them and produced 36 nodes with 12 edges
    -- a page of disconnected boxes."""
    g = SG.build(conn, ["gA1", "gB", "lone"])
    assert "lone" not in {n.node_id for n in g.nodes}
    assert g.omitted_nodes >= 1


def test_node_budget_is_respected(conn):
    g = SG.build(conn, ["gA1", "gA2", "gB"], max_nodes=3)
    assert len(g.nodes) <= 3


def test_best_connected_neighbours_survive_the_cap(conn):
    """s1 ties two generators together; s2 touches one. With room for a single
    neighbour, the informative one should win."""
    g = SG.build(conn, ["gA1", "gA2", "gB"], max_nodes=3)
    ids = {n.node_id for n in g.nodes}
    assert "s1" in ids


def test_omissions_are_reported_not_silent(conn):
    """An honest '+299 more' beats an unreadable complete picture, but only if
    the reader is told."""
    g = SG.build(conn, ["gA1", "gA2", "gB"], max_nodes=2)
    assert g.omitted_nodes > 0


def test_empty_input_is_safe(conn):
    g = SG.build(conn, [])
    assert len(g.nodes) == 0 and g.edges == []
    assert SG.to_dot(g).startswith("digraph")


def test_dot_escapes_latex_labels(conn):
    """Real labels contain $t\\bar{t}$ and quotes; unescaped they break the DOT."""
    conn2 = conn
    g = SG.build(conn2, ["gA1"])
    g.nodes[0].label = r'$t\bar{t}$ "weird" label'
    dot = SG.to_dot(g)
    assert "\\bar" not in dot
    assert dot.count('"') % 2 == 0, "quotes balanced, so graphviz can parse it"


def test_kinds_get_distinct_colours(conn):
    g = SG.build(conn, ["gA1", "gB"])
    dot = SG.to_dot(g)
    assert SG.KIND_COLOURS["generator"] in dot
    assert SG.KIND_COLOURS["sample"] in dot
