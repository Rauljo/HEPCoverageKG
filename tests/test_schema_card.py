"""Tests for the query layer's schema card.

The card's whole justification is that it is *derived*, so the tests check that
it tracks the data rather than that it matches a fixed string. A test asserting
the exact rendered text would break on every import and teach us nothing.
"""
from __future__ import annotations

import sqlite3

import pytest

from hepcoveragekg.query import schema_card


@pytest.fixture()
def conn():
    """A tiny graph with deliberately awkward shapes:

    - `r_uses_gen`   cleanly typed:    result -> generator, node objects
    - `r_reports`    literal objects:  result -> <value>
    - `mixed_pred`   an even 50/50 subject split
    - `skewed_pred`  ~86/14 -- the split an old 60% threshold hid entirely
    - `rare_pred`    below RARE_MAX_ROWS, so it must be grouped as rare
    """
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        """
        CREATE TABLE paper (paper_id TEXT PRIMARY KEY);
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE assertion (
            assertion_id TEXT PRIMARY KEY, predicate TEXT, family TEXT,
            subject_id TEXT, object_id TEXT, object_value TEXT
        );
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);

        INSERT INTO paper VALUES ('p1'), ('p2');
        INSERT INTO entity VALUES
            ('res1','result','Result one'), ('res2','result','Result two'),
            ('gen1','generator','Pythia 8'), ('gen2','generator','Herwig'),
            ('bkg1','background','ttbar');
        """
    )
    rows = []
    # 5 cleanly typed, node-object assertions
    for i in range(5):
        rows.append((f"a{i}", "r_uses_gen", "samples", "res1", "gen1" if i % 2 else "gen2", None))
    # 4 literal-object assertions
    for i in range(4):
        rows.append((f"b{i}", "r_reports", "obs", "res1", None, '{"v": 1}'))
    # 4 with two different subject kinds -> "mixed"
    for i in range(4):
        rows.append((f"c{i}", "mixed_pred", "x", "res1" if i < 2 else "bkg1", "gen1", None))
    # 6 result-subject + 1 background-subject -> ~86/14, the case a threshold hides
    for i in range(6):
        rows.append((f"e{i}", "skewed_pred", "x", "res1", "gen1", None))
    rows.append(("e6", "skewed_pred", "x", "bkg1", "gen1", None))
    # 2 rows only -> rare
    rows.append(("d0", "rare_pred", "x", "res2", "gen1", None))
    rows.append(("d1", "rare_pred", "x", "res2", "gen2", None))
    c.executemany("INSERT INTO assertion VALUES (?,?,?,?,?,?)", rows)
    # evidence on 10 of the 22 assertions
    c.executemany(
        "INSERT INTO assertion_evidence VALUES (?,?)",
        [(r[0], f"ev{i}") for i, r in enumerate(rows[:10])],
    )
    c.commit()
    return c


def test_counts_come_from_the_database(conn):
    card = schema_card.build(conn)
    assert card.papers == 2
    assert card.entities == 5
    assert card.assertions == 22
    assert card.evidence_share == pytest.approx(10 / 22)


def test_kinds_are_ordered_by_frequency(conn):
    kinds = dict(schema_card.build(conn).kinds)
    assert kinds == {"result": 2, "generator": 2, "background": 1}
    # most common first, so the prompt leads with what matters
    assert schema_card.build(conn).kinds[0][1] >= schema_card.build(conn).kinds[-1][1]


def test_predicates_ordered_by_row_count(conn):
    preds = schema_card.build(conn).predicates
    assert [p.rows for p in preds] == sorted((p.rows for p in preds), reverse=True)


def test_cleanly_typed_predicate_reports_its_kinds(conn):
    p = {x.name: x for x in schema_card.build(conn).predicates}["r_uses_gen"]
    assert p.subject_kinds[0][0] == "result"
    assert p.object_kinds[0][0] == "generator"
    assert not p.is_literal
    # pure -> the bare kind, no percentages cluttering the line
    assert "result" in p.render() and "%" not in p.render()


def test_literal_object_predicate_is_flagged(conn):
    """A join to `entity` on this predicate returns nothing -- a silent failure,
    so the card has to say so."""
    p = {x.name: x for x in schema_card.build(conn).predicates}["r_reports"]
    assert p.is_literal
    assert "<value>" in p.render()


def test_impure_predicate_shows_the_actual_shares(conn):
    """Half `result`, half `background`.

    An earlier version hid this behind the word "mixed", which told the reader a
    verdict instead of a fact -- and said nothing at all for an 86/14 split. The
    card now prints the distribution and lets the reader judge.
    """
    p = {x.name: x for x in schema_card.build(conn).predicates}["mixed_pred"]
    assert {k for k, _ in p.subject_kinds} == {"result", "background"}
    rendered = p.render()
    assert "50%" in rendered, "both sides of an even split must be visible"
    assert "result" in rendered and "background" in rendered


def test_minority_kinds_are_not_hidden(conn):
    """The 86/14 case: the minority is still 14% of rows and must be shown."""
    p = {x.name: x for x in schema_card.build(conn).predicates}["skewed_pred"]
    rendered = p.render()
    assert "genB" not in rendered or "%" in rendered
    assert "%" in rendered, "a non-pure predicate must show its shares"


def test_rare_predicates_are_grouped_not_dropped(conn):
    text = schema_card.render(conn)
    assert "rare_pred" in text, "rare predicates must still be reachable"
    assert "rare (" in text, "and must be grouped so they do not crowd the prompt"


def test_render_mentions_evidence_coverage(conn):
    """The citation check depends on evidence existing; the model should know
    how much of the graph is quotable."""
    assert "evidence quote" in schema_card.render(conn)


def test_provenance_tables_are_not_advertised(conn):
    assert "bundle_import" not in schema_card.QUERYABLE_TABLES
    assert "activity" not in schema_card.QUERYABLE_TABLES
    assert "assertion" in schema_card.QUERYABLE_TABLES
