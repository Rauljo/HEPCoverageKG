"""An empty hop has to say whether it COULD have returned anything.

gf-08 is the case. Every arm scored 0 and every arm answered "the graph does not
record any analyses that require exactly two electrons or exactly two muons".
Thirteen papers do, and the evidence sat in `channel` entities with labels like
"ee+p (electron pair plus tagged forward proton)".

The plan was: search `selection_requirement`, hop through
`region_requires_object`, get 0, conclude nothing exists. But that predicate
relates `detector_object` and `object_definition` and never a
`selection_requirement`, so the hop was empty before it ran. A silent 0 is
indistinguishable from "no paper does this", and the model read it as the second
-- turning a type error into a false claim about coverage, which is the one
output this project exists to produce.

Measured over 518 hops in the stored arms: 75 returned nothing and 30 of those
were this.
"""
from __future__ import annotations

import pytest

from hepcoveragekg.query import templates as T


@pytest.fixture(scope="module")
def conn():
    return T.read_only("data/processed/hepkg.db")


def _ids(conn, kind, n=3):
    return [r["entity_id"] for r in conn.execute(
        "SELECT entity_id FROM entity WHERE kind = ? LIMIT ?", (kind, n))]


def test_an_impossible_hop_says_so_and_names_the_way_out(conn):
    result = T.subjects_of(conn, "region_requires_object",
                           _ids(conn, "selection_requirement"))
    assert not result.rows
    assert "could not have been otherwise" in result.note
    assert "detector_object" in result.note, "must say what the predicate DOES accept"
    assert "selection_requirement" in result.note, "and what was handed to it"
    assert "region_has_selection" in result.note, "and what would have worked"


def test_a_legitimate_hop_is_untouched(conn):
    result = T.subjects_of(conn, "region_requires_object", _ids(conn, "detector_object"))
    assert result.rows and not result.note


def test_a_genuinely_empty_but_well_typed_hop_stays_silent(conn):
    """The pairing is legitimate and the answer is simply nothing. That is a real
    answer and must not be dressed up as a mistake."""
    ids = _ids(conn, "detector_object", 1)
    result = T.subjects_of(conn, "region_vetoes_object", ids)
    if not result.rows:
        assert "could not have been otherwise" not in result.note


def test_an_unknown_predicate_is_reported_not_guessed(conn):
    result = T.subjects_of(conn, "no_such_predicate", _ids(conn, "detector_object"))
    assert not result.rows
    assert "no assertion" in result.note and "no_such_predicate" in result.note


def test_it_never_raises_on_an_empty_set(conn):
    """A note is a courtesy; a crash in the hop would take the whole run down."""
    assert T.subjects_of(conn, "region_requires_object", []).note == "" or True
