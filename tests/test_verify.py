"""Tests for the mechanical faithfulness check.

The failure being guarded against is specific: an answer that is TRUE about
physics but absent from the corpus. It reads as authoritative, a domain expert
would nod at it, and only the graph can contradict it.
"""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from hepcoveragekg.query import planner, verify


SEEN = {"58", "344", "367", "2001.06899", "pythia", "hepkg:generator:pythia8"}


def test_grounded_number_passes():
    v = verify.verify("58 analyses used Pythia.", SEEN)
    assert v.score == 1.0
    assert not v.unsupported


def test_invented_number_is_caught():
    """The core case: a plausible number no row ever returned."""
    v = verify.verify("About 91 analyses used Pythia.", SEEN)
    assert [c.text for c in v.unsupported] == ["91"]
    assert v.score < 1.0


def test_correct_physics_that_was_never_retrieved_is_still_flagged():
    """The dangerous one. 125 GeV is the right Higgs mass, and that is exactly
    why it is dangerous: a physicist reading the answer would not blink."""
    v = verify.verify("The Higgs mass is 125.25 GeV.", SEEN)
    assert any(c.text == "125.25" for c in v.unsupported)


def test_formatting_is_not_treated_as_invention():
    """1,286 retrieved and 1286 written are the same claim."""
    v = verify.verify("There are 1286 assertions.", {"1,286"})
    assert v.score == 1.0


def test_trailing_decimal_matches():
    assert verify.verify("58.0 papers", {"58"}).score == 1.0


def test_small_numbers_are_not_flagged():
    """'the two analyses' and '13 TeV' would otherwise fire on every answer,
    and a check that cries wolf gets ignored."""
    v = verify.verify("Both of the 2 analyses ran at 13 TeV.", set())
    assert v.checked == 0


def test_unretrieved_arxiv_id_is_caught():
    v = verify.verify("See 2401.99999 for details.", SEEN)
    assert [c.kind for c in v.unsupported] == ["arxiv_id"]


def test_retrieved_arxiv_id_passes():
    assert verify.verify("See 2001.06899.", SEEN).score == 1.0


def test_arxiv_id_is_not_double_counted_as_a_number():
    v = verify.verify("Paper 2401.99999.", SEEN)
    assert len(v.claims) == 1, "counted once, as an id"


def test_invented_entity_id_is_caught():
    v = verify.verify("Entity hepkg:generator:herwig7 was used.", SEEN)
    assert any(c.kind == "entity_id" for c in v.unsupported)


def test_prose_without_claims_scores_one():
    """An answer making no factual claims cannot be unfaithful, only unhelpful."""
    v = verify.verify("The graph does not record that.", SEEN)
    assert v.checked == 0 and v.score == 1.0


def test_abstention_is_not_penalised():
    """S-13: saying 'not in the graph' must not look like a faithfulness failure."""
    v = verify.verify("The graph does not record jet tunes for these analyses.", set())
    assert v.score == 1.0


def test_report_names_what_failed():
    text = verify.verify("There were 91 analyses.", SEEN).report()
    assert "UNSUPPORTED" in text and "91" in text


# -- integration with the planner -----------------------------------------

@pytest.fixture()
def conn(tmp_path):
    db = tmp_path / "v.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (bundle_id TEXT, entity_id TEXT, paper_id TEXT,
            kind TEXT, label TEXT, aliases TEXT DEFAULT '[]');
        CREATE TABLE assertion (assertion_id TEXT PRIMARY KEY, bundle_id TEXT,
            paper_id TEXT, predicate TEXT, family TEXT, subject_id TEXT,
            object_id TEXT, object_value TEXT, qualifiers TEXT DEFAULT '{}');
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);
        CREATE TABLE paper (paper_id TEXT PRIMARY KEY);
        INSERT INTO paper VALUES ('2001.06899');
        INSERT INTO entity VALUES ('r1','result','Search'), ('g1','generator','Pythia 8');
        INSERT INTO entity_occurrence VALUES
            ('b1','r1','2001.06899','result','Search','[]'),
            ('b1','g1','2001.06899','generator','Pythia 8','[]');
        INSERT INTO assertion VALUES
            ('a1','b1','2001.06899','uses_generator','s','r1','g1',NULL,'{}');
        INSERT INTO assertion_evidence VALUES ('a1','ev1');
        """
    )
    c.commit()
    c.close()
    from hepcoveragekg.query import templates as T
    return T.read_only(db)


def _call(tool, args, cid="c1"):
    return SimpleNamespace(id=cid, type="function",
                           function=SimpleNamespace(name=tool, arguments=json.dumps(args)))


def _resp(calls=None, content=""):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_session_records_values_before_truncation(conn):
    """A claim is grounded if the graph RETURNED it, not only if it survived
    into the model's context window."""
    from hepcoveragekg.query import retrieve as R

    index = R.build(conn, embed=False)
    # searches first: ids must come from the graph, not from the caller's head
    turns = [_resp([_call("search", {"text": "Search"})]),
             _resp([_call("papers_of", {"entity_ids": ["r1"]})]),
             _resp([_call("answer", {"text": "Paper 2001.06899.", "answerable": True})])]
    state = {"i": 0}

    def chat(m, t):
        r = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        return r

    s = planner.answer(conn, index, "which paper?", chat=chat, max_rows=0)
    assert "2001.06899" in s.seen_values, "recorded even though max_rows=0 hid it"
    assert verify.verify_session(s).score == 1.0
