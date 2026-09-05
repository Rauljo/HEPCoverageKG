"""Tests for the evidence-backed truth trimmer.

Uses an in-memory sqlite db with the minimal schema the module reads:
assertion(paper_id, subject_id, object_id) and assertion_evidence(assertion_id).
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from hepcoveragekg.eval import verify_truth as VT


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE assertion (assertion_id TEXT, paper_id TEXT, "
             "subject_id TEXT, object_id TEXT)")
    c.execute("CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT)")
    # paper P1 has a QUOTED assertion linking it to entity E1
    c.execute("INSERT INTO assertion VALUES ('a1','P1','E1',NULL)")
    c.execute("INSERT INTO assertion_evidence VALUES ('a1','ev1')")
    # paper P2 has an assertion linking it to E1 but NO evidence row
    c.execute("INSERT INTO assertion VALUES ('a2','P2','E1',NULL)")
    # paper P3 has no assertion touching E1 at all
    yield c
    c.close()


def _q(papers, universe=None, entity_id="E1", **over):
    truth = {"kind": "set", "papers": papers, "value": len(papers)}
    if universe is not None:
        truth["universe"] = universe
    base = {"qid": "q1", "text": "x", "truth": truth,
           "provenance": {"entity_id": entity_id}, "truth_source": "sql"}
    base.update(over)
    return base


def test_supported_papers_requires_a_real_evidence_row(conn):
    got = VT.supported_papers(conn, "E1", ["P1", "P2", "P3"])
    assert got == {"P1"}  # only P1 has an assertion_evidence row


def test_trim_drops_unsupported_papers_keeps_supported(conn):
    q = _q(["P1", "P2"])
    out = VT.trim_question(conn, q)
    assert out["truth"]["papers"] == ["P1"]
    assert out["provenance"]["evidence_trimmed"] == ["P2"]


def test_trim_returns_none_when_nothing_survives(conn):
    q = _q(["P2", "P3"])
    assert VT.trim_question(conn, q) is None


def test_trim_leaves_gabriel_truth_untouched(conn):
    """Human-verified gold is never downgraded by an automated evidence check."""
    q = _q(["P1", "P2"], truth_source="gabriel")
    out = VT.trim_question(conn, q)
    assert out["truth"]["papers"] == ["P1", "P2"]  # unchanged
    assert "evidence_trimmed" not in out.get("provenance", {})


def test_trim_passes_through_questions_it_cannot_check(conn):
    """No entity_id, or no papers (a count-only question) -- not this filter's job."""
    no_entity = {"qid": "q2", "text": "x", "truth": {"kind": "count", "value": 3},
                "provenance": {}, "truth_source": "sql"}
    assert VT.trim_question(conn, no_entity) == no_entity


def test_trim_shrinks_universe_but_never_invents_a_confirmed_negative(conn):
    """A paper dropped from `papers` for lack of evidence must also leave the
    universe -- it was never confirmed one way or the other, and leaving it in
    the universe while removing it from gold would silently manufacture a
    confirmed-negative that no one ever checked."""
    q = _q(["P1", "P2"], universe=["P1", "P2", "P3"])
    out = VT.trim_question(conn, q)
    assert out["truth"]["papers"] == ["P1"]
    assert "P2" not in out["truth"]["universe"]


def test_trim_file_counts_unchanged_trimmed_and_dropped(conn, tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(x) for x in [
        _q(["P1"]),                              # unchanged (fully backed)
        _q(["P1", "P2"], **{"qid": "q2"}),        # trimmed
        _q(["P2", "P3"], **{"qid": "q3"}),        # dropped
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    counts = VT.trim_file(conn, src, out)
    assert counts == {"unchanged": 1, "trimmed": 1, "dropped": 1}
    survivors = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(survivors) == 2
    assert {s["qid"] for s in survivors} == {"q1", "q2"}


def test_backfill_fills_papers_from_evidence_backed_assertions(conn):
    """The retrieval bank's bug: text asks 'which analyses...', truth stores
    the entity, not the papers. Backfill must produce a real paper set."""
    q = {"qid": "r1", "text": "Which analyses used X?", "shape": "set",
        "truth": {"kind": "entity", "items": ["E1"], "papers": [], "value": 1},
        "provenance": {"entity_id": "E1"}, "truth_source": "generated"}
    # only 1 evidence-backed paper (P1) for E1 in the fixture -- below min_papers=3
    assert VT.backfill_retrieval_papers(conn, q, min_papers=3) is None
    out = VT.backfill_retrieval_papers(conn, q, min_papers=1)
    assert out["truth"]["papers"] == ["P1"]
    assert out["truth"]["kind"] == "set"
    assert out["shape"] == "set"


def test_backfill_skips_questions_below_min_papers():
    pass  # covered above; kept as a named marker for intent


def test_backfill_requires_an_entity_id(conn):
    q = {"qid": "r2", "text": "x", "truth": {"kind": "entity"}, "provenance": {}}
    assert VT.backfill_retrieval_papers(conn, q) is None
