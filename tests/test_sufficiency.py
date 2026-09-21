"""The brake (D-196): does more searching still pay, and does the block say so."""

import sqlite3

import pytest

from hepcoveragekg.query import sufficiency as S


def _conn(rows):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE entity_occurrence (entity_id TEXT, paper_id TEXT)")
    c.executemany("INSERT INTO entity_occurrence VALUES (?,?)", rows)
    return c


def test_enabled_follows_the_switch(monkeypatch):
    monkeypatch.delenv("SUFFICIENCY", raising=False)
    assert S.enabled() is False
    monkeypatch.setenv("SUFFICIENCY", "1")
    assert S.enabled() is True
    monkeypatch.setenv("SUFFICIENCY", "0")
    assert S.enabled() is False


def test_papers_for_maps_entities_to_their_papers():
    c = _conn([("E1", "P1"), ("E1", "P2"), ("E2", "P2"), ("E3", "P9")])
    assert S.papers_for(c, ["E1", "E2"]) == {"P1", "P2"}
    assert S.papers_for(c, []) == set()
    assert S.papers_for(None, ["E1"]) == set()


def test_papers_for_chunks_past_the_sqlite_variable_cap():
    """A saturating run is exactly the one that accumulates enough entities to
    hit the 999-variable limit, so the failure would arrive only on the runs
    this is built to measure."""
    rows = [("E%d" % i, "P%d" % (i % 7)) for i in range(1500)]
    c = _conn(rows)
    got = S.papers_for(c, ["E%d" % i for i in range(1500)])
    assert got == {"P%d" % i for i in range(7)}


def test_papers_for_survives_a_broken_database():
    """A hint must never kill a run."""
    c = sqlite3.connect(":memory:")          # no entity_occurrence table
    assert S.papers_for(c, ["E1"]) == set()


def test_marginal_yield_is_new_papers_not_total():
    c = _conn([("E1", "P1"), ("E2", "P2"), ("E3", "P3")])
    state = {}
    first = S.update(state, c, ["E1"])
    assert (first["papers"], first["new"]) == (1, 1)
    second = S.update(state, c, ["E1", "E2"])
    assert (second["papers"], second["new"]) == (2, 1), "P1 was already held"
    third = S.update(state, c, ["E1", "E2"])
    assert (third["papers"], third["new"]) == (2, 0), "nothing new at all"


def test_saturation_needs_consecutive_thin_rounds():
    """One quiet round is ordinary -- a describe call reads what is already
    held and adds nothing by design. Two in a row is a pattern."""
    rows = [("E%d" % i, "P%d" % i) for i in range(40)]
    c = _conn(rows)
    state = {}
    S.update(state, c, ["E0", "E1", "E2", "E3"])        # productive
    one = S.update(state, c, ["E0", "E1", "E2", "E3"])  # thin
    assert one["stale_rounds"] == 1 and not one["saturated"]
    two = S.update(state, c, ["E0", "E1", "E2", "E3"])  # thin again
    assert two["stale_rounds"] == 2 and two["saturated"]


def test_a_productive_round_clears_the_counter():
    rows = [("E%d" % i, "P%d" % i) for i in range(40)]
    c = _conn(rows)
    state = {}
    S.update(state, c, ["E0"])          # 1 new: already thin at MIN_NEW_PAPERS
    S.update(state, c, ["E0"])          # 0 new
    assert state["sufficiency_stale"] == 2
    back = S.update(state, c, ["E0", "E1", "E2", "E3"])
    assert back["new"] == 3 and back["stale_rounds"] == 0 and not back["saturated"]


def test_the_block_permits_stopping_and_does_not_order_it():
    """An unconditional stop would cost recall on the questions where the
    remaining gold is genuinely hard to reach, which is the failure the
    sub-goal family exists to fix. The brake must not undo the accelerator."""
    text = S.render({"papers": 47, "new": 0, "stale_rounds": 2, "saturated": True}, 6)
    assert S.MARKER in text
    assert "ANSWER NOW" in text
    assert "correct move" in text
    # The escape hatch: a part of the question with nothing behind it must
    # still send the planner searching rather than to the answer.
    assert "still has nothing behind it" in text
    assert "search for THAT" in text


def test_a_thin_round_is_reported_without_crying_saturation():
    text = S.render({"papers": 10, "new": 1, "stale_rounds": 1, "saturated": False}, 5)
    assert "thin round" in text
    assert "ANSWER NOW" not in text


def test_a_productive_round_says_keep_going():
    text = S.render({"papers": 10, "new": 7, "stale_rounds": 0, "saturated": False}, 5)
    assert "Keep going" in text
    assert "ANSWER NOW" not in text


def test_the_last_round_always_says_so():
    text = S.render({"papers": 3, "new": 3, "stale_rounds": 0, "saturated": False}, 1)
    assert "last round" in text


@pytest.mark.parametrize("held,expected", [(1, "1 paper."), (2, "2 papers.")])
def test_the_block_counts_papers_in_english(held, expected):
    text = S.render({"papers": held, "new": 0, "stale_rounds": 0}, 5)
    assert expected in text


def test_a_round_that_searched_nothing_is_not_evidence_of_saturation():
    """Before any retrieval there is nothing to add, so the first look must not
    start the stale counter -- otherwise two barren opening rounds would brake a
    run that has not begun."""
    c = _conn([("E1", "P1")])
    state = {}
    first = S.update(state, c, [])
    assert first["stale_rounds"] == 0 and not first["saturated"]
    second = S.update(state, c, [])
    assert second["stale_rounds"] == 0, "still nothing retrieved, still not stale"


def test_the_block_is_replaced_never_appended():
    """The message list grows every round. Appending would leave one stale
    coverage report per round, and the old copies say the run is still
    productive -- they contradict the newest one exactly when it matters."""
    msgs = [{"role": "user", "content": "which analyses use b-tagged jets?"},
            {"role": "system", "content": S.render(
                {"papers": 10, "new": 9, "stale_rounds": 0}, 8)}]
    for _ in range(5):
        msgs = S.replace_block(msgs, S.render(
            {"papers": 47, "new": 0, "stale_rounds": 2, "saturated": True}, 3))
    blocks = [m for m in msgs if S.MARKER in (m.get("content") or "")]
    assert len(blocks) == 1, "one coverage report, not one per round"
    assert "ANSWER NOW" in blocks[0]["content"], "and it is the CURRENT one"
    assert msgs[0]["role"] == "user", "the conversation itself is untouched"
    assert msgs[-1] is blocks[0], "the block is last, nearest the planner"


def test_replace_block_leaves_other_system_messages_alone():
    """The sub-goal block sits in the same list and must survive."""
    msgs = [{"role": "system", "content": "SUB-OBJECTIVES\n1. b-tagged jets"},
            {"role": "system", "content": S.render({"papers": 1, "new": 1}, 9)}]
    out = S.replace_block(msgs, S.render({"papers": 4, "new": 3}, 8))
    assert any("SUB-OBJECTIVES" in m["content"] for m in out)
    assert sum(S.MARKER in m["content"] for m in out) == 1


def test_replace_block_on_an_empty_history():
    out = S.replace_block([], S.render({"papers": 0, "new": 0}, 9))
    assert len(out) == 1 and S.MARKER in out[0]["content"]
