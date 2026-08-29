"""A call already made is answered from the record, not run again.

The worry this addresses, in Raul's words: persistence could make the agent
"just go in loops infinitely without actually finding anything". The retry
behaviour it would multiply already exists -- 68 of 4,479 steps in the stored
arms were exact repeats -- so it is closed BEFORE the loop is asked to persist
harder.

The reply says what the call RETURNED, not merely that it repeated. "You already
ran this" invites a third attempt; "you already ran this and it gave 0 rows" is
an argument for doing something else.
"""
from __future__ import annotations

import json

import pytest

from hepcoveragekg.query import graph as G
from hepcoveragekg.query import planner


def _state(calls):
    session = planner.Session(question="q")
    return {
        "session": session,
        "messages": [],
        "round": 1,
        "max_rows": 25,
        "pending_calls": calls,
        "last_content": "",
    }, session


def _call(cid, name, args):
    return {"id": cid, "name": name, "arguments": json.dumps(args)}


def _config(rows):
    class Result:
        def __init__(self, n):
            self.rows = [{"entity_id": f"e{i}", "label": f"l{i}"} for i in range(n)]
            self.note = ""
            self.evidence_ids = []
    return {"configurable": {"execute": lambda name, args: Result(rows),
                             "tools": [], "chat": None}}


def test_the_same_call_twice_runs_once():
    args = {"predicate": "region_requires_object", "object_set": "set_1"}
    state, session = _state([_call("1", "subjects_of", args)])
    G.execute(state, _config(rows=0))

    state["pending_calls"] = [_call("2", "subjects_of", args)]
    state["round"] = 2
    G.execute(state, _config(rows=0))

    duplicates = [s for s in session.steps if s.error == "duplicate_call"]
    assert len(duplicates) == 1, "the repeat should be short-circuited"
    reply = state["messages"][-1]["content"]
    assert "already ran this" in reply
    assert "0 rows" in reply, "it must say what the call returned, not just that it repeated"
    assert "try a different" in reply, "and point somewhere else"


def test_a_different_argument_is_not_a_duplicate():
    """Widening the search is exactly what we want it to do instead."""
    state, session = _state([_call("1", "subjects_of",
                                   {"predicate": "p", "object_set": "set_1"})])
    G.execute(state, _config(rows=0))
    state["pending_calls"] = [_call("2", "subjects_of",
                                    {"predicate": "p", "object_set": "set_2"})]
    state["round"] = 2
    G.execute(state, _config(rows=3))
    assert not [s for s in session.steps if s.error == "duplicate_call"]


def test_repeating_a_call_that_ERRORED_is_also_caught():
    """A failing call repeated is the same waste as an empty one repeated."""
    def boom(name, args):
        raise ValueError("no such column")
    cfg = {"configurable": {"execute": boom, "tools": [], "chat": None}}
    args = {"predicate": "p", "object_set": "set_1"}
    state, session = _state([_call("1", "subjects_of", args)])
    G.execute(state, cfg)
    state["pending_calls"] = [_call("2", "subjects_of", args)]
    state["round"] = 2
    G.execute(state, cfg)
    assert [s for s in session.steps if s.error == "duplicate_call"]
    assert "an error" in state["messages"][-1]["content"]


def test_answer_is_never_deduplicated():
    """`answer` is the exit. Blocking a repeat of it would trap the run in the
    loop this guard exists to prevent."""
    state, session = _state([_call("1", "answer", {"text": "x", "reason": "answered"})])
    G.execute(state, _config(rows=0))
    state["pending_calls"] = [_call("2", "answer", {"text": "x", "reason": "answered"})]
    state["round"] = 2
    G.execute(state, _config(rows=0))
    assert not [s for s in session.steps if s.error == "duplicate_call"]
