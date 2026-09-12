"""SUBGOAL_SEQUENTIAL=1: one objective at a time, advanced by the model."""
import types

import pytest

from hepcoveragekg.query import graph as G
from hepcoveragekg.query import planner as P
from hepcoveragekg.query import subgoals as sg

NS = types.SimpleNamespace
GOALS = ["Find analyses using b-tagged jets",
         "Find analyses using missing transverse momentum",
         "Intersect the two sets"]


def test_the_planner_sees_only_the_current_objective():
    first = sg.render_sequential(GOALS, 0, [], 6)
    assert GOALS[0] in first
    assert GOALS[1] not in first, "objective 2 must not be visible on objective 1"
    assert GOALS[2] not in first
    assert "Do not answer the question yet" in first

    last = sg.render_sequential(GOALS, 2, ["18 papers, set_1", "24 papers, set_2"], 3)
    assert GOALS[2] in last
    assert "LAST objective" in last
    assert "answer the question" in last
    # what earlier objectives established stays visible -- "intersect the two
    # sets" is unanswerable without them
    assert "set_1" in last and "set_2" in last


def test_the_index_is_clamped():
    assert GOALS[2] in sg.render_sequential(GOALS, 99, [], 1)
    assert GOALS[0] in sg.render_sequential(GOALS, -3, [], 1)
    assert sg.render_sequential([], 0, [], 1) == ""


def test_advance_is_declared_by_the_model():
    assert sg.wants_advance("OBJECTIVE COMPLETE: 18 papers")
    assert sg.wants_advance("blah\nobjective   complete\nmore")
    assert not sg.wants_advance("still working on the objective")
    assert not sg.wants_advance("")
    assert sg.note_from("OBJECTIVE COMPLETE: 18 papers in set_1") == "18 papers in set_1"
    assert sg.note_from("found lots\nOBJECTIVE COMPLETE") == "found lots"


def _state(**kw):
    s = P.Session(question="q")
    st = {"session": s, "round": 2, "max_rounds": 8, "messages": [],
          "pending_calls": [], "last_content": "", "sub_objectives": GOALS,
          "subgoal_index": 0, "subgoal_established": [], "objective": "",
          "review_cycles": 0, "max_places": 3, "max_rows": 25, "question": "q"}
    st.update(kw)
    return st


def test_the_pointer_advances_when_declared_and_is_counted(monkeypatch):
    monkeypatch.setenv("SUBGOAL_SEQUENTIAL", "1")
    st = _state()
    msg = NS(content="OBJECTIVE COMPLETE: 18 papers in set_1", tool_calls=[])
    G._advance_sequential(st, msg.content, msg)
    assert st["subgoal_index"] == 1
    assert st["subgoal_established"] == ["18 papers in set_1"]
    assert st["session"].subgoal_advances == 1
    assert st["session"].subgoal_forced_advances == 0


def test_it_advances_anyway_rather_than_stalling(monkeypatch):
    monkeypatch.setenv("SUBGOAL_SEQUENTIAL", "1")
    st = _state()
    msg = NS(content="still looking", tool_calls=[])
    for _ in range(sg.MAX_ROUNDS_PER_GOAL):
        G._advance_sequential(st, "still looking", msg)
    assert st["subgoal_index"] == 1, "a run must not stall on one objective"
    assert st["session"].subgoal_forced_advances == 1


def test_the_last_objective_never_advances(monkeypatch):
    monkeypatch.setenv("SUBGOAL_SEQUENTIAL", "1")
    st = _state(subgoal_index=2, subgoal_established=["a", "b"])
    msg = NS(content="OBJECTIVE COMPLETE", tool_calls=[])
    G._advance_sequential(st, "OBJECTIVE COMPLETE", msg)
    assert st["subgoal_index"] == 2
    assert st["session"].subgoal_advances == 0


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SUBGOAL_SEQUENTIAL", raising=False)
    st = _state()
    G._advance_sequential(st, "OBJECTIVE COMPLETE", NS(content="", tool_calls=[]))
    assert st["subgoal_index"] == 0 and st["session"].subgoal_advances == 0
