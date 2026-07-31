"""Tests for the planner loop.

The model is injected as a scripted `chat`, so what is tested is the LOOP --
batching, budgets, error handling, truncation, the trace -- not the model's
judgement. Model judgement is what the evaluation harness is for; a test that
depended on it would be flaky and would measure the wrong thing.
"""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from hepcoveragekg.query import planner, retrieve as R, templates as T


# -- a scripted model ------------------------------------------------------

def _call(tool: str, args: dict, cid: str = "c1"):
    return SimpleNamespace(
        id=cid, type="function",
        function=SimpleNamespace(name=tool, arguments=json.dumps(args)),
    )


def _response(calls=None, content=""):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=calls or None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def scripted(*turns):
    """A model that replays a fixed script, one entry per round."""
    state = {"i": 0}

    def chat(messages, tools):
        turn = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        return turn

    chat.calls_made = state  # inspectable
    return chat


@pytest.fixture()
def conn(tmp_path):
    db = tmp_path / "p.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT,
            label TEXT, aliases TEXT DEFAULT '[]');
        CREATE TABLE assertion (
            assertion_id TEXT PRIMARY KEY, bundle_id TEXT, paper_id TEXT,
            predicate TEXT, family TEXT, subject_id TEXT, object_id TEXT,
            object_value TEXT, qualifiers TEXT DEFAULT '{}');
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);
        CREATE TABLE paper (paper_id TEXT PRIMARY KEY);

        INSERT INTO paper VALUES ('p1');
        INSERT INTO entity VALUES ('r1','result','Search for X'), ('g1','generator','Pythia 8');
        INSERT INTO entity_occurrence VALUES ('b1','r1','p1','result','Search for X','[]');
        INSERT INTO entity_occurrence VALUES ('b1','g1','p1','generator','Pythia 8','[]');
        INSERT INTO assertion VALUES
            ('a1','b1','p1','uses_generator','samples','r1','g1',NULL,'{}');
        INSERT INTO assertion_evidence VALUES ('a1','ev1');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


@pytest.fixture()
def index(conn):
    return R.build(conn, embed=False)


# -- the loop --------------------------------------------------------------

def test_answer_tool_ends_the_loop(conn, index):
    chat = scripted(_response([_call("answer", {"text": "42 papers.", "answerable": True})]))
    s = planner.answer(conn, index, "how many?", chat=chat)
    assert s.answer == "42 papers."
    assert s.rounds == 1
    assert s.stopped_because == "answered"


def test_several_calls_run_in_one_round(conn, index):
    """The point of batching: N operations must not cost N model calls."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"}, "c1"),
                   _call("search", {"text": "result"}, "c2"),
                   _call("count", {"predicate": "uses_generator",
                                   "object_ids": ["g1"]}, "c3")]),
        _response([_call("answer", {"text": "done", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.llm_calls == 2, "two rounds, not four"
    assert s.tool_calls == 3, "all three operations ran"
    assert {st.round for st in s.steps} == {1}


def test_max_places_caps_a_round(conn, index):
    """The 1-vs-many dial: at 1 the planner is forced to be sequential, which
    makes batching an ablation rather than an architecture."""
    calls = [_call("search", {"text": f"q{i}"}, f"c{i}") for i in range(5)]
    chat = scripted(_response(calls),
                    _response([_call("answer", {"text": "x", "answerable": True})]))
    s = planner.answer(conn, index, "q", max_places=2, chat=chat)
    assert len([st for st in s.steps if st.round == 1]) == 2


def test_max_rounds_stops_a_wandering_loop(conn, index):
    """A model that never answers must stop, and must say why."""
    chat = scripted(_response([_call("search", {"text": "x"})]))  # repeats forever
    s = planner.answer(conn, index, "q", max_rounds=3, chat=chat)
    assert s.rounds == 3
    assert "max_rounds" in s.stopped_because
    assert s.answer == ""


def test_tool_error_is_reported_to_the_planner_not_raised(conn, index):
    """A failed call must come back as a message it can react to. Raising would
    lose the whole session over one bad argument."""
    chat = scripted(
        _response([_call("count", {"predicate": "uses_generator"})]),  # missing object_ids
        _response([_call("answer", {"text": "recovered", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.errors == 1
    assert s.answer == "recovered", "the loop continued after the error"


def test_malformed_arguments_are_survivable(conn, index):
    bad = SimpleNamespace(id="c1", type="function",
                          function=SimpleNamespace(name="count", arguments="{not json"))
    chat = scripted(_response([bad]),
                    _response([_call("answer", {"text": "ok", "answerable": True})]))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.steps[0].error == "bad_arguments"
    assert s.answer == "ok"


def test_prose_answer_without_the_tool_is_accepted(conn, index):
    """Form is not worth burning a round on."""
    chat = scripted(_response(content="The graph holds 3 analyses."))
    s = planner.answer(conn, index, "q", chat=chat)
    assert "3 analyses" in s.answer
    assert "without calling" in s.stopped_because


def test_abstention_is_recorded_not_treated_as_failure(conn, index):
    """S-13: 'the graph does not hold this' is an answer, and must be
    distinguishable in the trace from an answer that does hold."""
    chat = scripted(_response([_call("answer", {
        "text": "The graph does not record jet tunes.", "answerable": False})]))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.answerable is False
    assert s.answer


# -- the trace -------------------------------------------------------------

def test_session_records_cost_and_every_step(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "done", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.prompt_tokens == 200 and s.completion_tokens == 40
    assert s.seconds > 0
    assert s.steps[0].tool == "search"
    assert s.steps[0].seconds >= 0


def test_evidence_is_gathered_across_steps(conn, index):
    """Citations must accumulate over the whole session, not only the last call."""
    chat = scripted(
        _response([_call("count", {"predicate": "uses_generator", "object_ids": ["g1"]})]),
        _response([_call("answer", {"text": "1 paper", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.evidence_ids == ["ev1"]


def test_trace_serialises(conn, index):
    chat = scripted(_response([_call("answer", {"text": "x", "answerable": True})]))
    line = json.loads(planner.answer(conn, index, "q", chat=chat).to_jsonl())
    assert {"question", "answer", "rounds", "llm_calls", "prompt_tokens", "steps"} <= set(line)


# -- context control -------------------------------------------------------

def test_truncation_announces_itself():
    """A planner silently given 25 of 200 rows reasons as if it saw everything,
    and concludes something false with no way to notice."""
    rows = [{"i": i} for i in range(200)]
    text = planner._render_rows(rows, 25)
    assert "175 more rows not shown" in text
    assert text.count("\n") == 25


def test_empty_result_says_so():
    assert planner._render_rows([], 10) == "(no rows)"


# -- wiring ----------------------------------------------------------------

def test_answer_tool_is_exposed(conn):
    names = {t["name"] for t in planner.TOOL_SPECS}
    assert "answer" in names, "without it the planner can never stop deliberately"
    assert "search" in names, "without it nothing can start -- every other tool needs ids"


def test_executor_rejects_unknown_tools(conn, index):
    run = planner.build_executor(conn, index)
    with pytest.raises(ValueError):
        run("drop_everything", {})


def test_system_prompt_has_both_variants(conn):
    full = planner.system_prompt(conn, minimal=False)
    small = planner.system_prompt(conn, minimal=True)
    assert len(small) < len(full)
    assert "PREDICATES" in full and "PREDICATES" in small, "schema card is in both"
