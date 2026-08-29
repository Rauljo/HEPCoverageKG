"""An `answer` with no text and no citation is not an answer.

The schema never required `text` -- there was no `required` array at all -- and
Qwen filled it anyway, so nothing surfaced it. gpt-5.6-luna called `answer` with
no arguments on 24 of 24 of the supervisor's questions: `answered` went True,
the text stayed empty, and `judged_f1` scored the 54-paper retrieval footprint
at 0.635, which read like a win over Qwen's real 0.477.
"""
from __future__ import annotations

import json

from hepcoveragekg.query import graph as G
from hepcoveragekg.query import planner


def _run(args, contract="v3"):
    session = planner.Session(question="q")
    state = {"session": session, "messages": [], "round": 1, "max_rounds": 6,
             "max_places": 8, "max_rows": 25,
             "pending_calls": [{"id": "1", "name": "answer",
                                "arguments": json.dumps(args)}],
             "last_content": ""}
    cfg = {"configurable": {"execute": lambda n, a: None, "tools": [],
                            "chat": None, "contract": contract}}
    # a step, so the nudge path (no steps at all) is not what fires
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    G.execute(state, cfg)
    return session, state


def test_an_empty_answer_is_sent_back_once():
    session, state = _run({"reason": "answered"})
    assert session.answer_retried
    assert not session.answer, "it must not be accepted as an answer"
    assert "carried no answer" in state["messages"][-1]["content"]


def test_the_second_empty_answer_is_accepted_and_flagged():
    """Asked once, like the nudge. Twice would be a loop."""
    session, state = _run({"reason": "answered"})
    state["pending_calls"] = [{"id": "2", "name": "answer",
                               "arguments": json.dumps({"reason": "answered"})}]
    G.execute(state, {"configurable": {"execute": lambda n, a: None,
                                       "tools": [], "chat": None}})
    assert session.answer_retried
    assert session.reason == "answered", "the second one goes through"


def test_a_cited_set_counts_as_an_answer_even_with_no_prose():
    """v3's whole point is citing a set instead of retyping ids, so a citation
    with no prose is legitimate and must not be sent back."""
    session, _ = _run({"papers_from": "set_1_kept", "reason": "answered"})
    assert not session.answer_retried


def test_real_prose_is_untouched():
    session, _ = _run({"text": "Three analyses: 2106.01676, 2001.06899.",
                       "reason": "answered"})
    assert not session.answer_retried
    assert "Three analyses" in session.answer
