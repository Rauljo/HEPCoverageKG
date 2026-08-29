"""The last round is for answering, and only for answering.

A run that reaches max_rounds without calling `answer` was sent straight to
`finish` with an empty session answer -- every retrieved row and every token
discarded in silence, and the scorer then falling back to the retrieval
footprint so the number looked GOOD.

gpt-5.6-luna hit this on 24 of 24 of Gabriel's questions. It found gf-01's
answer in round 2 -- facets(objects=[BJet,MET], mode='all') returning 18 papers
-- then explored until the budget ran out. It scored judged_f1 0.666 with
judged_named_none 1: the score measured what it retrieved, not what it said.
Qwen never exposed this because Qwen stops at round 3.
"""
from __future__ import annotations

import json

from hepcoveragekg.query import graph as G
from hepcoveragekg.query import planner


def _reply(name, args):
    call = type("T", (), {"id": "1", "function": type("F", (), {
        "name": name, "arguments": json.dumps(args)})()})()
    msg = type("M", (), {"content": "", "tool_calls": [call]})()
    return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": None})()


def _state(round_, max_rounds, offered):
    def chat(messages, tools):
        offered.append([t["function"]["name"] for t in tools])
        offered.append(messages[-1].get("content", "")[:40])
        return _reply("search", {"text": "x"})
    return ({
        "session": planner.Session(question="q"),
        "messages": [{"role": "user", "content": "q"}],
        "round": round_ - 1, "max_rounds": max_rounds,
        "max_places": 8, "max_rows": 25,
    }, {"configurable": {"chat": chat, "tools": [
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": {"name": "count"}},
        {"type": "function", "function": {"name": "answer"}}]}})


def test_the_last_round_offers_only_answer():
    offered = []
    state, cfg = _state(6, 6, offered)
    G.plan(state, cfg)
    assert offered[0] == ["answer"], f"still offered {offered[0]}"
    assert "last" in offered[1].lower()


def test_earlier_rounds_keep_every_tool():
    offered = []
    state, cfg = _state(3, 6, offered)
    G.plan(state, cfg)
    assert set(offered[0]) == {"search", "count", "answer"}
    assert "last" not in offered[1].lower()


def test_it_does_not_strand_a_run_whose_contract_has_no_answer_tool():
    """If `answer` is somehow absent, fall back to every tool rather than
    handing the model an empty tool list, which some APIs reject outright."""
    offered = []
    state, cfg = _state(6, 6, offered)
    cfg["configurable"]["tools"] = [{"type": "function", "function": {"name": "search"}}]
    G.plan(state, cfg)
    assert offered[0] == ["search"]
