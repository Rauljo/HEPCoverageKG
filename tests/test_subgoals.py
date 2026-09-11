"""Decomposition and sub-objective status, as two nested removable arms."""
import types

import pytest

from hepcoveragekg.query import graph, planner
from hepcoveragekg.query import subgoals as SG


def _reply(content="", tool_calls=None):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=content, tool_calls=tool_calls))],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_a_numbered_list_is_parsed():
    out = SG.decompose(lambda m, t=None: _reply(
        "1. Find analyses using b-tagged jets\n"
        "2. Find those using missing transverse momentum\n"
        "3. Intersect them"), "q")
    assert len(out) == 3 and out[0].startswith("Find analyses")


def test_never_more_than_three():
    """93% of runs finish in <=4 rounds, so a fourth sub-objective leaves one
    round each and decomposition starves retrieval."""
    out = SG.decompose(lambda m, t=None: _reply(
        "\n".join(f"{i}. objective number {i}" for i in range(1, 8))), "q")
    assert len(out) <= SG.MAX_SUBGOALS


def test_decomposition_fails_open():
    """A run without sub-objectives is the baseline -- a worse arm, but a
    working one. A decomposition crash must not cost the question."""
    def boom(*a, **k):
        raise RuntimeError("endpoint down")
    assert SG.decompose(boom, "q") == []
    assert SG.decompose(lambda m, t=None: _reply("I'm not sure how to split this"), "q") == []


def test_status_is_taken_verbatim_not_parsed():
    """Parsing per-objective state out of prose is a second failure surface for
    no gain, and a status the model wrote is a status it can read."""
    got = SG.extract_status(
        "GOT: fine\nSTATUS:\n  1. 18 papers, set_1_kept\n  2. not started\n\nnow searching")
    assert "18 papers" in got and "not started" in got
    assert SG.extract_status("no status block here") == ""


def test_the_two_arms_render_differently():
    """--subgoals is the list and no upkeep; --subgoal-status adds the block
    that gets rewritten. E minus D is the whole measurement."""
    goals = ["find X", "find Y"]
    assert "STATUS:" not in SG.goals_only(goals)
    assert "STATUS:" in SG.render(goals)


def test_the_block_is_replaced_not_appended():
    """The message list grows every round; five stale status blocks would leave
    the model working out which one is current."""
    seen = []

    def chat(msgs, tools):
        seen.append(sum(1 for m in msgs if "SUB-OBJECTIVES" in (m.get("content") or "")))
        return _reply("STATUS:\n 1. done\n 2. not started",
                      [types.SimpleNamespace(id="c1", function=types.SimpleNamespace(
                          name="answer",
                          arguments='{"text":"x","answerable":true,"reason":"answered"}'))])

    session = planner.Session(question="q")
    graph.build().invoke(
        {"question": "q", "messages": [], "session": session, "round": 0,
         "max_rounds": 3, "max_places": 8, "max_rows": 25,
         "sub_objectives": ["find X", "find Y"], "subgoal_status": ""},
        config={"configurable": {
            "chat": chat, "tools": [], "subgoal_status": True, "contract": "v1",
            "execute": lambda n, a: types.SimpleNamespace(rows=[], shape="s", sql="")},
            "recursion_limit": 40})
    assert seen and max(seen) <= 1, f"the block duplicated: {seen}"


def test_off_by_default():
    from hepcoveragekg.eval import free_sql
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE paper(arxiv_id TEXT);
        CREATE TABLE entity(entity_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE assertion(assertion_id TEXT, subject_id TEXT,
                               predicate TEXT, object_id TEXT);
        CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT);
    """)
    s = free_sql.FreeSQLSystem(conn, None, chat=lambda *a, **k: None)
    assert s.config["subgoals"] is False and s.config["subgoal_status"] is False
    assert free_sql.FreeSQLSystem(conn, None, chat=lambda *a, **k: None,
                                  subgoal_status=True).config["subgoals"] is True


# --- LaTeX normalisation in the index --------------------------------------

def test_latex_is_indexed_in_both_forms():
    """2103.06956 has a ttZ control region, correctly extracted, with the
    supervisor's own quote confirming it -- stored as
    `{{\\mathup{{{t}}}}...{\\mathup{{{Z}}}}`. No search for "ttZ" or
    "t\\bar{t}Z" reaches that, so the paper scored as a miss on a fact the
    graph holds. Both spellings are indexed: the raw one for exact-token BM25,
    the normalised one for everything else.
    """
    from hepcoveragekg.query.retrieve import _normalised

    raw = r"$t\bar{t}Z$ control region"
    assert _normalised(raw) == "ttbarZ control region"
    # unchanged text yields nothing, so plain labels are not duplicated
    assert _normalised("b-tagged jet") == ""
    assert _normalised("") == ""


def test_the_mathup_case_becomes_matchable():
    from hepcoveragekg.query.retrieve import _normalised
    mathup = (r"a ${{\mathup{{{t}}}}{}{\mathup{{\overline{{{\mathup{{{t}}}}}}}}}}"
              r"{\mathup{{{Z}}}}$ control region is used")
    out = _normalised(mathup)
    assert "ttbar" in out and "mathup" not in out


def test_status_call_only_updates_status_and_fails_open(monkeypatch):
    """D-173: a dedicated call judges completeness; the planner then only plans."""
    from types import SimpleNamespace as NS
    from hepcoveragekg.query import subgoals as SG
    seen = {}

    def chat(msgs, tools=None):
        seen["prompt"] = msgs[0]["content"]; seen["tools"] = tools
        return NS(choices=[NS(message=NS(content="1. done: 2001.06899\n2. not started"))])

    steps = [NS(tool="search", args={"text": "b-jet"}, rows=12, error=None, preview="entity_id | label"),
             NS(tool="papers_of", args={"entity_set": "set_1"}, rows=0, error=None, preview="")]
    out = SG.status_call(chat, "which analyses use b-jets?", ["find b-jet entities", "get their papers"], "", steps)
    assert out.splitlines() == ["  1. done: 2001.06899", "  2. not started"]
    assert seen["tools"] is None                       # it must not be given tools
    assert "search({'text': 'b-jet'}) -> 12 rows" in seen["prompt"] and "papers_of" in seen["prompt"]
    assert "You do NOT plan" in seen["prompt"]

    def broken(msgs, tools=None): raise RuntimeError("judge down")
    assert SG.status_call(broken, "q", ["a"], "  1. earlier", steps) == ""


def test_readonly_block_does_not_ask_the_planner_to_rewrite():
    from hepcoveragekg.query import subgoals as SG
    rw = SG.render(["a", "b"], "  1. done")
    ro = SG.render_readonly(["a", "b"], "  1. done")
    assert "REWRITE THIS BLOCK" in rw and "REWRITE THIS BLOCK" not in ro
    assert "do not rewrite it" in ro and "1. done" in ro
