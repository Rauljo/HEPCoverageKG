"""Post-answer reflection judged against the graph (POST_REFLECT=1, D-177)."""
import sqlite3
from types import SimpleNamespace as NS

from hepcoveragekg.query import planner as P
from hepcoveragekg.query import reflect as R


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
      CREATE TABLE paper(arxiv_id TEXT);
      CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT, aliases TEXT);
      INSERT INTO paper VALUES('2001.06899'),('2004.14060');
      INSERT INTO entity_occurrence VALUES
        ('e1','2001.06899','b-tagged jet','["b-jet"]'),
        ('e2','2001.06899','Muon','[]');
    """)
    return c


def _session(entity_ids=("e1",)):
    s = P.Session(question="q")
    s.known_entity_ids = set(entity_ids)
    return s


Q = ("Which analyses are searches whose event selection uses both b-tagged jets "
     "and missing transverse momentum?")


def test_a_condition_with_nothing_retrieved_is_a_defect():
    """gf-01's shape: one condition retrieved, the other never looked for."""
    out = R.uncovered_conditions(Q, _conn(), _session())
    assert out == ["missing transverse momentum"]


def test_a_condition_covered_by_an_alias_is_not_a_defect():
    c = _conn()
    c.execute("INSERT INTO entity_occurrence VALUES('e3','2001.06899',"
              "'Missing transverse momentum (p_T^miss)','[\"MET\"]')")
    assert R.uncovered_conditions(Q, c, _session(("e1", "e3"))) == []


def test_single_condition_questions_are_left_to_the_ladder():
    assert R.uncovered_conditions("Which analyses use b-tagged jets?", _conn(), _session()) == []


def test_ids_the_graph_does_not_hold_are_a_defect():
    assert R.invented_papers("See 2001.06899 and 1606.05334.", _conn()) == ["1606.05334"]
    assert R.invented_papers("See 2001.06899 and 2004.14060.", _conn()) == []


def test_it_never_fires_on_an_abstention_path_or_without_rounds():
    s = _session()
    assert R.should_reflect(Q, "text", _conn(), s, rounds_left=1, enabled=True) is None
    assert R.should_reflect(Q, "text", _conn(), s, rounds_left=4, enabled=False) is None


def test_each_condition_is_offered_at_most_once_and_the_run_is_capped():
    s = _session()
    first = R.should_reflect(Q, "t", _conn(), s, rounds_left=4, enabled=True)
    assert first and first[0].kind == R.UNCOVERED
    s.reflect_offered.add(first[0].detail)
    assert R.should_reflect(Q, "t", _conn(), s, rounds_left=4, enabled=True) is None
    s.reflect_offered.clear(); s.reflections_used = R.MAX_REFLECTIONS
    assert R.should_reflect(Q, "t", _conn(), s, rounds_left=4, enabled=True) is None


def test_the_message_names_the_condition_and_never_judges_the_answer():
    d = R.should_reflect(Q, "See 2001.06899 and 1606.05334.", _conn(), _session(),
                         rounds_left=3, enabled=True)
    msg = R.message(d, 3)
    assert "missing transverse momentum" in msg and "1606.05334" in msg
    assert "3 rounds left" in msg
    for banned in ("wrong", "incorrect", "bad answer", "improve your answer"):
        assert banned not in msg.lower()
    assert "not a judgement of your answer" in msg


def test_it_survives_a_database_without_the_columns_it_wants():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT);"
                    "INSERT INTO entity_occurrence VALUES('e1','p','b-tagged jet');")
    assert R.uncovered_conditions(Q, c, _session()) == ["missing transverse momentum"]
    assert R.invented_papers("2001.06899", c) == []          # no paper table: claims nothing


def test_the_answer_handler_sends_the_run_back_once_and_then_lets_it_answer(monkeypatch):
    """End to end through graph.execute: the first answer attempt is bounced with
    the missing condition named, the second is accepted."""
    import json
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("POST_REFLECT", "1")
    conn = _conn()
    session = P.Session(question=Q)
    session.known_entity_ids = {"e1"}
    session.steps.append(P.Step(1, "search", {"text": "b-jet"}, rows=3))
    state = {"session": session, "round": 2, "max_rounds": 6,
             "max_rows": 25, "max_places": 3, "messages": [],
             "pending_calls": [{"id": "c1", "name": "answer",
                                "arguments": json.dumps({"text": "The analyses are 2001.06899.",
                                                         "reason": "answered"})}]}
    runtime = {"conn": conn, "post_reflect": True, "execute": lambda n, a: None,
               "persist": False, "push_further": False, "answer_critic": False}
    G.execute(state, {"configurable": runtime})
    sent = state["messages"][-1]["content"]
    assert "missing transverse momentum" in sent
    assert session.reflections_used == 1
    assert [s.tool for s in session.steps][-1] == "reflect"
    # second attempt: the same condition is not raised again
    state["pending_calls"] = [{"id": "c2", "name": "answer",
                               "arguments": json.dumps({"text": "The analyses are 2001.06899.",
                                                        "reason": "answered"})}]
    before = len(state["messages"])
    G.execute(state, {"configurable": runtime})
    assert session.reflections_used == 1
    assert len(state["messages"]) == before or "missing transverse momentum" not in state["messages"][-1].get("content", "")


def test_or_alternatives_count_as_one_condition():
    """gf-04: 'correct them back to particle level or truth level' is satisfied by
    either, and demanding both made the audit fire on a sound answer."""
    q = ("Which analyses unfold their measured distributions -- that is, correct "
         "them back to particle level or truth level?")
    groups = R.condition_groups(q)
    assert any(len(g) > 1 for g in groups), groups
    c = _conn()
    c.execute("INSERT INTO entity_occurrence VALUES('e9','2001.06899','particle level unfolding','[]')")
    c.execute("INSERT INTO entity_occurrence VALUES('e10','2001.06899','unfold measured distributions','[]')")
    # both groups covered: the OR group by ONE of its alternatives
    assert R.uncovered_conditions(q, c, _session(("e9", "e10"))) == []
    # and the OR group alone is never the thing reported
    assert "truth level" not in R.uncovered_conditions(q, c, _session(("e9",)))


def test_and_conditions_are_still_demanded_separately():
    c = _conn()
    assert R.uncovered_conditions(Q, c, _session(("e1",))) == ["missing transverse momentum"]


# --- the asked check (REFLECT_MODE=model) -----------------------------------

VERDICT_NOT_READY = """PART: searches using b-tagged jets -- HAVE: 47 entities, set_1
PART: missing transverse momentum -- HAVE: nothing yet
READY: no
NEXT: search "missing transverse momentum" """

VERDICT_READY = """PART: b-tagged jets -- HAVE: set_1, 47 entities
PART: missing transverse momentum -- HAVE: set_2, 12 entities
READY: yes
NEXT: -"""


def _boom(msgs, tools=None):
    raise AssertionError("the verdict for this round should have been reused")


def _chat(reply):
    def chat(msgs, tools=None):
        chat.prompts.append(msgs[0]["content"])
        return NS(choices=[NS(message=NS(content=reply))])
    chat.prompts = []
    return chat


def test_the_check_reads_the_rows_not_the_answer():
    s = _session(("e1",))
    s.steps.append(P.Step(1, "search", {"text": "b-jet"}, rows=47))
    c = _chat(VERDICT_NOT_READY)
    v = R.completeness(c, Q, _conn(), s)
    prompt = c.prompts[0]
    assert "WHAT HAS BEEN RETRIEVED SO FAR" in prompt and "b-tagged jet" in prompt
    assert "47 rows" in prompt
    flat = " ".join(prompt.split()).lower()
    assert "do not judge whether the retrieved rows are correct" in flat
    assert v.ready is False and v.missing == ["missing transverse momentum"]
    assert v.next_step.startswith('search "missing')


def test_a_ready_verdict_does_not_interfere():
    v = R.completeness(_chat(VERDICT_READY), Q, _conn(), _session())
    assert v.ready is True and v.missing == []


def test_an_unparseable_or_broken_check_never_blocks_an_answer():
    assert R.completeness(_chat("I think it looks fine!"), Q, _conn(), _session()) is None
    def broken(msgs, tools=None): raise RuntimeError("down")
    assert R.completeness(broken, Q, _conn(), _session()) is None


def test_mode_switch():
    import os
    for env, want in (({"REFLECT_MODE": "model"}, "model"), ({"REFLECT_MODE": "both"}, "both"),
                      ({"POST_REFLECT": "1"}, "graph"), ({}, "off")):
        os.environ.pop("REFLECT_MODE", None); os.environ.pop("POST_REFLECT", None)
        os.environ.update(env)
        assert R.mode() == want, (env, R.mode())
    os.environ.pop("REFLECT_MODE", None); os.environ.pop("POST_REFLECT", None)


def test_the_answer_is_bounced_once_with_the_missing_part_named(monkeypatch):
    import json
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("REFLECT_MODE", "model")
    conn = _conn()
    session = P.Session(question=Q)
    session.known_entity_ids = {"e1"}
    session.steps.append(P.Step(1, "search", {"text": "b-jet"}, rows=47))
    chat = _chat(VERDICT_NOT_READY)
    state = {"session": session, "round": 2, "max_rounds": 6, "max_rows": 25, "max_places": 3,
             "messages": [], "pending_calls": [{"id": "c1", "name": "answer",
                                                "arguments": json.dumps({"text": "The analyses are 2001.06899.",
                                                                         "reason": "answered"})}]}
    runtime = {"conn": conn, "chat": chat, "execute": lambda n, a: None,
               "persist": False, "push_further": False, "answer_critic": False}
    G.execute(state, {"configurable": runtime})
    sent = state["messages"][-1]["content"]
    assert "missing transverse momentum" in sent and "Retrieve that first" in sent
    assert "not a judgement of your answer" in sent
    assert session.reflections_used == 1 and session.reflect_checks == 1
    assert state["reflect_note"].startswith("PART:")


def test_the_prose_exit_is_checked_too_and_routes_back(monkeypatch):
    """30-45% of answers never call answer(); the check must see those as well."""
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("REFLECT_MODE", "model")
    session = P.Session(question=Q)
    session.known_entity_ids = {"e1"}
    session.steps.append(P.Step(1, "search", {"text": "b-jet"}, rows=47))
    state = {"session": session, "round": 2, "max_rounds": 6, "messages": [],
             "pending_calls": [], "last_content": "The analyses are 2001.06899."}
    assert G.after_plan(state) == "reflect_check"
    G.reflect_check(state, {"configurable": {"chat": _chat(VERDICT_NOT_READY), "conn": _conn()}})
    assert G.after_reflect(state) == "plan"
    assert state["messages"][-1]["role"] == "user"
    assert "missing transverse momentum" in state["messages"][-1]["content"]
    assert session.reflections_used == 1

    # the verdict belongs to the round, so a second exit in the same round
    # reuses it instead of paying for another call
    calls = session.llm_calls
    G.reflect_check(state, {"configurable": {"chat": _boom, "conn": _conn()}})
    assert session.llm_calls == calls

    # a ready verdict lets the prose answer through
    session2 = P.Session(question=Q)
    session2.known_entity_ids = {"e1"}
    session2.steps.append(P.Step(1, "search", {"text": "b"}, rows=4))
    state2 = {"session": session2, "round": 2, "max_rounds": 6, "messages": [],
              "pending_calls": [], "last_content": "The analyses are 2001.06899."}
    G.reflect_check(state2, {"configurable": {"chat": _chat(VERDICT_READY), "conn": _conn()}})
    assert G.after_reflect(state2) == "finish" and state2["messages"] == []


def test_the_prose_exit_is_untouched_when_the_budget_is_out(monkeypatch):
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("REFLECT_MODE", "model")
    session = P.Session(question=Q)
    session.steps.append(P.Step(1, "search", {"text": "b"}, rows=4))
    session.reflections_used = R.MAX_REFLECTIONS
    state = {"session": session, "round": 2, "max_rounds": 6, "messages": [],
             "pending_calls": [], "last_content": "answer"}
    assert G.after_plan(state) == "finish"


def test_the_check_runs_per_round_and_the_gate_reuses_it(monkeypatch):
    """One call per round, not one per exit: the planner sees the verdict before
    it decides, and the answer gate reuses the same verdict."""
    import json
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("REFLECT_MODE", "model")
    conn = _conn()
    session = P.Session(question=Q)
    session.known_entity_ids = {"e1"}
    session.steps.append(P.Step(1, "search", {"text": "b-jet"}, rows=47))
    chat = _chat(VERDICT_NOT_READY)

    def planner_chat(msgs, tools=None):
        if tools is None:
            return chat(msgs, None)
        planner_chat.seen.append("\n".join(m.get("content") or "" for m in msgs))
        return NS(choices=[NS(message=NS(content="AIM: keep going", tool_calls=[]))], usage=None)
    planner_chat.seen = []

    state = {"session": session, "question": Q, "round": 2, "max_rounds": 6,
             "max_places": 3, "max_rows": 25,
             "messages": [{"role": "user", "content": "q"}], "pending_calls": [],
             "last_content": "", "objective": "", "review_cycles": 0}
    runtime = {"chat": planner_chat, "conn": conn, "tools": [{"type": "function", "function": {"name": "answer"}}],
               "subgoal_status": False, "reviewer": False, "state_objective": False}
    G.plan(state, {"configurable": runtime})
    assert session.reflect_checks == 1                       # one check for the round
    assert "HOW MUCH OF THE QUESTION IS ANSWERED" in planner_chat.seen[0]
    assert "missing transverse momentum: nothing yet" in planner_chat.seen[0]

    # the gate now reuses it: no second call
    state["pending_calls"] = [{"id": "c1", "name": "answer",
                               "arguments": json.dumps({"text": "2001.06899", "reason": "answered"})}]
    runtime.update({"execute": lambda n, a: None, "persist": False,
                    "push_further": False, "answer_critic": False})
    G.execute(state, {"configurable": runtime})
    assert session.reflect_checks == 1                       # still one
    assert session.reflections_used == 1
    assert "Retrieve that first" in state["messages"][-1]["content"]
    assert state["_reflect_fresh"] is False                  # next round re-checks


def test_a_merely_unconfident_verdict_still_lets_the_planner_stop():
    """READY:no with every part covered must not read as 'still missing'.

    qwen3-32b returned READY:no on all eleven verdicts of the first live run
    while naming an empty HAVE on four. Rendering all eleven as a gap gives
    the planner a nudge every round and never a stopping signal.
    """
    covered = R.Verdict(ready=False, next_step="read the two ATLAS papers",
                        parts=[("analyses", "3 rows, ATLAS SUSY 2019"),
                               ("the variable", "1 row, E_T^miss")])
    out = R.render_verdict(covered)
    assert "you may answer" in out
    assert "Nothing retrieved yet speaks to" not in out
    assert "read the two ATLAS papers" in out      # NEXT is still offered

    gap = R.Verdict(ready=False, next_step="search for the generator",
                    parts=[("analyses", "3 rows"), ("the generator", "nothing yet")])
    assert "Nothing retrieved yet speaks to: the generator" in R.render_verdict(gap)
    assert "you may answer" not in R.render_verdict(gap)

    done = R.Verdict(ready=True, next_step="-",
                     parts=[("analyses", "3 rows"), ("the variable", "1 row")])
    assert "Everything the question asks for has something behind it" in R.render_verdict(done)


def test_the_check_uses_its_own_client_when_there_is_one(monkeypatch):
    """The first two model-mode arms timed out on 83% and 75% of records.

    The check ran on the planner's reasoning model, once a round. It is an
    extraction task and belongs on the small endpoint, like the answer critic.
    """
    from hepcoveragekg.query import graph as G
    monkeypatch.setenv("REFLECT_MODE", "model")
    small = _chat(VERDICT_READY)
    session = P.Session(question=Q)
    session.steps.append(P.Step(1, "search", {"text": "b"}, rows=4))
    state = {"session": session, "round": 2, "max_rounds": 6, "messages": [],
             "pending_calls": [], "last_content": "prose"}
    G.reflect_check(state, {"configurable": {
        "chat": _boom, "reflect_chat": small, "conn": _conn()}})
    assert small.prompts, "the small client was never called"
    assert session.reflect_checks == 1

    # with no separate endpoint it falls back to the planner's client
    assert G._reflect_chat({"chat": "planner"}) == "planner"
    assert G._reflect_chat({"chat": "planner", "reflect_chat": "small"}) == "small"


def test_the_record_says_which_arm_it_was(monkeypatch):
    """Three paper36 runs carried --subgoal-status and the same config hash.

    The knobs live in the environment, so the flag string and the config hash
    both miss them and the arm cannot be identified after the fact.
    """
    from hepcoveragekg.eval import systems

    monkeypatch.setenv("REFLECT_MODE", "model")
    monkeypatch.setenv("SUBGOAL_SCOPE", "question")
    session = P.Session(question=Q)
    rec = systems.from_session(session)
    assert rec.reflect_mode == "model"
    assert rec.subgoal_scope == "question"

    monkeypatch.delenv("REFLECT_MODE")
    monkeypatch.delenv("SUBGOAL_SCOPE")
    rec = systems.from_session(P.Session(question=Q))
    assert rec.reflect_mode == "off" and rec.subgoal_scope == ""
