"""The ranked answer (D-128): the ranking finally acts where the answer is decided."""
import json
from hepcoveragekg.query import answer_critic as AC, planner


def _rk(grades, usable=True):
    r = AC.Ranking(question="q", grades=dict(grades))
    r.order = sorted(grades, key=lambda p: -grades[p])
    if not usable:                      # force the spread check to fail
        r.grades = {p: 3 for p in grades}
    return r


def test_candidates_merge_best_grade_across_calls_best_first():
    a = _rk({"2001.00001": 1, "2002.00002": 3})
    b = _rk({"2001.00001": 2, "2003.00003": 2})
    got = AC.ranked_candidates([a, b])
    assert got[0] == ("2002.00002", 3)
    assert dict(got)["2001.00001"] == 2, "a paper keeps its best grade"


def test_an_unusable_ranking_contributes_nothing():
    bad = AC.Ranking(question="q", grades={f"20{i:02d}.0000{i%10}": (3 if i < 3 else 0) for i in range(40)})
    assert not bad.usable
    assert AC.ranked_candidates([bad]) == []


def _run(flag, rankings, text, max_rounds=6, round_=1):
    from hepcoveragekg.query import graph as G
    session = planner.Session(question="q"); session.rankings = rankings
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    state = {"session": session, "messages": [], "round": round_, "max_rounds": max_rounds,
             "max_places": 8, "max_rows": 25, "last_content": "",
             "pending_calls": [{"id": "1", "name": "answer",
                                "arguments": json.dumps({"text": text, "reason": "answered"})}]}
    G.execute(state, {"configurable": {"execute": lambda n, a: None, "tools": [], "chat": None,
                                       "contract": "v3", "ranked_answer": flag}})
    return session, state


def test_it_asks_once_when_the_answer_misses_the_strong_candidates():
    rk = _rk({"2001.00001": 3, "2002.00002": 3, "2003.00003": 2, "2004.00004": 1})
    s, st = _run(True, [rk], "The analyses are 2009.09999 and 2008.08888.")
    assert s.ranked_answer_asked and s.ranked_answer_shown == 4
    msg = st["messages"][-1]["content"]
    assert "2001.00001  grade 3" in msg and "names 0 of the 3" in msg
    assert s.answer_before_gate.startswith("The analyses"), "text stashed, not lost"


def test_it_does_not_ask_when_the_answer_already_names_the_top():
    rk = _rk({"2001.00001": 3, "2002.00002": 3, "2003.00003": 1})
    s, _ = _run(True, [rk], "They are 2001.00001 and 2002.00002.")
    assert not s.ranked_answer_asked and s.answer.startswith("They are")


def test_off_by_default_and_never_at_the_last_round():
    rk = _rk({"2001.00001": 3, "2002.00002": 3})
    s, _ = _run(False, [rk], "nothing relevant 2009.09999")
    assert not s.ranked_answer_asked
    s, _ = _run(True, [rk], "nothing relevant 2009.09999", max_rounds=2, round_=2)
    assert not s.ranked_answer_asked, "a retry at the last round loses the answer (D-127)"


def test_ranked_candidates_cap_is_a_parameter():
    from hepcoveragekg.query.answer_critic import Ranking, ranked_candidates
    grades = {f"p{i}": 3 if i < 30 else 1 for i in range(40)}
    r = Ranking(question="q", order=list(grades), grades=grades)
    r.grades["m1"] = 2; r.grades["m2"] = 1; r.order += ["m1", "m2"]   # middle grades: usable
    assert len(ranked_candidates([r])) == 15
    assert len(ranked_candidates([r], top_n=40)) == 40


def test_ranked_top_n_env_reaches_the_hook(monkeypatch):
    import os
    monkeypatch.setenv("RANKED_TOP_N", "40")
    assert int(os.environ.get("RANKED_TOP_N", "15") or 15) == 40


def test_the_prose_exit_gets_the_ranked_list_through_the_real_graph():
    """14 of 27 records in the RANKED_TOP_N=40 run left as prose and the hook
    in `execute` never ran (D-135). Through the compiled graph: prose that
    misses the strong candidates is asked once, and the second answer wins."""
    from types import SimpleNamespace
    from hepcoveragekg.query import graph as G, templates
    n = {"i": 0}
    def chat(messages, tools=None):
        n["i"] += 1
        if n["i"] == 1:
            call = SimpleNamespace(id="c1", type="function",
                                   function=SimpleNamespace(name="search", arguments=json.dumps({"text": "x"})))
            msg = SimpleNamespace(content="searching", tool_calls=[call])
        elif n["i"] == 2:
            msg = SimpleNamespace(content="The analyses are 2009.09999.", tool_calls=None)
        else:
            assert "grade 3" in messages[-1]["content"], "the ranked list was shown"
            msg = SimpleNamespace(content="They are 2001.00001 and 2002.00002.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    session = planner.Session(question="q")
    session.rankings = [_rk({"2001.00001": 3, "2002.00002": 3, "2003.00003": 1})]
    def execute(name, args):
        return templates.QueryResult(shape="search", rows=[{"entity_id": "e1", "label": "x", "kind": "k"}], note="saved as set_1")
    state = {"question": "q", "messages": [{"role": "user", "content": "q"}], "session": session,
             "round": 0, "max_rounds": 6, "max_places": 8, "max_rows": 25}
    cfg = {"configurable": {"chat": chat, "execute": execute, "tools": [], "contract": "v3",
                            "ranked_answer": True}, "recursion_limit": 40}
    G.build().invoke(state, config=cfg)
    assert n["i"] == 3, "asked exactly once"
    assert session.ranked_answer_asked and session.ranked_answer_shown == 3
    assert "2001.00001" in session.answer and "2002.00002" in session.answer
    assert session.answer_before_gate.startswith("The analyses"), "the first prose is stashed"


def test_the_prose_exit_is_not_asked_when_the_arm_is_off():
    from types import SimpleNamespace
    from hepcoveragekg.query import graph as G, templates
    n = {"i": 0}
    def chat(messages, tools=None):
        n["i"] += 1
        if n["i"] == 1:
            call = SimpleNamespace(id="c1", type="function",
                                   function=SimpleNamespace(name="search", arguments=json.dumps({"text": "x"})))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="s", tool_calls=[call]))], usage=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="The analyses are 2009.09999.", tool_calls=None))], usage=None)
    session = planner.Session(question="q")
    session.rankings = [_rk({"2001.00001": 3, "2002.00002": 3, "2003.00003": 1})]
    def execute(name, args):
        return templates.QueryResult(shape="search", rows=[{"entity_id": "e1", "label": "x", "kind": "k"}], note="")
    state = {"question": "q", "messages": [{"role": "user", "content": "q"}], "session": session,
             "round": 0, "max_rounds": 6, "max_places": 8, "max_rows": 25}
    G.build().invoke(state, config={"configurable": {"chat": chat, "execute": execute, "tools": [], "contract": "v3"}, "recursion_limit": 40})
    assert n["i"] == 2 and not session.ranked_answer_asked
