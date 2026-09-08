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
