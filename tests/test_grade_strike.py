"""The grade strike (D-142): shorten an over-long named answer with the ranker's grade 0."""
from hepcoveragekg.query import planner, graph as G, answer_critic as AC


def _session(text, grades, usable=True):
    s = planner.Session(question="q"); s.answer = text
    r = AC.Ranking(question="q", grades=dict(grades)); r.order = list(grades)
    if not usable:
        r.grades = {p: 3 for p in grades}
    s.rankings = [r]
    return s


def test_off_unless_the_variable_is_set(monkeypatch):
    monkeypatch.delenv("STRIKE_GRADE_MAX", raising=False)
    s = _session("2001.00001 and 2002.00002", {"2001.00001": 0, "2002.00002": 3, "2003.00003": 1})
    G._grade_strike(s)
    assert "2001.00001" in s.answer and not s.grade_struck


def test_strikes_grade_zero_keeps_ungraded_and_stashes(monkeypatch):
    monkeypatch.setenv("STRIKE_GRADE_MAX", "0")
    s = _session("They are 2001.00001, 2002.00002 and 2009.09999.", {"2001.00001": 0, "2002.00002": 3, "2003.00003": 1})
    G._grade_strike(s)
    assert "2001.00001" not in s.answer
    assert "2002.00002" in s.answer and "2009.09999" in s.answer, "grade 3 and ungraded kept"
    assert s.grade_struck == ["2001.00001"] and s.answer_before_strike.startswith("They are")


def test_never_empties_the_answer(monkeypatch):
    monkeypatch.setenv("STRIKE_GRADE_MAX", "1")
    s = _session("2001.00001 2003.00003", {"2001.00001": 0, "2003.00003": 1, "2002.00002": 2})
    G._grade_strike(s)
    assert "2001.00001" in s.answer and not s.grade_struck


def test_unusable_ranking_is_ignored(monkeypatch):
    monkeypatch.setenv("STRIKE_GRADE_MAX", "0")
    s = _session("2001.00001 and 2002.00002", {"2001.00001": 0, "2002.00002": 3}, usable=False)
    G._grade_strike(s)
    assert "2001.00001" in s.answer
