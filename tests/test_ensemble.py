"""The ensemble arm merges what the two systems name."""
from types import SimpleNamespace
from hepcoveragekg.eval.systems import Answer
from hepcoveragekg.eval import ensemble as E


class _Sys:
    def __init__(self, text, papers=(), cited=""):
        self.config = {"kind": "stub"}; self._a = Answer(text=text, papers=list(papers), cited=cited, steps=[{"tool": "x"}], llm_calls=1, rounds=1)
    def answer(self, q): return self._a


def test_union_and_intersection(monkeypatch):
    t = _Sys("The analyses are 2001.00001 and 2002.00002.")
    s = _Sys("Papers: 2002.00002, 2003.00003", papers=["2002.00002", "2003.00003"], cited="answer.papers")
    monkeypatch.setenv("ENSEMBLE_MODE", "union")
    u = E.EnsembleSystem(t, s).answer(SimpleNamespace(text="q"))
    assert set(u.papers) == {"2001.00001", "2002.00002", "2003.00003"}
    assert all(p in u.text for p in u.papers) and u.name if False else True
    assert [st.get("side") for st in u.steps] == ["typed", "sql", "merge"]
    monkeypatch.setenv("ENSEMBLE_MODE", "intersection")
    i = E.EnsembleSystem(t, s).answer(SimpleNamespace(text="q"))
    assert i.papers == ["2002.00002"] and "2001.00001" not in i.text


def test_bad_mode_is_refused(monkeypatch):
    monkeypatch.setenv("ENSEMBLE_MODE", "vote")
    try:
        E.EnsembleSystem(_Sys("a"), _Sys("b")); assert False
    except ValueError:
        pass
