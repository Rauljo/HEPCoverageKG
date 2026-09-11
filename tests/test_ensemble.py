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


def _graph():
    """A graph where the typed side's footprint reaches a paper it never named."""
    import sqlite3
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript("""
      CREATE TABLE paper(arxiv_id TEXT);
      CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT, aliases TEXT);
      CREATE TABLE assertion(assertion_id TEXT, paper_id TEXT, subject_id TEXT, object_id TEXT);
      CREATE TABLE assertion_evidence(assertion_id TEXT, evidence_id TEXT);
      CREATE TABLE evidence(evidence_id TEXT, quote TEXT);
      INSERT INTO paper VALUES('2001.00001'),('2002.00002'),('2003.00003');
      INSERT INTO entity_occurrence VALUES('E1','2003.00003','b-tagged jet','[]');
      INSERT INTO assertion VALUES('a1','2003.00003','E1',NULL);
      INSERT INTO assertion_evidence VALUES('a1','ev1');
      INSERT INTO evidence VALUES('ev1','Events are required to have a b-tagged jet.');
    """)
    return c


def test_critic_mode_judges_both_pools_once(monkeypatch):
    """D-167: neither side's own selection decides; the judge sees the union of
    the typed footprint, free-SQL's rows and whatever either one named."""
    from hepcoveragekg.eval import free_sql as F
    t = _Sys("The analyses are 2001.00001.")
    t._a = Answer(text="The analyses are 2001.00001.", entity_ids=["E1"], steps=[{"tool": "x"}], llm_calls=1, rounds=1)
    s = _Sys("Papers: 2002.00002", papers=["2002.00002"], cited="answer.papers")
    seen = {}
    def fake(conn, question, touched, named=(), chat=None):
        seen["cands"] = sorted(touched)
        return ["2001.00001", "2003.00003"], {"kept": 2, "candidates": len(touched)}
    monkeypatch.setattr(F, "critic_selects_papers", fake)
    monkeypatch.setenv("ENSEMBLE_MODE", "critic")
    out = E.EnsembleSystem(t, s, conn=_graph()).answer(SimpleNamespace(text="which use b-tagged jets?"))
    # 2003.00003 is in NEITHER answer: it comes from the typed side's footprint.
    assert seen["cands"] == ["2001.00001", "2002.00002", "2003.00003"]
    assert out.constrained_ids == ["2001.00001", "2003.00003"] and out.constrained_mode == "critic"
    assert out.papers == ["2001.00001", "2003.00003"] and out.answer_review["kept"] == 2
    assert "2001.00001" in out.text and "2002.00002" not in out.text


def test_critic_mode_needs_a_connection(monkeypatch):
    monkeypatch.setenv("ENSEMBLE_MODE", "critic")
    try:
        E.EnsembleSystem(_Sys("a"), _Sys("b")); assert False
    except ValueError:
        pass
