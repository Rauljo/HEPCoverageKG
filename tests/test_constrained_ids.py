"""Constrained selection of the answer's papers (CONSTRAINED_IDS=1)."""
import json, sqlite3
from types import SimpleNamespace
from hepcoveragekg.query import graph as G, planner


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""CREATE TABLE entity_occurrence(paper_id TEXT, entity_id TEXT, label TEXT, kind TEXT, bundle_id TEXT);
      INSERT INTO entity_occurrence VALUES('2001.00001','e1','b-jet','k','b1'),('2002.00002','e1','b-jet','k','b2'),('2003.00003','e2','MET','k','b3');""")
    return c


class _Client:
    def __init__(self, reply, refuse_guided=False): self.reply = reply; self.refuse = refuse_guided; self.calls = []
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.calls.append(kw)
        if "extra_body" in kw and self.refuse: raise RuntimeError("guided decoding not supported")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def _session():
    s = planner.Session(question="which use b-jets?"); s.answer = "Several analyses use b-jets."
    s.known_entity_ids = {"e1", "e2"}
    return s


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("CONSTRAINED_IDS", raising=False)
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_candidates == 0 and "Papers:" not in s.answer


def test_selects_only_from_the_footprint_and_writes_ids(monkeypatch):
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client(json.dumps({"papers": ["2001.00001", "2002.00002", "2099.09999"]}))
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_candidates == 3 and s.constrained_ids == ["2001.00001", "2002.00002"]
    assert "2099.09999" not in s.answer and s.answer.endswith("Papers: 2001.00001, 2002.00002")
    assert s.answer_papers == [] and s.constrained_mode == "guided_json"   # footprint kept for reach
    assert cl.calls[0]["extra_body"]["guided_json"]["properties"]["papers"]["items"]["enum"] == ["2001.00001", "2002.00002", "2003.00003"]


def test_falls_back_to_plain_json_when_guided_is_refused(monkeypatch):
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client('{"papers": ["2003.00003"]}', refuse_guided=True)
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_ids == ["2003.00003"] and s.constrained_mode == "json_object" and len(cl.calls) == 2


def test_bare_client_is_tolerated(monkeypatch):
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client('{"papers": ["2001.00001"]}')
    monkeypatch.setattr(planner, "_client", lambda: cl)
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_ids == ["2001.00001"]


def test_gated_by_question_shape(monkeypatch):
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client('{"papers": ["2001.00001"]}')
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    for shape, expect in (("count", 0), ("labels", 0), ("papers", 1), ("", 1)):
        s = _session(); G._constrained_ids({"conn": _conn(), "question_shape": shape}, s)
        assert len(s.constrained_ids) == expect, shape


def test_draft_ids_are_candidates_only_if_the_graph_holds_them(monkeypatch):
    """D-157: job 54296 let ids the draft had INVENTED into the enum (10 of 168
    records carried runs like 1606.05334, 1606.05335, ...). A draft-named id is
    a candidate only when the paper table holds it."""
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    c = _conn()
    c.executescript("""CREATE TABLE paper(arxiv_id TEXT);
      INSERT INTO paper VALUES('2001.00001'),('2002.00002'),('2003.00003'),('2050.00050');""")
    cl = _Client(json.dumps({"papers": ["2050.00050", "1606.05334"]}))
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    s = _session()
    s.answer = "See 2001.00001, 2050.00050 and 1606.05334, 1606.05335."   # one held, two invented
    G._constrained_ids({"conn": c}, s)
    enum = cl.calls[0]["extra_body"]["guided_json"]["properties"]["papers"]["items"]["enum"]
    assert enum == ["2001.00001", "2002.00002", "2003.00003", "2050.00050"]
    assert s.constrained_ids == ["2050.00050"] and "1606.05334" not in s.answer.split("Papers:")[-1]
