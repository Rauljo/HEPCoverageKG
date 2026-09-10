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
        if self.refuse and ("response_format" in kw and kw["response_format"].get("type") == "json_schema" or "guided_json" in kw.get("extra_body", {})):
            raise RuntimeError("schema decoding not supported")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def _enum(cl):
    return cl.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["papers"]["items"]["enum"]


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
    assert s.answer_papers == [] and s.constrained_mode == "json_schema"   # footprint kept for reach
    assert _enum(cl) == ["2001.00001", "2002.00002", "2003.00003"]


def test_falls_back_to_plain_json_when_guided_is_refused(monkeypatch):
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client('{"papers": ["2003.00003"]}', refuse_guided=True)
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_ids == ["2003.00003"] and s.constrained_mode == "json_object" and len(cl.calls) == 3


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
    assert _enum(cl) == ["2001.00001", "2002.00002", "2003.00003", "2050.00050"]
    assert s.constrained_ids == ["2050.00050"] and "1606.05334" not in s.answer.split("Papers:")[-1]


def test_critic_dropped_entities_leave_the_candidate_list(monkeypatch):
    """D-158: with CONSTRAINED_FROM_KEPT=1 an entity the search critic dropped
    contributes no candidates; without it the enum is the full footprint."""
    from types import SimpleNamespace as NS
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    review = NS(verdicts=[NS(entity_id="e2", kept=False), NS(entity_id="e1", kept=True)])
    for flag, enum, dropped in (("", ["2001.00001", "2002.00002", "2003.00003"], 0),
                                ("1", ["2001.00001", "2002.00002"], 1)):
        monkeypatch.setenv("CONSTRAINED_FROM_KEPT", flag)
        cl = _Client('{"papers": ["2001.00001"]}')
        monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
        s = _session(); s.reviews = [review]
        G._constrained_ids({"conn": _conn()}, s)
        assert _enum(cl) == enum
        assert s.constrained_critic_dropped == dropped


def test_answer_critic_runs_after_the_selector_when_constrained(monkeypatch):
    """D-158: the critic must judge the list the selector wrote, not a draft
    the selector then overwrites."""
    order = []
    monkeypatch.setattr(G, "_answer_critic", lambda rt, s: order.append("critic"))
    monkeypatch.setattr(G, "_grade_strike", lambda s: order.append("strike"))
    monkeypatch.setattr(G, "_constrained_ids", lambda rt, s: order.append("select"))
    monkeypatch.setenv("CONSTRAINED_IDS", "1")
    G._answer_exit({"answer_critic": True}, _session())
    assert order == ["strike", "select", "critic"]
    order.clear(); monkeypatch.delenv("CONSTRAINED_IDS")
    G._answer_exit({"answer_critic": True}, _session())
    assert order == ["critic", "strike", "select"]


def test_a_reply_that_ignores_the_schema_is_recorded_as_unparsed(monkeypatch):
    """D-159: vLLM 0.18 accepted `guided_json` and answered in prose."""
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client("Okay, let's see. The user is asking which analyses...")
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    s = _session(); G._constrained_ids({"conn": _conn()}, s)
    assert s.constrained_ids == [] and s.constrained_mode == "json_schema-unparsed" and "Papers:" not in s.answer


def test_critic_selects_writes_the_judges_kept_set(monkeypatch):
    """D-162: with CRITIC_SELECTS=1 the answer critic judges every candidate
    and its kept set is the list; the answerer's selection call is not made."""
    from types import SimpleNamespace as NS
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("CRITIC_SELECTS", "1"); monkeypatch.setenv("LLM_MODEL_NAME", "m")
    cl = _Client('{"papers": ["2001.00001"]}')
    monkeypatch.setattr(planner, "_client", lambda: (cl, "m"))
    from hepcoveragekg.query import answer_critic as AC
    monkeypatch.setattr(AC, "evidence_by_paper", lambda conn, ids, papers: {p: ({"b-jet"}, []) for p in papers})
    monkeypatch.setattr(AC, "judge_papers", lambda chat, q, ev: NS(kept=["2002.00002", "2003.00003"], dropped=["2001.00001"], defaulted=0, verdicts=[1, 2, 3]))
    s = _session(); G._constrained_ids({"conn": _conn(), "answer_critic_chat": lambda m: None}, s)
    assert s.constrained_ids == ["2002.00002", "2003.00003"] and s.constrained_mode == "critic"
    assert s.answer.endswith("Papers: 2002.00002, 2003.00003") and cl.calls == []
    order = []
    monkeypatch.setattr(G, "_answer_critic", lambda rt, s: order.append("critic"))
    monkeypatch.setattr(G, "_grade_strike", lambda s: None); monkeypatch.setattr(G, "_constrained_ids", lambda rt, s: order.append("select"))
    G._answer_exit({"answer_critic": True}, _session())
    assert order == ["select"]


def test_answer_critic_parse_survives_a_think_block():
    from hepcoveragekg.query import answer_critic as AC
    raw = '<think>Let me check {paper 1}... the set {a, b} is fine.</think>\n{"verdicts": [{"paper": "2001.00001", "keep": true, "why": "uses it"}, {"paper": "2002.00002", "keep": false, "why": "no"}]}'
    got = AC._parse(raw, {"2001.00001", "2002.00002"})
    assert got == {"2001.00001": (True, "uses it"), "2002.00002": (False, "no")}
