"""Tests for the answer-stage critic.

It sits on the 0.57-0.76 gap between what retrieval finds and what the answer
says, so the tests are about (a) judging the CONDITION not the topic, and
(b) never hiding its own failure -- the way a dead critic hid for weeks (D-105).
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from hepcoveragekg.query import answer_critic as AC


def _chat(payload, fail=False):
    """A stub judge. `payload` is what the model 'returns'."""
    class M:  # noqa: D401
        def __init__(s, c): s.content = c
    class C:
        def __init__(s, c): s.message = M(c)
    class R:
        def __init__(s, c): s.choices = [C(c)]
    def chat(messages):
        if fail:
            raise RuntimeError("judge is down")
        return R(payload)
    return chat


EV = {"2001.06899": ({"b-tagged jet"}, ["Events with >=1 b-tagged jet are selected."]),
      "2004.14060": ({"Muon"}, ["Two muons are required."])}


def test_keeps_and_drops_per_paper():
    payload = json.dumps({"verdicts": [
        {"paper": "2001.06899", "keep": True,  "why": "selects on b-jets"},
        {"paper": "2004.14060", "keep": False, "why": "muons only"}]})
    r = AC.judge_papers(_chat(payload), "Which analyses use b-tagged jets?", EV)
    assert r.kept == ["2001.06899"]
    assert r.dropped == ["2004.14060"]
    assert r.defaulted == 0


def test_a_missing_verdict_keeps_the_paper_and_is_counted():
    """Same asymmetry as critic.py: a drop the model never made is invisible in
    the output, so a missing verdict must keep. But it must also be COUNTED --
    that is the difference between recoverable and the D-105 silence."""
    payload = json.dumps({"verdicts": [
        {"paper": "2001.06899", "keep": False, "why": "no"}]})   # 2004 omitted
    r = AC.judge_papers(_chat(payload), "q", EV)
    assert "2004.14060" in r.kept
    assert r.defaulted == 1
    assert [v.defaulted for v in r.verdicts if v.paper_id == "2004.14060"] == [True]


def test_unparseable_output_keeps_everything_and_alarms(caplog):
    """The exact D-105 failure: judge returns nothing usable, everything is kept.
    It must be loud -- 33,836 candidates defaulted silently for weeks."""
    import logging
    with caplog.at_level(logging.WARNING):
        r = AC.judge_papers(_chat("I think all of them are relevant, actually."), "q", EV)
    assert r.kept == sorted(EV)          # nothing dropped
    assert r.defaulted == len(EV)
    assert any("not filtering" in m for m in caplog.messages)


def test_a_broken_judge_does_not_kill_the_run(caplog):
    r = AC.judge_papers(_chat("", fail=True), "q", EV)
    assert r.errors > 0 and r.kept == sorted(EV)


def test_verdicts_for_papers_not_in_the_batch_are_ignored():
    """A judge inventing an id must not add a paper to the answer."""
    payload = json.dumps({"verdicts": [
        {"paper": "9999.99999", "keep": True, "why": "invented"},
        {"paper": "2001.06899", "keep": True, "why": "ok"},
        {"paper": "2004.14060", "keep": False, "why": "no"}]})
    r = AC.judge_papers(_chat(payload), "q", EV)
    assert "9999.99999" not in r.kept
    assert set(v.paper_id for v in r.verdicts) == set(EV)


def test_chunking_covers_every_paper_exactly_once():
    ev = {f"20{i:02d}.0000{i%10}": ({f"e{i}"}, [f"quote {i}"]) for i in range(20)}
    seen = []
    def chat(messages):
        body = messages[1]["content"]
        ids = [p for p in ev if p in body]
        seen.extend(ids)
        class M: content = json.dumps({"verdicts": [{"paper": p, "keep": True} for p in ids]})
        class C: message = M()
        class R: choices = [C()]
        return R()
    r = AC.judge_papers(chat, "q", ev, chunk=8)
    assert sorted(seen) == sorted(ev)          # each paper judged once
    assert len(r.verdicts) == len(ev)
    assert r.calls == 3                        # 20 papers / chunk 8


def test_the_prompt_carries_the_quote_not_just_the_label():
    """gf-01-condition turns on 'veto = use', which lives in the sentence and
    not in the entity label. Judging on labels alone cannot get it right."""
    captured = {}
    def chat(messages):
        captured["user"] = messages[1]["content"]
        class M: content = '{"verdicts": []}'
        class C: message = M()
        class R: choices = [C()]
        return R()
    AC.judge_papers(chat, "Which analyses use b-tagged jets?", EV)
    assert "Events with >=1 b-tagged jet are selected." in captured["user"]
    assert "b-tagged jet" in captured["user"]


def test_evidence_by_paper_reads_only_retrieved_entities():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE entity_occurrence (entity_id TEXT, paper_id TEXT, label TEXT)")
    conn.execute("CREATE TABLE assertion (assertion_id TEXT, paper_id TEXT, subject_id TEXT, object_id TEXT)")
    conn.execute("CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT)")
    conn.execute("CREATE TABLE evidence (evidence_id TEXT, quote TEXT)")
    conn.execute("INSERT INTO entity_occurrence VALUES ('E1','P1','b-tagged jet')")
    conn.execute("INSERT INTO entity_occurrence VALUES ('E2','P1','Muon')")
    conn.execute("INSERT INTO assertion VALUES ('a1','P1','E1',NULL)")
    conn.execute("INSERT INTO assertion_evidence VALUES ('a1','ev1')")
    conn.execute("INSERT INTO evidence VALUES ('ev1','b-tagged jets are required.')")
    got = AC.evidence_by_paper(conn, ["E1"])          # E2 was NOT retrieved
    assert set(got) == {"P1"}
    labels, quotes = got["P1"][0], got["P1"][1]
    assert "b-tagged jet" in labels and "Muon" not in labels
    assert quotes == ["b-tagged jets are required."]


# --------------------------------------------------------------------------
# the wiring: what the critic is allowed to judge, and what it may change
# --------------------------------------------------------------------------

class _Judge:
    """A judge with a fixed opinion, so the wiring is what is under test."""
    def __init__(self, drop=()):
        self.drop = set(drop)
        self.seen = []

    def __call__(self, messages):
        import json as _json
        import re as _re
        body = messages[-1]["content"]
        papers = _re.findall(r"\b\d{4}\.\d{4,5}\b", body)
        self.seen.extend(papers)
        text = _json.dumps({"verdicts": [
            {"paper": p, "keep": p not in self.drop, "why": "t"}
            for p in dict.fromkeys(papers)]})
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": text})()})()]})()


def _session(text, papers=()):
    from hepcoveragekg.query import planner
    s = planner.Session(question="which analyses unfold?")
    s.answer = text
    s.answer_papers = list(papers)
    s.known_entity_ids = ["e1"]
    return s


def _conn_with(papers):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE entity_occurrence (paper_id TEXT, entity_id TEXT, label TEXT);
        CREATE TABLE assertion (assertion_id TEXT, paper_id TEXT,
                                subject_id TEXT, object_id TEXT);
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);
        CREATE TABLE evidence (evidence_id TEXT, quote TEXT);
    """)
    for p in papers:
        conn.execute("INSERT INTO entity_occurrence VALUES (?,?,?)", (p, "e1", "unfolding"))
    return conn


def test_the_critic_judges_the_ids_in_the_prose_not_just_the_citation():
    """D-107: `cited` was empty in 59 of 60 answers, so judging only
    `answer_papers` would have judged nothing while reporting itself as run."""
    from hepcoveragekg.query import graph as G

    judge = _Judge()
    s = _session("Two analyses unfold: 2004.14060 and 2006.05880.")
    G._answer_critic({"conn": _conn_with(["2004.14060", "2006.05880"]),
                      "answer_critic_chat": judge}, s)
    assert set(judge.seen) == {"2004.14060", "2006.05880"}


def test_a_dropped_paper_leaves_the_written_answer():
    """Filtering `answer_papers` alone is invisible to every set scorer."""
    from hepcoveragekg.query import graph as G

    s = _session("Two analyses unfold: 2004.14060 and 2006.05880.")
    G._answer_critic({"conn": _conn_with(["2004.14060", "2006.05880"]),
                      "answer_critic_chat": _Judge(drop=["2006.05880"])}, s)
    assert "2004.14060" in s.answer
    assert "2006.05880" not in s.answer
    assert s.answer_before_critic, "the original must be kept"


def test_a_judge_that_would_empty_the_answer_is_ignored():
    from hepcoveragekg.query import graph as G

    s = _session("Two analyses unfold: 2004.14060 and 2006.05880.")
    G._answer_critic({"conn": _conn_with(["2004.14060", "2006.05880"]),
                      "answer_critic_chat":
                          _Judge(drop=["2004.14060", "2006.05880"])}, s)
    assert "2004.14060" in s.answer and "2006.05880" in s.answer
    assert not s.answer_before_critic


def test_a_broken_judge_leaves_the_answer_alone():
    from hepcoveragekg.query import graph as G

    def boom(messages):
        raise RuntimeError("503")

    s = _session("Two analyses unfold: 2004.14060 and 2006.05880.")
    G._answer_critic({"conn": _conn_with(["2004.14060"]),
                      "answer_critic_chat": boom}, s)
    assert "2004.14060" in s.answer and "2006.05880" in s.answer


def test_no_named_papers_means_no_calls():
    from hepcoveragekg.query import graph as G

    judge = _Judge()
    s = _session("The graph does not record this.")
    G._answer_critic({"conn": _conn_with([]), "answer_critic_chat": judge}, s)
    assert judge.seen == []


def test_the_critic_runs_on_prose_ids_with_no_citation():
    """The outer guard used to require `answer_papers`, which only a resolved
    citation fills -- so the arm covered a third of answers while reporting
    itself as on. D-107 measured `cited` empty in 59 of 60 answers."""
    import json as _json
    from hepcoveragekg.query import graph as G

    judge = _Judge()
    session = _session("Two analyses unfold: 2004.14060 and 2006.05880.")
    session.answer_papers = []          # no citation resolved
    state = {"session": session, "messages": [], "round": 1, "max_rounds": 6,
             "max_places": 8, "max_rows": 25, "last_content": "",
             "pending_calls": [{"id": "1", "name": "answer", "arguments":
                                _json.dumps({"text": session.answer,
                                             "reason": "answered"})}]}
    cfg = {"configurable": {"execute": lambda n, a: None, "tools": [],
                            "chat": None, "contract": "v3",
                            "answer_critic": True,
                            "conn": _conn_with(["2004.14060", "2006.05880"]),
                            "answer_critic_chat": judge}}
    session.steps.append(__import__(
        "hepcoveragekg.query.planner", fromlist=["x"]).Step(1, "search", {}, rows=5))
    G.execute(state, cfg)
    assert set(judge.seen) == {"2004.14060", "2006.05880"}, \
        "the critic must judge ids written in prose, not only a cited set"


def test_evidence_shown_is_ranked_by_the_question():
    """D-163: the quote and label that carry the question's terms come first,
    not the first three in scan order or the first eight alphabetically."""
    from hepcoveragekg.query import answer_critic as AC
    labels = ["Muon", "Jet", "HistFitter framework", "b-jet", "Electron", "MET", "Photon", "Tau", "Vertex"]
    quotes = ["Jets are reconstructed with anti-kt.", "Muons must pass isolation.", "Electrons are calibrated.",
              "The statistical analysis uses the HistFitter framework.", "Photons are vetoed."]
    block = AC._render("2001.00001", labels, quotes, question="Which analyses use the HistFitter framework?")
    lines = block.splitlines()
    assert lines[1].startswith("  retrieved: HistFitter framework")
    assert lines[2] == "  quote: The statistical analysis uses the HistFitter framework."
    assert len([l for l in lines if l.startswith("  quote:")]) == 5


def test_quote_ranking_uses_the_matched_labels_and_aliases():
    """D-163: the corpus's spellings, not only the question's words."""
    from hepcoveragekg.query import answer_critic as AC
    labels = ["Missing transverse momentum (p_T^miss)"]
    quotes = ["Jets are calibrated.", "Events must have MET above 200 GeV.", "Muons are isolated.",
              "Photons are vetoed.", "Electrons pass tight id.", "Taus are reconstructed."]
    block = AC._render("2001.00001", labels, quotes, {"MET", "p_T^miss"},
                       question="Which analyses require missing transverse momentum in their selection?")
    assert block.splitlines()[2] == "  quote: Events must have MET above 200 GeV."


def test_evidence_by_paper_reads_aliases_when_present():
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE entity_occurrence (entity_id TEXT, paper_id TEXT, label TEXT, aliases TEXT)")
    conn.execute("CREATE TABLE assertion (assertion_id TEXT, paper_id TEXT, subject_id TEXT, object_id TEXT)")
    conn.execute("CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT)")
    conn.execute("CREATE TABLE evidence (evidence_id TEXT, quote TEXT)")
    conn.execute("INSERT INTO entity_occurrence VALUES ('E1','P1','Missing transverse momentum','[\"MET\",\"p_T^miss\"]')")
    got = AC.evidence_by_paper(conn, ["E1"])
    assert got["P1"][2] == {"MET", "p_T^miss"}


def test_judge_call_falls_back_to_a_small_cap_when_the_server_rejects_the_big_one():
    """D-164: the 9B judge's 8k window rejected max_tokens=8000 on a five-quote prompt."""
    from types import SimpleNamespace as NS
    from hepcoveragekg.query import planner
    calls = []

    class _C:
        chat = property(lambda self: self); completions = property(lambda self: self)
        def create(self, **kw):
            calls.append(kw["max_tokens"])
            if kw["max_tokens"] > 4000:
                raise RuntimeError("max_tokens exceeds the model's context length")
            return NS(choices=[NS(message=NS(content='{"verdicts": []}'))])
    out = planner.answer_critic_call(_C(), "m", [{"role": "user", "content": "q"}], 8000)
    assert out.choices[0].message.content == '{"verdicts": []}'
    assert calls == [8000, 8000, 1500]
