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
