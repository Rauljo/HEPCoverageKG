"""The free-SQL control: safety, fairness, and not scoring zero on a technicality."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hepcoveragekg.eval import free_sql as F
from hepcoveragekg.eval.questions import Question


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "t.db"
    rw = sqlite3.connect(path)
    rw.executescript("""
        CREATE TABLE paper (arxiv_id TEXT PRIMARY KEY, category TEXT);
        CREATE TABLE entity (entity_id TEXT, kind TEXT, label TEXT);
        CREATE TABLE assertion (assertion_id TEXT, predicate TEXT, paper_id TEXT);
        INSERT INTO paper VALUES ('2106.01676','search'),('2001.06899','measurement');
        INSERT INTO entity VALUES ('g1','generator','Pythia 8.230'),
                                  ('g2','generator','PYTHIA8');
        INSERT INTO assertion VALUES ('a1','uses_generator','2106.01676');
    """)
    rw.commit(); rw.close()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


@pytest.mark.parametrize("query", [
    "DELETE FROM paper",
    "DROP TABLE entity",
    "UPDATE paper SET category='x'",
    "PRAGMA table_info(paper)",
    "SELECT 1; DROP TABLE paper",
    "INSERT INTO paper VALUES ('x','y')",
])
def test_a_write_never_runs(conn, query):
    """Refused with a reason the model can act on. The read-only connection is
    the real barrier -- these checks exist to produce a useful error, not to be
    the defence."""
    assert F.run_sql(conn, query).error


@pytest.mark.parametrize("query", [
    "SELECT label FROM entity WHERE label LIKE '%update%'",
    "SELECT COUNT(*) FROM entity WHERE label LIKE '%drop%'",
    "SELECT label FROM entity WHERE label LIKE '%created%'",
])
def test_an_ordinary_read_is_not_mistaken_for_a_write(conn, query):
    """The first version carried a keyword blacklist and rejected these -- reads
    over a corpus whose labels contain English words. A control handicapped by a
    false positive is not measuring what it claims to."""
    assert not F.run_sql(conn, query).error


def test_the_connection_is_the_real_barrier(conn):
    """If the string checks were fooled, the connection must still refuse."""
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM paper")


def test_a_read_works_and_reports_its_row_count(conn):
    r = F.run_sql(conn, "SELECT arxiv_id FROM paper ORDER BY arxiv_id")
    assert not r.error and len(r.rows) == 2
    assert "2 rows" in r.render()


def test_truncation_is_reported_never_silent(conn):
    """A silent cap makes the model count a capped set with full confidence --
    the failure `count` was built to avoid."""
    r = F.run_sql(conn, "SELECT arxiv_id FROM paper", max_rows=1)
    assert r.truncated and "MORE EXIST" in r.render()


def test_an_error_reaches_the_model_verbatim(conn):
    """An agent that cannot see its own syntax error is being tested on one-shot
    SQL, which is not the question."""
    r = F.run_sql(conn, "SELECT nope FROM paper")
    assert "nope" in r.error and "ERROR" in r.render()


def test_papers_touched_are_collected_like_the_planners_footprint(conn):
    r = F.run_sql(conn, "SELECT arxiv_id FROM paper ORDER BY arxiv_id")
    assert F.papers_in(r) == ["2001.06899", "2106.01676"]


def test_entity_ids_are_collected_not_only_papers(conn):
    """The retrieval tier's truth IS an entity id, and `entity_retrieved` scores
    `Answer.entity_ids`. Collecting only papers scored all 16 of those questions
    0.000 for a reason with nothing to do with SQL -- a control must not lose on
    a field the harness forgot to fill."""
    result = F.SqlResult(rows=[("hepkg:generator:pythia8_v8p212", "Pythia 8.212"),
                               ("hepkg:region:sr_2j", "SR-2j")],
                         columns=["entity_id", "label"])
    assert F.entities_in(result) == ["hepkg:generator:pythia8_v8p212",
                                     "hepkg:region:sr_2j"]
    assert F.papers_in(result) == [], "an entity id is not a paper id"

    papers = F.SqlResult(rows=[("2106.01676",)], columns=["arxiv_id"])
    assert F.papers_in(papers) == ["2106.01676"]
    assert F.entities_in(papers) == [], "a paper id is not an entity id"


def test_the_schema_shows_real_values_not_just_column_names(conn):
    """The typed agent is handed closed vocabularies. A SQL agent facing
    `entity.kind` cannot guess the value is `generator` and not `Generator`, and
    that difficulty is about our column conventions, not about querying."""
    brief = F.schema_brief(conn)
    assert "generator" in brief and "uses_generator" in brief
    assert "entity_canonical" in brief or "NOT unified" in brief


def test_it_answers_through_the_same_System_protocol(conn):
    """One round, no tool call: the model replies in prose."""
    class Msg:
        content = "Two analyses: 2106.01676 and 2001.06899."
        tool_calls = None

    def chat(messages, tools):
        assert tools[0]["function"]["name"] == "sql"
        return type("R", (), {"choices": [type("C", (), {"message": Msg()})()],
                              "usage": None})()

    system = F.FreeSQLSystem(conn, chat=chat)
    a = system.answer(Question(qid="q", text="which analyses?"))
    assert a.answered and "2106.01676" in a.text
    assert system.config["kind"] == "free-sql"


def test_running_out_of_rounds_is_not_an_error(conn):
    """It looked and never concluded. That is a real behaviour and should score
    as one, not as a crash."""
    class Call:
        id = "1"
        function = type("F", (), {"name": "sql",
                                  "arguments": '{"query": "SELECT arxiv_id FROM paper"}'})()

    class Msg:
        content = ""
        tool_calls = [Call()]

    def chat(messages, tools):
        return type("R", (), {"choices": [type("C", (), {"message": Msg()})()],
                              "usage": None})()

    a = F.FreeSQLSystem(conn, chat=chat, max_rounds=2).answer(
        Question(qid="q", text="which analyses?"))
    assert not a.answered and not a.error
    assert a.rounds == 2
    assert a.papers == ["2001.06899", "2106.01676"], "what it touched is still recorded"


# --------------------------------------------------------------------------
# the upgraded control: two tools, and persistence to match the planner's
# --------------------------------------------------------------------------

def _scripted(replies):
    """A model that returns a canned sequence of (tool_name, args) or None."""
    state = {"i": 0, "offered": None}

    def chat(messages, tools):
        state["offered"] = [t["function"]["name"] for t in tools]
        i = state["i"]; state["i"] += 1
        spec = replies[i] if i < len(replies) else None
        if spec is None:
            msg = type("M", (), {"content": "no data found", "tool_calls": None})()
        else:
            name, args = spec
            call = type("T", (), {"id": str(i), "function": type("F", (), {
                "name": name, "arguments": json.dumps(args)})()})()
            msg = type("M", (), {"content": "", "tool_calls": [call]})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()],
                              "usage": None})()
    return chat, state


def test_it_only_offers_search_when_it_has_an_index(conn):
    """Without an index it is the SQL-only control; with one it is the
    same-raw-materials control. The config has to say which ran."""
    chat, state = _scripted([None])
    F.FreeSQLSystem(conn, chat=chat).answer(Question(qid="q", text="x"))
    assert state["offered"] == ["sql"]
    assert F.FreeSQLSystem(conn, chat=chat).config["tools"] == ["sql"]


def test_the_ladder_fires_only_when_nothing_was_found(conn):
    """Same gate as the planner's: rounds to spare, and empty-handed. A run that
    found rows and chose to answer is left alone."""
    chat, _ = _scripted([None, None, None, None, None])
    a = F.FreeSQLSystem(conn, chat=chat, max_rounds=6).answer(
        Question(qid="q", text="x"))
    assert a.rounds > 1, "it should have been pushed back at least once"

    # found something -> answered immediately, no pushback
    chat2, _ = _scripted([("sql", {"query": "SELECT arxiv_id FROM paper"}), None])
    b = F.FreeSQLSystem(conn, chat=chat2, max_rounds=6).answer(
        Question(qid="q", text="x"))
    assert b.answered and b.papers


def test_the_ladder_is_finite(conn):
    """It cannot loop: three rungs, each once, then the answer goes through."""
    chat, _ = _scripted([None] * 12)
    a = F.FreeSQLSystem(conn, chat=chat, max_rounds=6).answer(
        Question(qid="q", text="x"))
    assert a.rounds <= 6
    assert len(F.FREE_LADDER) == 3


def test_persistence_can_be_turned_off_for_a_paired_arm(conn):
    chat, _ = _scripted([None, None, None])
    a = F.FreeSQLSystem(conn, chat=chat, persist=False).answer(
        Question(qid="q", text="x"))
    assert a.rounds == 1, "with persist off it answers at the first opportunity"


def test_the_worked_examples_teach_the_join_path_that_actually_works(conn):
    """21 of 241 queries in the first run died on `entity.paper_id`, which does
    not exist. The examples and the schema both say so now."""
    prompt = F.SYSTEM_PROMPT.format(schema=F.schema_brief(conn),
                                    examples=F.WORKED_EXAMPLES)
    assert "entity has NO paper_id" in prompt
    assert "entity_occurrence" in prompt
    assert "entity_canonical" in prompt, "0 of 241 queries used it"
    assert "GROUP BY kind" in prompt
