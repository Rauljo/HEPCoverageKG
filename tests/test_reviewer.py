"""The plan reviewer and the stated objective, as removable arms.

The arms exist to be measured and possibly deleted, so the first thing tested is
that with the flags off nothing changes at all -- otherwise the baseline they
are measured against has already moved.
"""
import types

import pytest

from hepcoveragekg.query import graph, planner
from hepcoveragekg.query import reviewer as R


def _reply(content="", tool_calls=None, pin=10, pout=5):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        usage=types.SimpleNamespace(prompt_tokens=pin, completion_tokens=pout))


# --- the verdict parser ---------------------------------------------------

def test_approve_and_revise_are_read():
    assert R._parse("APPROVE -- fine")[0] is True
    assert R._parse("REVISE: search first")[0] is False
    assert R._parse("revise\nbecause...")[0] is False


def test_an_unreadable_verdict_approves_and_is_counted():
    """Fail open. A reviewer that failed closed would block a run on its own bad
    output, and an endless bounce-back is worse than an unreviewed plan."""
    approved, parsed = R._parse("I think maybe you should consider...")
    assert approved is True and parsed is False


def test_a_reviewer_that_raises_does_not_kill_the_run():
    def boom(*a, **k):
        raise RuntimeError("endpoint down")

    v = R.review_plan(boom, "q", "schema", "aim", "plan")
    assert v.approved is True and v.parsed is False


def test_the_sql_reviewer_is_told_to_check_the_sql():
    """The typed side judges a hop; here a wrong query is a wrong answer that
    looks right, so validity is part of the job."""
    seen = {}

    def chat(msgs, tools=None):
        seen["system"] = msgs[0]["content"]
        return _reply("APPROVE")

    R.review_plan(chat, "q", "schema", "aim", "SELECT 1", kind="sql")
    assert "SELECT" in seen["system"] and "joins connect" in seen["system"]
    assert "tables and columns" in seen["system"]


def test_the_reviewer_sees_the_schema_and_the_objective():
    seen = {}

    def chat(msgs, tools=None):
        seen["user"] = msgs[1]["content"]
        return _reply("APPROVE")

    R.review_plan(chat, "which regions?", "TABLE assertion(...)", "find the regions",
                  "search(text=region)", history="  search(x) -> 3 rows")
    assert "TABLE assertion" in seen["user"]
    assert "find the regions" in seen["user"]
    assert "ALREADY DONE THIS RUN" in seen["user"]


# --- the arms are off by default -----------------------------------------

def test_the_objective_block_is_absent_unless_asked(tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE paper(arxiv_id TEXT); CREATE TABLE entity(entity_id TEXT);")
    try:
        base = planner.system_prompt(conn, False, False)
        armed = planner.system_prompt(conn, False, True)
    except Exception:                       # schema_card needs the real graph
        pytest.skip("schema card needs a populated database")
    assert "AIM:" not in base and "GOT:" not in base
    assert "AIM:" in armed and "GOT:" in armed


def test_the_graph_skips_review_when_the_arm_is_off():
    """`after_plan` routes on state, because a routing function gets no config."""
    state = {"session": types.SimpleNamespace(steps=[], nudged=False),
             "pending_calls": [{"id": "1", "name": "search", "arguments": "{}"}],
             "_reviewing": False}
    assert graph.after_plan(state) == "execute"
    state["_reviewing"] = True
    assert graph.after_plan(state) == "review"


def test_a_rejection_does_not_spend_a_round():
    """The whole point of the arm: a bad plan costs a re-think, not a retrieval.

    If a bounce-back consumed a round the arm would be measuring the penalty
    rather than the review.
    """
    calls = []

    def chat(msgs, tools):
        calls.append(msgs)
        return _reply("AIM: look for regions", tool_calls=None)

    session = planner.Session(question="q")
    state = {"session": session, "round": 3, "max_rounds": 6, "max_places": 8,
             "messages": [], "replan": True, "review_feedback": "search first"}
    config = {"configurable": {"chat": chat, "tools": [], "state_objective": True}}
    out = graph.plan(state, config)
    assert out["round"] == 3, "a re-plan must not advance the round counter"
    assert "search first" in calls[0][-1]["content"]


def test_a_normal_plan_does_spend_a_round():
    def chat(msgs, tools):
        return _reply("AIM: x", tool_calls=None)

    session = planner.Session(question="q")
    state = {"session": session, "round": 3, "max_rounds": 6, "max_places": 8,
             "messages": []}
    out = graph.plan(state, {"configurable": {"chat": chat, "tools": []}})
    assert out["round"] == 4


def test_the_runaway_ceiling_breaks_the_loop():
    """Retries are uncapped by design; this only stops an overnight spin."""
    session = planner.Session(question="q")
    state = {"session": session, "round": 1, "question": "q",
             "pending_calls": [], "review_cycles": R.MAX_REVIEW_CYCLES}

    def chat(msgs, tools=None):
        raise AssertionError("must not call the reviewer past the ceiling")

    out = graph.review(state, {"configurable": {"review_chat": chat}})
    assert out.get("replan") is False
    assert session.review_ceiling_hit is True
    assert graph.after_review(out) == "execute"


def test_reviewer_cost_is_counted_apart_from_the_planner():
    """A reviewer that approves everything is a no-op at double the price, and
    that has to be readable in the run rather than inferred."""
    session = planner.Session(question="q")
    state = {"session": session, "round": 1, "question": "q",
             "pending_calls": [{"id": "1", "name": "search", "arguments": "{}"}]}

    def chat(msgs, tools=None):
        return _reply("REVISE: search for the region entity first", pin=100, pout=20)

    out = graph.review(state, {"configurable": {"review_chat": chat, "review_schema": "S"}})
    assert session.review_calls == 1
    assert session.reviews_rejected == 1
    assert session.review_prompt_tokens == 100
    assert session.review_completion_tokens == 20
    assert session.llm_calls == 0, "reviewer calls must not inflate the planner's"
    assert out["replan"] is True


# --- the free-SQL side ----------------------------------------------------

def test_the_free_sql_reviewer_runs_without_tripping_over_itself(tmp_path):
    """Exercises the real loop, because a scripted unit test of `_review` alone
    would not have caught what shipped: the call site passed `question`, which
    does not exist in `answer()` -- the variable is `q.text`. Every record in the
    first run came back `NameError`, scoring a clean 0 that looked like a result.
    """
    import sqlite3
    import types as _t

    from hepcoveragekg.eval import free_sql
    from hepcoveragekg.eval.questions import Question

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE paper(arxiv_id TEXT, title TEXT);
        CREATE TABLE entity(entity_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE assertion(assertion_id TEXT, subject_id TEXT,
                               predicate TEXT, object_id TEXT);
        CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT);
        INSERT INTO paper VALUES('2001.06899','a');
        INSERT INTO entity_occurrence VALUES('e1','2001.06899','BJet');
    """)

    def msg(content="", tool_calls=None):
        return _t.SimpleNamespace(
            choices=[_t.SimpleNamespace(message=_t.SimpleNamespace(
                content=content, tool_calls=tool_calls))],
            usage=_t.SimpleNamespace(prompt_tokens=5, completion_tokens=3))

    def call(name, args, i="c1"):
        return _t.SimpleNamespace(
            id=i, function=_t.SimpleNamespace(name=name, arguments=args))

    scripted = [
        msg("AIM: list the papers", [call("sql", '{"query":"SELECT paper_id FROM entity_occurrence"}')]),
        msg("APPROVE"),                       # the reviewer
        msg("done", [call("answer", '{"text":"2001.06899","papers":["2001.06899"]}')]),
    ]
    seen = {"i": 0}

    def chat(messages, tools=None):
        out = scripted[min(seen["i"], len(scripted) - 1)]
        seen["i"] += 1
        return out

    sys_ = free_sql.FreeSQLSystem(conn, None, chat=chat, reviewer=True,
                                  state_objective=True, max_rounds=4)
    ans = sys_.answer(Question(qid="q1", text="which papers use b-jets?"))
    assert not ans.error, f"the reviewer path raised: {ans.error}"
    assert ans.review_calls == 1, "the SQL should have been reviewed once"
    assert ans.papers == ["2001.06899"]


def test_review_stats_reset_between_questions():
    """They live on the instance, and one instance answers a whole run -- so
    without a reset the ceiling would trip once and silently disable the arm for
    every later question."""
    import sqlite3
    from hepcoveragekg.eval import free_sql

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE paper(arxiv_id TEXT);
        CREATE TABLE entity(entity_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE assertion(assertion_id TEXT, subject_id TEXT,
                               predicate TEXT, object_id TEXT);
        CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT);
    """)
    sys_ = free_sql.FreeSQLSystem(conn, None,
                                  chat=lambda *a, **k: None, reviewer=True)
    sys_._review_stats = {"calls": 9, "cycles": 20, "ceiling_hit": True}
    assert sys_._review_fields()["review_calls"] == 9
    sys_._review_stats = {}
    assert sys_._review_fields()["review_calls"] == 0
    assert sys_._review_fields()["review_ceiling_hit"] is False


# --- search sets ----------------------------------------------------------

def _sqlish_conn():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE paper(arxiv_id TEXT);
        CREATE TABLE entity(entity_id TEXT, label TEXT, kind TEXT);
        CREATE TABLE assertion(assertion_id TEXT, subject_id TEXT,
                               predicate TEXT, object_id TEXT);
        CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT);
    """)
    return conn


def test_a_search_set_is_queryable_and_holds_everything():
    """The bug: it saw 20 hits, pasted 3 ids, and capped recall at 3 of 279."""
    import types as _t
    from hepcoveragekg.eval import free_sql

    sys_ = free_sql.FreeSQLSystem(_sqlish_conn(), None, chat=lambda *a, **k: None,
                                  search_sets=True)
    hits = [_t.SimpleNamespace(entity_id=f"e{i}", kind="detector_object",
                               label=f"candidate {i}") for i in range(279)]
    name = sys_._materialise(hits)
    n = sys_._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    assert n == 279, "the whole match set must be queryable, not the shown few"
    got = sys_._conn.execute(
        f"SELECT COUNT(*) FROM {name} WHERE label LIKE '%candidate 1%'").fetchone()[0]
    assert got > 0, "the model must be able to FILTER the set in SQL"


def test_sets_do_not_leak_between_questions():
    """Temp tables live on the connection, and one connection answers the whole
    run -- so question 12 could otherwise join against question 3's search."""
    import types as _t
    from hepcoveragekg.eval import free_sql

    sys_ = free_sql.FreeSQLSystem(_sqlish_conn(), None, chat=lambda *a, **k: None,
                                  search_sets=True)
    sys_._materialise([_t.SimpleNamespace(entity_id="e1", kind="k", label="l")])
    sys_._drop_sets()
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        sys_._conn.execute("SELECT COUNT(*) FROM search_1")


def test_only_a_few_sets_stay_live():
    """Twenty temp tables is a second schema to reason about."""
    import types as _t
    from hepcoveragekg.eval import free_sql
    import sqlite3

    sys_ = free_sql.FreeSQLSystem(_sqlish_conn(), None, chat=lambda *a, **k: None,
                                  search_sets=True)
    for _ in range(sys_.MAX_LIVE_SETS + 2):
        sys_._materialise([_t.SimpleNamespace(entity_id="e", kind="k", label="l")])
    with pytest.raises(sqlite3.OperationalError):
        sys_._conn.execute("SELECT COUNT(*) FROM search_1")
    last = f"search_{sys_._set_n}"
    assert sys_._conn.execute(f"SELECT COUNT(*) FROM {last}").fetchone()[0] == 1


def test_search_sets_off_changes_nothing():
    """Default OFF: the arm must be deletable without moving the baseline."""
    from hepcoveragekg.eval import free_sql

    sys_ = free_sql.FreeSQLSystem(_sqlish_conn(), None, chat=lambda *a, **k: None)
    assert sys_.config["search_sets"] is False
    assert sys_._search_sets is False


# --- refusing an unfiltered search-set join --------------------------------

def test_joining_a_whole_search_set_is_refused():
    """Measured 2026-09-01: given the set as a table, the model joined it with
    no predicate in 10 of 20 statements -- a median of 25 rows against 5 for the
    filtered half, and precision 0.704 -> 0.602. The prompt already discouraged
    it and was ignored half the time, so this refuses instead."""
    from hepcoveragekg.eval.free_sql import unfiltered_set_join as u
    assert u("SELECT p FROM entity_occurrence eo JOIN search_1 s "
             "ON eo.entity_id = s.entity_id")
    assert not u("SELECT p FROM search_1 s JOIN entity_occurrence eo "
                 "ON eo.entity_id = s.entity_id WHERE s.label LIKE '%Higgs%'")
    assert not u("SELECT p FROM search_2 s WHERE s.kind = 'detector_object'")


def test_the_join_key_does_not_count_as_a_filter():
    """`ON eo.entity_id = s.entity_id` is in EVERY such query. Counting it made
    the guard accept exactly what it exists to refuse."""
    from hepcoveragekg.eval.free_sql import unfiltered_set_join as u
    assert u("SELECT DISTINCT eo.paper_id FROM search_2 s "
             "JOIN entity_occurrence eo ON eo.entity_id = s.entity_id")


def test_inspecting_the_set_is_always_allowed():
    """Checking before committing is the move the guard wants to encourage.
    A table-name regex of `[a-z_]+` stopped at the digit, read `search_1` as
    `search_`, and refused precisely those queries."""
    from hepcoveragekg.eval.free_sql import unfiltered_set_join as u
    assert not u("SELECT COUNT(*) FROM search_1")
    assert not u("SELECT kind, COUNT(*) FROM search_1 GROUP BY kind")
    assert not u("SELECT COUNT(*) FROM entity")
