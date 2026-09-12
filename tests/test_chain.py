"""Sequential sub-goal chaining: N short runs with history carried forward."""
import types

from hepcoveragekg.eval import chain as C
from hepcoveragekg.eval.systems import Answer, Question

NS = types.SimpleNamespace
GOALS = ["find b-jet analyses", "find MET analyses", "intersect them"]


def _q(text="analyses using b-jets and MET"):
    return Question(qid="q1", text=text, source="t", split="dev", shape="set",
                    needs=[], difficulty="easy", truth=None, truth_source="sql",
                    provenance={})


class FakeInner:
    """Records what each leg was asked; returns a distinct answer per leg."""
    name = "fake"
    config = {"kind": "planner"}

    def __init__(self):
        self.asked = []

    def answer(self, q):
        self.asked.append(q.text)
        i = len(self.asked)
        return Answer(text=f"leg{i} answer", answered=True,
                      papers=[f"200{i}.0000{i}"], entity_ids=[f"e{i}"],
                      evidence_ids=[f"v{i}"], steps=[{"tool": "search", "round": 1}],
                      llm_calls=2, rounds=3, seconds=1.0)


def _chat(goals):
    def chat(messages, tools=None):
        body = "\n".join("%d. %s" % (i, g) for i, g in enumerate(goals, 1))
        return NS(choices=[NS(message=NS(content=body))])
    return chat


def test_one_leg_per_goal_and_the_original_question_is_always_present():
    inner = FakeInner()
    out = C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    assert len(inner.asked) == 3
    assert out.chain_legs == 3
    for asked in inner.asked:
        assert "analyses using b-jets and MET" in asked, "leg lost the overall question"


def test_history_reaches_later_legs():
    inner = FakeInner()
    C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    first, second, third = inner.asked
    assert "ALREADY ESTABLISHED" not in first
    assert "leg1 answer" in second, "leg 2 cannot see what leg 1 found"
    assert "leg1 answer" in third and "leg2 answer" in third
    assert "2001.00001" in second, "established papers must carry forward"


def test_only_the_last_leg_is_allowed_to_answer():
    inner = FakeInner()
    C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    first, _, third = inner.asked
    assert "Do not try to answer the overall question yet" in first
    assert "FINAL STEP" in third
    assert "Do not try to answer the overall question yet" not in third


def test_the_record_describes_the_chain_not_the_last_leg():
    inner = FakeInner()
    out = C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    assert out.papers == ["2001.00001", "2002.00002", "2003.00003"]
    assert out.entity_ids == ["e1", "e2", "e3"]
    assert out.evidence_ids == ["v1", "v2", "v3"]
    assert len(out.steps) == 3
    assert out.rounds == 9
    assert out.llm_calls == 7          # 3 legs x 2 + the decomposition call
    assert out.text == "leg3 answer"   # the leg that answered


def test_it_fails_open_to_the_plain_system():
    inner = FakeInner()
    def empty(messages, tools=None):
        return NS(choices=[NS(message=NS(content="nothing parseable"))])
    out = C.ChainedSubgoalSystem(inner, empty).answer(_q())
    assert len(inner.asked) == 1 and out.chain_legs == 0
    assert inner.asked[0] == "analyses using b-jets and MET"


def test_one_goal_is_a_single_leg_that_answers():
    inner = FakeInner()
    out = C.ChainedSubgoalSystem(inner, _chat(["just find the papers"])).answer(_q())
    assert out.chain_legs == 1 and len(inner.asked) == 1
    assert "FINAL STEP" in inner.asked[0]


def test_the_cli_unpacks_the_client_pair_correctly():
    """_client() returns (client, model). Unpacking it as one value made every
    decomposition raise "'tuple' object has no attribute 'chat'", the chain
    fail open on all 27 records, and chain_legs=0 is what caught it."""
    import inspect
    from hepcoveragekg import cli
    from hepcoveragekg.query import planner

    assert len(planner._client()) == 2, "_client must stay a (client, model) pair"
    src = inspect.getsource(cli)
    assert "client, model = _pl._client()" in src
    assert "_pl._client(), os.environ" not in src, "the tuple bug is back"


def test_the_final_selection_sees_every_leg_not_just_the_last(monkeypatch):
    """Constrained selection picks from session.known_entity_ids -- this leg's
    retrieval. Each leg is a fresh session, so without a chain-level pass the
    judge cannot select the papers earlier legs found: 4.3 named of 38.8."""
    monkeypatch.setenv("CRITIC_SELECTS", "1")
    monkeypatch.setenv("CONSTRAINED_IDS", "1")
    seen = {}

    def fake_select(conn, question, touched, named=(), chat=None):
        seen["touched"] = set(touched)
        return sorted(touched), {}

    import hepcoveragekg.eval.free_sql as fs
    monkeypatch.setattr(fs, "critic_selects_papers", fake_select)

    inner = FakeInner()
    inner._conn = object()
    out = C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    assert seen["touched"] == {"2001.00001", "2002.00002", "2003.00003"}, \
        "the judge must see all three legs' papers"
    assert out.constrained_ids == ["2001.00001", "2002.00002", "2003.00003"]
    assert out.constrained_mode == "chain-critic"
    assert "Papers: 2001.00001, 2002.00002, 2003.00003" in out.text


def test_the_chain_survives_a_failing_selection(monkeypatch):
    monkeypatch.setenv("CRITIC_SELECTS", "1")
    monkeypatch.setenv("CONSTRAINED_IDS", "1")

    def boom(conn, question, touched, named=(), chat=None):
        raise RuntimeError("judge is down")

    import hepcoveragekg.eval.free_sql as fs
    monkeypatch.setattr(fs, "critic_selects_papers", boom)
    inner = FakeInner()
    inner._conn = object()
    out = C.ChainedSubgoalSystem(inner, _chat(GOALS)).answer(_q())
    assert out.chain_legs == 3 and out.text == "leg3 answer"


def test_free_sql_renders_one_objective_at_a_time(monkeypatch):
    """SUBGOAL_SEQUENTIAL lived only in the typed planner's graph, so free-SQL
    could not run the arm at all -- half of the comparison the chapter needs."""
    monkeypatch.setenv("SUBGOAL_SEQUENTIAL", "1")
    from hepcoveragekg.query import subgoals as sg

    first = sg.render_sequential(GOALS, 0, [], 6)
    assert GOALS[0] in first and GOALS[1] not in first
    import inspect
    from hepcoveragekg.eval import free_sql
    src = inspect.getsource(free_sql)
    assert "render_sequential" in src, "free-SQL must render the sequential block"
    assert "subgoal_advances=advances" in src, "free-SQL must record the pointer"


def test_the_chain_is_not_bolted_to_the_planner_branch():
    """It began inside `elif args.system == 'planner'`, so free-SQL could never
    be chained. The chain only needs .answer(Question) -> Answer."""
    import inspect
    from hepcoveragekg import cli
    src = inspect.getsource(cli)
    i_chain = src.index('SUBGOAL_CHAIN', src.index('def '))
    i_else = src.index("make_system = None")
    assert i_chain > i_else, "the chain must wrap whichever system was built"
