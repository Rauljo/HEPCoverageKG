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
