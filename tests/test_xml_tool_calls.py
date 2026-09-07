"""The model calls `answer` as an XML tag, and we were not listening (D-116).

Measured 2026-09-07 over three 164-question runs: of the 150 answers per run
that never reached `answer()`, 115 -- 77% -- end in

    <answer text="18 analyses use b-tagged jets ..." papers_from="the facets
            result" reason="answered" answerable="true"/>

`_TEXT_TOOL_CALL` wants `<tool_call>{json}</tool_call>` and matches none of it,
so the whole message fell through to `after_plan` and `last_content` became the
answer: raw, uncited, no `answer_papers`, no gate.

One missed syntax is the root of four separate findings -- the gate firing
twice in 164 questions, `cited` empty in 59 of 60 answers, the answer-critic
reaching 9% of its chances, and the citation mechanism reading as dead.
"""
import json

from hepcoveragekg.query import planner as P

TOOLS = {"answer", "search", "facets", "papers_of", "count"}


def _call(text, known=TOOLS):
    return P._recover_tool_calls(text, known)


def _args(call):
    return json.loads(call.function.arguments)


def test_the_xml_answer_call_is_recovered():
    """Verbatim from run 54242, gabriel-gf-01-condition."""
    got = _call('Some reasoning first.\n<answer text="18 analyses (papers) use '
                'b-tagged jets in their event selection." '
                'papers_from="set_1_kept" reason="answered" '
                'answerable="true"/>')
    assert len(got) == 1
    assert got[0].function.name == "answer"
    a = _args(got[0])
    assert a["papers_from"] == "set_1_kept"
    assert a["reason"] == "answered"
    assert a["answerable"] is True, "true/false become booleans, not strings"


def test_a_tag_that_is_not_a_tool_is_not_a_call():
    """Without the name check, any <b ...> or <sub ...> in prose is a call."""
    assert _call('The <sub scale="1">T</sub> subscript.') == []
    assert _call('<b class="x">bold</b>') == []


def test_the_json_form_still_wins():
    """A message holding both is the model correcting itself."""
    got = _call('<tool_call>{"name": "search", "arguments": {"text": "b-jet"}}'
                '</tool_call>\n<answer text="never mind" reason="answered"/>')
    assert [c.function.name for c in got] == ["search"]


def test_no_known_tools_means_no_xml_recovery():
    """The XML form is only a call because the tag names a tool."""
    assert _call('<answer text="x" reason="answered"/>', known=None) == []


def test_a_tag_with_no_attributes_is_not_a_call():
    assert _call("<answer/>") == []
    assert _call("<answer>") == []


def test_other_tools_are_recovered_too():
    got = _call('<facets field="objects" value="BJet"/>')
    assert len(got) == 1 and got[0].function.name == "facets"
    assert _args(got[0]) == {"field": "objects", "value": "BJet"}


def test_several_calls_in_one_message():
    got = _call('<search text="b-jet"/> then <papers_of entity_set="set_1"/>')
    assert [c.function.name for c in got] == ["search", "papers_of"]


def test_the_json_form_is_unchanged_by_the_addition():
    got = _call('<tool_call>{"name": "count", "arguments": {"predicate": "p"}}'
                '</tool_call>')
    assert len(got) == 1 and got[0].function.name == "count"
    assert _args(got[0]) == {"predicate": "p"}


def test_truncated_json_recovers_nothing_rather_than_guessing():
    assert _call('<tool_call>{"name": "search", "argum') == []


def test_recovered_calls_are_shaped_like_sdk_calls():
    """The loop must need no special case for them."""
    c = _call('<answer text="x" reason="answered"/>')[0]
    assert c.type == "function" and c.id.startswith("recovered-")
    assert isinstance(c.function.arguments, str)


# --------------------------------------------------------------------------
# harvesting by argument name, and recording WHICH notation was used
# --------------------------------------------------------------------------

OBSERVED = {
    "QwQ attrs":      ('<answer text="x" papers_from="set_1" reason="answered" '
                       'answerable="true"/>', "attrs"),
    "QwQ wrapper":    ('<answer>{"text": "x", "papers_from": "the facets '
                       'response", "reason": "answered"}</answer>', "json"),
    "qwen arg-tag":   ('<answerable>{"text": "18 analyses", "reason": '
                       '"answered"}</answerable>', "json"),
    "qwen per-arg":   ('<papers_from>set_1</papers_from>\n'
                       '<reason>answered</reason>', "tag_per_arg"),
    "plain prose":    ("The analyses are 2004.14060 and 2006.05880.", "prose"),
    "nothing":        ("I could not find anything.", "none"),
}


def test_every_observed_notation_is_classified():
    """A new form must arrive as a number in the record, not as a week of odd
    results. Four forms, two models, and the fifth will be someone else's."""
    for label, (text, want) in OBSERVED.items():
        assert P.answer_syntax(text) == want, label


def test_arguments_are_harvested_from_every_structured_form():
    for label, (text, kind) in OBSERVED.items():
        got = P.harvest_answer_args(text)
        if kind in ("prose", "none"):
            assert got == {}, f"{label}: prose has no arguments to harvest"
        else:
            assert got, f"{label}: nothing harvested"
            assert "reason" in got, label


def test_a_json_wrapper_is_not_read_as_an_argument_value():
    """`<answerable>{...}</answerable>` read as the VALUE of `answerable`
    yields False -- an abstention the model never made."""
    got = P.harvest_answer_args(
        '<answerable>{"text": "18 analyses", "reason": "answered"}</answerable>')
    assert got.get("answerable") is not False
    assert got["text"] == "18 analyses"


def test_a_real_abstention_still_reads_as_one():
    got = P.harvest_answer_args(
        '<answer text="none" answerable="false" reason="not_in_graph"/>')
    assert got["answerable"] is False and got["reason"] == "not_in_graph"


# --------------------------------------------------------------------------
# the prompt arm
# --------------------------------------------------------------------------

def test_v3_by_default_still_tells_the_model_not_to_write_ids():
    """The instruction that made this necessary. Kept as the control so the
    arm measures something, and because silently rewording v3 is D-062."""
    spec = [t for t in P.tools_for(contract="v3") if t["name"] == "answer"][0]
    assert "rather than writing arXiv ids" in \
        spec["parameters"]["properties"]["text"]["description"]


def test_name_ids_asks_for_the_thing_the_scorer_reads():
    spec = [t for t in P.tools_for(contract="v3", name_ids=True)
            if t["name"] == "answer"][0]
    desc = spec["parameters"]["properties"]["text"]["description"]
    assert "arXiv ids" in desc and "rather than writing" not in desc
    assert "2004.14060" in desc, "an example, because the format is the point"


def test_name_ids_leaves_v1_frozen():
    """v1 is the control every earlier result is measured against."""
    assert P.tools_for(False) == P.tools_for(False)
    v1 = [t for t in P.tools_for(False) if t["name"] == "answer"][0]
    assert v1["parameters"]["properties"]["text"]["description"] == \
        "the answer, citing what was retrieved"
