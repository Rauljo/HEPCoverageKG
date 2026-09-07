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
