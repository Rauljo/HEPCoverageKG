"""One worked call per tool, mined from runs that answered correctly."""
from __future__ import annotations

from hepcoveragekg.query import planner, tool_examples


def test_every_example_is_for_a_tool_that_exists():
    """A worked call for a tool nobody offers is instructions for a dead end."""
    names = {t["name"] for t in planner.tools_for(contract="v3")}
    names |= {t["name"] for t in planner.tools_for(contract="v2")}
    for tool in tool_examples.EXAMPLES:
        assert tool in names, f"{tool} is documented but never offered"


def test_the_block_covers_the_tools_actually_offered():
    block = tool_examples.render(planner.tools_for(contract="v3"))
    for tool in ("search", "facets", "subjects_of", "contents_of", "count"):
        assert f"{tool}(" in block


def test_it_names_the_two_measured_failures():
    """gf-08 filtered on the wrong kind; Qwen never reaches for facets, which
    answers gf-01 in one call."""
    block = tool_examples.render(planner.tools_for(contract="v3"))
    assert "several kinds" in block, "the wrong-kind trap must be named"
    assert "ONE call" in block, "the facets shortcut must be named"


def test_an_empty_tool_list_renders_to_nothing():
    assert tool_examples.render([]) == ""


def test_it_carries_no_answers_only_calls():
    """The reason this needs no contamination check: there is no gold in it."""
    block = tool_examples.render(planner.tools_for(contract="v3"))
    import re
    assert not re.search(r"\b\d{4}\.\d{4,5}\b", block.replace("2006.05880", "")), \
        "the only arXiv id may be the contents_of example's own argument"
