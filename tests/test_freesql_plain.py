"""FREESQL_PLAIN_ANSWER removes the scoring instruction from the free-SQL control."""
import sqlite3
from hepcoveragekg.eval import free_sql as F


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
      CREATE TABLE paper(arxiv_id TEXT, category TEXT); CREATE TABLE entity(entity_id TEXT, kind TEXT, label TEXT);
      CREATE TABLE entity_occurrence(paper_id TEXT, entity_id TEXT, kind TEXT, label TEXT, bundle_id TEXT);
      CREATE TABLE assertion(paper_id TEXT, predicate TEXT, subject_id TEXT, object_id TEXT);""")
    return c


def test_plain_tool_has_no_scoring_words():
    t = F.plain_answer_tool()
    assert "scored" not in t["function"]["description"]
    assert "scored" in F.ANSWER_TOOL["function"]["description"], "the default still carries it"
    assert "optionally" in t["function"]["parameters"]["properties"]["papers"]["description"]


def test_switch_is_recorded_and_default_off(monkeypatch):
    monkeypatch.delenv("FREESQL_PLAIN_ANSWER", raising=False)
    assert F.FreeSQLSystem(_conn(), None).config["plain_answer"] is False
    monkeypatch.setenv("FREESQL_PLAIN_ANSWER", "1")
    assert F.FreeSQLSystem(_conn(), None).config["plain_answer"] is True
