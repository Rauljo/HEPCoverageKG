"""Teaching the small critic where the large one draws the line.

Both judged the same 202 search steps on conceptB-200:
    8B    exact 11.1%  broader 20.5%  unrelated 68.4%  -> keeps 31.6%
    72B   exact 13.0%  broader 10.4%  unrelated 76.7%  -> keeps 23.4%

The 8B is the PERMISSIVE one, which is the opposite of the first guess. So the
examples are the 1,969 candidates it waved through and the 72B did not.
"""
from __future__ import annotations

import json

from hepcoveragekg.query import critic_examples as ce


def _run(tmp_path, name, reviews, question="which analyses use X?"):
    p = tmp_path / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"model": name}}) + "\n")
        fh.write(json.dumps({"qid": "q1", "question": question,
                             "answer": {"reviews": reviews}}) + "\n")
    return str(p)


def test_it_finds_what_the_small_critic_waved_through(tmp_path):
    small = _run(tmp_path, "small", [{"search_text": "X", "dropped": [
        {"id": "hepkg:a:one", "why": "clearly not"}]}])
    big = _run(tmp_path, "big", [{"search_text": "X", "dropped": [
        {"id": "hepkg:a:one", "why": "clearly not"},
        {"id": "hepkg:a:two", "why": "a different process entirely"}]}])
    got = ce.disagreements([small], [big])
    assert [g["entity_id"] for g in got] == ["hepkg:a:two"]
    assert got[0]["why"] == "a different process entirely"


def test_the_question_travels_with_the_example(tmp_path):
    """The critic judges a candidate against the QUESTION, not the search text.
    An example showing only the search text is unreadable -- the reason talks
    about something the reader cannot see."""
    small = _run(tmp_path, "small", [{"search_text": "ttbar", "dropped": []}],
                 question="which analyses estimate a nonprompt lepton background?")
    big = _run(tmp_path, "big", [{"search_text": "ttbar", "dropped": [
        {"id": "hepkg:bkg:x", "why": "not related to nonprompt leptons"}]}],
        question="which analyses estimate a nonprompt lepton background?")
    block = ce.render(ce.disagreements([small], [big]))
    assert "nonprompt lepton background?" in block
    assert "question:" in block


def test_agreement_produces_no_example(tmp_path):
    """Nothing to teach where both critics already agree."""
    rev = [{"search_text": "X", "dropped": [{"id": "hepkg:a:one", "why": "no"}]}]
    assert ce.disagreements([_run(tmp_path, "s", rev)], [_run(tmp_path, "b", rev)]) == []


def test_one_example_per_search_text(tmp_path):
    """A single verbose question must not supply the whole block."""
    small = _run(tmp_path, "small", [{"search_text": "X", "dropped": []}])
    big = _run(tmp_path, "big", [{"search_text": "X", "dropped": [
        {"id": f"hepkg:a:{i}", "why": "no"} for i in range(5)]}])
    assert len(ce.disagreements([small], [big])) == 1


def test_keep_examples_are_carried_too(tmp_path):
    """Teaching strictness can overshoot: a critic that drops everything looks
    fine on precision and scores zero on recall. `broader` exists because a
    wider version of what was asked is still worth keeping."""
    drops = [{"question": "q", "search_text": "X", "entity_id": "hepkg:a:1",
              "why": "unrelated"}]
    keeps = [{"question": "q", "search_text": "X", "entity_id": "hepkg:a:2"}]
    block = ce.render(drops, keeps)
    assert "unrelated" in block and "keep" in block
    assert "still worth counting" in block


def test_no_disagreements_renders_to_nothing():
    assert ce.render([]) == ""


def test_only_the_SMALL_critic_gets_the_examples(monkeypatch):
    """They are the 72B's own judgements. Showing them to the 72B would be
    showing it its own answers -- it measures nothing, and it would quietly
    contaminate the arm this is meant to be compared against."""
    from hepcoveragekg.query import critic, planner
    monkeypatch.setenv("CRITIC_EXAMPLES", "1")
    monkeypatch.setenv("CRITIC_EXAMPLES_SMALL", "eval/runs/conceptb-arms/*53972*.jsonl")
    monkeypatch.setenv("CRITIC_EXAMPLES_BIG", "eval/runs/conceptb-arms/*53970*.jsonl")
    planner._CRITIC_EXAMPLES = None
    assert planner._critic_prompt(False) is None, "the 72B uses the full PROMPT"
    small = planner._critic_prompt(True)
    assert small and len(small) > len(critic.SMALL_PROMPT)


def test_the_arm_is_off_without_its_variable(monkeypatch):
    from hepcoveragekg.query import critic, planner
    monkeypatch.delenv("CRITIC_EXAMPLES", raising=False)
    planner._CRITIC_EXAMPLES = None
    assert planner._critic_prompt(True) == critic.SMALL_PROMPT


def test_a_missing_run_file_does_not_kill_the_run(monkeypatch):
    """The examples are an optimisation. A run that cannot find them should
    judge without them, not crash forty minutes in."""
    from hepcoveragekg.query import critic, planner
    monkeypatch.setenv("CRITIC_EXAMPLES", "1")
    monkeypatch.setenv("CRITIC_EXAMPLES_SMALL", "/nonexistent/*.jsonl")
    monkeypatch.setenv("CRITIC_EXAMPLES_BIG", "/nonexistent/*.jsonl")
    planner._CRITIC_EXAMPLES = None
    assert planner._critic_prompt(True) == critic.SMALL_PROMPT
