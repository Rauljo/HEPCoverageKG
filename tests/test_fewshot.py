"""Few-shot exemplars, and the contamination rule that governs them."""
from __future__ import annotations

import json

import pytest

from hepcoveragekg.query import fewshot


def _run(tmp_path, records):
    p = tmp_path / "run.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"model": "m"}}) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _rec(qid, score, text="Two analyses: 2106.01676, 2001.06899.", steps=None):
    return {"qid": qid, "question": f"question for {qid}",
            "scores": {"judged_f1": score},
            "answer": {"text": text,
                       "steps": steps or [{"tool": "search",
                                           "args": {"text": "b-tagged jet"},
                                           "rows": 12}]}}


def test_an_exemplar_from_a_scored_question_is_REFUSED(tmp_path):
    """The whole reason this file is careful. Using sol's gf-02 answer as an
    exemplar and then scoring on gf-02 leaks the gold, and the leak is invisible
    in the results -- it just makes the number better."""
    pool = fewshot.candidates(_run(tmp_path, [_rec("gabriel-gf-02", 0.9)]))
    with pytest.raises(fewshot.Contaminated, match="gabriel-gf-02"):
        fewshot.select(pool, evaluation_qids={"gabriel-gf-02"})


def test_it_refuses_rather_than_silently_dropping(tmp_path):
    """Filtering the bad one out would let the same mistake through next time in
    a form that does not raise."""
    pool = fewshot.candidates(_run(tmp_path, [_rec("gabriel-gf-02", 0.9),
                                              _rec("gen-concept-1", 0.8)]))
    with pytest.raises(fewshot.Contaminated):
        fewshot.select(pool, evaluation_qids={"gabriel-gf-02"})


def test_a_disjoint_pool_is_accepted(tmp_path):
    pool = fewshot.candidates(_run(tmp_path, [_rec("gen-concept-1", 0.9),
                                              _rec("gen-paper-2", 0.7)]))
    got = fewshot.select(pool, evaluation_qids={"gabriel-gf-02"}, limit=2)
    assert [g["qid"] for g in got] == ["gen-concept-1", "gen-paper-2"]


def test_only_good_answers_become_exemplars(tmp_path):
    """An exemplar is a plan worth imitating. A run that scored 0.1 is not."""
    pool = fewshot.candidates(_run(tmp_path, [_rec("a", 0.9), _rec("b", 0.1)]),
                              min_score=0.6)
    assert [p["qid"] for p in pool] == ["a"]


def test_an_empty_answer_is_never_an_exemplar(tmp_path):
    pool = fewshot.candidates(_run(tmp_path, [_rec("a", 0.9, text="")]))
    assert pool == []


def test_the_two_variants_differ_in_exactly_one_thing(tmp_path):
    """Answer-only teaches what an answer looks like; with-plan teaches which
    tools produced it. Which transfers is an open question, so both exist and
    they are measured separately."""
    pool = fewshot.candidates(_run(tmp_path, [_rec("gen-1", 0.9)]))
    chosen = fewshot.select(pool, set())
    answers_only = fewshot.render(chosen)
    with_plan = fewshot.render(chosen, with_plan=True)

    assert "search(" not in answers_only, "answer-only must not leak the plan"
    assert "search(" in with_plan and "12 rows" in with_plan
    assert len(with_plan) > len(answers_only), "the plan costs tokens"
    for block in (answers_only, with_plan):
        assert "not the question you" in block, "both must warn against copying"


def test_no_examples_renders_to_nothing(tmp_path):
    assert fewshot.render([]) == ""
