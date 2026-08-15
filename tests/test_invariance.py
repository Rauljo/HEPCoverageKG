"""Paraphrase invariance: the check that needs no gold.

Tier B's truth was computed from the graph, so disagreeing with it is not proof
of being wrong. Two wordings of one question have the same true answer whatever
that answer is, so comparing the system to ITSELF sidesteps the gold entirely.
"""
from __future__ import annotations

import json

from hepcoveragekg.eval import invariance as I


def rec(qid, text, abstained=0, error=""):
    return {"qid": qid, "answer": {"text": text, "error": error},
            "scores": {"abstained": abstained}}


def test_pairs_are_read_from_the_relation_and_deduplicated(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(q) for q in [
        {"qid": "a", "relation": "paraphrase_of:b"},
        {"qid": "b", "relation": "paraphrase_of:a"},   # the same pair, reversed
        {"qid": "c", "relation": "paraphrase_of:d"},
        {"qid": "e"},                                   # no twin
    ]))
    assert I.paraphrase_pairs(p) == [("a", "b"), ("c", "d")]


def test_the_same_count_twice_is_agreement():
    out = I.measure([rec("a", "recorded in 38 papers"), rec("b", "38 analyses use it")],
                    [("a", "b")])
    assert out["count_agree"] == 1.0 and out["count_median_gap"] == 0


def test_different_counts_are_disagreement_and_the_gap_is_reported():
    """A flag says "wrong"; the gap says how wrong, which is the diagnosis."""
    out = I.measure([rec("a", "38 papers"), rec("b", "12 papers")], [("a", "b")])
    assert out["count_agree"] == 0.0 and out["count_median_gap"] == 26


def test_both_abstaining_is_excluded_not_counted_as_agreement():
    """A system that answers nothing is perfectly invariant and useless."""
    out = I.measure([rec("a", "no data", abstained=1), rec("b", "no data", abstained=1)],
                    [("a", "b")])
    assert out["both_abstained"] == 1 and out["usable"] == 0
    assert out["count_agree"] is None, "silence must not become a perfect score"


def test_one_side_abstaining_is_a_disagreement_of_its_own():
    out = I.measure([rec("a", "38 papers"), rec("b", "not in the graph", abstained=1)],
                    [("a", "b")])
    assert out["one_abstained"] == 1 and out["usable"] == 0


def test_an_error_counts_as_abstaining():
    out = I.measure([rec("a", "38 papers"), rec("b", "", error="TimeoutError")],
                    [("a", "b")])
    assert out["one_abstained"] == 1


def test_set_answers_are_compared_by_overlap_not_by_equality():
    """Two answers naming four of the same five papers agree far more than a
    same-or-not flag can express."""
    a = rec("a", "2308.02285, 2312.04450, 2004.01678")
    b = rec("b", "2308.02285, 2312.04450, 2011.07812")
    out = I.measure([a, b], [("a", "b")], shapes={"a": "set", "b": "set"})
    assert abs(out["set_jaccard"] - 0.5) < 1e-9      # 2 shared of 4 distinct


def test_a_missing_record_does_not_break_the_pair():
    out = I.measure([rec("a", "38 papers")], [("a", "b")])
    assert out["usable"] == 0 and out["pairs"] == 1


def test_the_report_says_what_invariance_cannot_prove():
    text = I.compare({"control": [rec("a", "38 papers"), rec("b", "38 papers")]},
                     [("a", "b")])
    assert "no gold consulted" in text
    assert "necessary but not sufficient" in text
