"""Merging the typed planner and the free-SQL control, which fail differently.

D-079: three questions tied, two clear wins each way, one mutual failure. Two
instruments with different blind spots, not two attempts at the same thing.

Measured on the supervisor's gold, union beats both inputs on all three models
and intersection is worse than either -- the same conclusion
`reader.merge_gathered` reached for sentences, now for papers.
"""
from __future__ import annotations

from hepcoveragekg.eval import merge


def _rec(qid, text="", papers=(), cited="", seconds=1.0, calls=1):
    return {"qid": qid, "repeat": 0,
            "answer": {"text": text, "papers": list(papers), "cited": cited,
                       "seconds": seconds, "llm_calls": calls}}


def test_union_keeps_what_either_side_found():
    a = _rec("q", text="2106.01676 and 2001.06899")
    b = _rec("q", text="2001.06899 and 2004.14060")
    got = merge.merge_answers(a, b, merge.UNION)
    assert got.papers == ["2001.06899", "2004.14060", "2106.01676"]


def test_intersection_keeps_only_agreement():
    a = _rec("q", text="2106.01676 and 2001.06899")
    b = _rec("q", text="2001.06899 and 2004.14060")
    got = merge.merge_answers(a, b, merge.INTERSECT)
    assert got.papers == ["2001.06899"]


def test_a_retrieval_footprint_is_not_merged():
    """D-080's rule, applied here too. Merging footprints would union two piles
    of everything the systems happened to touch, which is not an answer from
    either of them."""
    footprint = _rec("q", text="", papers=["1111.1111", "2222.2222"])   # no citation
    real = _rec("q", text="3333.3333")
    got = merge.merge_answers(footprint, real, merge.UNION)
    assert got.papers == ["3333.3333"]


def test_a_cited_set_IS_merged():
    """v3 cites a set instead of retyping ids, and free-SQL's answer tool does
    the same. Refusing citations would discard both systems' intended output."""
    cited = _rec("q", text="the papers in set_1", papers=["1111.1111"],
                 cited="papers=set_1")
    other = _rec("q", text="2222.2222")
    got = merge.merge_answers(cited, other, merge.UNION)
    assert got.papers == ["1111.1111", "2222.2222"]


def test_the_merge_carries_BOTH_costs():
    """A merge that doubles the bill has to show it."""
    a = _rec("q", text="1111.1111", seconds=10.0, calls=3)
    b = _rec("q", text="2222.2222", seconds=5.0, calls=2)
    got = merge.merge_answers(a, b, merge.UNION)
    assert got.seconds == 15.0 and got.llm_calls == 5


def test_the_judge_can_only_remove():
    """It filters the union; it cannot invent a paper neither side found."""
    a = _rec("q", text="1111.1111 2222.2222")
    b = _rec("q", text="3333.3333")
    got = merge.merge_answers(a, b, merge.JUDGED,
                              judge=lambda q, ps: [p for p in ps if p != "2222.2222"],
                              question="which analyses?")
    assert got.papers == ["1111.1111", "3333.3333"]


def test_an_empty_merge_is_not_an_answer():
    got = merge.merge_answers(_rec("q"), _rec("q"), merge.UNION)
    assert not got.answered and got.papers == []
