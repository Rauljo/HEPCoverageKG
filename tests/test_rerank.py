"""Ranking instead of filtering (D-113).

Measured on Gabriel's 253 verdicts: same judge, same calls, same evidence, and
only the use of the output changed --

    keep everything   F1 0.591        drop by verdict  F1 0.445
    rank, top-16      F1 0.654

The tests below pin the three properties that result depends on: membership is
never changed, the order is by grade, and an ordering from a judge that will
not use the scale is refused rather than applied.
"""
import pytest

from hepcoveragekg.query import answer_critic as AC
from hepcoveragekg.query import critic as C


# --------------------------------------------------------------------------
# the free half: the rung the critic already computes
# --------------------------------------------------------------------------

def _review(*pairs):
    return C.Review(question="q", search_text="t",
                    verdicts=[C.Verdict(i, r) for i, r in pairs])


def test_ranked_ids_puts_exact_before_broader():
    r = _review(("b1", C.BROADER), ("e1", C.EXACT), ("b2", C.BROADER),
                ("e2", C.EXACT))
    assert r.ranked_ids == ["e1", "e2", "b1", "b2"]


def test_ranking_never_changes_membership():
    """The whole point: a better order, not a smaller set."""
    r = _review(("b1", C.BROADER), ("e1", C.EXACT), ("u1", C.UNRELATED))
    assert set(r.ranked_ids) == set(r.kept_ids)
    assert "u1" not in r.ranked_ids


def test_ranking_is_stable_within_a_rung():
    """Ties keep retrieval order, which is itself a weak ranking."""
    r = _review(("e1", C.EXACT), ("e2", C.EXACT), ("e3", C.EXACT))
    assert r.ranked_ids == ["e1", "e2", "e3"]


def test_ids_at_reads_one_rung():
    r = _review(("e1", C.EXACT), ("b1", C.BROADER), ("u1", C.UNRELATED))
    assert r.ids_at(C.EXACT) == ["e1"]
    assert r.ids_at(C.UNRELATED) == ["u1"]


# --------------------------------------------------------------------------
# the graded ranker
# --------------------------------------------------------------------------

class _Grader:
    def __init__(self, grades, raw=None):
        self.grades = grades
        self.raw = raw
        self.calls = 0

    def __call__(self, messages):
        import json as _json
        import re as _re
        self.calls += 1
        body = messages[-1]["content"]
        seen = _re.findall(r"\b\d{4}\.\d{4,5}\b", body)
        text = self.raw if self.raw is not None else _json.dumps(
            {"grades": [{"paper": p, "grade": self.grades[p], "why": "t"}
                        for p in dict.fromkeys(seen) if p in self.grades]})
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": text})()})()]})()


def _evidence(*papers):
    return {p: ({"label"}, ["a quote"]) for p in papers}


def test_rank_papers_orders_by_grade():
    ev = _evidence("2001.00001", "2002.00002", "2003.00003")
    g = _Grader({"2001.00001": 0, "2002.00002": 3, "2003.00003": 2})
    r = AC.rank_papers(g, "which?", ev)
    assert r.order == ["2002.00002", "2003.00003", "2001.00001"]


def test_rank_papers_never_drops_a_paper():
    """A set question's answer IS the set; a ranker that removes one is a
    filter, and filtering measured worse than doing nothing."""
    ev = _evidence("2001.00001", "2002.00002", "2003.00003")
    g = _Grader({"2001.00001": 0, "2002.00002": 0, "2003.00003": 0})
    r = AC.rank_papers(g, "which?", ev)
    assert set(r.order) == set(ev)


def test_an_ungraded_paper_goes_to_the_back_not_away():
    """A judge must not be able to delete a candidate by staying silent."""
    ev = _evidence("2001.00001", "2002.00002")
    g = _Grader({"2001.00001": 2})            # says nothing about the second
    r = AC.rank_papers(g, "which?", ev)
    assert r.order == ["2001.00001", "2002.00002"]
    assert r.ungraded == ["2002.00002"]


def test_a_broken_judge_leaves_every_paper_present():
    def boom(messages):
        raise RuntimeError("503")
    ev = _evidence("2001.00001", "2002.00002")
    r = AC.rank_papers(boom, "which?", ev)
    assert set(r.order) == set(ev) and r.errors == 1


def test_unparseable_output_does_not_invent_an_order():
    ev = _evidence("2001.00001", "2002.00002")
    r = AC.rank_papers(_Grader({}, raw="I think they are all fine, really."),
                       "which?", ev)
    assert set(r.order) == set(ev) and not r.grades


# --------------------------------------------------------------------------
# the diagnostic that llama-3.1-8b fails
# --------------------------------------------------------------------------

def test_a_judge_that_refuses_the_middle_is_unusable():
    """llama-3.1-8b put 14 of 253 papers in grades 1-2 and scored 0.575 --
    worse than keeping everything (0.591). It is binary in a grader's prompt."""
    grades = {f"20{i:02d}.0000{i % 10}": (3 if i < 5 else 0) for i in range(50)}
    r = AC.Ranking(question="q", grades=grades)
    assert not r.usable
    assert r.spread[2] == 0 and r.spread[1] == 0


def test_a_judge_that_uses_the_scale_is_usable():
    """gpt-4.1-mini's real spread over Gabriel's 253: 38/31/64/120."""
    grades = {}
    for g, n in ((3, 38), (2, 31), (1, 64), (0, 120)):
        for i in range(n):
            grades[f"g{g}-{i}"] = g
    r = AC.Ranking(question="q", grades=grades)
    assert r.usable
    assert r.spread == {3: 38, 2: 31, 1: 64, 0: 120}


def test_no_grades_at_all_is_not_usable():
    assert not AC.Ranking(question="q").usable


def test_the_spread_travels_in_the_record():
    r = AC.Ranking(question="q", grades={"a": 3, "b": 1}, calls=1)
    d = r.to_dict()
    assert d["spread"] == {"3": 1, "2": 0, "1": 1, "0": 0}
    assert d["usable"] is True and d["graded"] == 2


# --------------------------------------------------------------------------
# the wiring: papers_of is where the truncation happens
# --------------------------------------------------------------------------

def test_the_paper_ranker_is_off_unless_both_flags_are_on():
    """The rung reordering is free and always applies under --rerank; the
    graded pass costs a call per `papers_of` and is the thing being measured."""
    from hepcoveragekg.query import planner

    assert planner._paper_ranker(None, None, False, False) is None
    assert planner._paper_ranker(None, None, True, False) is None
    assert planner._paper_ranker(None, None, False, True) is None


def test_an_unusable_ranking_is_not_applied():
    """Acting on a binary judge's order would be a known regression (D-113)."""
    from hepcoveragekg.query import templates

    order = ["2003.00003", "2001.00001", "2002.00002"]
    result = templates.QueryResult(
        shape="papers_of",
        rows=[{"paper_id": p} for p in
              ["2001.00001", "2002.00002", "2003.00003"]])
    before = [r["paper_id"] for r in result.rows]
    ranking = AC.Ranking(question="q",
                         grades={p: (3 if p == order[0] else 0) for p in before},
                         order=order)
    assert not ranking.usable
    # the guard in `_paper_ranker`: usable is False, so rows are left alone
    if ranking.usable:
        rank_of = {p: n for n, p in enumerate(ranking.order)}
        result.rows.sort(key=lambda r: rank_of.get(r["paper_id"], 10 ** 6))
    assert [r["paper_id"] for r in result.rows] == before


def test_a_usable_ranking_reorders_without_losing_rows():
    from hepcoveragekg.query import templates

    papers = ["2001.00001", "2002.00002", "2003.00003", "2004.00004"]
    result = templates.QueryResult(shape="papers_of",
                                   rows=[{"paper_id": p} for p in papers])
    ranking = AC.Ranking(
        question="q",
        grades={"2001.00001": 0, "2002.00002": 3, "2003.00003": 1,
                "2004.00004": 2},
        order=["2002.00002", "2004.00004", "2003.00003", "2001.00001"])
    assert ranking.usable
    rank_of = {p: n for n, p in enumerate(ranking.order)}
    result.rows.sort(key=lambda r: rank_of.get(str(r.get("paper_id")), 10 ** 6))
    assert [r["paper_id"] for r in result.rows] == ranking.order
    assert len(result.rows) == len(papers), "ranking must not remove a row"
