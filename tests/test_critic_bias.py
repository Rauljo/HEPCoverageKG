"""The ordering arms (D-060).

What these pin down is the BOOKKEEPING, because the measurement is worthless if
a shuffled arm's verdicts do not line up with the ranked arm's on the same
candidate. A flip rate computed over misaligned pairs would look like bias and
would be an indexing bug.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from hepcoveragekg.eval import critic_bias as B
from hepcoveragekg.query import critic as C


def _review(qid, rungs):
    return C.Review(question="q", search_text="s",
                    verdicts=[C.Verdict(f"e{i}", rung) for i, rung in enumerate(rungs)])


def test_two_identical_arms_show_no_bias():
    a = B.ArmResult("a", {"q1": _review("q1", [C.EXACT, C.UNRELATED, C.BROADER])})
    b = B.ArmResult("b", {"q1": _review("q1", [C.EXACT, C.UNRELATED, C.BROADER])})
    stats = B.compare(a, b)
    assert stats["rung_flip"] == 0.0 and stats["kept_flip"] == 0.0


def test_a_rung_move_that_does_not_change_the_set_is_reported_separately():
    """`exact` -> `broader` changes the diagnosis; `broader` -> `unrelated`
    changes what gets counted. One number cannot carry both."""
    a = B.ArmResult("a", {"q1": _review("q1", [C.EXACT, C.BROADER])})
    b = B.ArmResult("b", {"q1": _review("q1", [C.BROADER, C.UNRELATED])})
    stats = B.compare(a, b)
    assert stats["rung_flip"] == 1.0, "both moved rung"
    assert stats["kept_flip"] == 0.5, "only one left the set"


def test_comparison_is_by_candidate_not_by_position():
    """The bug this exists to catch: comparing arm A's slot 0 against arm B's
    slot 0 when the arms judged in different orders."""
    a = B.ArmResult("a", {"q1": _review("q1", [C.EXACT, C.UNRELATED])})
    flipped = C.Review(question="q", search_text="s", verdicts=[
        C.Verdict("e1", C.UNRELATED), C.Verdict("e0", C.EXACT)])
    b = B.ArmResult("b", {"q1": flipped})
    assert B.compare(a, b)["rung_flip"] == 0.0, "same verdicts, listed differently"


def test_keep_rate_by_rank_finds_a_slope():
    """A critic doing its job keeps the head and drops the tail."""
    arm = B.ArmResult("a", {"q1": _review("q1", [C.EXACT] * 8 + [C.UNRELATED] * 8)})
    assert B.keep_rate_by_rank(arm, buckets=2) == [1.0, 0.0]


def test_a_flat_keep_rate_is_visible():
    """Which is the failure an overall keep-rate hides: 50% kept looks the same
    whether the dropped half was the tail or was scattered."""
    arm = B.ArmResult("a", {"q1": _review("q1", [C.EXACT, C.UNRELATED] * 8)})
    assert B.keep_rate_by_rank(arm, buckets=2) == [0.5, 0.5]


def test_cases_come_from_real_search_texts(tmp_path):
    """One case per distinct search text, and empty searches are not cases."""
    run = tmp_path / "run.jsonl"
    run.write_text("\n".join(json.dumps(r) for r in [
        {"_meta": {}},
        {"qid": "a", "question": "how many use Pythia 8?", "answer": {"steps": [
            {"tool": "search", "args": {"text": "Pythia"}, "rows": 60},
            {"tool": "count", "args": {}, "rows": 1}]}},
        {"qid": "b", "question": "another", "answer": {"steps": [
            {"tool": "search", "args": {"text": "Pythia"}, "rows": 60}]}},
        {"qid": "c", "question": "third", "answer": {"steps": [
            {"tool": "search", "args": {"text": "nothing here"}, "rows": 0}]}},
        {"qid": "d", "question": "fourth", "answer": {"steps": [
            {"tool": "search", "args": {"text": "top squark"}, "rows": 60}]}},
    ]))
    cases = B.load_cases(run)
    assert [c.search_text for c in cases] == ["Pythia", "top squark"]
    assert cases[0].question == "how many use Pythia 8?"


def test_no_overlap_is_not_a_crash():
    assert B.compare(B.ArmResult("a"), B.ArmResult("b")) == {"n": 0}


def test_the_report_names_what_each_pair_measures():
    arms = {n: B.ArmResult(n, {"q1": _review("q1", [C.EXACT, C.UNRELATED])})
            for n in ("ranked", "shuffleA", "shuffleB", "chunk60")}
    text = B.report(arms)
    assert "pure position bias" in text
    assert "how far it restates the retriever" in text
    assert "should slope down" in text
