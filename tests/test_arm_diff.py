"""Tests for the per-paper arm diff.

The point of this module is to see a fix a noisy mean cannot, so the tests
are about direction correctness and about not overclaiming.
"""
from __future__ import annotations

import json

import pytest

from hepcoveragekg.eval import arm_diff as AD


def _run(tmp_path, name, per_repeat):
    """per_repeat: list of (qid, [papers named in prose])."""
    p = tmp_path / f"{name}.jsonl"
    lines = ['{"_meta": {}}']
    for qid, papers in per_repeat:
        lines.append(json.dumps({
            "qid": qid, "scores": {},
            "answer": {"text": "See " + ", ".join(papers) if papers else "None found.",
                       "steps": [], "entity_ids": [], "papers": [], "cited": ""},
        }))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


QS = {"q1": {"truth": {"papers": ["1111.1111", "2222.2222"],
                       "universe": ["1111.1111", "2222.2222", "3333.3333"]}}}


def test_naming_a_missed_gold_paper_is_FIXED(tmp_path):
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "2222.2222"])] * 3)
    fl = AD.flips(ctrl, arm, QS)
    assert [(f.paper, f.direction) for f in fl] == [("2222.2222", "FIXED")]


def test_dropping_a_judged_negative_is_also_FIXED(tmp_path):
    """3333.3333 is in the universe but NOT gold -- Gabriel judged it wrong.
    An arm that stops naming it has improved precision."""
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111", "3333.3333"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111"])] * 3)
    fl = AD.flips(ctrl, arm, QS)
    assert [(f.paper, f.direction) for f in fl] == [("3333.3333", "FIXED")]


def test_newly_naming_a_judged_negative_is_BROKE(tmp_path):
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "3333.3333"])] * 3)
    fl = AD.flips(ctrl, arm, QS)
    assert [(f.paper, f.direction) for f in fl] == [("3333.3333", "BROKE")]


def test_a_wobble_on_both_sides_is_filtered_as_noise(tmp_path):
    """Run-to-run divergence is infrastructural (temperature is 0 and identical
    runs still differ on the first tool call). When BOTH sides are already
    inconsistent, a change between them carries no information -- unlike the
    `unlocked` case, where the control is unanimous and the arm breaks new
    ground."""
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111", "2222.2222"]),
                                ("q1", ["1111.1111"]),
                                ("q1", ["1111.1111"])])
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "2222.2222"]),
                               ("q1", ["1111.1111", "2222.2222"]),
                               ("q1", ["1111.1111"])])
    fl = [f for f in AD.flips(ctrl, arm, QS) if f.paper == "2222.2222"]
    assert fl and not fl[0].stable and not fl[0].unlocked and not fl[0].lost
    assert "0 fixed, 0 broke" in AD.report(ctrl, arm, QS)   # filtered


def test_papers_outside_the_judged_universe_are_ignored(tmp_path):
    """Naming an unjudged paper is not evidence either way -- D-072."""
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "9999.9999"])] * 3)
    assert AD.flips(ctrl, arm, QS) == []


def test_report_separates_predicted_from_unpredicted(tmp_path):
    """With 253 verdicts some flip by chance. A predicted hit is evidence; an
    unpredicted one is a hypothesis for the next run."""
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "2222.2222"])] * 3)
    out = AD.report(ctrl, arm, QS, predicted=["q1"])
    assert "PREDICTED targets: 1 fixed" in out
    assert "*PREDICTED*" in out
    out2 = AD.report(ctrl, arm, QS, predicted=["q_other"])
    assert "unpredicted:       1 fixed" in out2


def test_trace_diff_surfaces_the_first_call(tmp_path):
    p1 = tmp_path / "c.jsonl"; p2 = tmp_path / "a.jsonl"
    def w(p, args):
        p.write_text('{"_meta": {}}\n' + json.dumps({
            "qid": "q1", "scores": {},
            "answer": {"text": "x", "entity_ids": [], "papers": [], "cited": "",
                       "steps": [{"tool": "facets", "args": args}]}}) + "\n", encoding="utf-8")
    w(p1, {"values": ["MET"]}); w(p2, {"values": ["MET"], "category": "search"})
    d = AD.trace_diff(p1, p2, "q1")
    assert d["control"]["first_calls"] != d["arm"]["first_calls"]
    assert "category" in d["arm"]["first_calls"][0]


def test_a_capability_the_control_never_had_is_not_filtered_as_noise(tmp_path):
    """gf-05, 2026-09-04: control scored 0.000 in EVERY repeat of every run ever
    made; chATLAS named 11 of 16 gold papers on one repeat in three. Requiring
    unanimity on both sides hid the most informative event in the experiment."""
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111", "2222.2222"]),
                               ("q1", ["1111.1111"]),
                               ("q1", ["1111.1111"])])
    fl = [f for f in AD.flips(ctrl, arm, QS) if f.paper == "2222.2222"]
    assert fl and fl[0].unlocked and not fl[0].stable
    out = AD.report(ctrl, arm, QS)
    assert "1 fixed" in out and "UNLOCKED" in out


def test_losing_a_paper_the_control_always_had_is_reported(tmp_path):
    ctrl = _run(tmp_path, "c", [("q1", ["1111.1111", "2222.2222"])] * 3)
    arm = _run(tmp_path, "a", [("q1", ["1111.1111"]),
                               ("q1", ["1111.1111", "2222.2222"]),
                               ("q1", ["1111.1111"])])
    fl = [f for f in AD.flips(ctrl, arm, QS) if f.paper == "2222.2222"]
    assert fl and fl[0].lost
    assert "LOST" in AD.report(ctrl, arm, QS)
