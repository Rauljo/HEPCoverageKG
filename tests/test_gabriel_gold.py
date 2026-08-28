"""Gabriel's verdicts as a question set, and the scoring that partial gold needs.

The failure this guards against is quiet: gold that looks complete, is not, and
charges the system for papers a human never looked at.
"""
from __future__ import annotations

import pytest

from hepcoveragekg.eval import gabriel_gold as G
from hepcoveragekg.eval import questions as Q
from hepcoveragekg.eval import scoring
from hepcoveragekg.eval.systems import Answer


@pytest.fixture(scope="module")
def built():
    return G.build()


def test_the_strongest_verdict_wins_when_a_paper_has_several_rows():
    """A sheet row is (question, paper, ONE cited sentence). The builder failed
    to dedupe by (qid, paper), so 13 papers were asked about twice -- all 13
    with DIFFERENT quotes, none a repeat.

    So "no" judges the SENTENCE, not the paper: this quote does not establish
    the claim. If another quote does, the paper is yes. Treating the pair as a
    contradiction and dropping the paper cost gf-01 five of its eight gold
    papers."""
    assert G.resolve(["yes", "no"]) == "yes"
    assert G.resolve(["no", "yes"]) == "yes"
    assert G.resolve(["yes", "unsure"]) == "yes"
    assert G.resolve(["unsure", "no"]) == "unsure", "neither row established it"
    assert G.resolve(["no", "no"]) == "no"
    assert G.resolve([]) is None


def test_a_paper_with_one_supporting_quote_is_gold(built):
    """The concrete case: 2006.05880 was judged no on one quote and yes on
    another, and belongs in gf-01's gold."""
    gf01 = next(q for q in built if q["provenance"]["reader_qid"] == "gf-01")
    assert "2006.05880" in gf01["truth"]["papers"]
    assert len(gf01["truth"]["papers"]) == 8


def test_unsure_is_neither_gold_nor_a_negative(built):
    """Four of the unsure rows are 'the evidence is thin but the paper is
    probably true' (D-066) -- a different axis. Scoring them either way invents
    a verdict he declined to give."""
    verdicts = G.read_verdicts()
    for q in built:
        judged = verdicts[q["provenance"]["reader_qid"]]
        unsure = {p for p, given in judged.items() if G.resolve(given) == "unsure"}
        assert not (unsure & set(q["truth"]["universe"]))


def test_the_set_loads_through_the_real_loader(tmp_path, built):
    path = G.write(tmp_path / "gabriel.jsonl", built)
    qset = Q.load(path)
    assert len(qset) == len(built)
    for q in qset:
        assert q.shape == "set"
        assert q.truth.papers and q.truth.universe
        assert set(q.truth.papers) <= set(q.truth.universe)


def test_an_unjudged_paper_is_not_a_false_positive():
    """The whole reason `universe` exists. He only saw papers our system
    surfaced, so an unjudged paper is unknown, not wrong -- and charging for it
    would measure how the review sheet was sampled."""
    q = Q.parse({
        "qid": "t", "text": "which analyses?", "shape": "set", "split": "dev",
        "truth_source": "gabriel",
        "truth": {"kind": "set", "papers": ["1111.1111"],
                  "universe": ["1111.1111", "2222.2222"]},
    })
    clean = scoring.judged_set_f1(q, Answer(text="1111.1111"))
    unjudged = scoring.judged_set_f1(q, Answer(text="1111.1111 and 9999.9999"))
    assert unjudged["judged_precision"] == clean["judged_precision"] == 1.0

    rejected = scoring.judged_set_f1(q, Answer(text="1111.1111 and 2222.2222"))
    assert rejected["judged_precision"] == 0.5, "a REJECTED paper must count against"


def test_set_f1_and_judged_set_f1_never_both_fire():
    """Two measures under one name is how a month of numbers became
    uninterpretable last time (see `retrieval_reach`)."""
    q = Q.parse({
        "qid": "t", "text": "which analyses?", "shape": "set", "split": "dev",
        "truth_source": "gabriel",
        "truth": {"kind": "set", "papers": ["1111.1111"],
                  "universe": ["1111.1111", "2222.2222"]},
    })
    a = Answer(text="1111.1111")
    assert scoring.set_f1(q, a) is None
    assert scoring.judged_set_f1(q, a) is not None

    plain = Q.parse({
        "qid": "t2", "text": "which analyses?", "shape": "set", "split": "dev",
        "truth_source": "sql", "truth": {"kind": "set", "papers": ["1111.1111"]},
    })
    assert scoring.set_f1(plain, a) is not None
    assert scoring.judged_set_f1(plain, a) is None


def test_the_questions_stay_his_questions(built):
    """Reworded from 'does THIS paper' to 'which papers', and nothing else. A
    question whose physics drifted is no longer scoreable against his verdicts."""
    for q in built:
        assert q["text"].lower().startswith("which analyses")
        assert q["truth_source"] == "gabriel"
        assert q["provenance"]["conditions"] >= 1
