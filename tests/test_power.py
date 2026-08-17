"""How many questions a conclusion needs, measured rather than assumed."""
from __future__ import annotations

import statistics

from hepcoveragekg.eval import power


def rec(score, metric="count_correct"):
    return {"scores": {metric: score}}


def test_repeats_collapse_before_resampling():
    """Three readings of one question are ONE observation of the effect. Treating
    them as three shrinks the apparent spread and talks the analysis into a
    smaller test set than the data supports."""
    a = {("q1", 0): rec(0.0), ("q1", 1): rec(1.0), ("q1", 2): rec(1.0)}
    b = {("q1", 0): rec(1.0), ("q1", 1): rec(1.0), ("q1", 2): rec(1.0)}
    paired = power.by_question(a, b, "count_correct")
    assert list(paired) == ["q1"]
    assert paired["q1"] == (statistics.mean([0.0, 1.0, 1.0]), 1.0)


def test_a_large_effect_reproduces_at_a_small_size():
    paired = {f"q{i}": (0.0, 1.0) for i in range(400)}     # every question moves
    curve = power.power_curve(paired, sizes=(25, 50), bootstrap=200)
    assert all(p.reproduced == 1.0 for p in curve)
    assert power.smallest_reliable(curve) == 25


def test_a_marginal_effect_needs_more():
    """Five questions in a hundred move, so a sample of 25 often contains none."""
    paired = {f"q{i}": ((0.0, 1.0) if i % 20 == 0 else (0.0, 0.0)) for i in range(800)}
    curve = power.power_curve(paired, sizes=(25, 800), bootstrap=300)
    assert curve[0].reproduced < curve[-1].reproduced
    assert curve[-1].reproduced > 0.9


def test_an_effect_below_the_noise_floor_never_reproduces():
    """A difference smaller than run-to-run variation is not a finding at any
    sample size, and the curve should say so rather than converging on it."""
    paired = {f"q{i}": (0.0, 0.0001) for i in range(800)}
    curve = power.power_curve(paired, sizes=(100, 800), bootstrap=200)
    assert all(p.reproduced == 0.0 for p in curve)
    assert power.smallest_reliable(curve) is None


def test_the_sign_must_match_not_just_the_magnitude():
    """A subsample that finds a big effect the WRONG way has not reproduced."""
    paired = {f"q{i}": ((0.0, 1.0) if i < 60 else (1.0, 0.0)) for i in range(100)}
    curve = power.power_curve(paired, sizes=(25,), bootstrap=300)
    assert 0.0 < curve[0].reproduced < 1.0, "a mixed effect must not read as certain"


def test_sizes_beyond_the_data_are_dropped():
    paired = {f"q{i}": (0.0, 1.0) for i in range(30)}
    curve = power.power_curve(paired, sizes=(25, 50, 100), bootstrap=50)
    assert [p.n for p in curve] == [25]


def test_render_names_the_smallest_reliable_size():
    paired = {f"q{i}": (0.0, 1.0) for i in range(200)}
    text = power.render("critic", 1.0, power.power_curve(paired, sizes=(25, 50), bootstrap=50))
    assert "smallest reliable size: 25" in text
