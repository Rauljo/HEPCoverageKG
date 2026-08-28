"""Subsetting a question set without breaking what it measures."""
from __future__ import annotations

import json
from collections import Counter

from hepcoveragekg.eval import subset


def _records(groups: dict[str, tuple[str, int]]) -> list[dict]:
    out = []
    for name, (shape, n) in groups.items():
        for i in range(n):
            out.append({"qid": f"{name}-{i}", "shape": shape, "group": name})
    return out


def test_a_metamorphic_group_is_never_split():
    """Paraphrase invariance is measured WITHIN a group, so a group split across
    the subset boundary destroys that measure for every group it splits -- and
    the file still looks fine, which is what makes it dangerous."""
    records = _records({f"g{i}": ("count", 4) for i in range(50)})
    picked = subset.sample(records, target=100)
    src = Counter(r["group"] for r in records)
    sub = Counter(r["group"] for r in picked)
    assert all(sub[g] == src[g] for g in sub), "a group was split"


def test_the_shape_mix_survives():
    """Tier B is 75% count / 25% set, and the critic moves the two very
    differently -- roughly doubling set F1 while barely moving counting. A
    subset that drifted toward one shape would move the headline number for a
    reason that has nothing to do with the arm."""
    records = _records({**{f"c{i}": ("count", 4) for i in range(75)},
                        **{f"s{i}": ("set", 4) for i in range(25)}})
    picked = subset.sample(records, target=200)
    mix = Counter(r["shape"] for r in picked)
    ratio = mix["count"] / (mix["count"] + mix["set"])
    assert 0.70 <= ratio <= 0.80, f"shape mix drifted to {ratio:.2f}"


def test_it_is_deterministic():
    records = _records({f"g{i}": ("count", 4) for i in range(100)})
    a = [r["qid"] for r in subset.sample(records, 100)]
    b = [r["qid"] for r in subset.sample(records, 100)]
    assert a == b
    c = [r["qid"] for r in subset.sample(records, 100, seed=999)]
    assert c != a, "a different seed should draw a different subset"


def test_a_set_smaller_than_the_target_comes_back_whole():
    records = _records({f"g{i}": ("count", 4) for i in range(10)})
    assert len(subset.sample(records, target=200)) == 40


def test_an_ungrouped_question_is_its_own_group():
    """Falling back to a shared '?' bucket would make every ungrouped question
    travel together -- all or nothing, for no reason."""
    records = [{"qid": "a", "shape": "count"}, {"qid": "b", "shape": "count"}]
    assert subset.group_of(records[0]) == "a"
