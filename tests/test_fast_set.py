"""The fast mixed set: small, but not blind in the way one tier would be."""
from __future__ import annotations

from collections import Counter

from hepcoveragekg.eval import fast_set


def test_every_tier_is_present():
    """Three findings today came from a shape or tier split that a single-tier
    set could not see. A fast set drawn from one tier would reproduce that
    blindness at higher speed."""
    rows = fast_set.build()
    truths = Counter((r.get("truth") or {}).get("kind") for r in rows)
    for kind in ("set", "count", "labels", "entity"):
        assert truths[kind] > 0, f"no {kind} questions -- that scorer is unexercised"
    assert any(r.get("source") == "gabriel" for r in rows), "human gold missing"


def test_it_stays_small_enough_to_be_useful_within_the_hour():
    rows = fast_set.build()
    assert 50 <= len(rows) <= 90, f"{len(rows)} questions is not a fast set"


def test_it_is_deterministic():
    assert [r["qid"] for r in fast_set.build()] == [r["qid"] for r in fast_set.build()]


def test_metamorphic_groups_survive_the_sampling():
    """Paraphrase invariance is measured within a group. A split group is a
    larger fraction of the evidence at 24 questions than at 436."""
    rows = fast_set.build()
    picked = Counter(r.get("group") for r in rows if r.get("group"))
    import json
    from pathlib import Path
    for src in ("eval/questions/dev-conceptB-200.jsonl",):
        full = Counter()
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line).get("group")
                if g:
                    full[g] += 1
        for g, n in picked.items():
            if g in full:
                assert n == full[g], f"group {g} was split: {n} of {full[g]}"
