"""A small mixed question set, for a day of iterating rather than a night of proving.

Every powered run so far has been one tier at a time -- 200 concept questions,
or 200 per-paper ones -- because that is what a defensible ablation needs. It is
the wrong instrument for the other job: changing something at 11am and wanting to
know by 1pm whether it fired, whether it broke a different tier, and whether it
cost time.

FOUR TIERS, BECAUSE A REGRESSION HIDES IN THE ONE YOU LEFT OUT. Three separate
findings today came from a shape split that a single-tier set could not see:
ranked-vs-shuffled reverses between count and set questions, the 8B critic looks
fine on set questions and halves counting accuracy, and gf-08 failed on a path
nothing else exercises. A fast set that sampled one tier would reproduce exactly
that blindness at higher speed.

  gabriel   the only human gold there is        -> judged_f1
  paperA    per-paper, exact truth              -> count_correct, label_recall
  conceptB  concepts, whole metamorphic groups  -> count_correct, set_f1
  retrieval entity-id truth, ambiguity-proof    -> entity_retrieved

Sized for roughly 200 records at 3 repeats, which two arms sharing one server
finish in about ninety minutes -- against nine hours for conceptB-200.

WHAT IT IS NOT FOR. Nothing here has the power to settle a small effect; the
noise floor is 0.0008 and this set cannot see anything near it. It is a smoke
test with real metrics: it says "this fired, and nothing else fell over". A
result that matters still goes to a powered set before it is believed.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import subset

#: (source, how many questions) -- Tier B gets the most because it is where the
#: critic acts and where the count/set split lives.
SOURCES: tuple[tuple[str, int], ...] = (
    ("eval/questions/gabriel-gold-2026-08-25.jsonl", 8),      # all of it
    ("eval/questions/dev-2026-08-03-paperA-200.jsonl", 20),
    ("eval/questions/dev-conceptB-200.jsonl", 24),
    ("eval/questions/dev-2026-08-03-retrieval.jsonl", 16),
)

#: The ITERATION set: Gabriel's eight plus a handful from each other tier.
#: Sized for ~25 minutes an arm rather than ~90, because the bottleneck today is
#: how many ideas can be tried in an afternoon, not how tightly any one of them
#: is measured. Nothing here has the power to settle a small effect -- it is for
#: "did this fire, and did anything else fall over".
PROBE: tuple[tuple[str, int], ...] = (
    ("eval/questions/gabriel-gold-2026-08-25.jsonl", 8),
    ("eval/questions/dev-2026-08-03-paperA-200.jsonl", 6),
    ("eval/questions/dev-conceptB-200.jsonl", 8),
    ("eval/questions/dev-2026-08-03-retrieval.jsonl", 4),
)


def build(sources=SOURCES, *, seed: int = 20260829) -> list[dict]:
    out: list[dict] = []
    for path, target in sources:
        rows = [json.loads(l) for l in
                Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        # `subset.sample` keeps metamorphic groups whole and preserves the shape
        # mix, which matters as much in 24 questions as in 200 -- more, since a
        # single split group is a larger fraction of the evidence.
        out.extend(subset.sample(rows, target, seed=seed))
    return out


def write(path: Path | str = "eval/questions/dev-fast-mixed.jsonl", **kw) -> tuple[Path, dict]:
    rows = build(**kw)
    path = Path(path)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    from collections import Counter
    return path, {
        "questions": len(rows),
        "by_source": dict(Counter(r.get("source", "?") for r in rows)),
        "by_shape": dict(Counter(r["shape"] for r in rows)),
        "by_truth": dict(Counter((r.get("truth") or {}).get("kind", "none") for r in rows)),
    }
