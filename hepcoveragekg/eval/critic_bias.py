"""
Does the candidate critic read the question, or does it follow the retriever?

Three arms over identical input (D-060). They answer different questions and the
middle one is the one that matters:

  shuffle A vs shuffle B   PURE POSITION BIAS. Same candidates, both arms in a
      meaningless order, so a verdict that changes flipped because of where the
      candidate sat and nothing else.

  ranked vs shuffled       HOW FAR THE CRITIC MERELY RESTATES BM25. If verdicts
      barely move when the retriever's ordering is destroyed, the critic is
      reading the question. If they move a lot, it was leaning on rank -- and a
      relevance step that agrees with the ranking is an expensive way to have no
      relevance step.

  chunk 15 vs chunk 60     whether chunking earns its extra calls, or whether
      "lost in the middle" is not biting at this scale anyway.

WHY NOT JUST SHUFFLE IN PRODUCTION. Shuffling makes bias measurable, but it
relocates the lost-in-the-middle risk onto the best candidate: ranked, the middle
of the list holds mid-relevance items where a miss costs least; shuffled, the
middle can hold rank 1. Ranked stays the default and this module is where the
cost of that choice gets measured instead of assumed.

TWO DIAGNOSTICS FALL OUT, and they separate failures a single number cannot:

  keep-rate by ORIGINAL RETRIEVAL RANK should slope DOWN -- rank 60 really is
      worse than rank 1 on average. A flat line means the critic is not
      discriminating at all.

  keep-rate by POSITION IN THE PROMPT should be FLAT once shuffled. A step means
      position bias, and where the step falls says whether it is the middle
      being lost or the tail being skimmed.

The cases come from real planner-authored search texts in `eval/runs/`, because
the search text is something the MODEL invented and no reconstruction of it from
the question would be honest.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from hepcoveragekg.query import critic as C

logger = logging.getLogger(__name__)

# Enough searches to separate a real effect from noise, few enough to run while
# something else holds the GPU. At 60 candidates and chunk 15 one case is 4
# calls per arm, so 40 cases across four arms is ~640 calls.
DEFAULT_CASES = 40


@dataclass
class Case:
    """One real search, ready to re-judge under different orderings."""

    qid: str
    question: str
    search_text: str


@dataclass
class ArmResult:
    name: str
    reviews: dict[str, C.Review] = field(default_factory=dict)

    def rungs(self) -> dict[tuple[str, str], str]:
        """(qid, entity_id) -> rung, so two arms can be compared elementwise."""
        return {(qid, v.entity_id): v.rung
                for qid, review in self.reviews.items()
                for v in review.verdicts}

    def kept(self) -> dict[tuple[str, str], bool]:
        return {(qid, v.entity_id): v.kept
                for qid, review in self.reviews.items()
                for v in review.verdicts}


def load_cases(run_path: Path | str, limit: int = DEFAULT_CASES) -> list[Case]:
    """Real (question, search text) pairs from a run's trace.

    One case per distinct search text: repeating the same words with a different
    question measures the question, and repeating the same question measures
    nothing. Searches that returned nothing are skipped -- there is no ordering
    to bias.
    """
    cases: list[Case] = []
    seen: set[str] = set()
    for line in Path(run_path).read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = record.get("question")
        if not question:
            continue
        for step in (record.get("answer") or {}).get("steps") or []:
            if step.get("tool") != "search" or step.get("error"):
                continue
            if not step.get("rows"):
                continue
            text = (step.get("args") or {}).get("text")
            if not text or text in seen:
                continue
            seen.add(text)
            cases.append(Case(record.get("qid", ""), question, text))
            if len(cases) >= limit:
                return cases
    return cases


def run_arm(chat: Callable, index, conn, cases: Sequence[Case], *,
            name: str, seed: Optional[int] = None, chunk: int = C.CHUNK,
            limit: int = 60) -> ArmResult:
    """Judge every case once, under one ordering."""
    from hepcoveragekg.query import retrieve

    arm = ArmResult(name=name)
    for case in cases:
        hits = retrieve.search(index, case.search_text, conn=conn, limit=limit)
        if not hits:
            continue
        arm.reviews[case.qid or case.search_text] = C.judge_candidates(
            chat, case.question, case.search_text, hits, chunk=chunk, seed=seed)
    return arm


def compare(a: ArmResult, b: ArmResult) -> dict:
    """How much two arms disagree, on the same candidates.

    `rung_flip` is the strict reading and `kept_flip` the actionable one: a
    candidate moving `exact` -> `broader` changes the diagnosis but not the set,
    while one moving `broader` -> `unrelated` changes what gets counted.
    """
    left, right = a.rungs(), b.rungs()
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"n": 0}

    kept_a, kept_b = a.kept(), b.kept()
    rung_flips = sum(1 for k in shared if left[k] != right[k])
    kept_flips = sum(1 for k in shared if kept_a[k] != kept_b[k])
    moved = Counter((left[k], right[k]) for k in shared if left[k] != right[k])
    return {
        "n": len(shared),
        "rung_flip": rung_flips / len(shared),
        "kept_flip": kept_flips / len(shared),
        "kept_a": sum(1 for k in shared if kept_a[k]) / len(shared),
        "kept_b": sum(1 for k in shared if kept_b[k]) / len(shared),
        "moves": dict(moved.most_common(6)),
    }


def keep_rate_by_rank(arm: ArmResult, buckets: int = 4) -> list[float]:
    """Keep-rate down the ORIGINAL retrieval order, in equal buckets.

    Should slope down. Flat means the critic is not discriminating -- which a
    single overall keep-rate would hide completely, because "kept 70%" looks the
    same whether the 30% dropped were the tail or were scattered at random.
    """
    per_bucket: list[list[bool]] = [[] for _ in range(buckets)]
    for review in arm.reviews.values():
        n = len(review.verdicts)
        if not n:
            continue
        for position, verdict in enumerate(review.verdicts):
            per_bucket[min(position * buckets // n, buckets - 1)].append(verdict.kept)
    return [round(sum(b) / len(b), 3) if b else float("nan") for b in per_bucket]


def report(arms: dict[str, ArmResult]) -> str:
    """The three comparisons and the two diagnostics, as text."""
    lines = ["critic ordering arms", "=" * 60]
    for name, arm in arms.items():
        n = sum(len(r.verdicts) for r in arm.reviews.values())
        kept = sum(len(r.kept_ids) for r in arm.reviews.values())
        defaulted = sum(r.defaulted for r in arm.reviews.values())
        calls = sum(r.calls for r in arm.reviews.values())
        lines.append(f"{name:16s} searches={len(arm.reviews):3d} candidates={n:5d} "
                     f"kept={kept:5d} ({100 * kept / max(n, 1):4.1f}%) "
                     f"defaulted={defaulted:4d} calls={calls:4d}")
        lines.append(f"{'':16s} keep-rate by original rank: "
                     f"{keep_rate_by_rank(arm)}  (should slope down)")

    pairs = [
        ("shuffleA", "shuffleB", "pure position bias"),
        ("ranked", "shuffleA", "how far it restates the retriever"),
        ("ranked", "chunk60", "whether chunking earns its calls"),
    ]
    lines += ["", "comparisons", "-" * 60]
    for left, right, why in pairs:
        if left not in arms or right not in arms:
            continue
        stats = compare(arms[left], arms[right])
        if not stats.get("n"):
            continue
        lines.append(f"{left} vs {right}  ({why})")
        lines.append(f"    n={stats['n']}  rung_flip={stats['rung_flip']:.3f}  "
                     f"kept_flip={stats['kept_flip']:.3f}  "
                     f"kept {stats['kept_a']:.3f} -> {stats['kept_b']:.3f}")
        if stats["moves"]:
            moves = ", ".join(f"{a}->{b}: {n}" for (a, b), n in stats["moves"].items())
            lines.append(f"    {moves}")
    return "\n".join(lines)


def measure(chat: Callable, index, conn, cases: Sequence[Case]) -> dict[str, ArmResult]:
    """All four arms over one set of cases."""
    return {
        "ranked": run_arm(chat, index, conn, cases, name="ranked"),
        "shuffleA": run_arm(chat, index, conn, cases, name="shuffleA", seed=11),
        "shuffleB": run_arm(chat, index, conn, cases, name="shuffleB", seed=97),
        "chunk60": run_arm(chat, index, conn, cases, name="chunk60", chunk=60),
    }
