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


# ---------------------------------------------------------------------------
# The two controls, built from the graph itself.
# ---------------------------------------------------------------------------
#
# These GATE the arms above: a flip rate means nothing from a critic that cannot
# produce both extreme answers.
#
#   all-keep  a question and a search that agree exactly, so every candidate
#             belongs. A critic that still drops some is hedging rather than
#             judging -- AgentRivet's Claude-Opus never once returned "approved"
#             even when instructed to.
#
#   all-drop  a question and a search about DIFFERENT things, so no candidate
#             belongs. A critic that keeps some is filling a quota, and ranked
#             order makes that worse: the weakest candidates arrive together in
#             the last chunk, so one forced keep per chunk inflates every count.

# The all-keep cases must be searches whose hits really ARE all the same thing.
# The first attempt used "jet energy scale" and failed at 63% -- correctly: BM25
# returns `electron-energy-scale`, `tau-h-energy-scale` and
# `unclustered-energy-scale` for it, and those are genuinely different
# systematics. The critic was right and the CONTROL was wrong. Generator names
# are safe in a way systematic names are not, because the graph holds one entity
# per version and tune of a single named program.
CONTROLS = [
    # (name, question, search text, what a working critic must return)
    ("all-keep", "Which analyses use the Pythia generator?", "Pythia", "keep"),
    ("all-keep", "Which analyses use the Sherpa generator?", "Sherpa", "keep"),
    ("all-drop", "Which analyses use the Pythia generator?", "jet energy scale", "drop"),
    ("all-drop", "Which analyses apply a jet energy scale uncertainty?",
     "Pythia", "drop"),
    # The one the first run failed on, kept as a THIRD kind of case: a family
    # question where the near-neighbours are different members of that family.
    # Neither all-keep nor all-drop -- what it must not do is keep everything.
    ("mixed-family", "Which analyses apply a jet energy scale uncertainty?",
     "jet energy scale", "mixed"),
]


def run_controls(chat: Callable, index, conn, *, limit: int = 30) -> str:
    """Can the critic say "all of them" and "none of them"?"""
    from hepcoveragekg.query import retrieve

    lines = ["controls (these gate everything else)", "=" * 60]
    verdict = True
    for name, question, text, expected in CONTROLS:
        hits = retrieve.search(index, text, conn=conn, limit=limit)
        review = C.judge_candidates(chat, question, text, hits)
        rate = len(review.kept_ids) / max(len(review.verdicts), 1)
        if expected == "keep":
            ok = rate >= 0.9
        elif expected == "drop":
            ok = rate <= 0.1
        else:                      # mixed: must discriminate, not answer wholesale
            ok = 0.1 < rate < 0.9
        verdict &= ok
        lines.append(
            f"{'PASS' if ok else 'FAIL'}  {name:9s} kept {len(review.kept_ids):3d}/"
            f"{len(review.verdicts):3d} ({rate:5.1%})   q={question[:44]!r} "
            f"search={text!r}")
        if not ok and expected in ("keep", "drop"):
            # the verdicts that broke it: unexpected drops, or unexpected keeps
            wanted_kept = expected == "keep"
            worst = [v for v in review.verdicts if v.kept != wanted_kept][:3]
            for v in worst:
                lines.append(f"          e.g. {v.entity_id[:44]}: {v.reason[:60]}")
    lines.append("")
    lines.append("GATE: " + ("passed -- the arms below are meaningful"
                            if verdict else
                            "FAILED -- fix the prompt before reading anything else"))
    return "\n".join(lines)


def main(argv=None) -> int:
    """Run the controls, then the four ordering arms, and print both."""
    import argparse
    import os
    import sqlite3

    from hepcoveragekg.query import planner, retrieve, templates

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/processed/hepkg.db")
    ap.add_argument("--run", default="eval/runs/20260803T234732-hepkg-48227.jsonl",
                    help="a trace to take real planner-authored search texts from")
    ap.add_argument("--cases", type=int, default=DEFAULT_CASES)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    conn = templates.read_only(args.db)
    index = retrieve.build(conn, embed=False)
    client, model = planner._client()

    def chat(messages):
        return client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            max_tokens=planner.MAX_COMPLETION_TOKENS)

    report_text = run_controls(chat, index, conn)
    print(report_text, flush=True)

    cases = load_cases(args.run, args.cases)
    print(f"\n{len(cases)} real search texts from {args.run}\n", flush=True)
    arms = measure(chat, index, conn, cases)
    report_text += "\n\n" + report(arms)
    print(report(arms), flush=True)

    if args.out:
        Path(args.out).write_text(report_text)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
