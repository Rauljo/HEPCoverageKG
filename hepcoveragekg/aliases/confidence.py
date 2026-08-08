"""
HEPCoverageKG aliases: how sure is the adjudicator, really?

Two different things get called "consistency", and conflating them cost a
measurement on 2026-08-02.

  temperature 0, repeated.  111 of 111 verdicts identical across three runs.
      That measures whether the SERVING STACK is deterministic -- worth knowing,
      and not guaranteed, since vLLM's batching can reorder floating-point
      reductions. It says nothing about whether the model is sure. Greedy
      decoding hides doubt by construction: a pair the model is 51/49 on and one
      it is 99/1 on both come back identical every time.

  temperature > 0, sampled.  Ask the same pair five times and count how often the
      verdict agrees. THAT measures the model's own uncertainty, and it is a
      behavioural signal rather than a self-report -- much harder to be
      miscalibrated about than the `confidence` field, which the model writes
      about itself and which currently clusters on 0.85/0.90/0.95.

The technique is self-consistency (Wang et al.) and its stronger form semantic
entropy (Farquhar et al., Nature 2024) -- already S-52 in the vault for the query
layer, applied here to the aliases layer.

**Why it is worth the calls.** There are 2,262 approved merges and no basis for
deciding which can be confirmed without a human. Agreement rate under sampling is
exactly that ranking: auto-confirm what 5/5 samples agree on, review the rest.

**And the caveat that makes the hand-labelling step non-negotiable.** A model can
be *consistently wrong*. Agreement measures stability, not correctness. So the
sample deliberately covers the same pairs as `review_pairs.tsv`, and the claim
"agreement predicts correctness" is a hypothesis to be tested against those hand
labels -- not an assumption to build on.

Production verdicts stay at temperature 0: reproducible, auditable, and a merge
can always be explained. This is a separate measurement layered on top, never a
replacement.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# High enough that a genuinely uncertain pair will flip, low enough that a
# confident one will not wander. 0.7 is the usual self-consistency setting.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLES = 5


def agreement(verdicts: list[Optional[bool]]) -> float:
    """Share of usable samples that landed on the majority verdict.

    Errors are dropped rather than counted as disagreement -- a timeout is not
    the model changing its mind, and treating it as one would make a flaky
    network look like an uncertain model.
    """
    usable = [v for v in verdicts if v is not None]
    if not usable:
        return 0.0
    return Counter(usable).most_common(1)[0][1] / len(usable)


def measure(conn, proposals_path: Path | str, out_path: Path | str,
            per_kind: int = 6, samples: int = DEFAULT_SAMPLES,
            temperature: float = DEFAULT_TEMPERATURE,
            concurrency: int = 16, seed: int = 0) -> dict:
    """Re-adjudicate the review sample at temperature > 0 and record agreement.

    Uses `stratified_matches` with the same arguments as the review file, so the
    pairs a human labels are the pairs whose agreement is known. Without that
    alignment a disagreement between human and model is unreadable: it could mean
    the model is wrong or that it is unstable, and those need different fixes.
    """
    from hepcoveragekg.aliases import adjudicate, context, run as run_module

    records = run_module.load_proposals(proposals_path)
    sample = run_module.stratified_matches(records, per_kind=per_kind, seed=seed)
    if not sample:
        return {"sampled": 0}

    context.load_section_priors(conn)
    wanted = {i for r in sample for i in (r["id_a"], r["id_b"])}
    blocks = {i: context.fetch(conn, i).render("ENTITY A") for i in wanted}

    logger.info(f"Sampling {len(sample)} pairs x {samples} at temperature "
                f"{temperature} ({len(sample) * samples} calls)")

    async def run_all():
        sem = asyncio.Semaphore(concurrency)

        async def one(rec, _n):
            async with sem:
                ctx_a = blocks.get(rec["id_a"], "")
                ctx_b = blocks.get(rec["id_b"], "")
                if ctx_b.startswith("ENTITY A:"):
                    ctx_b = "ENTITY B:" + ctx_b[len("ENTITY A:"):]
                res = await adjudicate.adjudicate_pair(
                    rec["term_a"], rec["term_b"], kind=rec.get("kind", "concept"),
                    context_a=ctx_a, context_b=ctx_b, temperature=temperature,
                )
                return res["is_match"] if res["status"] == "ok" else None

        tasks = [one(r, n) for r in sample for n in range(samples)]
        flat = await asyncio.gather(*tasks)
        return [flat[i * samples:(i + 1) * samples] for i in range(len(sample))]

    per_pair = asyncio.run(run_all())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buckets: Counter = Counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for rec, verdicts in zip(sample, per_pair):
            rate = agreement(verdicts)
            buckets[round(rate, 2)] += 1
            fh.write(json.dumps({
                "id_a": rec["id_a"], "id_b": rec["id_b"],
                "term_a": rec["term_a"], "term_b": rec["term_b"],
                "kind": rec.get("kind", ""),
                # The temperature-0 verdict this pair was approved on.
                "greedy_is_match": rec.get("is_match"),
                "greedy_confidence": rec.get("confidence"),
                "sampled_verdicts": verdicts,
                "agreement": round(rate, 4),
                "unanimous": rate == 1.0,
                "transitivity_paradox": rec.get("transitivity_paradox", False),
            }, ensure_ascii=False) + "\n")

    unanimous = sum(1 for v in per_pair if agreement(v) == 1.0)
    logger.info(f"Agreement: {unanimous}/{len(sample)} pairs unanimous "
                f"({unanimous / len(sample):.1%}); distribution "
                f"{dict(sorted(buckets.items(), reverse=True))}")
    if unanimous == len(sample):
        # Not automatically good news -- it can mean the sampling temperature is
        # too low to reveal anything, so the measurement failed rather than the
        # model being certain.
        logger.warning("Every pair unanimous: either the model is genuinely "
                       "certain, or the temperature is too low to separate "
                       "anything. Try a higher temperature before trusting it.")

    return {"sampled": len(sample), "samples_each": samples,
            "temperature": temperature, "unanimous": unanimous,
            "unanimous_share": round(unanimous / len(sample), 4),
            "distribution": dict(sorted(buckets.items(), reverse=True)),
            "out": str(out_path)}
