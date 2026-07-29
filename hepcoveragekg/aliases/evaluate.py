"""
HEPCoverageKG aliases: run an adjudicator over the trial set and score it.

Answers the question we could not answer before: did this change help?

Scoring is reported PER SOURCE, never pooled, because the three sources are not
equally reliable (see trialset.py). Pooling them would produce one confident
number resting on assumptions of very different strength.

  tier1_positive    a miss is a real loss -- these were merged by deterministic
                    rules we trust. This is the recall number to quote.
  same_id_positive  a miss is AMBIGUOUS: the model may be wrong, or it may have
                    found one of the id collisions the contract warns about
                    (752 shared ids with conflicting kind/label). Reported as
                    "disagreements to review", not as errors.
  negative          a "same" verdict is a SUSPECTED false positive, but may be a
                    genuine discovery: Tier 1 only catches spelling variants, so
                    two clusters can still be the same concept. Reported with
                    that caveat.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 32


async def _adjudicate_all(conn, pairs, concurrency: int, use_context: bool):
    from hepcoveragekg.aliases import adjudicate, context

    # Context is read from SQLite, which is not safe to touch from many threads;
    # fetch it up front, then the async phase is pure network.
    blocks: dict[str, str] = {}
    if use_context:
        for p in pairs:
            for key, name in ((p["id_a"], "ENTITY A"), (p["id_b"], "ENTITY B")):
                if key not in blocks:
                    blocks[key] = context.fetch(conn, key).render(name)

    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            ca = blocks.get(p["id_a"], "") if use_context else ""
            cb = blocks.get(p["id_b"], "") if use_context else ""
            # render() stamps the name at fetch time; re-label B defensively
            if cb.startswith("ENTITY A:"):
                cb = "ENTITY B:" + cb[len("ENTITY A:"):]
            res = await adjudicate.adjudicate_pair(
                p["term_a"], p["term_b"], kind=p["kind"], context_a=ca, context_b=cb
            )
            return {**p, **{f"got_{k}": v for k, v in res.items()}}

    return await asyncio.gather(*(one(p) for p in pairs))


def run(conn, pairs: list[dict], out_path: Path | str,
        concurrency: int | None = None, use_context: bool = True) -> dict:
    """Adjudicate every pair, write the raw results, return the scorecard."""
    if concurrency is None:
        concurrency = int(os.environ.get("LLM_CONCURRENCY", DEFAULT_CONCURRENCY))
    results = asyncio.run(_adjudicate_all(conn, pairs, concurrency, use_context))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return score(results)


def score(results: list[dict]) -> dict:
    """Per-source tallies. Errors are excluded from accuracy, never counted as
    wrong answers -- a failed call is not a verdict."""
    out: dict = {}
    for src in sorted({r["source"] for r in results}):
        rows = [r for r in results if r["source"] == src]
        ok = [r for r in rows if r["got_status"] == "ok"]
        errs = len(rows) - len(ok)
        agree = [r for r in ok
                 if (r["expected"] == "same") == bool(r["got_is_match"])]
        out[src] = {
            "pairs": len(rows),
            "answered": len(ok),
            "errors": errs,
            "agreed": len(agree),
            "disagreed": len(ok) - len(agree),
            "agreement": (len(agree) / len(ok)) if ok else 0.0,
        }
    out["_calibration"] = calibration(results)
    return out


CONFIDENCE_BANDS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01))


def calibration(results: list[dict]) -> list[dict]:
    """Is the model's stated confidence meaningful?

    For each band, how often was it actually right. A calibrated model answering
    0.9 is correct about 90% of the time; the 8B put every one of 2,712 matches
    above 0.9, so its number carried no information at all. This is the check
    that tells us whether confidence can be used as a filter -- measured, not
    assumed.
    """
    ok = [r for r in results if r["got_status"] == "ok"]
    bands: list[dict] = []
    for lo, hi in CONFIDENCE_BANDS:
        rows = [r for r in ok if lo <= float(r.get("got_confidence") or 0.0) < hi]
        if not rows:
            continue
        right = sum(1 for r in rows
                    if (r["expected"] == "same") == bool(r["got_is_match"]))
        bands.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "n": len(rows),
            "accuracy": right / len(rows),
            "share": len(rows) / len(ok),
        })
    return bands


def report(scorecard: dict) -> str:
    lines = [f"{'source':20s} {'pairs':>6s} {'answered':>9s} {'agreed':>7s} "
             f"{'disagreed':>10s} {'rate':>8s}", "-" * 64]
    for src, s in scorecard.items():
        if src.startswith("_"):
            continue
        lines.append(f"{src:20s} {s['pairs']:6d} {s['answered']:9d} {s['agreed']:7d} "
                     f"{s['disagreed']:10d} {s['agreement']:7.1%}")
        if s["errors"]:
            lines.append(f"{'':20s} {s['errors']} calls FAILED (no verdict; excluded)")

    bands = scorecard.get("_calibration") or []
    if bands:
        lines += ["", "confidence calibration -- is the number meaningful?",
                  f"  {'band':12s} {'n':>6s} {'share':>7s} {'accuracy':>9s}", "  " + "-" * 38]
        for b in bands:
            lines.append(f"  {b['band']:12s} {b['n']:6d} {b['share']:6.1%} {b['accuracy']:9.1%}")
        lines.append("  (calibrated = accuracy tracks the band; flat/all-in-one-band = useless)")

    lines += [
        "",
        "reading it:",
        "  tier1_positive   disagreement = a real synonym the model would lose (recall)",
        "  same_id_positive disagreement = model error OR a genuine id collision -> review",
        "  negative         disagreement = suspected false positive, or a real discovery",
    ]
    return "\n".join(lines)
