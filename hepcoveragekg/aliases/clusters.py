"""
HEPCoverageKG aliases: checking a whole cluster at once.

Pairwise adjudication asks *"is A the same as B?"*. That question cannot see the
failure it causes. Chaining 2,262 approved pairs on 2026-08-02 produced 330
clusters, the largest holding **75 statistical methods** -- profile likelihood,
the CLs limit-setting procedure and log-normal nuisance constraints welded into
one node, because every NEIGHBOURING pair looked similar enough. Nobody ever
judged the two ends of that chain, and they are plainly different things.

So this module asks a different question, with the whole set visible at once:

    "Are all of these one thing, or should they be split?"

Sizes decide who answers, and the split is very favourable:

    174 clusters of 2      already one verdict -- nothing to re-check
    122 clusters of 3-8    ONE model call each, ~2 minutes for all of them
     34 clusters of 9+     663 entities, too varied for one call -- a human
                           reading list (`run.cluster_report`)

**Deliberately NOT part of `propose_deep_semantics`.** Deduplication answers a
question about IDENTITY and emits pairs; deciding that several versions belong
to one FAMILY is the grouping layer's job (S-28). Keeping the two apart is what
stopped the chain failure, so re-merging them into one command would undo that.
It is a separate CLI action reading a proposals file, which also means it can be
re-run after a prompt change without repeating the expensive pairwise pass.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Small enough that one prompt can hold every member with room to think. Above
# this the members are too varied for a single verdict to mean anything, and the
# question stops being answerable rather than merely getting harder.
MAX_CHECKABLE = 8
MIN_CHECKABLE = 3

# Per member, inside a cluster prompt. Far tighter than pairwise adjudication
# (10 quotes) because eight members share the budget rather than two.
QUOTES_PER_MEMBER = 2

_TASK = """You are given a group of entities from one high-energy-physics knowledge graph.
They were merged into one group by a chain of pairwise decisions: A was judged the
same as B, B the same as C, and so on. Nobody checked the ends of the chain.

Decide whether the WHOLE group refers to a single thing.

Judge identity, not topic. Two entities are the same only if a physicist would
say they are literally the same object, sample, region, method or uncertainty --
not merely that they are related, belong to the same family, or appear together.

Common ways a chain goes wrong:
  - different versions or tunes of one tool (a family, not one thing)
  - different channels or selections (electron vs muon, leading vs subleading)
  - a general method and a specific application of it
  - a background process and a broader category containing it"""

_SCHEMA = """Answer with JSON only:
{
  "verdict": "one" or "split",
  "confidence": <number between 0.0 and 1.0>,
  "groups": [[0, 2], [1, 3]],
  "explanation": "<one sentence naming the specific distinction, or why they are one thing>"
}
`groups` lists the member numbers that belong together; give it only when the
verdict is "split", and make every member appear exactly once."""


def render_member(ctx, index: int) -> str:
    """One member of a cluster, compact. Eight of these share a prompt."""
    lines = [f"[{index}] {ctx.labels[0] if ctx.labels else ctx.entity_id}"]
    if len(ctx.labels) > 1:
        lines.append("     also written as: " + "; ".join(ctx.labels[1:4]))
    if ctx.aliases:
        lines.append("     aliases: " + "; ".join(ctx.aliases[:4]))
    if ctx.attributes:
        lines.append(f"     attributes: {json.dumps(ctx.attributes, ensure_ascii=False)[:200]}")
    for _section, quote in ctx.quotes[:QUOTES_PER_MEMBER]:
        lines.append(f'     quote: "{quote[:200]}"')
    return "\n".join(lines)


def build_prompt(kind: str, members: list[str]) -> str:
    return "\n".join([_TASK, "", f"Entity kind: {kind}", "",
                      "GROUP:", *members, "", _SCHEMA])


def _parse(result: Any, size: int) -> dict:
    """Read the verdict defensively; a malformed `groups` must not silently
    become an approval."""
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict not in ("one", "split"):
        raise ValueError(f"unusable verdict: {result!r}")

    groups = result.get("groups") or []
    if verdict == "split":
        flat = [i for g in groups for i in g]
        if sorted(flat) != list(range(size)):
            # The model said split but could not say how. That is still useful --
            # "do not merge this" -- so it is kept, with the partition dropped.
            logger.debug(f"split verdict with unusable groups {groups!r}")
            groups = []
    else:
        groups = []

    return {
        "verdict": verdict,
        "confidence": float(result.get("confidence") or 0.0),
        "groups": groups,
        "explanation": str(result.get("explanation", ""))[:400],
    }


async def check_one(conn_blocks: dict[str, str], cluster: list[str], kind: str) -> dict:
    """Ask about one cluster. Returns the verdict, or an error record."""
    from hepcoveragekg.aliases import adjudicate

    try:
        # Inside the try on purpose. Building the prompt can fail -- a missing
        # context block, an entity that vanished between passes -- and an
        # exception raised here escapes `asyncio.gather` and kills every other
        # cluster in flight. One bad cluster must cost one error record.
        client, model = adjudicate._get_llm_client()
        prompt = build_prompt(kind, [conn_blocks[e] for e in cluster])

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty response")
        parsed = _parse(json.loads(content), len(cluster))
        return {"status": "ok", **parsed}
    except Exception as exc:  # noqa: BLE001
        # A failed call is NOT a verdict. Recording it as "one" would silently
        # approve a chain nobody judged -- the exact failure this module exists
        # to catch.
        return {"status": "error", "verdict": None, "confidence": 0.0,
                "groups": [], "explanation": f"{type(exc).__name__}: {exc}"}


def check_clusters(conn, proposals_path: Path | str, out_path: Path | str,
                   min_size: int = MIN_CHECKABLE, max_size: int = MAX_CHECKABLE,
                   concurrency: int = 16) -> dict:
    """Check every cluster small enough to fit in one prompt.

    Reads a proposals file rather than re-running the pairwise pass, so a prompt
    change costs minutes instead of an hour.
    """
    from hepcoveragekg.aliases import context, run as run_module

    plan = run_module.writeback_plan(proposals_path)
    targets = [sorted(c) for c in plan["clusters"] if min_size <= len(c) <= max_size]
    if not targets:
        return {"checked": 0}

    context.load_section_priors(conn)
    wanted = {e for c in targets for e in c}
    blocks: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for eid in wanted:
        ctx = context.fetch(conn, eid)
        kinds[eid] = ctx.kind or ""
        blocks[eid] = ctx  # rendered per cluster, where the index is known

    logger.info(f"Checking {len(targets)} clusters of {min_size}-{max_size} members "
                f"({len(wanted)} entities, concurrency {concurrency})")

    async def run_all():
        sem = asyncio.Semaphore(concurrency)

        async def one(cluster):
            async with sem:
                rendered = {e: render_member(blocks[e], i) for i, e in enumerate(cluster)}
                kind = kinds.get(cluster[0], "")
                return cluster, await check_one(rendered, cluster, kind)

        return await asyncio.gather(*[one(c) for c in targets])

    results = asyncio.run(run_all())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"one": 0, "split": 0, "error": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for cluster, verdict in results:
            counts[verdict["verdict"] or "error"] = \
                counts.get(verdict["verdict"] or "error", 0) + 1
            fh.write(json.dumps({
                "members": cluster,
                "size": len(cluster),
                "kind": kinds.get(cluster[0], ""),
                "labels": [
                    (blocks[e].labels[0] if blocks[e].labels else e) for e in cluster
                ],
                **verdict,
            }, ensure_ascii=False) + "\n")

    total = len(results)
    logger.info(f"Cluster check: {counts['one']} hold together, {counts['split']} "
                f"should be split, {counts['error']} errored (of {total})")
    return {"checked": total, **counts, "out": str(out_path)}
