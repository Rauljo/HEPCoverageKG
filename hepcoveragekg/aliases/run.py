# =============================================================================
# HEPCoverageKG aliases: orchestration
#
# Two entry points, deliberately kept separate:
#
#   build(conn)                 -- Tiers 1 + 1.5. Deterministic, offline, cheap.
#                                  No model, no network. Writes same_as proposals.
#   propose_deep_semantics(...) -- Tiers 2 / 2.5 / 3. Loads an embedding model and
#                                  calls an LLM endpoint. Writes a review JSON only.
#
# They are NOT chained. build() used to call the deep pass, which meant the cheap
# deterministic tier could not run without a GPU and a live LLM server, and made
# the test suite load a model and hit the network.
#
# propose_deep_semantics takes its output path as a REQUIRED argument. It used to
# be hardcoded relative to the working directory, so running pytest from the repo
# root silently overwrote real pipeline results.
# =============================================================================
from __future__ import annotations

import asyncio
import collections
import json
import logging
from pathlib import Path

from hepcoveragekg.aliases import guards, normalize, spelling, store
from hepcoveragekg.aliases.cluster import connected_components

# `semantics` (sentence-transformers) and `adjudicate` (openai) are imported inside
# propose_deep_semantics, not here -- importing this module must stay cheap so that
# build() and the test suite never pull in a model or an HTTP client.

logger = logging.getLogger(__name__)

# The CLI's default. Never applied implicitly inside the library.
DEFAULT_DEEP_OUT = Path("data/processed/aliases_proposed.json")


def build(conn) -> dict[str, int]:
    """Tiers 1 + 1.5 only. Deterministic, offline, non-destructive.

    Writes same_as proposals; nothing resolves until `aliases confirm` runs.
    Tiers 2/2.5/3 are a separate call -- see propose_deep_semantics().
    """
    ents = store.entities_with_kind(conn)
    results: dict[str, int] = {}

    # Tier 1 — deterministic normalization (separators, case, version dot)
    results["normalize"] = store.write_proposals(conn, normalize.propose(ents))

    # Tier 1.5 — deterministic US/UK spelling + data-driven plural
    results["spelling"] = store.write_proposals(conn, spelling.propose(ents))

    return results


def propose_deep_semantics(conn, out_path: Path | str) -> dict[str, int]:
    """Tiers 2 (candidates), 2.5 (guards) and 3 (LLM adjudication) over entity labels.

    Requires an embedding model and a reachable LLM endpoint. Results are written
    to `out_path` as JSON for human review -- deliberately NOT into same_as yet.
    """
    # Deferred so that importing this module (and therefore build(), and the test
    # suite) never loads sentence-transformers or an HTTP client.
    from hepcoveragekg.aliases import adjudicate, semantics

    out_path = Path(out_path)
    logger.info("Starting Deep Semantic Pipeline (Tiers 2-3)...")

    cursor = conn.execute("SELECT entity_id, label, kind FROM entity")
    kind_to_labels = collections.defaultdict(set)
    label_to_ids = collections.defaultdict(list)

    for row in cursor:
        eid, label, kind = row["entity_id"], row["label"], row["kind"]
        label = label or ""
        kind_to_labels[kind].add(label)
        label_to_ids[label].append(eid)

    all_candidates = []

    # Phase A: Generate Candidates
    for kind, labels in kind_to_labels.items():
        if len(labels) < 2:
            continue
        candidates = semantics.generate_candidates(list(labels))
        for a, b in candidates:
            all_candidates.append((a, b, kind))

    logger.info(f"Phase A (Embeddings + Jaccard) proposed {len(all_candidates)} candidates.")

    # Phase B: Semantic Guards
    surviving_candidates = []
    for a, b, kind in all_candidates:
        passed, reason = guards.passes_semantic_guards(a, b)
        if passed:
            surviving_candidates.append((a, b, kind))
        else:
            logger.debug(f"Vetoed '{a}' vs '{b}': {reason}")

    logger.info(f"Phase B (Semantic Guards) surviving candidates: {len(surviving_candidates)}")

    # Phase C: LLM Adjudication
    final_proposals = []
    approved_edges = []
    rejected_edges = []

    logger.info(f"Invoking LLM on {len(surviving_candidates)} pairs...")

    async def process_candidates():
        # Limit concurrent requests so we don't overwhelm vLLM or hit socket limits
        sem = asyncio.Semaphore(100)

        async def fetch(a, b, kind):
            async with sem:
                res = await adjudicate.adjudicate_pair(a, b, kind=kind)
                return a, b, kind, res

        tasks = [fetch(a, b, kind) for a, b, kind in surviving_candidates]
        return await asyncio.gather(*tasks)

    results = asyncio.run(process_candidates())

    for a, b, kind, res in results:
        entry = {
            "term_a": a,
            "term_b": b,
            "kind": kind,
            "is_match": res["is_match"],
            "confidence": res["confidence"],
            "explanation": res["explanation"],
        }

        if res["is_match"]:
            approved_edges.append((a, b))
        else:
            rejected_edges.append((a, b))

        final_proposals.append(entry)

    # Phase D: Transitivity Consistency Checker
    clusters = connected_components(approved_edges)
    for cluster_set in clusters:
        for a, b in rejected_edges:
            if a in cluster_set and b in cluster_set:
                logger.warning(
                    f"TRANSITIVITY PARADOX: LLM rejected '{a}' == '{b}', but they were "
                    f"bridged transitively by other approvals in cluster {cluster_set}"
                )

    # Sort and output JSON
    final_proposals.sort(key=lambda x: x.get("confidence", 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_proposals, f, indent=2)

    logger.info(f"Wrote {len(final_proposals)} adjudicated proposals to {out_path}")
    return {
        "candidates": len(all_candidates),
        "after_guards": len(surviving_candidates),
        "matched": len(approved_edges),
        "rejected": len(rejected_edges),
        "written": len(final_proposals),
    }
