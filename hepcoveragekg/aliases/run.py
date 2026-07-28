# =============================================================================
# HEPCoverageKG aliases: orchestration
#
# build()  -> run the enabled tiers, write proposals (nothing resolves yet)
# Tier 1 (normalize) is the only tier wired now; ngram / embed / llm slot in here.
# =============================================================================
from __future__ import annotations

import collections
import json
import logging
import asyncio
from pathlib import Path

from hepcoveragekg.aliases import normalize, spelling, store
from hepcoveragekg.aliases import semantics, guards, adjudicate
from hepcoveragekg.aliases.cluster import connected_components

logger = logging.getLogger(__name__)


def propose_deep_semantics(conn) -> int:
    """Run Tiers 2 (candidates), 2.5 (guards), and 3 (LLM) on entity labels."""
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
            "explanation": res["explanation"]
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
                logger.warning(f"🚨 TRANSITIVITY PARADOX 🚨: LLM rejected '{a}' == '{b}', but they were bridged transitively by other approvals in cluster {cluster_set}!")
                
    # Sort and output JSON
    final_proposals.sort(key=lambda x: x.get("confidence", 0.0))
    
    out_path = Path("data/processed/aliases_proposed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_proposals, f, indent=2)
        
    logger.info(f"Wrote {len(final_proposals)} adjudicated proposals to {out_path}")
    return len(final_proposals)


def build(conn) -> dict[str, int]:
    """Run the tiers over the current entities, writing proposals. Non-destructive."""
    ents = store.entities_with_kind(conn)
    results: dict[str, int] = {}

    # Tier 1 — deterministic normalization (separators, case, version dot)
    results["normalize"] = store.write_proposals(conn, normalize.propose(ents))

    # Tier 1.5 — deterministic US/UK spelling + data-driven plural
    results["spelling"] = store.write_proposals(conn, spelling.propose(ents))

    # Tiers 2 / 2.5 / 3 — Deep Semantics (outputs to JSON for human review)
    results["deep_semantics"] = propose_deep_semantics(conn)

    return results
