# =============================================================================
# HEPCoverageKG aliases: orchestration
#
# build()  -> run the enabled tiers, write proposals (nothing resolves yet)
# Tier 1 (normalize) is the only tier wired now; ngram / embed / llm slot in here.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import normalize, store


def build(conn) -> dict[str, int]:
    """Run the tiers over the current entities, writing proposals. Non-destructive."""
    ents = store.entities_with_kind(conn)
    results: dict[str, int] = {}

    # Tier 1 — deterministic normalization
    results["normalize"] = store.write_proposals(conn, normalize.propose(ents))

    # Tiers 2 / 2.5 / 3 slot in here (ngram, embed, llm) as they are built.
    return results
