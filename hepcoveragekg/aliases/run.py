# =============================================================================
# HEPCoverageKG aliases: orchestration
#
# build()  -> run the enabled tiers, write proposals (nothing resolves yet)
# Tiers 1 + 1.5 (deterministic) are wired now; ngram / embed / llm slot in here.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import normalize, spelling, store


def build(conn) -> dict[str, int]:
    """Run the tiers over the current entities, writing proposals. Non-destructive."""
    ents = store.entities_with_kind(conn)
    results: dict[str, int] = {}

    # Tier 1 — deterministic normalization (separators, case, version dot)
    results["normalize"] = store.write_proposals(conn, normalize.propose(ents))

    # Tier 1.5 — deterministic US/UK spelling + data-driven plural
    results["spelling"] = store.write_proposals(conn, spelling.propose(ents))

    # Tiers 2 / 2.5 / 3 slot in here (ngram, embed, llm) as they are built.
    return results
