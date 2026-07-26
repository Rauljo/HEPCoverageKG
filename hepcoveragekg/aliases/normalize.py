# =============================================================================
# HEPCoverageKG aliases: Tier 1 — deterministic slug normalization
#
# The cheap, safe tier. Two entities of the SAME kind whose id-slugs match once
# hyphens/underscores/case are folded are the same concept
# (b_jet / b-jet / bjet). Digits are preserved, so version-numbered ids stay
# apart automatically (sherpa_2_2_1 vs sherpa_2_2_2) — the fuzzy version-guard
# problem only appears at Tier 2. Blocking by kind prevents cross-kind merges.
# =============================================================================
from __future__ import annotations

import re
from typing import Iterable, NamedTuple


class Proposal(NamedTuple):
    entity_id_a: str
    entity_id_b: str
    method: str
    score: float


def slug_of(entity_id: str) -> str:
    """Everything after the namespace: hepkg:object:b_jet -> 'b_jet'."""
    parts = entity_id.split(":", 2)
    return parts[2] if len(parts) >= 3 else entity_id


def normalize_slug(entity_id: str) -> str:
    """Fold case and strip separators (including the '.' version dot), keeping
    digits: 'b-Jet' -> 'bjet', 'sherpa_2_2_1' -> 'sherpa221',
    'pythia8.210' / 'pythia8-210' / 'pythia8210' -> 'pythia8210'.

    Digits are preserved, so distinct version numbers never merge
    (8.212 -> 8212 != 8.230 -> 8230). Note: the 'p'-as-decimal convention in
    energies (2p76tev = 2.76 TeV) is a Tier-2 rule, not handled here.
    """
    return re.sub(r"[.\-_\s]+", "", slug_of(entity_id).lower())


def propose(entities: Iterable[tuple[str, str]]) -> list[Proposal]:
    """entities = (entity_id, kind). Group by (kind, normalized slug); any group
    with >1 member yields star edges to the lexicographically-smallest member
    (a deterministic anchor — enough to connect the cluster)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for entity_id, kind in entities:
        groups.setdefault((kind, normalize_slug(entity_id)), []).append(entity_id)

    proposals: list[Proposal] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        anchor = min(members)
        for member in members:
            if member != anchor:
                proposals.append(Proposal(member, anchor, "normalize", 1.0))
    return proposals
