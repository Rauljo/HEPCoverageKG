# =============================================================================
# HEPCoverageKG aliases: Tier 1.5 — deterministic spelling + plural
#
# Still exact-match (no fuzzy similarity), just a RICHER normalization applied
# per token: US/UK spelling (modelling->modeling, colour->color, hadronisation->
# hadronization) and data-driven singular/plural. It proposes a merge only when
# this richer key unites entities Tier 1 kept apart -- so it adds the safe
# residual (colour/color, dimension/dimensions) without the fuzzy-matching danger
# that would conflate w/z polarization, s/t channel, or version numbers.
#
# Plural is data-driven: a trailing 's' is dropped ONLY if the singular token
# actually occurs elsewhere in the corpus -- so 'dimensions'->'dimension' (both
# exist) but 'higgs'/'mass'/'analysis' are left alone. See backlog (2026-07-26).
# =============================================================================
from __future__ import annotations

import re
from typing import Iterable

from hepcoveragekg.aliases.normalize import Proposal, normalize_slug, slug_of

# UK -> US, whole-token. Explicit list = safe (no mid-word false positives like
# precise->precize). HEP-weighted.
_UK_US: dict[str, str] = {
    "modelling": "modeling", "modelled": "modeled", "modeller": "modeler",
    "labelling": "labeling", "labelled": "labeled",
    "colour": "color", "colours": "colors", "coloured": "colored",
    "flavour": "flavor", "flavours": "flavors", "flavoured": "flavored",
    "behaviour": "behavior", "neighbour": "neighbor",
    "normalise": "normalize", "normalised": "normalized", "normalising": "normalizing",
    "optimise": "optimize", "optimised": "optimized", "optimising": "optimizing",
    "parametrise": "parametrize", "parametrised": "parametrized",
    "factorise": "factorize", "factorised": "factorized",
    "hadronise": "hadronize", "hadronised": "hadronized",
    "polarise": "polarize", "polarised": "polarized",
    "centre": "center", "centred": "centered", "fibre": "fiber", "metre": "meter",
    "calibre": "caliber", "aluminium": "aluminum", "sulphur": "sulfur", "grey": "gray",
    "analogue": "analog", "catalogue": "catalog", "dialogue": "dialog",
    "defence": "defense", "licence": "license", "mould": "mold",
}


def _spell(token: str) -> str:
    """Canonicalize one token's spelling to US. Explicit map first, then two safe
    suffix rules (…isation→…ization, …yse→…yze)."""
    if token in _UK_US:
        return _UK_US[token]
    if token.endswith("isation"):
        return token[:-7] + "ization"
    if token.endswith("isations"):
        return token[:-8] + "izations"
    if token.endswith("yse"):
        return token[:-3] + "yze"
    if token.endswith("ysed"):
        return token[:-4] + "yzed"
    return token


_NON_PLURAL = ("ss", "us", "is", "as", "os", "ys")


def _singular(token: str, vocab: set[str]) -> str:
    """Drop a trailing 's' only if it looks like a plural AND the singular occurs
    in the corpus (so 'higgs'/'mass'/'analysis' are never mangled)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(_NON_PLURAL):
        if token[:-1] in vocab:
            return token[:-1]
    return token


def _tokens(entity_id: str) -> list[str]:
    return [t for t in re.split(r"[._\-\s]+", slug_of(entity_id).lower()) if t]


def spelling_key(entity_id: str, singular_vocab: set[str]) -> str:
    """Richer normalization key: US-spelled, de-pluralized tokens, joined."""
    return "".join(_singular(_spell(t), singular_vocab) for t in _tokens(entity_id))


def propose(entities: Iterable[tuple[str, str]]) -> list[Proposal]:
    """Propose merges only where the spelling/plural key unites entities that
    Tier-1 normalization left in different clusters (block by kind)."""
    entities = list(entities)

    # vocab of US-spelled tokens, so plural-stripping only merges pairs that exist
    vocab = {_spell(t) for e, _ in entities for t in _tokens(e)}

    groups: dict[tuple[str, str], list[str]] = {}
    for entity_id, kind in entities:
        groups.setdefault((kind, spelling_key(entity_id, vocab)), []).append(entity_id)

    proposals: list[Proposal] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # only a NEW bridge: members already share a Tier-1 slug -> skip (Tier 1 has it)
        if len({normalize_slug(m) for m in members}) < 2:
            continue
        anchor = min(members)
        for member in members:
            if member != anchor:
                proposals.append(Proposal(member, anchor, "spelling", 1.0))
    return proposals
