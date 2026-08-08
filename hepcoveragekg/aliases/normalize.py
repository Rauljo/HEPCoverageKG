# =============================================================================
# HEPCoverageKG aliases: Tier 1 — deterministic normalization
#
# The cheap, safe tier. Two entities of the SAME kind are the same concept when
# either their id-slugs or their labels match once case and separators are
# folded. Digits are preserved, so version-numbered entities stay apart
# automatically (sherpa_2_2_1 vs sherpa_2_2_2) — the fuzzy version-guard problem
# only appears at Tier 2. Blocking by kind prevents cross-kind merges.
#
# Two independent keys, because they miss different things (D-050):
#
#   the ID key catches variant spellings of one id — b_jet / b-jet / bjet. It is
#       blind whenever two papers chose unrelated ids for the same concept.
#   the LABEL key catches exactly that. Seven entities carry the byte-identical
#       label "MadGraph5_aMC@NLO" under the ids madgraph5_amcnlo, mg5-amcnlo and
#       madgraph5-amcatnlo; the id key folds them into four separate clusters,
#       because the ids genuinely differ. Measured before this rule existed: 41
#       groups of byte-identical labels split across clusters.
#
# Under-merging is not the safe direction. It inflates every "how many distinct
# X" answer, which is the coverage map's headline number — so the free merges
# (identical strings, no judgement) are taken, and only the genuinely ambiguous
# cases are left for review.
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


def normalize_label(label: str) -> str:
    """Fold case and strip separators from a free-text label:
    'B-Tagged Jet' / 'b tagged jet' / 'b_tagged_jet' -> 'btaggedjet'.

    The decimal point is deliberately KEPT, unlike normalize_slug. Slugs are
    short identifiers where 'pythia8.210' and 'pythia8210' are the same thing,
    but labels carry measurements, and stripping the dot would fold
    'pT > 2.0 GeV' onto 'pT > 20 GeV' — a false merge that changes a cut value.
    """
    return re.sub(r"[\-_\s]+", "", label.strip().lower())


def _star_edges(groups: dict, method: str, skip: set[frozenset] | None = None) -> list[Proposal]:
    """Star edges from every member of a >1 group to its lexicographically
    smallest member (a deterministic anchor — enough to connect the cluster).
    Pairs already present in `skip` are not re-emitted under a weaker method.
    """
    proposals: list[Proposal] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        anchor = min(members)
        for member in members:
            if member == anchor:
                continue
            if skip is not None and frozenset((member, anchor)) in skip:
                continue
            proposals.append(Proposal(member, anchor, method, 1.0))
    return proposals


def propose(entities: Iterable[tuple[str, str]]) -> list[Proposal]:
    """entities = (entity_id, kind). Group by (kind, normalized id-slug)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for entity_id, kind in entities:
        groups.setdefault((kind, normalize_slug(entity_id)), []).append(entity_id)
    return _star_edges(groups, "normalize")


def propose_labels(entities: Iterable[tuple[str, str, str]]) -> list[Proposal]:
    """entities = (entity_id, kind, label). Two rules, strongest first.

    `label_exact` — same kind, byte-identical label. Cannot be wrong: identical
        strings denote the same concept or the kind itself is broken.
    `label_norm`  — same kind, labels equal after folding case and separators.
        Emitted only for pairs `label_exact` did not already cover, so the two
        counts read as "exact found N, folding found M more" rather than
        double-counting one edge under two methods.
    """
    entities = list(entities)

    exact: dict[tuple[str, str], list[str]] = {}
    folded: dict[tuple[str, str], list[str]] = {}
    for entity_id, kind, label in entities:
        if not label or not label.strip():
            continue
        exact.setdefault((kind, label), []).append(entity_id)
        folded.setdefault((kind, normalize_label(label)), []).append(entity_id)

    exact_edges = _star_edges(exact, "label_exact")
    seen = {frozenset((p.entity_id_a, p.entity_id_b)) for p in exact_edges}
    return exact_edges + _star_edges(folded, "label_norm", skip=seen)
