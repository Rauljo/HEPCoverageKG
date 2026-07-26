# =============================================================================
# HEPCoverageKG aliases: clustering + canonical pick (pure, no DB)
#
# same_as edges -> connected-component clusters -> one canonical representative
# per cluster. Deterministic, so re-running gives byte-identical results.
# =============================================================================
from __future__ import annotations

import collections
from typing import Iterable, Mapping, Optional


def connected_components(
    edges: Iterable[tuple[str, str]], nodes: Optional[Iterable[str]] = None
) -> list[set[str]]:
    """Union-find over the edges; returns clusters of size >= 2 (singletons omitted)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    for n in nodes or ():
        find(n)

    groups: dict[str, set[str]] = collections.defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    return [members for members in groups.values() if len(members) > 1]


def pick_canonical(
    members: Iterable[str],
    paper_counts: Mapping[str, int],
    overrides: Optional[Mapping[str, str]] = None,
) -> str:
    """Canonical = the member in the most papers; ties broken lexicographically.

    A human override for any member of the cluster wins outright, so the auto
    pick is never binding.
    """
    members = list(members)
    if overrides:
        for member in members:
            if member in overrides:
                return overrides[member]
    return sorted(members, key=lambda e: (-paper_counts.get(e, 0), e))[0]
