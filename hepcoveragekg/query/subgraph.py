"""
HEPCoverageKG query layer: the subgraph a run actually touched.

What the user sees after asking a question -- the corner of the graph the answer
was built from. system.md calls this the user's subgraph (§2.4); this is its
read-only ancestor, before editing exists.

The whole difficulty is deciding what NOT to draw. Measured on "how many
analyses used Pythia?": 56 entities retrieved, 367 assertions touching them, 339
distinct subjects pointing at them. Drawn naively that is ~395 nodes of
spaghetti -- the failure every graph browser demonstrates by default, and worse
than no picture at all, because it looks like insight.

Three reductions, in order of how much they buy:

  collapse canonical clusters.  56 Pythia entities are 31 distinct things once
      deduplication is applied. The node says "Pythia 8.2 (+3 spellings)", so
      the merge is visible rather than hidden -- consistent with S-28: a merge
      that changes what you see should say so.

  cap the far side by connectivity.  339 samples pointing at Pythia is not a
      picture. Keep the best-connected few and state plainly how many were left
      out; an honest "+299 more" beats an unreadable complete one.

  keep the predicate on the edge.  Without it the graph says two things are
      related, which is the least interesting fact available about them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_NODES = 36

# Enough to tell kinds apart at a glance; deliberately muted so the highlighted
# retrieved nodes stand out against them.
KIND_COLOURS = {
    "result": "#ffd8a8",
    "paper": "#d0bfff",
    "generator": "#a5d8ff",
    "sample": "#b2f2bb",
    "systematic_uncertainty": "#ffc9c9",
    "detector_object": "#99e9f2",
    "physics_process": "#eebefa",
    "event_region": "#ffec99",
    "observable": "#c0eb75",
    "bsm_model": "#fcc2d7",
    "background": "#e9ecef",
    "channel": "#bac8ff",
}
DEFAULT_COLOUR = "#f1f3f5"


@dataclass
class Node:
    node_id: str
    label: str
    kind: str
    members: int = 1      # how many entities collapsed into this one
    retrieved: bool = False  # was it in what the query returned


@dataclass
class Subgraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, predicate, dst)
    omitted_nodes: int = 0
    omitted_edges: int = 0

    def __len__(self) -> int:
        return len(self.nodes)


def _canonical_map(conn, entity_ids: list[str]) -> dict[str, str]:
    """entity id -> its cluster id, defaulting to itself."""
    if not entity_ids:
        return {}
    marks = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT entity_id, canonical_id FROM entity_canonical WHERE entity_id IN ({marks})",
        tuple(entity_ids),
    ).fetchall()
    mapping = {r["entity_id"]: r["canonical_id"] for r in rows}
    return {e: mapping.get(e, e) for e in entity_ids}


def _labels(conn, ids: list[str]) -> dict[str, tuple[str, str]]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {
        r["entity_id"]: (r["label"] or r["entity_id"], r["kind"] or "")
        for r in conn.execute(
            f"SELECT entity_id, label, kind FROM entity WHERE entity_id IN ({marks})",
            tuple(ids),
        )
    }


def build(conn, entity_ids: list[str], max_nodes: int = DEFAULT_MAX_NODES) -> Subgraph:
    """The neighbourhood of what a run retrieved, collapsed and capped."""
    graph = Subgraph()
    if not entity_ids:
        return graph

    clusters = _canonical_map(conn, entity_ids)
    members: dict[str, int] = {}
    for cluster in clusters.values():
        members[cluster] = members.get(cluster, 0) + 1

    marks = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        "SELECT DISTINCT a.subject_id AS subject_id, a.predicate AS predicate,"
        "       a.object_id AS object_id"
        f"  FROM assertion a WHERE a.object_id IN ({marks}) OR a.subject_id IN ({marks})",
        (*entity_ids, *entity_ids),
    ).fetchall()

    retrieved = set(clusters.values())

    # Score EVERY candidate by connectivity -- retrieved and neighbour alike --
    # rather than reserving the whole budget for the retrieved set.
    #
    # An earlier version did reserve it, and produced 36 nodes with 12 edges:
    # every Pythia cluster present, almost none of the samples that use them, so
    # most nodes sat isolated. A page of disconnected boxes is worse than no
    # picture, because it looks like the graph has no structure.
    #
    # Retrieved nodes get a bonus so they are preferred at equal connectivity,
    # but a retrieved node with no surviving neighbour loses to a sample that
    # ties five of them together.
    RETRIEVED_BONUS = 2
    degree: dict[str, int] = {}
    for r in rows:
        for side in (r["subject_id"], r["object_id"]):
            if side:
                node = clusters.get(side, side)
                degree[node] = degree.get(node, 0) + 1

    def score(node: str) -> tuple[int, str]:
        return (degree.get(node, 0) + (RETRIEVED_BONUS if node in retrieved else 0), node)

    candidates = set(degree) | retrieved
    keep = {n for n in sorted(candidates, key=score, reverse=True)[:max_nodes]}
    graph.omitted_nodes = max(0, len(candidates) - len(keep))

    wanted = list(keep)
    info = _labels(conn, wanted)
    for node_id in wanted:
        label, kind = info.get(node_id, (node_id, ""))
        graph.nodes.append(Node(
            node_id=node_id, label=label, kind=kind,
            members=members.get(node_id, 1),
            retrieved=node_id in retrieved,
        ))

    present = {n.node_id for n in graph.nodes}
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        src = clusters.get(r["subject_id"], r["subject_id"])
        dst = clusters.get(r["object_id"], r["object_id"])
        if not src or not dst or src not in present or dst not in present:
            graph.omitted_edges += 1
            continue
        key = (src, r["predicate"], dst)
        if key in seen:
            continue
        seen.add(key)
        graph.edges.append(key)

    # Drop anything left isolated. A node with no surviving edge tells the
    # reader nothing and crowds out what does -- better to say it was omitted.
    connected = {n for edge in graph.edges for n in (edge[0], edge[2])}
    isolated = [n for n in graph.nodes if n.node_id not in connected]
    if isolated and graph.edges:
        graph.nodes = [n for n in graph.nodes if n.node_id in connected]
        graph.omitted_nodes += len(isolated)

    return graph


def to_dot(graph: Subgraph, title: str = "") -> str:
    """Graphviz source. DOT rather than a JS library so it renders offline."""
    out = [
        "digraph {",
        "rankdir=LR; bgcolor=\"transparent\"; splines=true; overlap=false;",
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 '
        'color="#ced4da"];',
        'edge [color="#adb5bd" fontname="Helvetica" fontsize=8 fontcolor="#868e96"];',
    ]
    if title:
        out.append(f'label="{_esc(title)}"; labelloc=t; fontname="Helvetica"; fontsize=11;')

    for node in graph.nodes:
        label = _esc(_trim(node.label))
        if node.members > 1:
            label += f"\\n(+{node.members - 1} spellings)"
        colour = KIND_COLOURS.get(node.kind, DEFAULT_COLOUR)
        pen = ' penwidth=2 color="#ff4b4b"' if node.retrieved else ""
        out.append(f'"{node.node_id}" [label="{label}" fillcolor="{colour}"{pen}];')

    for src, predicate, dst in graph.edges:
        out.append(f'"{src}" -> "{dst}" [label="{_esc(predicate)}"];')

    out.append("}")
    return "\n".join(out)


def _trim(text: str, width: int = 34) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _esc(text: str) -> str:
    return str(text).replace("\\", "").replace('"', "'")
