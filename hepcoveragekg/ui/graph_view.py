"""An interactive graph behind an answer: drag, zoom, click to expand.

Rendered with vis-network in the browser, the same library the Neo4j browser
family of tools uses, so the graph behaves the way a physicist who has seen
one expects: nodes can be dragged, the view zoomed, and a click on a node
reveals what it connects to.

Two kinds of expansion, both served from data embedded in the page so a
click costs no round trip:

  click a paper    -> the rest of that paper's entities that bear on the
                      question (beyond the ones shown at first)
  click an entity  -> the other papers in the corpus that carry it, which is
                      the coverage question in miniature: who else uses this?

Everything is capped and the caps are stated, because a 60-paper corpus
with 5,000 entities would otherwise be one grey ball.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field

from ..query import answer_critic as AC
from ..query.subgraph import DEFAULT_COLOUR, KIND_COLOURS

VISIBLE_ENTITIES = 20        # shown at first
HIDDEN_ENTITIES = 120        # revealed by clicking a paper
OTHER_PAPERS_PER_ENTITY = 10 # revealed by clicking an entity
VIS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


@dataclass
class GraphData:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    n_papers: int = 0
    n_entities_visible: int = 0
    n_entities_hidden: int = 0
    n_other_papers: int = 0


def _label(text: str, width: int = 28) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    words, out = text.split(), ""
    for w in words:
        if len(out) + len(w) + 1 > width * 2:
            return out.rstrip() + "…"
        out += w + " "
        if len(out.rstrip()) > width and "\n" not in out:
            out = out.rstrip() + "\n"
    return out.rstrip()


def build(conn, question: str, papers: list[str], retrieved_entity_ids: list[str] = ()) -> GraphData:
    g = GraphData()
    papers = [str(p) for p in papers][:40]
    if not papers:
        return g
    marks = ",".join("?" * len(papers))
    titles = {pid: (t or pid) for pid, t in conn.execute(
        f"SELECT arxiv_id, title FROM paper WHERE arxiv_id IN ({marks})", papers)}
    terms = AC._question_terms(question)
    retrieved = {str(e) for e in retrieved_entity_ids}

    # every (paper, predicate, entity) among the answer's papers
    rows = conn.execute(
        f"SELECT a.paper_id, a.predicate, a.subject_id, a.object_id FROM assertion a "
        f"WHERE a.paper_id IN ({marks})", papers).fetchall()
    cand: dict[str, dict] = {}
    for pid, pred, subj, obj in rows:
        for eid in (subj, obj):
            if not eid or eid.startswith("hepkg:paper:"):
                continue
            c = cand.setdefault(eid, {"papers": set(), "edges": {}})
            c["papers"].add(pid)
            c["edges"].setdefault(pid, pred)
    if not cand:
        return g
    ids = list(cand)
    info: dict[str, tuple[str, str]] = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        m = ",".join("?" * len(chunk))
        for eid, label, kind in conn.execute(
                f"SELECT entity_id, label, kind FROM entity WHERE entity_id IN ({m})", chunk):
            info[eid] = (label or eid, kind or "")

    def score(eid):
        label = info.get(eid, (eid, ""))[0]
        return ((3 if eid in retrieved else 0)
                + 2 * min(AC._overlap(label, terms), 3)
                + min(len(cand[eid]["papers"]), 5))
    ranked = sorted(cand, key=lambda e: -score(e))
    visible = [e for e in ranked if score(e) > 0][:VISIBLE_ENTITIES]
    hidden = [e for e in ranked if e not in set(visible)][:HIDDEN_ENTITIES]

    # paper nodes (the answer)
    for pid in papers:
        g.nodes.append({"id": f"p:{pid}", "label": pid, "group": "paper",
                        "title": html.escape(titles.get(pid, pid)), "shape": "box",
                        "color": {"background": "#fff3bf", "border": "#f59f00"},
                        "font": {"size": 15, "face": "Helvetica"}, "mass": 3,
                        "borderWidth": 2, "hidden": False})
    g.n_papers = len(papers)

    def entity_node(eid, hid):
        label, kind = info.get(eid, (eid, ""))
        colour = KIND_COLOURS.get(kind, DEFAULT_COLOUR)
        border = "#ff4b4b" if eid in retrieved else "#adb5bd"
        return {"id": eid, "label": _label(label), "group": kind or "entity",
                "title": html.escape(f"{label}\n[{kind}] in {len(cand[eid]['papers'])} of the answer's papers"
                                     + ("\nretrieved by the run" if eid in retrieved else "")),
                "shape": "ellipse", "color": {"background": colour, "border": border},
                "borderWidth": 3 if eid in retrieved else 1,
                "font": {"size": 12, "face": "Helvetica"}, "hidden": hid, "kind": kind}

    for eid in visible:
        g.nodes.append(entity_node(eid, False))
    for eid in hidden:
        g.nodes.append(entity_node(eid, True))
    g.n_entities_visible, g.n_entities_hidden = len(visible), len(hidden)

    for eid in visible + hidden:
        hid = eid not in set(visible)
        for pid, pred in cand[eid]["edges"].items():
            g.edges.append({"from": f"p:{pid}", "to": eid, "title": pred.replace("_", " "),
                            "label": "", "hidden": hid,
                            "color": {"color": "#adb5bd"}, "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}})

    # other papers in the corpus carrying a visible entity (revealed on click)
    shown = set(papers)
    other: dict[str, set] = {}
    vis_ids = visible[:VISIBLE_ENTITIES]
    if vis_ids:
        m = ",".join("?" * len(vis_ids))
        for pid, eid in conn.execute(
                f"SELECT DISTINCT a.paper_id, e FROM (SELECT paper_id, subject_id AS e FROM assertion "
                f"UNION ALL SELECT paper_id, object_id FROM assertion) a WHERE e IN ({m})", vis_ids):
            if pid and pid not in shown:
                other.setdefault(eid, set()).add(pid)
    extra_titles: dict[str, str] = {}
    extra = sorted({p for s in other.values() for p in sorted(s)[:OTHER_PAPERS_PER_ENTITY]})
    if extra:
        m = ",".join("?" * len(extra))
        extra_titles = {pid: (t or pid) for pid, t in conn.execute(
            f"SELECT arxiv_id, title FROM paper WHERE arxiv_id IN ({m})", extra)}
    for pid in extra:
        g.nodes.append({"id": f"p:{pid}", "label": pid, "group": "other-paper",
                        "title": html.escape("Not in the answer: " + extra_titles.get(pid, pid)),
                        "shape": "box", "color": {"background": "#f8f9fa", "border": "#ced4da"},
                        "font": {"size": 12, "face": "Helvetica", "color": "#495057"},
                        "hidden": True})
    for eid, s in other.items():
        for pid in sorted(s)[:OTHER_PAPERS_PER_ENTITY]:
            g.edges.append({"from": f"p:{pid}", "to": eid, "title": "also carries", "hidden": True,
                            "dashes": True, "color": {"color": "#ced4da"}})
    g.n_other_papers = len(extra)
    return g


_TEMPLATE = """
<div id="legend" style="font: 11px Helvetica, sans-serif; color:#495057; margin: 0 0 6px 0;">
  <span style="background:#fff3bf;border:2px solid #f59f00;padding:1px 6px;border-radius:3px;">paper in the answer</span>
  &nbsp;<span style="background:#f8f9fa;border:1px solid #ced4da;padding:1px 6px;border-radius:3px;">other paper (click an entity)</span>
  &nbsp;<span style="border:3px solid #ff4b4b;padding:1px 6px;border-radius:12px;">retrieved by the run</span>
  &nbsp; drag to move · scroll to zoom · click a paper for more of its entities · click an entity for the other papers that carry it · double-click to hide again
  <span id="stats" style="float:right;"></span>
</div>
<div id="net" style="width:100%; height:__HEIGHT__px; border:1px solid #e9ecef; border-radius:6px; background:#ffffff;"></div>
<script src="__CDN__"></script>
<script>
  const NODES = __NODES__;
  const EDGES = __EDGES__;
  const nodes = new vis.DataSet(NODES);
  const edges = new vis.DataSet(EDGES.map((e, i) => Object.assign({id: "e" + i}, e)));
  const container = document.getElementById("net");
  const network = new vis.Network(container, {nodes, edges}, {
    physics: {solver: "forceAtlas2Based",
              forceAtlas2Based: {gravitationalConstant: -60, springLength: 120, springConstant: 0.05, avoidOverlap: 0.8},
              stabilization: {iterations: 300, fit: true}},
    interaction: {hover: true, tooltipDelay: 120, multiselect: false, navigationButtons: true, keyboard: false},
    edges: {smooth: {type: "continuous"}, font: {size: 9, color: "#868e96"}, width: 1},
    nodes: {shapeProperties: {borderRadius: 6}}
  });
  function connected(id) {
    return edges.get({filter: e => e.from === id || e.to === id});
  }
  function reveal(id) {
    const es = connected(id);
    const ids = [];
    es.forEach(e => { const other = e.from === id ? e.to : e.from; ids.push(other); });
    nodes.update(ids.map(n => ({id: n, hidden: false})));
    edges.update(es.map(e => ({id: e.id, hidden: false})));
    stats();
  }
  function collapse(id) {
    const es = connected(id);
    es.forEach(e => {
      const other = e.from === id ? e.to : e.from;
      const stillLinked = edges.get({filter: x => !x.hidden && x.id !== e.id && (x.from === other || x.to === other)});
      const n = nodes.get(other);
      if (n && n.group !== "paper" && stillLinked.length === 0) { nodes.update({id: other, hidden: true}); }
      if (n && n.group !== "paper") { edges.update({id: e.id, hidden: true}); }
    });
    stats();
  }
  function stats() {
    const vn = nodes.get({filter: n => !n.hidden}).length;
    const hn = nodes.get({filter: n => n.hidden}).length;
    document.getElementById("stats").textContent = vn + " shown, " + hn + " more on click";
  }
  // The layout settles once and then stays put: physics runs only while the
  // initial layout stabilises and, briefly, to place newly revealed nodes.
  // Left on, the solver keeps every node trembling for ever.
  function freeze() { network.setOptions({physics: {enabled: false}}); }
  function settle() {
    network.setOptions({physics: {enabled: true}});
    network.stabilize(120);
    network.once("stabilized", freeze);
    setTimeout(freeze, 1500);
  }
  network.on("click", p => { if (p.nodes.length) { reveal(p.nodes[0]); settle(); } });
  network.on("doubleClick", p => { if (p.nodes.length) collapse(p.nodes[0]); });
  network.once("stabilizationIterationsDone", () => { freeze(); network.fit({animation: false}); });
  stats();
</script>
"""


def to_html(g: GraphData, height: int = 560) -> str:
    return (_TEMPLATE.replace("__NODES__", json.dumps(g.nodes))
            .replace("__EDGES__", json.dumps(g.edges))
            .replace("__HEIGHT__", str(height)).replace("__CDN__", VIS_CDN))
