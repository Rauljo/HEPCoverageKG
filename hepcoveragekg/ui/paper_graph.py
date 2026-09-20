"""The subgraph behind an answer, drawn as papers connected to entities.

The existing entity subgraph (`query.subgraph`) shows what the retrieval
touched. This one shows what the *answer* stands on: the papers it names,
the entities of those papers that bear on the question, and the predicate
that links each pair -- a region that requires an object, a sample that uses
a generator. Papers are the answer's unit, so they are the nodes a physicist
reads first.

Built from the assertion table, so it works for either agent: the typed
agent contributes the entities it retrieved, the free-SQL agent contributes
none, and the paper's own entities that match the question fill in either
way. Capped so it stays readable; the cap is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..query import answer_critic as AC
from ..query.subgraph import DEFAULT_COLOUR, KIND_COLOURS, _esc, _trim


@dataclass
class PaperGraph:
    papers: dict[str, str] = field(default_factory=dict)          # id -> title
    entities: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (label, kind)
    edges: list[tuple[str, str, str]] = field(default_factory=list)     # (paper, predicate, entity)
    omitted_entities: int = 0


def build(conn, question: str, papers: list[str], retrieved_entity_ids: list[str] = (),
          max_entities: int = 24) -> PaperGraph:
    g = PaperGraph()
    papers = [str(p) for p in papers][:40]
    if not papers:
        return g
    marks = ",".join("?" * len(papers))
    for pid, title in conn.execute(
            f"SELECT arxiv_id, title FROM paper WHERE arxiv_id IN ({marks})", papers):
        g.papers[pid] = title or pid

    terms = AC._question_terms(question)
    retrieved = {str(e) for e in retrieved_entity_ids}
    # Every (paper, predicate, entity) among the answer's papers, with the
    # entity's label and kind.
    rows = conn.execute(
        f"""SELECT a.paper_id, a.predicate, a.subject_id, a.object_id
            FROM assertion a WHERE a.paper_id IN ({marks})""", papers).fetchall()
    cand: dict[str, dict] = {}
    for pid, pred, subj, obj in rows:
        for eid in (subj, obj):
            if not eid or eid.startswith("hepkg:paper:"):
                continue
            c = cand.setdefault(eid, {"papers": set(), "edges": set()})
            c["papers"].add(pid)
            c["edges"].add((pid, pred))
    if not cand:
        return g
    ids = list(cand)
    lab: dict[str, tuple[str, str]] = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        m = ",".join("?" * len(chunk))
        for eid, label, kind in conn.execute(
                f"SELECT entity_id, label, kind FROM entity WHERE entity_id IN ({m})", chunk):
            lab[eid] = (label or eid, kind or "")
    # Score: retrieved by the run, matches the question, shared by several
    # of the answer's papers. Keep the best `max_entities`.
    def score(eid):
        label = lab.get(eid, (eid, ""))[0]
        return ((3 if eid in retrieved else 0)
                + 2 * min(AC._overlap(label, terms), 3)
                + min(len(cand[eid]["papers"]), 5))
    ranked = sorted(cand, key=lambda e: -score(e))
    keep = [e for e in ranked if score(e) > 0][:max_entities]
    g.omitted_entities = len(cand) - len(keep)
    for eid in keep:
        g.entities[eid] = lab.get(eid, (eid, ""))
        for pid, pred in sorted(cand[eid]["edges"]):
            g.edges.append((pid, pred, eid))
    return g


def to_dot(g: PaperGraph, retrieved_entity_ids: list[str] = ()) -> str:
    retrieved = {str(e) for e in retrieved_entity_ids}
    out = ["digraph {", 'rankdir=LR; bgcolor="transparent"; splines=true; overlap=false; nodesep=0.15;',
           'node [fontname="Helvetica" fontsize=9 style="rounded,filled" shape=box color="#ced4da"];',
           'edge [color="#adb5bd" fontname="Helvetica" fontsize=7 fontcolor="#868e96" arrowsize=0.5];']
    out.append("subgraph cluster_papers { label=\"papers in the answer\"; fontname=\"Helvetica\"; "
               "fontsize=10; color=\"#dee2e6\";")
    for pid, title in g.papers.items():
        out.append(f'"p:{pid}" [label="{_esc(pid)}\\n{_esc(_trim(title, 40))}" fillcolor="#fff3bf" '
                   f'color="#f59f00" penwidth=1.5];')
    out.append("}")
    for eid, (label, kind) in g.entities.items():
        colour = KIND_COLOURS.get(kind, DEFAULT_COLOUR)
        pen = ' penwidth=2 color="#ff4b4b"' if eid in retrieved else ""
        out.append(f'"{_esc(eid)}" [label="{_esc(_trim(label))}\\n({_esc(kind)})" fillcolor="{colour}"{pen}];')
    seen = set()
    for pid, pred, eid in g.edges:
        if (pid, eid) in seen:
            continue
        seen.add((pid, eid))
        out.append(f'"p:{pid}" -> "{_esc(eid)}" [label="{_esc(pred.replace("_", " "))}"];')
    out.append("}")
    return "\n".join(out)
