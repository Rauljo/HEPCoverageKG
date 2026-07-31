"""
HEPCoverageKG — ask the coverage map a question.

Two audiences, one page, and the split matters:

  a physicist wants the question answered and the evidence shown. Not spans,
      not token counts. So the answer and its quotes come first, in plain
      language, and everything else is behind a toggle.

  we want to see what the system DID -- which tools, in what order, what it
      reasoned, what it cost, and whether every claim traced back to a row.
      That is the debugging view, and it is off by default.

The graph redraws with the current node highlighted as the run proceeds. It is
generated from the compiled graph, so it cannot drift from the code, and it is
the clearest way to show a non-technical reader that the system is *looking
things up* rather than answering from memory.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sqlite3
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:8000/v1")

DB = "data/processed/hepkg.db"
INDEX_CACHE = "data/processed/retrieval_index.npz"

st.set_page_config(page_title="HEPCoverageKG", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# loading (cached across reruns -- Streamlit re-executes the whole file on
# every interaction, so an uncached index rebuild would cost 20s per keypress)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the knowledge graph…")
def load():
    from hepcoveragekg.query import retrieve, templates
    conn = templates.read_only(DB)
    return conn, retrieve.build(conn, cache=INDEX_CACHE)


@st.cache_data(show_spinner=False)
def graph_stats():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    return {
        "papers": q("SELECT COUNT(*) FROM paper"),
        "entities": q("SELECT COUNT(*) FROM entity"),
        "assertions": q("SELECT COUNT(*) FROM assertion"),
        "quotes": q("SELECT COUNT(*) FROM evidence"),
    }


def dot(active: str | None) -> str:
    """The planner graph, with the node currently running picked out.

    Written as DOT rather than the Mermaid LangGraph emits, because Streamlit
    renders DOT natively -- no CDN, so it works offline on the cluster.
    """
    nodes = ["plan", "execute", "nudge", "finish"]
    lines = ['digraph {', 'rankdir=LR;', 'bgcolor="transparent";',
             'node [shape=box style="rounded,filled" fontname="Helvetica" '
             'fillcolor="#eeeeee" color="#bbbbbb" fontcolor="#333333"];',
             'edge [color="#999999" fontname="Helvetica" fontsize=9];',
             'start [shape=circle label="" width=0.2 fillcolor="#666666"];',
             'done [shape=doublecircle label="" width=0.2 fillcolor="#666666"];']
    for n in nodes:
        if n == active:
            lines.append(f'{n} [fillcolor="#ff4b4b" fontcolor="white" color="#ff4b4b" penwidth=2];')
        else:
            lines.append(f'{n};')
    lines += [
        'start -> plan;',
        'plan -> execute [label="tool calls"];',
        'plan -> nudge [label="nothing retrieved"];',
        'plan -> finish [label="answered"];',
        'nudge -> plan;',
        'execute -> plan [label="keep going"];',
        'execute -> finish [label="answer"];',
        'finish -> done;',
        '}',
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# what a step looks like to a reader
# --------------------------------------------------------------------------

def describe_step(step) -> str:
    """One tool call, in words rather than JSON."""
    a = step.args
    if step.tool == "search":
        kind = f" ({a['kind']})" if a.get("kind") else ""
        return f"Looked up **{a.get('text', '')}**{kind} — found {step.rows} entities"
    if step.tool == "count":
        return f"Counted analyses linked by **{a.get('predicate', '')}**"
    if step.tool in ("subjects_of", "list"):
        return f"Followed **{a.get('predicate', '')}** back to what uses it — {step.rows} results"
    if step.tool == "papers_of":
        return f"Resolved to papers — {step.rows} found"
    if step.tool == "describe":
        return f"Read everything attached to those entities — {step.rows} facts"
    if step.tool == "crosstab":
        return f"Built a coverage grid: {a.get('predicate_a')} × {a.get('predicate_b')}"
    if step.tool == "quotes":
        return "Fetched the sentence a fact came from"
    return f"{step.tool} — {step.rows} rows"


def quote_rows(conn, evidence_ids, limit=8):
    if not evidence_ids:
        return []
    marks = ",".join("?" * len(evidence_ids[:200]))
    return conn.execute(
        f"SELECT quote, section_title FROM evidence WHERE evidence_id IN ({marks})"
        " AND LENGTH(quote) > 60 LIMIT ?",
        (*evidence_ids[:200], limit),
    ).fetchall()


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

conn, index = load()
stats = graph_stats()

st.title("🔎 HEPCoverageKG")
st.caption(
    f"A coverage map of {stats['papers']} ATLAS/CMS papers — "
    f"{stats['entities']:,} entities, {stats['assertions']:,} facts, "
    f"{stats['quotes']:,} verbatim quotes. "
    "Every answer is built from what the papers actually say."
)

with st.sidebar:
    st.subheader("Settings")
    show_detail = st.toggle("Show what the system did", value=False,
                            help="Tool calls, reasoning, cost, and the faithfulness check")
    max_rounds = st.slider("Max rounds", 2, 10, 6,
                           help="How many times the planner may think before stopping")
    max_places = st.slider("Max operations per round", 1, 8, 8,
                           help="Set to 1 to force it to work one step at a time")
    minimal_prompt = st.toggle("Minimal instructions", value=False,
                               help="Give the model only the facts it cannot infer")
    st.divider()
    st.caption("**Ask things like**")
    for example in ("How many analyses used Pythia?",
                    "Which analyses searched for top squarks?",
                    "What systematic uncertainties are recorded?",
                    "Which processes were studied at 13 TeV?"):
        st.caption(f"· {example}")

question = st.text_input("Ask the coverage map", placeholder="How many analyses used Pythia?")

if question:
    graph_box = st.empty()
    steps_box = st.container()
    seen = 0
    session = None

    from hepcoveragekg.query import planner

    graph_box.graphviz_chart(dot("plan"))
    with steps_box:
        progress = st.status("Working…", expanded=True)

    try:
        for node, session in planner.stream(
            conn, index, question,
            max_rounds=max_rounds, max_places=max_places,
            minimal_prompt=minimal_prompt,
        ):
            graph_box.graphviz_chart(dot(node))
            for step in session.steps[seen:]:
                progress.write(("⚠️ " if step.error else "✅ ") + describe_step(step))
            seen = len(session.steps)
        progress.update(label=f"Done in {session.seconds:.0f}s", state="complete",
                        expanded=False)
    except Exception as exc:  # noqa: BLE001 -- a dead server must not blank the page
        progress.update(label="Failed", state="error")
        st.error(f"Could not reach the model: {exc}\n\n"
                 "Is the vLLM server running and the SSH tunnel open?")
        st.stop()

    graph_box.graphviz_chart(dot(None))

    # -- the answer ------------------------------------------------------
    if session.answerable:
        st.success(session.answer or "_no answer produced_")
    else:
        st.warning(session.answer or "_no answer produced_")
        st.caption("The graph could not answer this — which is itself information "
                   "about what the literature covers." if session.reason == "not_in_graph"
                   else "This question is outside what these papers cover.")

    # -- the graph it was built from -------------------------------------
    #
    # The entities behind the answer, drawn. This is the read-only ancestor of
    # the user subgraph in system.md 2.4, and for a non-technical reader it is
    # the most convincing artefact on the page: it shows the answer standing on
    # named things in the literature rather than on the model's memory.
    from hepcoveragekg.query import subgraph as SG

    all_ids = sorted({i for ids in session.sets.values() for i in ids})
    if all_ids:
        sg = SG.build(conn, all_ids)
        if sg.nodes:
            caption = f"{len(sg.nodes)} of {len(sg.nodes) + sg.omitted_nodes} entities"
            with st.expander(f"🕸️ What the answer is built on — {caption}",
                             expanded=not show_detail):
                st.graphviz_chart(SG.to_dot(sg))
                st.caption(
                    "Red outline = matched your question. Others = what they connect to. "
                    "Entities merged by deduplication are drawn once, with their spelling "
                    "count." +
                    (f" {sg.omitted_nodes} less-connected entities omitted to keep this "
                     "readable." if sg.omitted_nodes else ""))

    # -- the evidence ----------------------------------------------------
    rows = quote_rows(conn, session.evidence_ids)
    if rows:
        with st.expander(f"📄 Evidence — {len(session.evidence_ids)} quotes support this",
                         expanded=not show_detail):
            for quote, section in rows:
                st.markdown(f"> {quote}")
                st.caption(f"— {section or 'unattributed section'}")

    # -- the debugging view ----------------------------------------------
    if show_detail:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rounds", session.rounds)
        c2.metric("Model calls", session.llm_calls)
        c3.metric("Tokens", f"{session.prompt_tokens + session.completion_tokens:,}")
        c4.metric("Seconds", f"{session.seconds:.0f}")

        v = session.verification
        if v is not None:
            if v.checked and not v.unsupported:
                st.success(f"✅ Every claim traced to a retrieved row ({v.checked} checked)")
            elif v.unsupported:
                st.error(f"⚠️ {len(v.unsupported)} claim(s) not supported by anything retrieved")
                for claim in v.unsupported:
                    st.caption(f"· `{claim.text}` — {claim.note}")
            else:
                st.info("No checkable claims in the answer")

        if session.thoughts:
            with st.expander("🧠 What it was thinking"):
                for t in session.thoughts:
                    st.markdown(f"**Round {t.round}** → `{', '.join(t.tools_called)}`")
                    st.caption(t.text)

        with st.expander("🔧 Tool calls"):
            st.json([{"round": s.round, "tool": s.tool, "args": s.args,
                      "rows": s.rows, "error": s.error, "seconds": round(s.seconds, 3)}
                     for s in session.steps])

        flags = []
        if session.invented_ids:
            flags.append(f"invented ids rejected: {session.invented_ids}")
        if session.recovered_calls:
            flags.append(f"tool calls recovered from text: {session.recovered_calls}")
        if session.nudged:
            flags.append("nudged — it tried to answer before retrieving anything")
        if not session.grounded_in_tools:
            flags.append("**answered without retrieving anything**")
        if flags:
            st.warning("  \n".join("· " + f for f in flags))

        if session.sets:
            st.caption("Sets held: " +
                       ", ".join(f"`{k}` ({len(v)})" for k, v in session.sets.items()))
