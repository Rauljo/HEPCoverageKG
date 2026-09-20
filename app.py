"""
HEPCoverageKG -- ask the coverage map, in conversation.

What the page does, and why each part is there:

  a physicist asks a question and gets an answer in prose with the list of
      papers it commits to. Under each paper: the sentences from that paper
      that support it, with their section, looked up in the graph the way the
      answer critic reads them. That is the provenance requirement delivered
      to the user rather than measured in a run.

  the model, the agent and the effort are the ones the results chapter kept.
      Three planners; standard effort is constrained decoding with the answer
      critic, high effort chains sub-goals. Which agent a model runs by default
      follows the chapter: the typed agent for the two 32B models, the free-SQL
      agent for Qwen3.8-flash.

  a follow-up is first offered the earlier answers. A one-call router either
      answers from them or asks for a retrieval, and a retrieval is given the
      earlier turns as context the way a chain leg is given its earlier steps.

  the graph the answer stands on is drawn as papers connected to entities,
      and what the system did -- rounds, calls, tokens, the estimated cost at
      list prices, and the judge's verdicts -- is behind a toggle.

Run:  streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time

# Quiet the libraries before they are imported. The embedding model is cached
# locally, so the Hub need not be asked (and would warn about it); its weight
# loading bar and the clients' per-request INFO lines belong in a run log, not
# on the terminal behind a demo. The judge's call ladder logs a WARNING every
# time its first rung is refused, which is normal, so the package sits at ERROR.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _name, _level in (("httpx", logging.WARNING), ("httpcore", logging.WARNING),
                      ("openai", logging.WARNING), ("hepcoveragekg", logging.ERROR),
                      ("sentence_transformers", logging.ERROR), ("transformers", logging.ERROR),
                      ("huggingface_hub", logging.ERROR)):
    logging.getLogger(_name).setLevel(_level)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
    _hf_logging.set_verbosity_error()
except Exception:  # noqa: BLE001 -- transformers is optional here
    pass

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB = "data/processed/hepkg.db"
INDEX_CACHE = "data/processed/retrieval_index.npz"

st.set_page_config(page_title="HEPCoverageKG", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# loading (cached: Streamlit re-runs the whole file on every interaction)
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
    return {"papers": q("SELECT COUNT(*) FROM paper"),
            "entities": q("SELECT COUNT(*) FROM entity"),
            "assertions": q("SELECT COUNT(*) FROM assertion"),
            "quotes": q("SELECT COUNT(*) FROM evidence")}


@st.cache_data(ttl=60, show_spinner=False)
def endpoint_status(label: str) -> tuple[bool, str]:
    from hepcoveragekg.ui import models as M
    return M.endpoint_ok(M.MODELS[label])


def describe_step(step: dict) -> str:
    """One tool call, in words rather than JSON."""
    a = step.get("args") or {}
    tool, rows = step.get("tool"), step.get("rows")
    if tool == "search":
        kind = f" ({a['kind']})" if a.get("kind") else ""
        return f"Looked up **{a.get('text', '')}**{kind} — {rows} entities"
    if tool == "sql":
        return f"Ran a query — {rows} rows"
    if tool == "count":
        return f"Counted analyses linked by **{a.get('predicate', '')}**"
    if tool in ("subjects_of", "list"):
        return f"Followed **{a.get('predicate', '')}** back to what uses it — {rows} results"
    if tool == "papers_of":
        return f"Resolved to papers — {rows} found"
    if tool == "contents_of":
        return f"Read what {len(a.get('paper_ids', []) or [])} paper(s) contain — {rows} facts"
    if tool == "facets":
        return f"Took the papers tagged **{', '.join(a.get('values', []) or [])}** — {rows}"
    if tool == "describe":
        return f"Read everything attached to those entities — {rows} facts"
    if tool == "quotes":
        return "Fetched the sentence a fact came from"
    return f"{tool} — {rows} rows"


# --------------------------------------------------------------------------
# sidebar: model, agent, effort
# --------------------------------------------------------------------------

from hepcoveragekg.ui import models as M  # noqa: E402

conn, index = load()
stats = graph_stats()

with st.sidebar:
    st.subheader("Model")
    model_label = st.selectbox("Planner", list(M.MODELS), index=1,
                               help="The three models the evaluation measured.")
    spec = M.MODELS[model_label]
    ok, why = endpoint_status(model_label)
    (st.success if ok else st.error)(f"{'Endpoint ' + why if ok else 'Endpoint: ' + why}", icon="🟢" if ok else "🔴")
    st.caption(spec.note)

    st.subheader("Effort")
    if len(spec.efforts) == 1:
        effort = spec.efforts[0]
        st.caption("One level for this model: chaining adds nothing to it.")
    else:
        effort = st.radio("How hard should it work?", spec.efforts, index=0,
                          format_func=lambda e: {"standard": "Standard", "high": "High (chained sub-goals)"}[e],
                          help="\n\n".join(f"**{k}**: {v}" for k, v in M.EFFORT_HELP.items()))
    if effort == "high":
        st.caption(M.EFFORT_HELP["high"])

    st.subheader("Agent")
    agent_choice = st.radio("Which way of querying the graph?",
                            ["recommended", "typed", "freesql"], index=0,
                            format_func=lambda a: {"recommended": f"Recommended for this model ({spec.default_agent})",
                                                   "typed": "Typed tools", "freesql": "Free SQL"}[a],
                            help="Typed: twelve fixed operations over the graph, the walk the "
                                 "chapter analyses; it is the agent whose trace can be checked. "
                                 "Free SQL: the model writes its own queries; cheaper, and the "
                                 "better choice on Qwen3.8-flash.")
    agent = spec.default_agent if agent_choice == "recommended" else agent_choice
    if agent == "freesql" and effort == "high":
        st.caption("Chaining is a typed-agent mechanism; free SQL runs at standard effort.")
        effort = "standard"

    with st.expander("Advanced"):
        max_rounds = st.slider("Max rounds", 2, 16, spec.max_rounds,
                               help="What the chapter's runs allowed this model.")
        show_detail = st.toggle("Show what the system did", value=False)
        show_graph = st.toggle("Draw the graph behind each answer", value=True)
    st.divider()
    turns = st.session_state.setdefault("turns", [])
    spent = sum(t.cost for t in turns)
    st.caption(f"Conversation: {len(turns)} turn(s), about ${spent:.4f} at list prices")
    if st.button("Start a new conversation", use_container_width=True):
        st.session_state["turns"] = []
        st.rerun()
    st.caption("**Ask things like**")
    for ex in ("Which analyses use b-tagged jets in their event selection?",
               "How many analyses used Pythia?",
               "Which of those also require missing transverse momentum?",
               "What backgrounds does 2001.06899 estimate?"):
        st.caption(f"· {ex}")


# --------------------------------------------------------------------------
# rendering a turn
# --------------------------------------------------------------------------

def graph_frame(html_text: str, height: int) -> None:
    """Embed the interactive graph. `st.iframe` takes a file, so the page is
    written once per graph to a temp file keyed by its content; the old
    `components.html` is the fallback on Streamlit versions without it."""
    import hashlib
    import tempfile
    from pathlib import Path
    if hasattr(st, "iframe"):
        d = Path(tempfile.gettempdir()) / "hepcoveragekg_graphs"
        d.mkdir(exist_ok=True)
        f = d / (hashlib.sha1(html_text.encode()).hexdigest()[:16] + ".html")
        if not f.exists():
            f.write_text("<!doctype html><html><head><meta charset='utf-8'></head><body style='margin:0'>"
                         + html_text + "</body></html>", encoding="utf-8")
        st.iframe(f, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html_text, height=height, scrolling=False)


def render_turn(t, *, detail: bool, graph: bool):
    if t.error:
        st.error(f"The run failed: {t.error}")
        return
    if t.answered:
        st.markdown(t.text or "_no answer produced_")
    else:
        st.warning(t.text or "_no answer produced_")
        st.caption("The graph could not answer this, which is itself information about "
                   "what the literature covers.")

    tags = [t.model, {"typed": "typed tools", "freesql": "free SQL", "memory": "from earlier turns"}.get(t.agent, t.agent)]
    if t.effort == "high":
        tags.append("chained")
    if t.from_memory:
        tags.append("no retrieval")
    else:
        tags.append(f"{t.rounds} rounds")
    tags.append(f"${t.cost:.4f}")
    st.caption(" · ".join(tags))

    # -- what it cost, always visible --------------------------------------
    if not t.from_memory:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rounds", t.rounds)
        c2.metric("Model calls", f"{t.llm_calls} + {t.judge_calls}",
                  help="Planner calls plus the judge's batched verdict calls")
        c3.metric("Tokens", f"{t.prompt_tokens + t.completion_tokens:,}",
                  help=f"{t.prompt_tokens:,} in, {t.completion_tokens:,} out, planner only")
        c4.metric("Cost", f"${t.cost:.4f}", help="At OpenRouter list prices, judge included")
        c5.metric("Seconds", f"{t.seconds:.0f}")

    if t.goals:
        with st.expander(f"🪜 Split into {len(t.goals)} steps", expanded=False):
            for i, g in enumerate(t.goals, 1):
                st.markdown(f"{i}. {g}")

    # -- provenance: the papers and the sentences behind them ---------------
    if t.provenance:
        label = f"📄 {len(t.provenance)} paper(s) in this answer, with the sentences that support them"
        with st.expander(label, expanded=True):
            for pe in t.provenance:
                st.markdown(f"**{pe.paper_id}** — {pe.title}")
                if pe.labels:
                    st.caption("Matches: " + "; ".join(pe.labels[:6]))
                if pe.verdict:
                    st.caption(f"Judge: _{pe.verdict}_")
                if pe.quotes:
                    for q in pe.quotes:
                        st.markdown(f"> {q.text}")
                        st.caption(f"— {q.section or 'section not recorded'}")
                else:
                    st.caption("No sentence in the graph ties this paper to the question's terms.")
                st.markdown("")
        if t.dropped:
            head = (f"Judge kept {t.judge_kept} of {t.judge_candidates} candidates; rejected ({len(t.dropped)})"
                    if t.judge_candidates else f"Papers the judge rejected ({len(t.dropped)})")
            with st.expander(head, expanded=False):
                for pid in t.dropped:
                    why = t.verdicts.get(pid, "")
                    st.caption(f"· {pid}" + (f" — {why}" if why else ""))

    # -- the graph it stands on -------------------------------------------
    if graph and t.papers and not t.from_memory:
        from hepcoveragekg.ui import graph_view as GV
        g = GV.build(conn, t.question, t.papers, t.entity_ids)
        if g.n_entities_visible:
            with st.expander(f"🕸️ What the answer is built on — {g.n_papers} papers, "
                             f"{g.n_entities_visible} entities shown, "
                             f"{g.n_entities_hidden + g.n_other_papers} more on click", expanded=False):
                graph_frame(GV.to_html(g), height=600)

    # -- what the system did ------------------------------------------------
    if detail and not t.from_memory:
        with st.expander("🔧 What the system did", expanded=False):
            v = t.verification
            if v is not None:
                if getattr(v, "checked", 0) and not getattr(v, "unsupported", []):
                    st.success(f"Every claim traced to a retrieved row ({v.checked} checked)")
                elif getattr(v, "unsupported", []):
                    st.warning(f"{len(v.unsupported)} claim(s) not found in anything retrieved")
                    for c in v.unsupported[:8]:
                        st.caption(f"· `{c.text}` — {c.note}")
            for s in t.steps:
                st.write(("⚠️ " if s.get("error") else "✅ ") + describe_step(s))
            if t.thoughts:
                for th in t.thoughts:
                    st.caption(f"**Round {th.round}** → `{', '.join(th.tools_called)}`  \n{th.text[:600]}")


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

st.title("🔎 HEPCoverageKG")
st.caption(f"A coverage map of {stats['papers']} ATLAS/CMS papers — {stats['entities']:,} entities, "
           f"{stats['assertions']:,} facts, {stats['quotes']:,} verbatim quotes. "
           "Every paper in an answer comes with the sentences that put it there.")

for t in turns:
    with st.chat_message("user"):
        st.markdown(t.question)
    with st.chat_message("assistant"):
        render_turn(t, detail=show_detail, graph=show_graph)

question = st.chat_input("Ask the coverage map, or follow up on the answer above")

if question:
    from hepcoveragekg.ui import runner as R

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        M.apply_env(spec)
        started = time.time()
        turn = None
        status = st.status("Working…", expanded=True)

        # -- a follow-up: try the earlier turns first ------------------------
        if turns:
            status.write("Checking whether the earlier answers already cover this…")
            try:
                from_memory = R.route_followup(turns, question)
            except Exception as exc:  # noqa: BLE001
                from_memory = None
                status.write(f"Router unavailable ({type(exc).__name__}); retrieving.")
            if from_memory:
                turn = R.memory_turn(conn, question, from_memory, spec=spec, seconds=time.time() - started)
                status.update(label="Answered from the earlier turns", state="complete", expanded=False)
        run_text = R.followup_question(turns, question) if turns else question

        # -- a retrieval -------------------------------------------------------
        if turn is None:
            seen = {"n": 0}

            def on_node(node, session):
                for s in session.steps[seen["n"]:]:
                    status.write(("⚠️ " if s.error else "✅ ") + describe_step(
                        {"tool": s.tool, "args": s.args, "rows": s.rows, "error": s.error}))
                seen["n"] = len(session.steps)
                if node == "review":
                    status.write("Judging the candidate papers…")

            try:
                if agent == "freesql":
                    status.write("Free-SQL agent: searching, then writing queries…")
                    answer = R.run_freesql(conn, index, run_text, max_rounds=max_rounds)
                    turn = R.to_turn(conn, question, answer, spec=spec, agent=agent, effort=effort,
                                     seconds=time.time() - started)
                elif effort == "high":
                    def on_goals(goals):
                        status.write("Split into steps: " + " → ".join(goals) if goals else "No decomposition; running as one question.")

                    def on_leg(i, goal, last):
                        status.write(f"**Step {i + 1}**: {goal}" + (" (final)" if last else ""))
                        seen["n"] = 0
                    answer, goals, review = R.run_chained(conn, index, run_text, on_goals=on_goals,
                                                          on_leg=on_leg, on_node=on_node)
                    turn = R.to_turn(conn, question, answer, spec=spec, agent="typed", effort=effort,
                                     review=review, goals=goals, seconds=time.time() - started)
                else:
                    answer, session = R.run_typed(conn, index, run_text, max_rounds=max_rounds, on_node=on_node)
                    turn = R.to_turn(conn, question, answer, spec=spec, agent="typed", effort=effort,
                                     session=session, seconds=time.time() - started)
                status.update(label=f"Done in {turn.seconds:.0f}s", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 -- a dead endpoint must not blank the page
                status.update(label="Failed", state="error")
                turn = R.Turn(question=question, model=spec.label, agent=agent, effort=effort,
                              error=f"{type(exc).__name__}: {exc}")

        turns.append(turn)
        render_turn(turn, detail=show_detail, graph=show_graph)
