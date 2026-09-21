"""
HEPCoverageKG query layer: the planner as a state graph.

The same loop `planner.answer()` has always run, declared instead of implied.
LangGraph orchestrates -- nodes, edges, state, checkpoints -- and nothing else:
the prompts, the LLM call, the tools, retrieval and verification all stay in our
own code (system.md S-16). What a framework would hide is exactly what we need
to keep visible.

  plan      call the model; pull out tool calls, including any written as text
  execute   guard the ids, run the batch, feed results back
  nudge     "you have not retrieved anything yet"
  finish    verify the answer against what was actually returned

The loop is the `execute -> plan` edge. There is no `while`.

Why bother, given the hand-rolled loop worked: **checkpointing**. State is saved
after every node, so a run can be paused, resumed, inspected step by step, or
rewound -- which is what a web session needs when a user asks something, wanders
off, and comes back. That is the one part genuinely annoying to hand-roll, and
it is also how multi-turn conversation arrives later: same `thread_id`, same
accumulated state, so "how many of *those* used Herwig?" has a referent.

`planner.answer()` keeps its exact signature and delegates here, so the 34 tests
that pin the loop's behaviour are the acceptance criterion for the port rather
than something rewritten alongside it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from hepcoveragekg.query import anchor as _anchor
from hepcoveragekg.query import sufficiency as _suff

logger = logging.getLogger(__name__)


class PlannerState(TypedDict, total=False):
    """Everything one question needs, carried between nodes.

    `session` is the existing trace object rather than a flattened copy: it
    already accumulates steps, thoughts, named sets, known ids and token counts,
    and duplicating that into graph state would give two things to keep in step.
    """

    question: str
    messages: list
    session: Any          # planner.Session
    round: int
    max_rounds: int
    max_places: int
    max_rows: int

    # NOTE what is NOT here: the model callable, the tool executor and the tool
    # schemas. State is serialised at every checkpoint, and functions do not
    # serialise -- putting them here made checkpointing fail outright. Runtime
    # dependencies belong in `config`, which is not persisted; only data that
    # describes the run belongs in state.

    # Handed from `plan` to `execute`. Two rules learned the hard way:
    #
    #   they must be DECLARED -- LangGraph keeps only the keys its schema names,
    #       so anything passed between nodes without being here is silently
    #       dropped and the next node fails on a missing key.
    #
    #   they must be PLAIN DATA -- every checkpoint serialises the state, and
    #       SDK objects do not serialise. So tool calls are normalised to
    #       {id, name, arguments} dicts and the message to its text, rather than
    #       carrying the response objects around.
    pending_calls: list   # [{"id": str, "name": str, "arguments": str}]
    last_content: str

    # The reviewer arm. `replan` is the one that matters: a bounce-back must NOT
    # consume a round, so `plan` skips its round increment when it is set. Any
    # other arrangement makes a rejected plan cost the same as a wasted retrieval
    # and the arm would be measuring the penalty, not the review.
    sub_objectives: list
    subgoal_status: str
    #: SUBGOAL_SEQUENTIAL=1: which objective the planner is currently shown,
    #: and the one-line notes the finished ones left behind.
    subgoal_index: int
    subgoal_established: list
    #: SUFFICIENCY=1: the papers reachable from everything retrieved so far, and
    #: how many consecutive rounds have added almost none. Carried in state
    #: because the marginal yield is the signal and it cannot be recovered from
    #: a single round's view.
    sufficiency_papers: set
    sufficiency_stale: int
    objective: str
    review_feedback: str
    review_cycles: int
    replan: bool
    # `after_plan` is a routing function and receives no config, so whether the
    # reviewer is on has to travel in state. Set by `plan` from runtime.
    _reviewing: bool
    # Set by `after_plan` when the model answered in prose instead of
    # calling `answer`. `finish` then applies what the answer branch would
    # have: harvested arguments, citations, the gate, the critic.
    _prose_answer: bool
    _ranked_prose_ask: bool


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def _advance_sequential(state: PlannerState, reasoning: str, message) -> None:
    """Move the sequential pointer on when the current objective is finished.

    The MODEL declares it, by writing OBJECTIVE COMPLETE, so the mechanism
    stays in its hands rather than resting on a heuristic for "done" the graph
    could not justify. Two guards stop a run stalling: MAX_ROUNDS_PER_GOAL
    advances anyway, and the last objective never advances -- it is the one
    carrying the instruction to answer.
    """
    if os.environ.get("SUBGOAL_SEQUENTIAL", "") != "1":
        return
    goals = state.get("sub_objectives") or []
    if not goals:
        return
    from hepcoveragekg.query import subgoals as _sg

    session = state["session"]
    idx = int(state.get("subgoal_index") or 0)
    spent = int(state.get("_subgoal_rounds") or 0) + 1
    state["_subgoal_rounds"] = spent
    text = reasoning or ""
    content = getattr(message, "content", "") or ""
    declared = _sg.wants_advance(text) or _sg.wants_advance(content)
    forced = spent >= _sg.MAX_ROUNDS_PER_GOAL
    if not (declared or forced) or idx >= len(goals) - 1:
        return
    note = _sg.note_from(text or content, "advanced without a note")
    state["subgoal_established"] = list(state.get("subgoal_established") or []) + [note]
    state["subgoal_index"] = idx + 1
    state["_subgoal_rounds"] = 0
    session.subgoal_advances += 1
    session.subgoal_forced_advances += 0 if declared else 1
    logger.info("sub-objective %d/%d done (%s): %s", idx + 1, len(goals),
                "declared" if declared else "forced", note[:70])


def _reflect_chat(runtime):
    """The client the completeness check uses: its own, or the planner's.

    Set up in `planner._prepare` when REFLECT_BASE_URL or CRITIC_BASE_URL is
    present. Falling back to the planner's client keeps every existing run and
    every test behaving exactly as before.
    """
    return runtime.get("reflect_chat") or runtime["chat"]


def _runtime(config) -> dict:
    """The callables for this run, from config rather than from state."""
    return (config or {}).get("configurable", {})


def plan(state: PlannerState, config=None) -> PlannerState:
    """Ask the model what to do next."""
    from hepcoveragekg.query import reflect as reflect_mod
    from hepcoveragekg.query import planner

    runtime = _runtime(config)
    session = state["session"]
    if state.pop("replan", False):
        # A rejected plan is re-planned inside the SAME round, with the
        # reviewer's objection appended so the model sees what it must fix.
        state["messages"] = state["messages"] + [{
            "role": "user",
            "content": ("A reviewer looked at that plan before it ran and asked "
                        "for a revision:\n\n" + (state.get("review_feedback") or "")
                        + "\n\nRevise and state your objective again.")}]
    else:
        state["round"] = state.get("round", 0) + 1
    session.rounds = state["round"]

    # THE LAST ROUND IS FOR ANSWERING, and only for answering.
    #
    # `after_execute` sends a run that reaches max_rounds straight to `finish`
    # with whatever is in the session -- which, for a model that never called
    # `answer`, is nothing. Every retrieved row, every token, discarded in
    # silence. gpt-5.6-luna hit this on 24 of 24 questions: it found gf-01's
    # answer in round 2 with facets(objects=[BJet,MET]) and then explored until
    # the budget ran out, and the run scored 0.666 purely because the scorer
    # fell back to the retrieval footprint. Qwen never showed it because Qwen
    # stops at round 3.
    #
    # So on the final round the model is given the `answer` tool and no other,
    # with a message saying why. It cannot keep searching, and the run ends with
    # a conclusion drawn from what it actually found rather than with an empty
    # string. A model with nothing to say can still answer "not in the graph",
    # which is a real answer and scoreable; silence is neither.
    # Reviewed only when there is a RETRIEVAL to review. `answer` is a different
    # act -- judging a conclusion is not judging a route, the reviewer's prompt
    # is written for routes, and a reviewer that can reject an answer can trap a
    # run in a loop at the very end. Matches the free-SQL side, which reviews
    # only when a `sql` call is present.
    # THE SUB-OBJECTIVE BLOCK IS REPLACED, NEVER APPENDED. The message list
    # grows every round, so appending would leave one stale copy per round and
    # the model would have to work out which status is current.
    goals = state.get("sub_objectives") or []
    split_status = (runtime.get("subgoal_status")
                    and os.environ.get("SUBGOAL_STATUS_CALL", "") == "1")
    if goals:
        from hepcoveragekg.query import subgoals as _sg
        question_scope = runtime.get("subgoal_scope") == "question"
        if split_status and session.steps and question_scope:
            fresh = _sg.status_call_question(runtime["chat"], state["question"],
                                             state.get("subgoal_status", ""), session.steps)
            session.llm_calls += 1
            session.subgoal_status_calls += 1
            if fresh:
                state["subgoal_status"] = fresh
        elif split_status and session.steps:
            # ONE CALL THAT ONLY JUDGES COMPLETENESS (D-173), before the call
            # that plans. The combined version asks the planner to report on
            # the last results and choose the next ones in the same breath.
            fresh = _sg.status_call(runtime["chat"], state["question"], goals,
                                    state.get("subgoal_status", ""), session.steps)
            session.llm_calls += 1
            session.subgoal_status_calls += 1
            if fresh:
                state["subgoal_status"] = fresh
        if os.environ.get("SUBGOAL_SEQUENTIAL", "") == "1" and not question_scope:
            # ONE OBJECTIVE AT A TIME. The planner never sees the ones it has
            # not reached, so it cannot satisfy itself on the first condition
            # and answer -- the gf-01 failure this whole family targets.
            idx = int(state.get("subgoal_index") or 0)
            block = _sg.render_sequential(
                goals, idx, state.get("subgoal_established") or [],
                state["max_rounds"] - state["round"])
        elif runtime.get("subgoal_status") and question_scope:
            block = _sg.render_question(state.get("subgoal_status", ""), readonly=split_status)
        elif runtime.get("subgoal_status"):
            block = (_sg.render_readonly(goals, state.get("subgoal_status", ""))
                     if split_status else _sg.render(goals, state.get("subgoal_status", "")))
        else:
            block = _sg.goals_only(goals)
        # Both markers: the whole-question block (D-176) carries no
        # "SUB-OBJECTIVES" heading, so filtering on that alone would leave one
        # stale progress block per round -- the exact failure this replaces.
        msgs = [m for m in state["messages"]
                if not (m.get("role") == "system"
                        and ("SUB-OBJECTIVES" in (m.get("content") or "")
                             or "PROGRESS ON THE QUESTION" in (m.get("content") or "")))]
        state["messages"] = msgs + [{"role": "system", "content": block}]
    # THE BRAKE (D-196). Every block above this one tells the planner to keep
    # going; this one tells it what more searching is still buying. Placed
    # after the sub-goal block so that when both are fitted the planner reads
    # "here is what you have not covered" and then "here is what your last
    # round added", which is the order the two have to be read in.
    if _suff.enabled():
        stats = _suff.update(state, runtime.get("conn"),
                             sorted(session.known_entity_ids))
        session.sufficiency_new.append(stats["new"])
        if stats["saturated"]:
            session.sufficiency_saturated += 1
        state["messages"] = _suff.replace_block(
            state["messages"],
            _suff.render(stats, state["max_rounds"] - state["round"]))

    # THE CHECK RUNS WHEN A ROUND'S RESULTS ARE IN, NOT WHEN AN ANSWER IS
    # TRIED (D-178). One call reads the rows retrieved so far and says which
    # parts of the question they establish; the planner sees that verdict
    # before it decides what to do next, and the answer gate later reuses it
    # rather than asking again. Devil's Advocate runs its alignment after
    # every action; this is the same cadence at the granularity of a round.
    if (reflect_mod.mode() in ("model", "both") and session.steps
            and state["round"] > 1 and not state.get("_reflect_fresh")):
        fresh = reflect_mod.completeness(
            _reflect_chat(runtime), session.question, runtime.get("conn"), session,
            previous=state.get("reflect_note", ""))
        if fresh is not None:
            session.llm_calls += 1
            session.reflect_checks += 1
            session.reflect_note = fresh.raw
            state["reflect_note"] = fresh.raw
            state["reflect_ready"] = fresh.ready
            state["_reflect_fresh"] = True
    note = state.get("reflect_note", "")
    if note and reflect_mod.mode() in ("model", "both"):
        v = reflect_mod.completeness_from_raw(note)
        if v is not None:
            msgs = [m for m in state["messages"]
                    if not (m.get("role") == "system"
                            and "HOW MUCH OF THE QUESTION IS ANSWERED" in (m.get("content") or ""))]
            state["messages"] = msgs + [{"role": "system", "content": reflect_mod.render_verdict(v)}]
    state["_reviewing"] = bool(runtime.get("reviewer"))
    tools = runtime["tools"]
    if state["round"] >= state["max_rounds"]:
        tools = [t for t in tools if t.get("function", {}).get("name") == "answer"] or tools
        state["messages"] = state["messages"] + [{
            "role": "user",
            "content": (f"This is round {state['round']} of {state['max_rounds']} "
                        "-- your last. No more searching: answer now from what "
                        "you have already retrieved. If it is not enough, say "
                        "so with `not_in_graph` and name what you looked for.")}]

    response = runtime["chat"](state["messages"], tools)
    session.llm_calls += 1
    usage = getattr(response, "usage", None)
    if usage:
        session.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        session.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    message = response.choices[0].message
    calls = list(getattr(message, "tool_calls", None) or [])
    if not calls:
        known = {t.get("function", {}).get("name")
                 for t in (tools or []) if isinstance(t, dict)}
        calls = planner._recover_tool_calls(message.content or "", known - {None})
        if calls:
            session.recovered_calls += len(calls)
            logger.info(f"round {state['round']}: recovered {len(calls)} tool call(s) "
                        "written as text")

    reasoning = planner.clean_content(message.content)
    if reasoning and calls:
        session.thoughts.append(planner.Thought(
            round=state["round"], text=reasoning,
            tools_called=[c.function.name for c in calls]))  # SDK objects here, pre-normalisation

    # The stated objective for this round IS the prose the model wrote before
    # its calls. Captured rather than re-asked: a second call to ask "what were
    # you trying to do" would cost a round and could disagree with the first.
    state["last_content"] = planner.clean_content(message.content)
    if runtime.get("state_objective") and reasoning:
        state["objective"] = reasoning
    if runtime.get("subgoal_status") and not split_status:
        from hepcoveragekg.query import subgoals as _sg
        fresh = (_sg.extract_progress(reasoning)
                 if runtime.get("subgoal_scope") == "question"
                 else _sg.extract_status(reasoning))
        if fresh:
            state["subgoal_status"] = fresh
    _advance_sequential(state, reasoning, message)
    state["pending_calls"] = [
        {"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
        for c in calls[: state["max_places"]]
    ]
    if len(calls) > state["max_places"]:
        logger.info(f"round {state['round']}: {len(calls)} calls "
                    f"capped to {state['max_places']}")
    # SET HERE, NOT IN after_plan (D-127). `after_plan` is a routing function:
    # LangGraph reads its return value and discards its state mutations, so
    # the flag it set never reached `finish`. Measured on 38 + 9 prose exits
    # across two live runs: gate_kind set 0 times, cited 0 times -- the whole
    # prose-exit service (harvest, citations, gate, critic) was inert while
    # its tests passed, because the tests called both functions on one dict.
    # A node's returned state persists; this is a node.
    state["_prose_answer"] = (not calls) and bool(session.steps)
    # THE RANKED ANSWER ON THE PROSE EXIT (D-135). 14 of 27 records in the
    # RANKED_TOP_N=40 run left as prose and the hook in `execute` never ran.
    # Decided here, where `runtime` is in hand; routed by `after_plan`;
    # the message is written by the `ranked_ask` node, which loops to `plan`.
    state["_ranked_prose_ask"] = False
    if (state["_prose_answer"] and runtime.get("ranked_answer") and session.rankings
            and not session.ranked_answer_asked
            and state["round"] < state["max_rounds"]):
        cands, strong, named_now, ask = _ranked_ask_plan(
            session, state.get("last_content") or "")
        session.ranked_answer_shown = len(cands)
        if ask:
            state["_ranked_prose_ask"] = True
            state["_prose_answer"] = False
    return state


def _ranked_ask_plan(session, text):
    """What the ranked answer would show, and whether to ask (D-128, D-135).

    Shared by the `answer` tool branch and the prose exit. `strong` is the
    graded-3-or-2 candidates; RANKED_TOP_N caps the list shown;
    RANKED_ASK_MIN_MISSING, when set, asks whenever at least that many strong
    candidates are unnamed (the D-132 rule, fewer than half named, fired on 2
    of 27 records once the cap was 40 -- the model names most of the page).
    """
    from hepcoveragekg.query import answer_critic as AC
    try:
        top_n = int(os.environ.get("RANKED_TOP_N", "15") or 15)
    except ValueError:
        top_n = 15
    cands = AC.ranked_candidates(session.rankings, top_n=top_n)
    strong = [p for p, g in cands if g >= 2]
    named_now = set(re.findall(r"\b\d{4}\.\d{4,5}\b", text or ""))
    missing = len(set(strong) - named_now)
    try:
        min_missing = int(os.environ.get("RANKED_ASK_MIN_MISSING", "") or 0)
    except ValueError:
        min_missing = 0
    should_ask = (missing >= min_missing) if min_missing > 0 else (
        len(named_now & set(strong)) < len(strong) / 2)
    return cands, strong, named_now, bool(strong and should_ask)


def _ranked_message(cands, strong, named_now):
    from hepcoveragekg.query import answer_critic as AC
    listing = "\n".join(f"  {p}  grade {g}" for p, g in cands)
    return AC.RANKED_MESSAGE.format(listing=listing, named=len(named_now & set(strong)),
                                    top=len(strong))


def _papers_resolver(runtime):
    """Entity ids -> the papers they appear in, through the ordinary tool.

    Goes through `papers_of` rather than a second query path so a cited answer
    and a `papers_of` call in the trace can never disagree about the same set.
    """
    def resolve(entity_ids):
        if not entity_ids:
            return []
        # THE WHOLE RESOLUTION, not just the call. The `try` used to stop at
        # the tool call, so a `papers_of` that came back shaped wrong -- None,
        # or rows without a paper column -- raised out of `resolve_citations`
        # and killed the run, which is the exact thing this guard exists to
        # prevent. A citation that cannot resolve is an uncited answer, never a
        # crash.
        try:
            result = runtime["execute"]("papers_of", {"entity_ids": list(entity_ids)})
            seen, out = set(), []
            for row in result.rows:
                paper = row.get("paper_id") or row.get("arxiv_id")
                if paper and paper not in seen:
                    seen.add(paper)
                    out.append(str(paper))
            return out
        except Exception:  # noqa: BLE001 -- a citation must not crash the answer
            logger.warning("papers_of did not resolve the cited set; "
                           "the answer stands uncited")
            return None
    return resolve


#: The ids the answer names, in the text. Must match `scoring._arxiv_ids_in`
#: or the critic would filter a set the scorer never reads.
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


def _answer_named(session) -> list:
    """What the answer actually claims, by the same rule the scorer uses.

    WHICHEVER SIDE THE ANSWER USED. The critic's first wiring judged only
    `answer_papers`, which is filled by a resolved citation -- and D-107
    measured `cited` empty in 59 of 60 Gabriel answers. So on the ordinary
    path, where the model writes ids into prose, it would have judged an empty
    list and reported itself as having run.
    """
    in_text = []
    seen = set()
    for pid in _ARXIV.findall(session.answer or ""):
        if pid not in seen:
            seen.add(pid)
            in_text.append(pid)
    return in_text or list(session.answer_papers)


def _constrained_ids(runtime, session) -> None:
    """Constrained selection of the answer's papers (2026-09-09, literature:
    Willard & Louf 2023). Off unless CONSTRAINED_IDS=1.

    --name-ids asks for arXiv ids in prose and gets the footprint written out
    (D-153). This removes the formatting question altogether: the candidate
    list is every paper reachable from the entities the run retrieved, and
    the model is asked once more for a JSON object whose `papers` items are
    constrained by schema (vLLM `guided_json`) to that list. The model cannot
    invent an id or write a summary; it can only choose. Selected ids are
    appended to the text (the field the scorers read) and set as
    `answer_papers`; the original text is kept.
    """
    if os.environ.get("CONSTRAINED_IDS", "") != "1":
        return
    # Paper-set questions only: a count wants a number and a per-paper
    # question wants the paper's labels; an id list would replace both.
    if (runtime.get("question_shape") or "") not in ("", "papers"):
        return
    conn = runtime.get("conn")
    ids = sorted(getattr(session, "known_entity_ids", set()) or [])
    # ANCHOR=1 (D-197): the judge sees what the planner NOMINATED, not
    # everything it touched. The footprint above is still what the record
    # reports as retrieved; only the candidate list changes.
    # CONSTRAINED_FROM_KEPT=1 (D-158): the search critic's verdicts shape the
    # candidate list. Without it the critic and the selector are disconnected:
    # `known_entity_ids` holds every row a tool returned, dropped or not, so
    # the enum offered the papers the critic had just called unrelated.
    if os.environ.get("CONSTRAINED_FROM_KEPT", "") == "1":
        dropped = {v.entity_id for r in (getattr(session, "reviews", []) or [])
                   for v in getattr(r, "verdicts", []) if not v.kept}
        if dropped:
            before = len(ids)
            ids = [i for i in ids if i not in dropped]
            session.constrained_critic_dropped = before - len(ids)
    if conn is None or not ids:
        return
    from hepcoveragekg.query import planner as _p
    try:
        rows = conn.execute(
            f"SELECT DISTINCT eo.paper_id, eo.label FROM entity_occurrence eo "
            f"WHERE eo.entity_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("constrained ids: footprint query failed: %s", exc)
        return
    by: dict = {}
    for pid, label in rows:
        by.setdefault(str(pid), []).append(str(label))
    # ANCHOR=1 (D-197): the judge is shown the papers the planner NOMINATED,
    # not everything it touched. `ids` stays the footprint so the judge's
    # EVIDENCE for each candidate is everything the run retrieved about it;
    # only the candidate list narrows. Papers nominated at paper level (an
    # intersection's rows) enter as-is, so the AND the planner computed is
    # still an AND at the judge's door.
    anchor_pool = None
    if _anchor.enabled():
        anchor_pool, session.anchor_fallback = _anchor.candidate_pool(session, conn)
        if not session.anchor_fallback:
            by = {p: l for p, l in by.items() if p in anchor_pool}
            for p in anchor_pool:
                by.setdefault(str(p), [])
    # Ids the draft already named are eligible too, but ONLY if the graph
    # holds them: job 54296 (D-157) admitted runs of consecutive ids the
    # draft had invented, and the enum then made them legal choices.
    named = set(_answer_named(session))
    # UNDER ANCHOR THE ANSWER'S IDS DO NOT WIDEN THE POOL (v6). The override
    # -- "name a paper in answer() to include it" -- was meant for a paper the
    # planner specifically wants. qwen3.8-flash uses it as a dump: on gf-01 it
    # named 38 ids in its answer and the pool went from the 18-paper
    # intersection to 38 (54569). 32b names what it nominated and is
    # unaffected. So a named id counts only if it was nominated; the planner
    # widens by making a structural call on the question's terms, not by
    # writing ids down.
    if anchor_pool is not None and not session.anchor_fallback:
        named &= set(anchor_pool)
    if named - set(by):
        try:
            held = {str(r[0]) for r in conn.execute(
                f"SELECT arxiv_id FROM paper WHERE arxiv_id IN ({','.join('?' * len(named))})",
                sorted(named)).fetchall()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("constrained ids: paper lookup failed (%s); draft ids dropped", exc)
            held = set()
        named &= held
    cands = sorted(set(by) | named)[:300]
    if not cands:
        return
    # CRITIC_SELECTS=1 (D-162): the answer critic judges EVERY candidate with
    # its evidence and its kept set is the answer's list; the answerer makes
    # no selection. The answerer's own constrained call keeps about half of
    # the list either way (gold rate 0.21 kept vs 0.10 left out); this asks
    # whether a judge reading the evidence does better when it decides alone.
    if os.environ.get("CRITIC_SELECTS", "") == "1" and runtime.get("answer_critic_chat"):
        from hepcoveragekg.query import answer_critic as AC
        try:
            if os.environ.get("CRITIC_EVIDENCE", "") == "paper":
                evidence = AC.evidence_by_paper_wide(conn, cands, ids)
            else:
                evidence = AC.evidence_by_paper(conn, ids, cands)
            review = AC.judge_papers(runtime["answer_critic_chat"], session.question, evidence)
        except Exception as exc:  # noqa: BLE001 -- a judge must not kill a run
            logger.warning("critic-selects failed, no list written: %s", exc)
            return
        kept = set(review.kept)
        picked = [p for p in cands if p in kept]
        session.answer_review = review
        session.constrained_candidates = len(cands)
        session.constrained_ids = picked
        session.constrained_mode = "critic"
        session.answer_before_constrained = session.answer
        if picked:
            session.answer = (session.answer or "").rstrip() + "\n\nPapers: " + ", ".join(picked)
        logger.info("critic selects: %d of %d candidates kept (%d defaulted)",
                    len(picked), len(cands), review.defaulted)
        return
    listing = "\n".join(f"{p}: " + "; ".join(sorted(set(by.get(p, [])))[:3])[:160] for p in cands)
    schema = {"type": "object",
              "properties": {"papers": {"type": "array", "items": {"type": "string", "enum": cands}}},
              "required": ["papers"], "additionalProperties": False}
    messages = [
        {"role": "system", "content": "You select papers from a candidate list. Reply with JSON only, "
                                      "of the form {\"papers\": [\"2106.01676\", ...]}, using only ids from the list."},
        {"role": "user", "content": (
            f"Question: {session.question}\n\nDraft answer:\n{(session.answer or '')[:3000]}\n\n"
            f"Candidate papers this run retrieved (arXiv id: entities that matched):\n{listing}\n\n"
            "List every candidate that answers the question. Leave out candidates that merely "
            "mention the concept without satisfying the question.")}]
    made = _p._client()
    # `_client()` returns (client, model); tolerate a bare client too.
    if isinstance(made, tuple):
        client, model = made[0], (made[1] if len(made) > 1 else os.environ.get("LLM_MODEL_NAME", ""))
    else:
        client, model = made, os.environ.get("LLM_MODEL_NAME", "")
    content = ""
    # THREE FORMS, STANDARD FIRST (D-159). vLLM 0.8.5 honoured the legacy
    # `guided_json` field; 0.18 silently ignores it and the model answered in
    # free chain of thought, so a whole night's selections came back empty
    # while `constrained_mode` still said guided_json. `response_format`
    # json_schema is enforced on both, and is what hosted providers accept.
    attempts = (
        ("json_schema", dict(
            response_format={"type": "json_schema", "json_schema": {
                "name": "papers", "schema": schema, "strict": True}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})),
        ("guided_json", dict(
            extra_body={"guided_json": schema, "chat_template_kwargs": {"enable_thinking": False}})),
        ("json_object", dict(response_format={"type": "json_object"})),
    )
    for mode, kw in attempts:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=2000, **kw)
            content = resp.choices[0].message.content or ""
            session.constrained_mode = mode
            break
        except Exception as exc:  # noqa: BLE001 -- try the next form
            logger.info("constrained ids: %s refused (%s)", mode, str(exc)[:80])
    else:
        logger.warning("constrained ids: every selection form was refused")
        return
    picked: list = []
    parsed = False
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            for p in (json.loads(m.group(0)).get("papers") or []):
                p = str(p).strip()
                if p in set(cands) and p not in picked:
                    picked.append(p)
            parsed = True
        except (ValueError, AttributeError):
            pass
    if not parsed:
        # The server accepted the request but did not enforce the schema:
        # recorded, so a run cannot report a constraint that never applied.
        session.constrained_mode += "-unparsed"
        logger.warning("constrained ids: reply was not JSON under %s (%d chars)",
                       session.constrained_mode, len(content))
    session.constrained_candidates = len(cands)
    session.constrained_ids = picked
    session.answer_before_constrained = session.answer
    if picked:
        # The TEXT carries the selection (the field the scorers read). The
        # record's `papers` stays the retrieval footprint, so retrieval_reach
        # remains comparable with the control (job 54292 overwrote it and
        # read 0.53 against 0.74 for the same retrieval).
        session.answer = (session.answer or "").rstrip() + "\n\nPapers: " + ", ".join(picked)
    logger.info("constrained ids: %d of %d candidates selected (%s)",
                len(picked), len(cands), getattr(session, "constrained_mode", ""))


def _answer_exit(runtime, session) -> None:
    """The answer-side steps, in the one order that lets each see the other.

    Without constrained selection: answer critic, grade strike, then the
    (inactive) selector -- the order every run before D-158 used. With it, the
    selector runs BEFORE the answer critic: the critic judges the list the
    selector wrote, instead of striking ids from a draft the selector then
    re-adds from the full candidate list, which is what the old order did.
    """
    if os.environ.get("CONSTRAINED_IDS", "") == "1":
        _grade_strike(session)
        _constrained_ids(runtime, session)
        # Under CRITIC_SELECTS the critic already judged the list it wrote.
        if runtime.get("answer_critic") and os.environ.get("CRITIC_SELECTS", "") != "1":
            _answer_critic(runtime, session)
        return
    if runtime.get("answer_critic"):
        _answer_critic(runtime, session)
    _grade_strike(session)
    _constrained_ids(runtime, session)


def _grade_strike(session) -> None:
    """Strike from the named answer the papers the ranker graded at or below
    STRIKE_GRADE_MAX (D-142). Off unless the variable is set.

    The stack names ~21 ids per set answer against truth sets of 2-8 on the
    generated questions: precision 0.19 to the control's 0.27. The 32B judge's
    grade 0 ("does not satisfy") carries P(gold) 0.15 against 0.49 at grade 3
    (D-135), so it is the one signal in hand that can shorten a list without a
    second judge pass. Rules: only USABLE rankings count; an ungraded paper is
    kept (no evidence is not evidence against, D-105); the strike never empties
    the answer; the original text is kept in `answer_before_strike`.
    """
    raw = os.environ.get("STRIKE_GRADE_MAX", "")
    if raw == "":
        return
    try:
        max_grade = int(raw)
    except ValueError:
        return
    best: dict = {}
    for r in getattr(session, "rankings", []) or []:
        if not getattr(r, "usable", False):
            continue
        for pid, g in r.grades.items():
            if pid not in best or g > best[pid]:
                best[pid] = g
    named = _answer_named(session)
    if not best or not named:
        return
    dropped = [p for p in named if p in best and best[p] <= max_grade]
    if not dropped or len(dropped) >= len(named):
        return
    session.answer_before_strike = session.answer
    for pid in dropped:
        session.answer = session.answer.replace(pid, "")
    if session.answer_papers:
        session.answer_papers = [p for p in session.answer_papers if p not in set(dropped)]
    session.grade_struck = list(dropped)
    logger.info("grade strike: removed %d of %d named papers graded <= %d",
                len(dropped), len(named), max_grade)


def _answer_critic(runtime, session) -> None:
    """Drop the papers the answer names that do not satisfy the question.

    Never raises and never adds a paper: a judge that can only remove is safe
    to leave on, and the failure mode measured in D-105 -- a reasoning model
    returning empty content and defaulting every verdict to KEEP -- degrades
    to the unfiltered answer rather than to an empty one.

    THE TEXT IS EDITED, NOT JUST THE LIST. The scorer reads ids out of the
    prose whenever there are any, so filtering `answer_papers` alone would
    leave the arm invisible to every set metric -- it would cost calls and
    change no number. Only the dropped ids are struck; the sentence they sat in
    is left alone, and `answer_before_critic` keeps the original so the edit is
    auditable rather than silent.
    """
    from hepcoveragekg.query import answer_critic as AC

    conn = runtime.get("conn")
    chat = runtime.get("answer_critic_chat")
    named = _answer_named(session)
    if conn is None or chat is None or not named:
        return
    try:
        if os.environ.get("CRITIC_EVIDENCE", "") == "paper":
            evidence = AC.evidence_by_paper_wide(conn, named, session.known_entity_ids)
        else:
            evidence = AC.evidence_by_paper(conn, session.known_entity_ids, named)
        review = AC.judge_papers(chat, session.question, evidence)
    except Exception as exc:  # noqa: BLE001 -- a judge must not kill a run
        logger.warning("answer-critic failed, keeping the answer as is: %s", exc)
        return
    if not review.verdicts:
        return
    session.answer_review = review
    kept = set(review.kept)
    dropped = review.dropped
    # A judge that would empty the answer is a judge that is wrong, not an
    # answer that is empty. Observed on the fast test: at high candidate purity
    # the critic costs more than it saves (gf-04, 0.88 -> 0.60).
    if not kept:
        logger.warning("answer-critic would drop all %d papers; keeping them",
                       len(named))
        return
    if not dropped:
        return
    session.answer_before_critic = session.answer
    for pid in dropped:
        session.answer = session.answer.replace(pid, "")
    if session.answer_papers:
        session.answer_papers = [p for p in session.answer_papers if p in kept]
    logger.info("answer-critic kept %d of %d named papers (%d defaulted)",
                len(kept), len(review.verdicts), review.defaulted)


def execute(state: PlannerState, config=None) -> PlannerState:
    """Run this round's batch of tool calls and feed the results back."""
    from hepcoveragekg.query import planner, reflect, widen

    runtime = _runtime(config)
    session = state["session"]
    calls = state["pending_calls"]

    state["messages"].append({
        "role": "assistant",
        "content": state.get("last_content", ""),
        "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls
        ],
    })

    for call in calls:
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"],
                "content": f"ERROR: arguments were not valid JSON ({exc})"})
            session.steps.append(planner.Step(state["round"], name, {}, error="bad_arguments"))
            continue

        # A call already made in this run is answered from the record instead of
        # re-executed. The concern this addresses is real and pre-dates any
        # persistence work: a model that has just been told a route is empty
        # sometimes tries the same route again, and every retry it spends is a
        # round it cannot spend on a different one.
        #
        # Small today -- 68 of 4,479 steps (1.5%) -- and worth closing before
        # asking the loop to persist harder, because persistence multiplies
        # whatever the retry behaviour already is.
        #
        # The reply says what the call RETURNED, not merely that it repeated.
        # "You already ran this" invites running it a third time; "you already
        # ran this and it gave 0 rows" is an argument for doing something else.
        if name != "answer":
            fingerprint = (name, json.dumps(args, sort_keys=True, default=str))
            previous = getattr(session, "calls_made", {}).get(fingerprint)
            if previous is not None:
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": (f"You already ran this exact call earlier and it "
                                f"returned {previous}. Running it again cannot "
                                f"give a different result -- try a different "
                                f"predicate, a broader search, or a different "
                                f"tool.")})
                session.steps.append(planner.Step(state["round"], name, args,
                                                  error="duplicate_call"))
                continue

        if name == "answer":
            claimed = str(args.get("reason", "answered"))
            if not session.steps and not session.nudged:
                session.nudged = True
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": planner.NUDGE})
                continue

            # An abstention made while HOLDING relevant rows gets challenged --
            # once, and for a REASON, never for a different answer.
            #
            # Two measured false negatives look exactly like this: the critic
            # kept 13 candidates and the system still said "the graph does not
            # record any papers", and on f_CP^Htt it rejected 41 of 42 with
            # individually correct reasons and then reported nothing found.
            #
            # But challenging an abstention is dangerous in a way challenging an
            # answer is not: "no paper here covers that" is this project's actual
            # output, and a system taught never to abstain would fabricate
            # coverage instead. So the challenge asks it to say WHY the rows it
            # holds do not answer the question. A sound abstention survives that
            # and is recorded with its justification, which makes it *more*
            # trustworthy; only the ones that simply gave up change.
            held = planner.rows_held(session)
            if (runtime.get("challenge_abstention") and claimed == "not_in_graph"
                    and held and not session.abstention_challenged):
                session.abstention_challenged = True
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": planner.ABSTENTION_CHALLENGE.format(held=held)})
                continue

            # About to give up while holding rows, with rounds to spare. Offer
            # ONE concrete untried route -- see query/widen.py for why the
            # ladder is finite and why that is what stops this looping.
            rounds_left = state["max_rounds"] - state["round"]
            widened = widen.should_widen(
                session, reason=claimed, rounds_left=rounds_left,
                enabled=bool(runtime.get("persist")))
            # And the mirror case: an ANSWER reached after looking once. Qwen
            # made exactly one search on 135 of 207 records and none on 69, then
            # concluded -- stopping at 3.25 rounds of a budget of 6 it is never
            # denied. The round it declines is the valuable one: 4-round runs
            # scored count_correct 0.158 against 0.063 for 3-round runs.
            #
            # Its own flag, because the risk runs the other way. Pushing an
            # abstention can only turn a refusal into an answer; pushing an
            # ANSWER can replace a precise set with a broader one. So it is
            # separable, and separately measured.
            if widened is None:
                widened = widen.should_push_further(
                    session, reason=claimed, rounds_left=rounds_left,
                    enabled=bool(runtime.get("push_further")))
                if widened is not None:
                    session.widenings_used.add(widened.rung)
                    session.widenings_offered += 1
                    logger.info("widening ladder: rung %s offered (%d so far)",
                                widened.rung, session.widenings_offered)
                    state["messages"].append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": widen.PUSH_MESSAGE.format(
                            searches=len([x for x in session.steps
                                          if x.tool == "search" and not x.error]),
                            rounds_left=rounds_left,
                            suggestion=widened.message)})
                    session.steps.append(planner.Step(state["round"], "push",
                                                      {"rung": widened.rung}))
                    continue

            # POST-ANSWER REFLECTION (D-177), checked against the GRAPH, not
            # against the model's opinion of its own answer. It may send the
            # run back to retrieve a condition nothing covers; it may never
            # edit or re-judge the answer, so a right answer cannot be turned
            # into a wrong one (Pan et al. 2024's warning about post-hoc
            # correction degrading correct responses).
            if claimed != "not_in_graph" and reflect.mode() in ("model", "both"):
                # THE ASKED CHECK (D-178). One call looks at the rows and says
                # which parts of the question they establish; if a part has
                # nothing behind it and there are rounds left, the answer is
                # bounced once with that part named.
                if session.reflections_used < reflect.MAX_REFLECTIONS and rounds_left >= reflect.MIN_ROUNDS_LEFT:
                    # Reuse the verdict this round already produced; only ask
                    # here if the run answered on its first round, before any
                    # check had been made.
                    v = reflect.completeness_from_raw(state.get("reflect_note", ""))
                    if v is None:
                        v = reflect.completeness(_reflect_chat(runtime), session.question,
                                                 runtime.get("conn"), session)
                        if v is not None:
                            session.llm_calls += 1
                            session.reflect_checks += 1
                            session.reflect_note = v.raw
                            state["reflect_note"] = v.raw
                    if v is not None:
                        state["reflect_ready"] = v.ready
                        if not v.ready and v.missing:
                            session.reflections_used += 1
                            session.reflect_defects.append(
                                {"kind": "unfulfilled", "detail": "; ".join(v.missing)})
                            state["messages"].append({
                                "role": "tool", "tool_call_id": call["id"],
                                "content": reflect.bounce_message(v, rounds_left)})
                            session.steps.append(planner.Step(
                                state["round"], "reflect", {"missing": v.missing}))
                            state["_reflect_fresh"] = False
                            continue
            if claimed != "not_in_graph" and reflect.mode() in ("graph", "both"):
                defects = reflect.should_reflect(
                    session.question, str(args.get("text") or session.answer or ""),
                    runtime.get("conn"), session, rounds_left,
                    enabled=True)
                if defects:
                    session.reflections_used += 1
                    for d in defects:
                        session.reflect_offered.add(
                            d.detail if d.kind == reflect.UNCOVERED else d.kind)
                        session.reflect_defects.append({"kind": d.kind, "detail": d.detail})
                    state["messages"].append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": reflect.message(defects, rounds_left)})
                    session.steps.append(planner.Step(
                        state["round"], "reflect",
                        {"defects": [d.kind for d in defects]}))
                    continue
                widened = None
            if widened is not None:
                session.widenings_used.add(widened.rung)
                session.widenings_offered += 1
                logger.info("widening ladder: rung %s offered (%d so far)",
                            widened.rung, session.widenings_offered)
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": widen.WIDEN_MESSAGE.format(
                        rounds_left=state["max_rounds"] - state["round"],
                        suggestion=widened.message)})
                session.steps.append(planner.Step(state["round"], "widen",
                                                  {"rung": widened.rung}))
                continue

            # A schema can be ignored, so the loop checks too. An `answer` with
            # no text and no citation is not an answer: `answered` goes True,
            # the text stays empty, and the scorer falls back to the retrieval
            # footprint -- which is how a run that said nothing scored 0.635.
            # Asked once, exactly like the nudge, then accepted and flagged.
            if (not str(args.get("text", "")).strip()
                    and not args.get("papers_from")
                    and not session.answer_retried):
                session.answer_retried = True
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": ("That call carried no answer -- `text` was empty "
                                "and no set was cited. Say it in prose, naming "
                                "the papers, or cite a set in `papers_from`. If "
                                "the graph does not hold it, answer that with "
                                "reason=not_in_graph.")})
                continue

            session.answer = str(args.get("text", "")).strip()
            session.answerable = bool(args.get("answerable", True))
            session.reason = claimed
            session.stopped_because = f"answered ({claimed})"
            if runtime.get("simple_answer"):
                # A literal list, so there is nothing to resolve: what the model
                # wrote IS the citation. Malformed ids are dropped rather than
                # trusted -- the same rule the free-SQL control uses.
                import re as _re
                asserted = [str(x).strip() for x in (args.get("papers") or [])]
                asserted = [x for x in asserted
                            if _re.fullmatch(r"\d{4}\.\d{4,5}", x)]
                if asserted:
                    session.answer_papers = sorted(asserted)
                    # `answer_cited`, NOT `cited`. systems.py reads
                    # `answer_cited`; a typo here meant every --simple-answer
                    # answer reached the scorer with cited="" and so lost the
                    # legitimate cited-set path in `judged_set_f1`. See D-107.
                    session.answer_cited = "answer.papers"
            else:
                planner.resolve_citations(session, args, _papers_resolver(runtime))
            # A refused citation is corrected once, like an invented id: the
            # model gets told why and can narrow the set or state the number.
            if session.citation_refused and not session.citation_corrected:
                session.citation_corrected = True
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": "ERROR: " + session.citation_refused})
                session.citation_refused = ""
                session.answer = ""
                session.stopped_because = ""
                continue

            # THE ANSWER GATE (D-107). Deterministic, one retry, arm-gated
            # until it is measured. It asks only whether the answer NAMES
            # anything -- the semantic "is this the right set" question is the
            # answer-critic's, and mixing the two would make a formatting fix
            # look like a reasoning result.
            if runtime.get("answer_gate"):
                from hepcoveragekg.query import answer_gate as _ag
                verdict = _ag.check(session.answer, session.answer_cited,
                                    session.answerable, claimed)
                session.answer_gate_kind = verdict.kind
                if (not verdict.ok and not session.answer_gate_retried
                        and state["round"] < state["max_rounds"]):
                    # A retry costs a round. At the last round it would blank
                    # the answer and then hit max_rounds with nothing to show:
                    # gf-04 scored 0.00 with reach 1.00 exactly that way. So
                    # ask only with a round to spare, and keep the text.
                    session.answer_gate_retried = True
                    session.answer_before_gate = session.answer
                    logger.info("answer gate: %s -- asking once", verdict.kind)
                    state["messages"].append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": verdict.message})
                    session.answer = ""
                    session.stopped_because = ""
                    continue
                if not verdict.ok:
                    # Asked and still nothing. Accepted, and LOUD: an arm whose
                    # gate fires twice is not fixing the answers, and that has
                    # to be visible in the run rather than inferred from a
                    # score that looks the same as the control's.
                    session.answer_gate_failed = True
                    logger.warning("answer gate: still no ids after retry (%s)",
                                   verdict.kind)

            # THE RANKED ANSWER (D-128, arm). The ranking reorders rows at
            # execute time and the answer, written rounds later, never sees it.
            # Once per run, with a round to spare, if the answer names fewer
            # than half of the top graded candidates, show the list and ask.
            # Nothing is removed; the text is stashed like the gate's.
            if (runtime.get("ranked_answer") and session.rankings
                    and not session.ranked_answer_asked
                    and state["round"] < state["max_rounds"]):
                cands, strong, named_now, should_ask = _ranked_ask_plan(
                    session, session.answer or "")
                session.ranked_answer_shown = len(cands)
                if should_ask:
                    session.ranked_answer_asked = True
                    session.answer_before_gate = session.answer
                    state["messages"].append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": _ranked_message(cands, strong, named_now)})
                    logger.info("ranked answer: named %d of %d strong candidates -- asking once",
                                len(named_now & set(strong)), len(strong))
                    session.answer = ""
                    session.stopped_because = ""
                    continue

            # THE ANSWER CRITIC (D-106, arm). Judges each cited PAPER against
            # the question. It only ever narrows `answer_papers`, so it cannot
            # invent coverage, and it runs after the gate so it never judges a
            # list the gate was about to reject.
            # NOT `and session.answer_papers`. That guard was left over from
            # the first wiring, which judged only a RESOLVED CITATION -- and
            # `_answer_named` was then changed to read the ids out of the
            # prose, because D-107 measured `cited` empty in 59 of 60 answers.
            # The inner fix was applied and the outer guard was not, so the arm
            # silently covered only the third of answers that cite a set: on
            # 54245 the critic ran 8 times in 22 chances, and 9 of the skips
            # had evidence available and prose ids to judge.
            _answer_exit(runtime, session)

            return state

        # Papers handed to an entity tool go to `contents_of` before the id
        # guard sees them -- the guard's correction is right for entities and
        # sent 14.2% of Tier A into a false "not in the graph" (see
        # planner.resolve_paper_calls).
        asked_for = name
        name, args, redirect = planner.resolve_paper_calls(name, args)

        unknown = [] if (args.get("object_set") or args.get("entity_set")) \
            else planner._check_ids(args, session.known_entity_ids)
        if unknown:
            session.invented_ids.extend(unknown)
            session.steps.append(planner.Step(state["round"], name, args,
                                              error="unknown_entity_id"))
            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"],
                "content": planner.UNKNOWN_ID_MESSAGE.format(unknown=unknown)})
            continue

        started = time.perf_counter()
        try:
            result = runtime["execute"](name, args)
            elapsed = time.perf_counter() - started
            session.evidence_ids.extend(getattr(result, "evidence_ids", []) or [])
            # A corpus-sized read is cut BEFORE its ids are collected, so the
            # cap shrinks the footprint and not just the transcript.
            _anchor.cap_expansion(name, result)
            session.seen_values |= planner._values_in(result.rows)
            session.known_entity_ids |= planner._collect_ids(result.rows, result.note)
            # ANCHORED CANDIDATES (D-197): classify the call against the
            # planner's own first intersection and nominate only if it was
            # finding things on the question's terms. Reads never nominate.
            anchor_tag = ""
            if _anchor.enabled():
                anchor_tag = _anchor.observe(session, state["round"], name, args,
                                             result.rows, result.note or "",
                                             planner._collect_ids)
            if runtime.get("rerank"):
                # THE CUT IS HERE, so the order has to be decided here (D-113,
                # D-118 class B). Search rows carry the critic's rung as
                # `bears_on` and arrive in retrieval order: with 60 hits and a
                # 25-row window, an `exact` hit ranked 40th by BM25 is cut and
                # an `unrelated` one ranked 3rd is shown. A stable sort by rung
                # changes only which side of the window a row lands on --
                # membership, counts and sets are untouched.
                planner.order_by_rung(result.rows, kept={
                    e for k, v in session.sets.items() if k.endswith("_kept") for e in v})
            body = planner._render_rows(result.rows, state["max_rows"])
            if result.note:
                body += f"\n[{result.note}]"
            if redirect:
                body += f"\n[{redirect}]"
            if anchor_tag in (_anchor.DRIFT, _anchor.RELAX):
                body += "\n" + _anchor.feedback(anchor_tag, session, name, args,
                                                len(result.rows))
            session.steps.append(planner.Step(
                state["round"], name, args,
                rows=len(result.rows), seconds=elapsed, preview=body[:200],
                redirected_from=asked_for if redirect else None,
                anchor=anchor_tag,
                # kept only for tools whose single row IS an answer, so
                # `answer(value_from=...)` can lift the number rather than the
                # model retyping it or the harness re-deriving it
                result=(dict(result.rows[0]) if name in planner.ANSWER_SHAPED
                        and result.rows else None)))
            session.calls_made[(asked_for, json.dumps(args, sort_keys=True,
                                                      default=str))] = (
                f"{len(result.rows)} rows")
            state["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                      "content": body})
        except Exception as exc:  # noqa: BLE001 -- the planner must see any failure
            elapsed = time.perf_counter() - started
            session.calls_made[(asked_for, json.dumps(args, sort_keys=True,
                                                      default=str))] = (
                f"an error: {str(exc)[:80]}")
            session.steps.append(planner.Step(state["round"], name, args,
                                              error=str(exc)[:200], seconds=elapsed,
                                              redirected_from=asked_for if redirect else None))
            state["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                      "content": f"ERROR: {exc}"})
    return state


def nudge(state: PlannerState) -> PlannerState:
    """The model tried to answer without retrieving anything.

    Reached only when it produced prose rather than calling `answer` -- the
    tool-call path handles its own nudge inside `execute`, because it has to
    reply on that call's id.
    """
    from hepcoveragekg.query import planner

    session = state["session"]
    session.nudged = True
    state["messages"].append({"role": "user", "content": planner.NUDGE})
    return state


def ranked_ask(state: PlannerState) -> PlannerState:
    """The model wrote its answer as prose and left graded candidates unnamed.

    Show the ranked list once and ask for the answer again, as a user turn
    (there is no tool call to reply to). The prose is stashed like the gate's
    so a worse second answer cannot erase it (`finish` restores it).
    """
    session = state["session"]
    text = (state.get("last_content") or "").strip()
    cands, strong, named_now, _ = _ranked_ask_plan(session, text)
    session.ranked_answer_asked = True
    session.ranked_answer_shown = len(cands)
    session.answer_before_gate = text
    state["messages"].append({"role": "user", "content": _ranked_message(cands, strong, named_now)})
    logger.info("ranked answer on a prose exit: named %d of %d strong candidates -- asking once",
                len(named_now & set(strong)), len(strong))
    return state


def finish(state: PlannerState, config=None) -> PlannerState:
    """Terminal. Verifies the answer against what the tools actually returned.

    Verification lives here rather than being left to the caller so that no
    answer can leave unchecked -- it is part of the machine, not a courtesy the
    caller extends.

    It is also where an answer written as PROSE is serviced (D-116). The model
    calls retrieval tools properly and improvises the answer call, differently
    per model -- QwQ writes `<answer .../>` and `<answer>{json}</answer>`,
    qwen3-32b writes `<answerable>{json}</answerable>` and one tag per
    argument. Chasing each form with another regex fails on the next model, so
    the arguments are harvested BY NAME and handed to exactly the code an
    `answer` call would have reached.
    """
    from hepcoveragekg.query import verify

    runtime = _runtime(config)
    session = state["session"]

    if not (session.answer or "").strip() and getattr(session, "answer_before_gate", ""):
        # The gate asked again and nothing came back (max_rounds, or the model
        # abandoned the call). The first answer was weak, not absent.
        session.answer = session.answer_before_gate
        session.answer_gate_failed = True
        session.stopped_because = session.stopped_because or "gate retry yielded nothing; first answer kept"

    # CLASSIFIED HERE, FOR EVERY EXIT. Setting it in the answer branch recorded
    # "tool_call" for runs that ENTERED that branch and then left through the
    # prose path anyway -- a gate retry or a refused citation both `continue`.
    # Measured on a control run: 17 records marked tool_call against 8 that
    # actually ended in an `answer` call.
    from hepcoveragekg.query import planner as _p
    session.answer_syntax = ("tool_call"
                             if not state.get("_prose_answer")
                             else _p.answer_syntax(session.answer))

    if state.get("_prose_answer"):
        harvested = _p.harvest_answer_args(session.answer)
        if harvested:
            session.reason = str(harvested.get("reason") or session.reason or "")
            if "answerable" in harvested:
                session.answerable = bool(harvested["answerable"])
            if harvested.get("papers_from") or harvested.get("value_from"):
                _p.resolve_citations(session, harvested,
                                     _papers_resolver(runtime))
            logger.info("harvested %s from an answer written as prose",
                        sorted(harvested))
        from hepcoveragekg.query import answer_gate as _ag
        verdict = _ag.check(session.answer, session.answer_cited,
                            session.answerable, session.reason)
        session.answer_gate_kind = verdict.kind
        if not verdict.ok:
            # NOT retried: there is no tool call to reply to and the graph is
            # already terminal. Recorded, which is the difference between a
            # known rate and an invisible one.
            session.answer_gate_failed = True
            logger.warning("answer written as prose: %s, %d chars",
                           verdict.kind, len(session.answer))
        _answer_exit(runtime, session)

    # SELECT AT THE CEILING (SELECT_AT_CEILING=1). A run that spends every
    # round and never calls answer() leaves with no list at all, and scores
    # zero on every metric -- not because it found nothing but because the
    # selection step never ran over what it found. On the anchored 3.8-flash
    # arm that was 5 of 18 records against 2 on the control (the feedback line
    # makes the planner search longer), and the whole difference between an
    # F1 of 0.493 and 0.683. Under CRITIC_SELECTS the judge decides the list
    # from the evidence anyway, so it can do that at the ceiling as well as
    # after an answer() call; the answer text stays empty and the record says
    # the list was selected without one.
    if (os.environ.get("SELECT_AT_CEILING", "") == "1"
            and not (session.answer or "").strip()
            and not state.get("_prose_answer")
            and os.environ.get("CONSTRAINED_IDS", "") == "1"
            and os.environ.get("CRITIC_SELECTS", "") == "1"
            and getattr(session, "known_entity_ids", None)):
        session.selected_at_ceiling = True
        session.stopped_because = ((session.stopped_because or "hit max_rounds")
                                   + "; list selected at the ceiling")
        _answer_exit(runtime, session)

    # A new round of rows means the last verdict is stale (D-178).
    state["_reflect_fresh"] = False
    session.evidence_ids = sorted(set(session.evidence_ids))
    session.verification = verify.verify_session(session)
    return state


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------


def review(state: PlannerState, config=None) -> PlannerState:
    """A second model judges the plan before a round is spent running it."""
    from hepcoveragekg.query import reviewer as R

    runtime = _runtime(config)
    session = state["session"]
    cycles = state.get("review_cycles", 0)

    # THE RUNAWAY GUARD. Retries are uncapped by design for this arm -- the
    # question is how many get used -- but a stubborn planner meeting a strict
    # reviewer must not spin all night, which is what a retry storm cost on
    # 2026-08-30. Hitting this is a finding, so it is loud and recorded.
    if cycles >= R.MAX_REVIEW_CYCLES:
        logger.warning(f"review cycle ceiling ({R.MAX_REVIEW_CYCLES}) hit on round "
                       f"{state['round']}; approving to break the loop")
        session.review_ceiling_hit = True
        state["replan"] = False
        return state

    verdict = R.review_plan(
        chat=runtime["review_chat"],
        question=state["question"],
        schema=runtime.get("review_schema", ""),
        objective=state.get("objective", ""),
        plan=R.describe_calls(state["pending_calls"]),
        kind="typed",
        history=R.describe_history(session),
    )
    session.review_calls += 1
    session.review_prompt_tokens += verdict.prompt_tokens
    session.review_completion_tokens += verdict.completion_tokens
    if not verdict.parsed:
        session.review_unparsed += 1
    # An approval the reviewer never gave. Counted separately so an arm where
    # every call 400s can never again be read as "the reviewer never objected".
    if getattr(verdict, "failed", False):
        session.review_failures += 1
    if verdict.approved:
        state["replan"] = False
    else:
        session.reviews_rejected += 1
        state["review_cycles"] = cycles + 1
        state["review_feedback"] = verdict.feedback
        state["replan"] = True
    return state


def after_review(state: PlannerState) -> str:
    """Run the plan, or send it back to be rewritten inside the same round."""
    return "plan" if state.get("replan") else "execute"


def reflect_check(state: PlannerState, config=None) -> PlannerState:
    """The asked completeness check on the PROSE exit (D-178).

    About 30-45% of answers never call `answer()` -- the model writes prose and
    the loop ends -- so the check placed in the tool-call branch could not see
    them. A routing function gets no config and cannot call a model, so the
    decision is made here, in a node, and routed afterwards.
    """
    from hepcoveragekg.query import planner, reflect

    runtime = _runtime(config)
    session = state["session"]
    state["_reflect_bounce"] = False
    rounds_left = state["max_rounds"] - state["round"]
    v = reflect.completeness_from_raw(state.get("reflect_note", ""))
    if v is None:
        v = reflect.completeness(_reflect_chat(runtime), session.question,
                                 runtime.get("conn"), session)
        if v is None:
            _accept_prose(state)
            return state
        session.llm_calls += 1
        session.reflect_checks += 1
        session.reflect_note = v.raw
        state["reflect_note"] = v.raw
    if v.ready or not v.missing:
        _accept_prose(state)
        return state
    session.reflections_used += 1
    session.reflect_defects.append({"kind": "unfulfilled", "detail": "; ".join(v.missing)})
    # No tool call to reply to on this exit, so the bounce is a user turn.
    state["messages"] = state["messages"] + [
        {"role": "user", "content": reflect.bounce_message(v, rounds_left)}]
    session.steps.append(planner.Step(state["round"], "reflect", {"missing": v.missing}))
    state["_reflect_fresh"] = False        # the next round re-checks
    state["_reflect_bounce"] = True
    return state


def _accept_prose(state: PlannerState) -> None:
    """Record a prose exit's answer on the session.

    EXTRACTED BECAUSE THE REFLECTION JUMPED OVER IT (2026-09-12). `after_plan`
    recorded the answer on the two lines immediately after the reflection's
    early `return "reflect_check"`, so with REFLECT_MODE on, every record
    leaving through the prose exit lost its answer entirely -- `session.answer`
    empty, no stopping reason, and the scorer reading it as unanswered. It cost
    `answered` 0.889 -> 0.167 on the first clean pair, and I read that as the
    mechanism discouraging the planner from committing. It was a node put in
    front of an exit that did bookkeeping on the way out.

    Both paths call this now, so an exit cannot be added in front of it again
    without the answer coming too.
    """
    session = state["session"]
    session.answer = (state.get("last_content") or "").strip()
    session.stopped_because = ("answered from memory, no tool calls"
                               if not session.steps
                               else "answered without calling answer()")


def after_reflect(state: PlannerState) -> str:
    return "plan" if state.get("_reflect_bounce") else "finish"


def after_plan(state: PlannerState) -> str:
    """Tool calls to run, a prose answer to accept, or a nudge to send."""
    session = state["session"]
    if state["pending_calls"]:
        retrieval = [c for c in state["pending_calls"] if c.get("name") != "answer"]
        return "review" if (state.get("_reviewing") and retrieval) else "execute"

    # No tool calls: the model wrote prose. If nothing has been retrieved this
    # is an answer from memory, which is the failure the project exists to
    # prevent -- so ask once, then accept and flag it.
    if not session.steps and not session.nudged:
        return "nudge"
    if state.get("_ranked_prose_ask"):
        return "ranked_ask"

    # THE OTHER WAY AN ANSWER GETS OUT, and until 2026-09-06 nothing checked
    # it. The model wrote prose and made no tool call, so `last_content`
    # becomes the answer -- and for a reasoning model `last_content` is its
    # CHAIN OF THOUGHT. Measured on the D-108 arms: 23 of 125 set answers in
    # the gate arm reached the scorer this way, every one of them opening
    # "Okay, let's tackle this question step by step", every one scored as an
    # answer, and the gate never saw a single one because it lives in the
    # `answer` tool branch.
    #
    # So the gate is applied here too. It cannot ask again -- there is no tool
    # call to reply to and the graph is on its way to `finish` -- but it can
    # record what it found, which is the difference between a known 18% and an
    # invisible one.
    from hepcoveragekg.query import reflect as _reflect
    if (_reflect.mode() in ("model", "both") and session.steps
            and session.reflections_used < _reflect.MAX_REFLECTIONS
            and state["max_rounds"] - state["round"] >= _reflect.MIN_ROUNDS_LEFT):
        return "reflect_check"

    _accept_prose(state)

    # The gate, the citation resolver and the critic all need `runtime`, and a
    # routing function receives no config -- the same constraint that put
    # `_reviewing` in state. So this exit is MARKED here and serviced in
    # `finish`, which does get config. Before D-116 it was serviced nowhere,
    # and it is where about 70% of answers leave.
    # `_prose_answer` is set by `plan`; a routing function's writes are lost.
    return "finish"


def after_execute(state: PlannerState) -> str:
    """Keep going unless the answer is in, or the round budget is spent."""
    session = state["session"]
    if session.answer:
        return "finish"
    if state["round"] >= state["max_rounds"]:
        session.stopped_because = f"hit max_rounds ({state['max_rounds']})"
        return "finish"
    return "plan"


# Our own dataclasses travel in the checkpointed state. LangGraph deserialises
# unregistered types with a warning today and will REFUSE them in a future
# release, which would break resume silently and late. Declaring them now makes
# the trust explicit and survives the upgrade.
ALLOWED_MSGPACK_MODULES = (
    ("hepcoveragekg.query.planner", "Session"),
    ("hepcoveragekg.query.planner", "Step"),
    ("hepcoveragekg.query.planner", "Thought"),
    ("hepcoveragekg.query.verify", "Verification"),
    ("hepcoveragekg.query.verify", "Claim"),
)


def serde():
    """Serialiser that accepts our trace types and nothing else unexpected."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


def build(checkpointer=None):
    """Compile the graph. Pass a checkpointer to make runs resumable."""
    g = StateGraph(PlannerState)
    g.add_node("plan", plan)
    g.add_node("review", review)
    g.add_node("execute", execute)
    g.add_node("nudge", nudge)
    g.add_node("ranked_ask", ranked_ask)
    g.add_node("finish", finish)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan,
                            {"review": "review", "execute": "execute",
                             "nudge": "nudge", "ranked_ask": "ranked_ask",
                             "finish": "finish", "reflect_check": "reflect_check"})
    g.add_conditional_edges("reflect_check", after_reflect,
                            {"plan": "plan", "finish": "finish"})
    g.add_conditional_edges("review", after_review,
                            {"plan": "plan", "execute": "execute"})
    g.add_node("reflect_check", reflect_check)
    g.add_edge("nudge", "plan")
    g.add_edge("ranked_ask", "plan")
    g.add_conditional_edges("execute", after_execute, {"plan": "plan", "finish": "finish"})
    g.add_edge("finish", END)

    return g.compile(checkpointer=checkpointer)


def diagram() -> str:
    """Mermaid source for the graph.

    Generated from the compiled graph, so a figure made from it cannot drift
    away from what the code actually does.
    """
    return build().get_graph().draw_mermaid()
