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
import time
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

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


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def _runtime(config) -> dict:
    """The callables for this run, from config rather than from state."""
    return (config or {}).get("configurable", {})


def plan(state: PlannerState, config=None) -> PlannerState:
    """Ask the model what to do next."""
    from hepcoveragekg.query import planner

    runtime = _runtime(config)
    session = state["session"]
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
        calls = planner._recover_tool_calls(message.content or "")
        if calls:
            session.recovered_calls += len(calls)
            logger.info(f"round {state['round']}: recovered {len(calls)} tool call(s) "
                        "written as text")

    reasoning = (message.content or "").strip()
    if reasoning and calls:
        session.thoughts.append(planner.Thought(
            round=state["round"], text=reasoning,
            tools_called=[c.function.name for c in calls]))  # SDK objects here, pre-normalisation

    state["last_content"] = message.content or ""
    state["pending_calls"] = [
        {"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
        for c in calls[: state["max_places"]]
    ]
    if len(calls) > state["max_places"]:
        logger.info(f"round {state['round']}: {len(calls)} calls "
                    f"capped to {state['max_places']}")
    return state


def _papers_resolver(runtime):
    """Entity ids -> the papers they appear in, through the ordinary tool.

    Goes through `papers_of` rather than a second query path so a cited answer
    and a `papers_of` call in the trace can never disagree about the same set.
    """
    def resolve(entity_ids):
        if not entity_ids:
            return []
        try:
            result = runtime["execute"]("papers_of", {"entity_ids": list(entity_ids)})
        except Exception:  # noqa: BLE001 -- a citation must not crash the answer
            return None
        seen, out = set(), []
        for row in result.rows:
            paper = row.get("paper_id") or row.get("arxiv_id")
            if paper and paper not in seen:
                seen.add(paper)
                out.append(str(paper))
        return out
    return resolve


def execute(state: PlannerState, config=None) -> PlannerState:
    """Run this round's batch of tool calls and feed the results back."""
    from hepcoveragekg.query import planner, widen

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
            widened = widen.should_widen(
                session, reason=claimed,
                rounds_left=state["max_rounds"] - state["round"],
                enabled=bool(runtime.get("persist")))
            if widened is not None:
                session.widenings_used.add(widened.rung)
                session.widenings_offered += 1
                state["messages"].append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": widen.WIDEN_MESSAGE.format(
                        rounds_left=state["max_rounds"] - state["round"],
                        suggestion=widened.message)})
                session.steps.append(planner.Step(state["round"], "widen",
                                                  {"rung": widened.rung}))
                continue

            session.answer = str(args.get("text", "")).strip()
            session.answerable = bool(args.get("answerable", True))
            session.reason = claimed
            session.stopped_because = f"answered ({claimed})"
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
            session.seen_values |= planner._values_in(result.rows)
            session.known_entity_ids |= planner._collect_ids(result.rows, result.note)
            body = planner._render_rows(result.rows, state["max_rows"])
            if result.note:
                body += f"\n[{result.note}]"
            if redirect:
                body += f"\n[{redirect}]"
            session.steps.append(planner.Step(
                state["round"], name, args,
                rows=len(result.rows), seconds=elapsed, preview=body[:200],
                redirected_from=asked_for if redirect else None,
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


def finish(state: PlannerState) -> PlannerState:
    """Terminal. Verifies the answer against what the tools actually returned.

    Verification lives here rather than being left to the caller so that no
    answer can leave unchecked -- it is part of the machine, not a courtesy the
    caller extends.
    """
    from hepcoveragekg.query import verify

    session = state["session"]
    session.evidence_ids = sorted(set(session.evidence_ids))
    session.verification = verify.verify_session(session)
    return state


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

def after_plan(state: PlannerState) -> str:
    """Tool calls to run, a prose answer to accept, or a nudge to send."""
    session = state["session"]
    if state["pending_calls"]:
        return "execute"

    # No tool calls: the model wrote prose. If nothing has been retrieved this
    # is an answer from memory, which is the failure the project exists to
    # prevent -- so ask once, then accept and flag it.
    if not session.steps and not session.nudged:
        return "nudge"

    session.answer = (state.get("last_content") or "").strip()
    session.stopped_because = ("answered from memory, no tool calls"
                               if not session.steps
                               else "answered without calling answer()")
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
    g.add_node("execute", execute)
    g.add_node("nudge", nudge)
    g.add_node("finish", finish)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan,
                            {"execute": "execute", "nudge": "nudge", "finish": "finish"})
    g.add_edge("nudge", "plan")
    g.add_conditional_edges("execute", after_execute, {"plan": "plan", "finish": "finish"})
    g.add_edge("finish", END)

    return g.compile(checkpointer=checkpointer)


def diagram() -> str:
    """Mermaid source for the graph.

    Generated from the compiled graph, so a figure made from it cannot drift
    away from what the code actually does.
    """
    return build().get_graph().draw_mermaid()
