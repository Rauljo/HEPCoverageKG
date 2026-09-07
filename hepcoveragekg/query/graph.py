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
import re
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

    # The reviewer arm. `replan` is the one that matters: a bounce-back must NOT
    # consume a round, so `plan` skips its round increment when it is set. Any
    # other arrangement makes a rejected plan cost the same as a wasted retrieval
    # and the arm would be measuring the penalty, not the review.
    sub_objectives: list
    subgoal_status: str
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
    if goals:
        from hepcoveragekg.query import subgoals as _sg
        block = (_sg.render(goals, state.get("subgoal_status", ""))
                 if runtime.get("subgoal_status") else _sg.goals_only(goals))
        msgs = [m for m in state["messages"]
                if not (m.get("role") == "system" and "SUB-OBJECTIVES" in (m.get("content") or ""))]
        state["messages"] = msgs + [{"role": "system", "content": block}]
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
    if runtime.get("subgoal_status"):
        from hepcoveragekg.query import subgoals as _sg
        fresh = _sg.extract_status(reasoning)
        if fresh:
            state["subgoal_status"] = fresh
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
                widened = None
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
                if not verdict.ok and not session.answer_gate_retried:
                    session.answer_gate_retried = True
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
            if runtime.get("answer_critic"):
                _answer_critic(runtime, session)

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

    if state.get("_prose_answer"):
        from hepcoveragekg.query import planner as _p
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
        if runtime.get("answer_critic"):
            _answer_critic(runtime, session)

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
    session.answer = (state.get("last_content") or "").strip()
    session.stopped_because = ("answered from memory, no tool calls"
                               if not session.steps
                               else "answered without calling answer()")

    # The gate, the citation resolver and the critic all need `runtime`, and a
    # routing function receives no config -- the same constraint that put
    # `_reviewing` in state. So this exit is MARKED here and serviced in
    # `finish`, which does get config. Before D-116 it was serviced nowhere,
    # and it is where about 70% of answers leave.
    state["_prose_answer"] = True
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
    g.add_node("finish", finish)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan,
                            {"review": "review", "execute": "execute",
                             "nudge": "nudge", "finish": "finish"})
    g.add_conditional_edges("review", after_review,
                            {"plan": "plan", "execute": "execute"})
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
