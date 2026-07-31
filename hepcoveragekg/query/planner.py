"""
HEPCoverageKG query layer: the planner.

The part that decides. Everything else in `query/` is machinery a planner drives:
retrieval turns words into ids, templates turn ids into rows, and this chooses
which to call, in what order, and when to stop.

Shape (system.md S-27): **one planner, batched operations, one LLM call per
round.** It emits a batch of tool calls, they all execute, it sees every result
together, and it plans again. Measured on the pilot: a template query is 0.29 ms
and the heaviest is 16 ms, while one LLM call is seconds -- so one LLM call costs
3,000-17,000 queries. Parallelising operations buys nothing; the only cost that
matters is the number of round-trips to the model.

No separate "executor" agent: tool calling already emits structured calls, so a
translator agent would be a second LLM call per round doing a job the schema
does.

Three things are budgeted, and each is also a measurement:
    max_rounds   how many times it may think. Stops a wandering loop.
    max_places   how many operations per round. This is the 1-vs-many dial --
                 with it at 1 the planner is forced to be sequential, which
                 makes "does batching help?" a parameter sweep rather than a
                 second architecture.
    max_rows     how much of each result goes back into the context. The
                 planner's context is the constraint that would eventually
                 justify a worker hierarchy, so it is capped and logged rather
                 than allowed to grow silently.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_PLACES = 8
DEFAULT_MAX_ROWS = 25

# Tool descriptions are the part the planner actually reasons over -- more so
# than the SQL behind them -- so they say WHEN to reach for something, not just
# what it does.
TOOL_SPECS: list[dict] = [
    {
        "name": "search",
        "description": (
            "Turn words into entity ids. ALWAYS the first step: every other tool takes ids, "
            "not text. Returns the whole set a concept covers -- asking about 'Pythia' returns "
            "all 56 Pythia entities, because the graph holds one per version and tune."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "what to look for, in words"},
                "kind": {"type": "string", "description": "optional: restrict to one entity kind"},
                "limit": {"type": "integer", "description": "how many distinct entities (default 50)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "describe",
        "description": (
            "What an entity points AT. The forward hop. Use on a result to see what analysis "
            "it was: its final state, dataset, objects, systematics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_ids": {"type": "array", "items": {"type": "string"}},
                "predicate": {"type": "string", "description": "optional: only this relation"},
            },
            "required": ["entity_ids"],
        },
    },
    {
        "name": "subjects_of",
        "description": (
            "What points AT these entities. The backward hop. Use to go from a thing "
            "(a generator, a systematic) to the analyses that used it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicate": {"type": "string"},
                "object_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["predicate", "object_ids"],
        },
    },
    {
        "name": "papers_of",
        "description": "Which papers these entities appear in. Ends a chain.",
        "parameters": {
            "type": "object",
            "properties": {"entity_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["entity_ids"],
        },
    },
    {
        "name": "count",
        "description": (
            "How many papers and how many distinct facts link to these entities by this "
            "predicate. Returns papers, facts and assertions separately -- 'how many analyses' "
            "means PAPERS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicate": {"type": "string"},
                "object_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["predicate", "object_ids"],
        },
    },
    {
        "name": "compare",
        "description": "What two entities share and what only one has, through one predicate.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject_a": {"type": "string"},
                "subject_b": {"type": "string"},
                "predicate": {"type": "string"},
            },
            "required": ["subject_a", "subject_b", "predicate"],
        },
    },
    {
        "name": "crosstab",
        "description": (
            "A coverage grid: for each analysis, what it has under two predicates. Rows mean "
            "ONE analysis did both. Combinations absent from the grid are what nobody covered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicate_a": {"type": "string"},
                "predicate_b": {"type": "string"},
            },
            "required": ["predicate_a", "predicate_b"],
        },
    },
    {
        "name": "quotes",
        "description": "The verbatim sentence a fact came from. Use to support a claim.",
        "parameters": {
            "type": "object",
            "properties": {"assertion_id": {"type": "string"}},
            "required": ["assertion_id"],
        },
    },
    {
        "name": "answer",
        "description": (
            "Give the final answer and stop. Call this only when the retrieved rows support "
            "it. If the graph does not contain what was asked, say so here -- that is a valid "
            "and useful answer, not a failure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the answer, citing what was retrieved"},
                "answerable": {
                    "type": "boolean",
                    "description": "false if you cannot answer from this graph",
                },
                "reason": {
                    "type": "string",
                    "enum": ["answered", "not_in_graph", "out_of_scope"],
                    "description": (
                        "answered: the rows support an answer. "
                        "not_in_graph: the question IS about these physics papers, but the "
                        "graph does not record it -- you must search before claiming this. "
                        "out_of_scope: the question is not about high-energy-physics papers "
                        "at all (chemistry, general knowledge, the weather), so there is "
                        "nothing here to search for."
                    ),
                },
            },
            "required": ["text", "answerable", "reason"],
        },
    },
]


@dataclass
class Step:
    """One tool call and what came back. The unit of the trace."""

    round: int
    tool: str
    args: dict
    rows: int = 0
    error: Optional[str] = None
    seconds: float = 0.0
    preview: str = ""


@dataclass
class Thought:
    """What the planner said before acting, in one round.

    Models emit reasoning alongside their tool calls, and discarding it throws
    away the only record of WHY a chain was chosen. Two things need it: debugging
    a wrong answer (was the plan bad, or the execution?), and the evaluation,
    where "did it understand the question" is a different failure from "did it
    pick the right tool".
    """

    round: int
    text: str
    tools_called: list[str] = field(default_factory=list)


@dataclass
class Session:
    """A whole question, and every measurable thing about answering it."""

    question: str
    answer: str = ""
    answerable: bool = True
    # "answered" | "not_in_graph" | "out_of_scope".
    # The last two are different kinds of no, and conflating them would ruin the
    # signal this project produces: "no HEP paper here covers that" is COVERAGE
    # INFORMATION, while "that is not physics" is just the wrong tool. Only the
    # first is a fact about the literature.
    reason: str = "answered"
    steps: list[Step] = field(default_factory=list)
    rounds: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    stopped_because: str = ""

    thoughts: list[Thought] = field(default_factory=list)
    nudged: bool = False

    # Every literal value that came back from a tool, for the faithfulness check
    # (S-14). Accumulated as results arrive rather than reconstructed afterwards:
    # results are truncated before they enter the context, so replaying the steps
    # later would verify against less than the planner actually saw.
    seen_values: set[str] = field(default_factory=set)

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    @property
    def errors(self) -> int:
        return sum(1 for s in self.steps if s.error)

    @property
    def grounded_in_tools(self) -> bool:
        """Did anything get retrieved before the answer was given?

        False means the answer came from the model, not the graph -- which is a
        distinct failure from being wrong, and one the faithfulness check would
        only catch if the answer happened to contain a checkable number.
        """
        return any(not s.error for s in self.steps)

    @property
    def tool_usage(self) -> dict[str, int]:
        """How often each tool was called.

        Aggregated across a question set this is a direct read on whether the
        model understands the toolkit: a planner that only ever calls `search`
        and `count` has not grasped that the graph can be walked, and one that
        never calls `quotes` is answering without checking its evidence. Tool
        choice is diagnosable in a way that answer quality alone is not.
        """
        usage: dict[str, int] = {}
        for step in self.steps:
            usage[step.tool] = usage.get(step.tool, 0) + 1
        return dict(sorted(usage.items(), key=lambda kv: -kv[1]))

    def to_jsonl(self) -> str:
        """One line per session, for the trace log (S-22)."""
        return json.dumps({
            "question": self.question,
            "answer": self.answer,
            "answerable": self.answerable,
            "reason": self.reason,
            "rounds": self.rounds,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
            "evidence_ids": self.evidence_ids,
            "stopped_because": self.stopped_because,
            "tool_usage": self.tool_usage,
            "grounded_in_tools": self.grounded_in_tools,
            "nudged": self.nudged,
            "thoughts": [{"round": t.round, "text": t.text, "tools": t.tools_called}
                         for t in self.thoughts],
            "steps": [
                {"round": s.round, "tool": s.tool, "args": s.args, "rows": s.rows,
                 "error": s.error, "seconds": round(s.seconds, 4)}
                for s in self.steps
            ],
        }, ensure_ascii=False)


_VALUE_TOKEN = re.compile(r"[0-9][0-9,.]*|[A-Za-z0-9_:\-]{3,}")


def _values_in(rows: list[dict]) -> set[str]:
    """Every literal token a result contained, for grounding an answer against.

    Deliberately crude and over-inclusive: the check this feeds asks "could the
    answer have got this from the data", and a false *pass* costs less than a
    false accusation of hallucination. Numbers are normalised (commas dropped,
    trailing .0 removed) so "1,286" in a row grounds "1286" in an answer.
    """
    out: set[str] = set()
    for row in rows:
        for value in row.values():
            for token in _VALUE_TOKEN.findall(str(value)):
                token = token.strip(".,:").lower()
                if not token:
                    continue
                out.add(token)
                if token.replace(",", "").replace(".", "").isdigit():
                    out.add(token.replace(",", "").rstrip("0").rstrip(".") or "0")
                    out.add(token.replace(",", ""))
    return out


def _render_rows(rows: list[dict], max_rows: int) -> str:
    """Results as compact text for the planner's context.

    Truncated and SAID to be truncated: a planner that silently receives 25 of
    200 rows will reason as though it saw everything, and conclude something
    false with no way to notice.
    """
    if not rows:
        return "(no rows)"
    shown = rows[:max_rows]
    lines = [json.dumps(r, ensure_ascii=False, default=str)[:400] for r in shown]
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows not shown "
                     f"(narrow the query if you need them)")
    return "\n".join(lines)


def build_executor(conn, index) -> Callable[[str, dict], Any]:
    """Bind the tools to this database and index.

    Returned as a closure so the planner never holds a connection itself and
    cannot be handed a writable one by accident.
    """
    from hepcoveragekg.query import retrieve, templates

    def run(tool: str, args: dict):
        if tool == "search":
            hits = retrieve.search(index, args["text"], conn=conn,
                                   kind=args.get("kind"), limit=int(args.get("limit", 50)))
            ids = retrieve.concept(index, args["text"], conn=conn,
                                   kind=args.get("kind"), limit=int(args.get("limit", 50)))
            return templates.QueryResult(
                shape="search",
                rows=[{"entity_id": h.entity_id, "label": h.label, "kind": h.kind}
                      for h in hits],
                note=f"{len(ids)} entity ids after expanding canonical clusters: {ids}",
            )
        if tool == "describe":
            return templates.describe(conn, args["entity_ids"], args.get("predicate"))
        if tool == "subjects_of":
            return templates.subjects_of(conn, args["predicate"], args["object_ids"])
        if tool == "papers_of":
            return templates.papers_of(conn, args["entity_ids"])
        if tool == "count":
            return templates.count(conn, args["predicate"], args["object_ids"])
        if tool == "compare":
            return templates.compare(conn, args["subject_a"], args["subject_b"],
                                     args["predicate"])
        if tool == "crosstab":
            return templates.crosstab(conn, args["predicate_a"], args["predicate_b"])
        if tool == "quotes":
            return templates.quotes(conn, args["assertion_id"])
        raise ValueError(f"unknown tool: {tool}")

    return run


def system_prompt(conn, minimal: bool = False) -> str:
    """Purpose + what the graph contains. Assembled fresh so it cannot go stale."""
    from hepcoveragekg.query import prompts, schema_card

    purpose = prompts.PURPOSE_MINIMAL if minimal else prompts.PURPOSE
    return "\n\n".join([
        purpose,
        schema_card.render(conn),
        "Work by calling tools. You may call several in one turn when they do not depend on "
        "each other -- one round of several calls costs far less than several rounds of one. "
        "Call `answer` when the retrieved rows support an answer, or to say the graph does not "
        "hold what was asked.",
    ])


def _client():
    """OpenAI-compatible client, configured exactly as the aliases layer's."""
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "dummy"),
        timeout=float(os.environ.get("LLM_TIMEOUT", 120)),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", 3)),
    ), os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")


def answer(
    conn,
    index,
    question: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_places: int = DEFAULT_MAX_PLACES,
    max_rows: int = DEFAULT_MAX_ROWS,
    minimal_prompt: bool = False,
    chat: Optional[Callable] = None,
) -> Session:
    """Answer one question, returning the answer and the whole trace.

    `chat` is injectable so the loop can be tested without a model: it takes
    (messages, tools) and returns an object shaped like an OpenAI response.
    """
    session = Session(question=question)
    started = time.perf_counter()
    execute = build_executor(conn, index)

    tools = [{"type": "function", "function": spec} for spec in TOOL_SPECS]
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(conn, minimal_prompt)},
        {"role": "user", "content": question},
    ]

    if chat is None:
        client, model = _client()

        def chat(msgs, tls):  # noqa: E306
            return client.chat.completions.create(
                model=model, messages=msgs, tools=tls, temperature=0.0
            )

    for round_no in range(1, max_rounds + 1):
        session.rounds = round_no
        response = chat(messages, tools)
        session.llm_calls += 1
        usage = getattr(response, "usage", None)
        if usage:
            session.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            session.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])

        if not calls:
            # Answered in prose instead of calling a tool.
            #
            # If NOTHING has been retrieved yet, this is the failure the whole
            # project is built to prevent: an answer from the model's own
            # knowledge of physics rather than from this corpus. It reads as
            # authoritative and a domain expert would not blink at it.
            #
            # Even "the graph does not contain this" has to be ESTABLISHED by
            # looking. An abstention that was assumed rather than checked is a
            # guess wearing the costume of caution.
            #
            # So: nudge once, then accept and flag. A second refusal is a
            # finding, not something to keep spending rounds on.
            if not session.steps and not session.nudged:
                session.nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You have not retrieved anything yet. Answer only from this graph, "
                        "never from your own knowledge of physics. Use the tools to look, "
                        "then answer -- including to establish that the graph does not "
                        "contain what was asked."
                    ),
                })
                continue

            session.answer = (message.content or "").strip()
            session.stopped_because = ("answered from memory, no tool calls"
                                       if not session.steps
                                       else "answered without calling answer()")
            break

        if len(calls) > max_places:
            logger.info(f"round {round_no}: {len(calls)} calls capped to {max_places}")
            calls = calls[:max_places]

        reasoning = (message.content or "").strip()
        if reasoning:
            session.thoughts.append(Thought(
                round=round_no, text=reasoning,
                tools_called=[c.function.name for c in calls]))

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        })

        finished = False
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"ERROR: arguments were not valid JSON ({exc})"})
                session.steps.append(Step(round_no, name, {}, error="bad_arguments"))
                continue

            if name == "answer":
                claimed = str(args.get("reason", "answered"))
                # ONE RULE: look before answering, whatever the reason.
                #
                # An earlier version exempted `out_of_scope`, on the grounds
                # that a chemistry question has nothing to check against a HEP
                # corpus. True, but it made "not physics" a **cheap exit** -- a
                # verdict that skips all work -- and a model offered a free way
                # out will sometimes take it for questions that were perfectly
                # answerable. Detecting a wrong out_of_scope claim would itself
                # require searching, so the exemption removed the only evidence
                # that could have contradicted it.
                #
                # The exemption bought one LLM call on absurd questions, which
                # are near-absent from a physics question set. `reason` is still
                # recorded -- it separates a real coverage gap from a wrong-tool
                # question -- it just no longer buys a skip.
                if not session.steps and not session.nudged:
                    session.nudged = True
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": (
                        "You have not retrieved anything yet. Search first, then answer. "
                        "This applies even when you expect to find nothing: 'no paper here "
                        "covers that' is a claim about the literature and has to be checked. "
                        "If the question is not about high-energy-physics papers at all, "
                        "search anyway (it costs nothing) and then answer with "
                        "reason='out_of_scope'.")})
                    continue

                session.answer = str(args.get("text", "")).strip()
                session.answerable = bool(args.get("answerable", True))
                session.reason = claimed
                session.stopped_because = f"answered ({claimed})"
                finished = True
                break

            t0 = time.perf_counter()
            try:
                result = execute(name, args)
                elapsed = time.perf_counter() - t0
                session.evidence_ids.extend(getattr(result, "evidence_ids", []) or [])
                # BEFORE truncation: the planner is shown max_rows, but a claim
                # is grounded if the graph returned it at all.
                session.seen_values |= _values_in(result.rows)
                body = _render_rows(result.rows, max_rows)
                if result.note:
                    body += f"\n[{result.note}]"
                session.steps.append(Step(round_no, name, args, rows=len(result.rows),
                                          seconds=elapsed, preview=body[:200]))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": body})
            except Exception as exc:  # noqa: BLE001 -- the planner must see any failure
                elapsed = time.perf_counter() - t0
                session.steps.append(Step(round_no, name, args, error=str(exc)[:200],
                                          seconds=elapsed))
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": f"ERROR: {exc}"})

        if finished:
            break
    else:
        session.stopped_because = f"hit max_rounds ({max_rounds})"

    session.evidence_ids = sorted(set(session.evidence_ids))
    session.seconds = time.perf_counter() - started
    return session
