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
from types import SimpleNamespace
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_PLACES = 8
DEFAULT_MAX_ROWS = 25

# How many distinct entities a `search` returns. NOT exposed to the model.
#
# It used to be, and that is a decision a model should not be making: breadth
# decides correctness here. `concept("Pythia", limit=6)` yields 45 papers;
# limit=60 yields 58, the true answer. A model economising on breadth produces a
# wrong answer with a *clean* trace -- it searched, it got results, nothing looks
# amiss.
#
# The value below is a considered default, NOT a measured one, and the measured
# thing is that no single value can be right. Relative-score cutoffs behave
# completely differently by concept:
#   "Pythia"      rank 20 scores 50% of the best hit and is still a real Pythia
#   "top squark"  rank 20 scores 71% and is already "single top" -- a different
#                 particle entirely
# A narrow concept with many spellings and a broad phrase that brushes many
# neighbours cannot share a threshold. 60 is set to cover the observed concept
# sizes (Pythia 31 clusters, b-jet 47, jet energy scale 109) while stopping the
# broad ones running to 192.
#
# Tuning this is a job for the question set: sweep it, and see which questions
# change answer. Guessing harder now would only look like rigour.
SEARCH_BREADTH = 60

# Hard cap on one completion. Without it a single call can run to the context
# limit, and on 2026-08-02 that stalled an evaluation run for an hour on one
# question ("How many analyses define the object Electron?").
#
# The arithmetic is the point, because the failure is silent and looks like a
# hang rather than an error: the client allows 120s per request with 3 retries,
# and a 72B AWQ on one A100 generates ~26 tokens/s. So ANY completion beyond
# ~3,100 tokens times out, is retried, generates again from scratch, and the
# server ends up working on stacked abandoned requests while the client makes no
# progress. Six rounds x 4 attempts x 120s = 48 minutes for one question.
#
# 800 is generous for what a planner legitimately emits: a few tool calls and a
# short prose answer. S-29 exists precisely so the model never retypes entity
# ids, which was the only thing that ever needed a long completion -- so a
# completion running past this cap means something has gone wrong, and truncating
# it surfaces that as a visible bad answer instead of an hour of silence.
MAX_COMPLETION_TOKENS = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", 800))

# Sent when the model tries to answer having retrieved nothing. Shared with
# graph.py so the two cannot drift apart.
NUDGE = (
    "You have not retrieved anything yet. Search first, then answer. "
    "This applies even when you expect to find nothing: 'no paper here covers that' "
    "is a claim about the literature and has to be checked. If the question is not "
    "about high-energy-physics papers at all, search anyway (it costs nothing) and "
    "then answer with reason='out_of_scope'."
)

UNKNOWN_ID_MESSAGE = (
    "ERROR: these ids were never returned by a search: {unknown}. "
    "Do not invent ids. Call `search` FIRST, wait for its results, and then use the "
    "exact entity_id values it gives you -- or better, pass the set name it saved "
    "(object_set / entity_set). A tool that depends on another tool's output cannot "
    "be called in the same turn as it."
)

# Tool descriptions are the part the planner actually reasons over -- more so
# than the SQL behind them -- so they say WHEN to reach for something, not just
# what it does.
TOOL_SPECS: list[dict] = [
    {
        "name": "search",
        "description": (
            "Find the entities a concept covers, and SAVE them under a short name you can "
            "reuse. ALWAYS the first step. Returns something like 'saved as set_1 (31 "
            "entities)'; pass set_1 to other tools as `object_set` or `entity_set` instead of "
            "listing ids. Asking about 'Pythia' finds all of them at once, because the graph "
            "holds one entity per version and tune."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "what to look for, in words"},
                "kind": {"type": "string", "description": "optional: restrict to one entity kind"},
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
                "entity_ids": {"type": "array", "items": {"type": "string"},
                    "description": "explicit ids; prefer entity_set"},
                "entity_set": {"type": "string",
                    "description": "a set name returned by search, e.g. set_1"},
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
                "object_ids": {"type": "array", "items": {"type": "string"},
                    "description": "explicit ids; prefer object_set"},
                "object_set": {"type": "string",
                    "description": "a set name returned by search, e.g. set_1"},
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
        "name": "contents_of",
        "description": (
            "What one paper says. Give the arXiv id directly -- do NOT search for a paper "
            "and do NOT pass a paper id to describe. This is the only way to answer "
            "'which generators / systematics / regions does analysis 2001.06899 use?'. "
            "Optionally filter to one predicate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"},
                    "description": "arXiv ids, e.g. ['2001.06899']"},
                "predicate": {"type": "string",
                    "description": "optional; omit to get everything the paper says"},
            },
            "required": ["paper_ids"],
        },
    },
    {
        "name": "facets",
        "description": (
            "Papers whose analysis card carries these closed-vocabulary values. THE CHEAPEST "
            "ROUTE: no search, no spelling, a set operation over a fixed menu. Use it whenever "
            "the question names a standard object, technique or process family that appears in "
            "the FACET VOCABULARY -- 'which searches select b-jets and missing transverse "
            "momentum' is facets(field='objects', values=['BJet','MET'], mode='all', "
            "category='search'). Values must come from the vocabulary EXACTLY as written "
            "('BJet', not 'b-jet'). "
            "Returns each paper with the labels that caused the match: read them. The tag is a "
            "family, the label is what the paper actually did -- six papers tagged ABCD use six "
            "different ABCD variants. The vocabulary is closed and misses roughly a quarter of "
            "entities, so this is a CANDIDATE SET, not a final count; confirm with the labels, "
            "or with search/contents_of when the question is not about a standard category."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string",
                    "description": "one of: objects, generators, process_families, "
                                   "background_methods, statistical_methods, "
                                   "systematic_sources, observable_types"},
                "values": {"type": "array", "items": {"type": "string"},
                    "description": "vocabulary keys, exactly as spelled in the schema card"},
                "mode": {"type": "string", "enum": ["all", "any"],
                    "description": "'all' = the paper has every value (AND); 'any' = at least "
                                   "one (OR). Default 'all'."},
                "category": {"type": "string", "enum": ["search", "measurement"],
                    "description": "optional filter on the paper's category"},
                "experiment": {"type": "string",
                    "description": "optional, e.g. 'ATLAS' or 'CMS'"},
            },
            "required": ["field", "values"],
        },
    },
    {
        "name": "facet_entities",
        "description": (
            "The distinct THINGS carrying one facet tag, with their labels -- not the papers. "
            "`facets` answers 'which papers use an ABCD-family estimate'; this answers 'and what "
            "are they', which is the question a coverage map is usually really being asked: "
            "how many different ways does this literature do X, and what are they. "
            "Returns three counts because they differ: papers, raw entity records, and distinct "
            "things after alias merging. Use it for 'how many different / which different / what "
            "kinds of' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "the facet field"},
                "value": {"type": "string",
                    "description": "one key, taken from a search hit's facets"},
                "kind": {"type": "string",
                    "description": "optional entity-kind filter"},
            },
            "required": ["field", "value"],
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
                "object_ids": {"type": "array", "items": {"type": "string"},
                    "description": "explicit ids; prefer object_set"},
                "object_set": {"type": "string",
                    "description": "a set name returned by search, e.g. set_1"},
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

    # Entity ids the graph has actually handed back this session. Anything else
    # passed to a tool was invented -- see _check_ids.
    known_entity_ids: set[str] = field(default_factory=set)
    invented_ids: list[str] = field(default_factory=list)
    # Tool calls the model wrote as text and we parsed back out. A count worth
    # watching: if it is high, the serving-side tool parser is underperforming.
    recovered_calls: int = 0

    # Named result sets. The model referred to 31 entities by writing out 31
    # ids -- 863 completion tokens for one call, every one an opportunity to
    # mistype or invent. A handle costs three tokens and cannot be misspelt
    # into something that silently returns zero.
    sets: dict[str, list[str]] = field(default_factory=dict)

    # Filled by the graph's `finish` node, so no answer leaves unchecked.
    verification: Any = None

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
            "invented_ids": self.invented_ids,
            "recovered_calls": self.recovered_calls,
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


# A tool call the model wrote as TEXT instead of emitting structurally.
#
# Observed live: after the id guard corrected it, Qwen produced exactly the right
# call -- correct predicate, correct ids -- but wrapped in <tool_call> tags in
# the message body, where vLLM's parser did not pick it up. The loop then saw
# "no tool calls", assumed the model had answered in prose, and recorded raw
# JSON as the final answer while the model was still mid-work.
#
# "No structured tool calls" is not the same as "answered in prose". Recovering
# the call is a few lines and turns a wasted session into a working one.
_TEXT_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)


class _RecoveredCall:
    """Shaped like an SDK tool call, so the loop needs no special case."""

    def __init__(self, name: str, arguments: str, index: int):
        self.id = f"recovered-{index}"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)


def _recover_tool_calls(content: str) -> list:
    """Tool calls the model wrote into the message body."""
    out: list = []
    for i, blob in enumerate(_TEXT_TOOL_CALL.findall(content or "")):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue  # truncated mid-JSON; nothing safe to recover
        name = parsed.get("name")
        if not name:
            continue
        args = parsed.get("arguments", parsed.get("parameters", {}))
        out.append(_RecoveredCall(
            name, args if isinstance(args, str) else json.dumps(args), i))
    return out


# Arguments that must name entities the graph already returned.
_ID_ARGS = ("entity_ids", "object_ids", "subject_a", "subject_b")


def _check_ids(args: dict, known: set[str]) -> list[str]:
    """Ids in this call that the graph never returned.

    The failure this catches, observed on the first live run: the model batched
    `search` and `count` in ONE round, so when it wrote `count` the search
    results did not exist yet -- and rather than wait, it invented a
    plausible-looking id ("gen-223") and got a confident zero back.

    Batching is right for independent operations and wrong for dependent ones,
    and the model cannot always tell which it has. The prompt says so in words;
    this says so in a way that cannot be overlooked. A wrong answer becomes a
    corrective message instead.
    """
    unknown: list[str] = []
    for name in _ID_ARGS:
        value = args.get(name)
        if value is None:
            continue
        for candidate in ([value] if isinstance(value, str) else value):
            if isinstance(candidate, str) and candidate not in known:
                unknown.append(candidate)
    return unknown


# Every column a template can return an entity id in. Narrower than this is a
# guard bug, not caution: `known_entity_ids` is what `_check_ids` accepts, so an
# id the graph itself returned but that is not listed here gets rejected as
# invented. `contents_of` exposed it on 2026-08-03 -- its rows carry `object_id`,
# so nothing it returned was ever recognised -- and `describe` had the same
# latent problem, which is why chaining one describe into another failed.
_ID_COLUMNS = ("entity_id", "canonical_id", "object_id", "subject_id")


def _collect_ids(rows: list[dict], note: str) -> set[str]:
    """Entity ids a result handed back, from its rows and its note."""
    found = {str(r[k]) for r in rows for k in _ID_COLUMNS
             if isinstance(r, dict) and r.get(k)}
    found |= set(re.findall(r"hepkg:[a-z_]+:[A-Za-z0-9_.\-]+", note or ""))
    return found


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


def build_executor(conn, index, sets: Optional[dict] = None) -> Callable[[str, dict], Any]:
    """Bind the tools to this database, index and set of named results.

    Returned as a closure so the planner never holds a connection itself and
    cannot be handed a writable one by accident.
    """
    from hepcoveragekg.query import retrieve, templates

    sets = {} if sets is None else sets

    def ids_for(args: dict, id_arg: str, set_arg: str) -> list[str]:
        """Ids from a saved set name, or written out explicitly.

        Set names exist because the alternative was measured and is bad: asked
        about 31 Pythia entities, the model wrote out every id -- 863 completion
        tokens on one call, and on a retry it looped and produced 6,521
        characters before being cut off mid-identifier. A handle is three tokens
        and cannot be mistyped into something that silently returns nothing.
        """
        name = args.get(set_arg)
        if name:
            if name not in sets:
                raise ValueError(
                    f"no set named '{name}'. Available: {sorted(sets) or 'none yet'}. "
                    "Run `search` first; it tells you the name it saved.")
            return sets[name]
        explicit = args.get(id_arg)
        if not explicit:
            raise ValueError(f"give either {set_arg} (preferred) or {id_arg}")
        return [explicit] if isinstance(explicit, str) else list(explicit)

    def run(tool: str, args: dict):
        if tool == "search":
            # The model may not narrow this; see SEARCH_BREADTH.
            limit = SEARCH_BREADTH
            hits = retrieve.search(index, args["text"], conn=conn,
                                   kind=args.get("kind"), limit=limit)
            ids = retrieve.concept(index, args["text"], conn=conn,
                                   kind=args.get("kind"), limit=limit)
            name = f"set_{len(sets) + 1}"
            sets[name] = ids
            # `facets` is included on every hit that has one. This is how the
            # model learns which closed-vocabulary key covers a concept -- from
            # an entity that demonstrably exists, rather than from a vocabulary
            # list in the prompt. Shipping the list instead would have put five
            # of the seven Tier 1 gold keys into the system prompt (D-054).
            rows = []
            for h in hits:
                row = {"entity_id": h.entity_id, "label": h.label, "kind": h.kind}
                if h.facets:
                    row["facets"] = h.facets
                rows.append(row)
            tagged = sum(1 for h in hits if h.facets)
            note = (f"saved as {name} ({len(ids)} entities, canonical clusters expanded). "
                    f"Pass {name} as object_set/entity_set -- do not retype the ids.")
            if tagged:
                note += (f" {tagged}/{len(hits)} hits carry facet tags; those keys are"
                         " what the `facets` tool takes.")
            return templates.QueryResult(shape="search", rows=rows, note=note)
        if tool == "describe":
            return templates.describe(conn, ids_for(args, "entity_ids", "entity_set"),
                                      args.get("predicate"))
        if tool == "subjects_of":
            return templates.subjects_of(conn, args["predicate"],
                                         ids_for(args, "object_ids", "object_set"))
        if tool == "papers_of":
            return templates.papers_of(conn, ids_for(args, "entity_ids", "entity_set"))
        if tool == "contents_of":
            papers = args.get("paper_ids") or args.get("paper_id") or []
            if isinstance(papers, str):
                papers = [papers]
            return templates.contents_of(conn, papers, args.get("predicate"))
        if tool == "facets":
            values = args.get("values") or args.get("value") or []
            if isinstance(values, str):
                values = [values]
            return templates.facets(
                conn, args["field"], values,
                mode=args.get("mode", "all"),
                category=args.get("category"),
                experiment=args.get("experiment"),
            )
        if tool == "facet_entities":
            value = args.get("value")
            if isinstance(value, list):      # a values=[...] slip; one key is meant
                value = value[0] if value else None
            if not value:
                raise ValueError("facet_entities needs a single `value`")
            return templates.facet_entities(conn, args["field"], value,
                                            kind=args.get("kind"))
        if tool == "count":
            return templates.count(conn, args["predicate"],
                                   ids_for(args, "object_ids", "object_set"))
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


def _prepare(conn, index, question, max_rounds, max_places, max_rows,
             minimal_prompt, chat, thread_id):
    """The state and runtime config a run needs. Shared by answer() and stream()."""
    session = Session(question=question)
    if chat is None:
        client, model = _client()

        def chat(msgs, tls):  # noqa: E306
            return client.chat.completions.create(
                model=model, messages=msgs, tools=tls, temperature=0.0,
                max_tokens=MAX_COMPLETION_TOKENS)

    state = {
        "question": question,
        "messages": [
            {"role": "system", "content": system_prompt(conn, minimal_prompt)},
            {"role": "user", "content": question},
        ],
        "session": session,
        "round": 0,
        "max_rounds": max_rounds,
        "max_places": max_places,
        "max_rows": max_rows,
    }
    runtime = {
        "thread_id": thread_id,
        "chat": chat,
        "execute": build_executor(conn, index, session.sets),
        "tools": [{"type": "function", "function": spec} for spec in TOOL_SPECS],
    }
    config = {"configurable": runtime, "recursion_limit": max_rounds * 3 + 6}
    return session, state, config


def stream(
    conn,
    index,
    question: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_places: int = DEFAULT_MAX_PLACES,
    max_rows: int = DEFAULT_MAX_ROWS,
    minimal_prompt: bool = False,
    chat: Optional[Callable] = None,
    checkpointer: Any = None,
    thread_id: str = "default",
):
    """Yield `(node_name, session)` after each node completes.

    The same run as `answer()`, surfaced step by step. Exists so a UI can show
    what the system is doing while it does it -- which for a non-technical
    reader is the difference between a spinner and an explanation.
    """
    from hepcoveragekg.query import graph as graph_module

    session, state, config = _prepare(conn, index, question, max_rounds,
                                      max_places, max_rows, minimal_prompt,
                                      chat, thread_id)
    started = time.perf_counter()
    app = graph_module.build(checkpointer=checkpointer)
    for update in app.stream(state, config=config, stream_mode="updates"):
        for node_name in update:
            session.seconds = time.perf_counter() - started
            yield node_name, session
    session.seconds = time.perf_counter() - started


def answer(
    conn,
    index,
    question: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_places: int = DEFAULT_MAX_PLACES,
    max_rows: int = DEFAULT_MAX_ROWS,
    minimal_prompt: bool = False,
    chat: Optional[Callable] = None,
    checkpointer: Any = None,
    thread_id: str = "default",
) -> Session:
    """Answer one question, returning the answer and the whole trace.

    The loop itself lives in `graph.py` as a LangGraph state machine; this stays
    the public entry point so callers -- and the tests that pin the loop's
    behaviour -- do not change.

    `chat` is injectable so the loop can be tested without a model: it takes
    (messages, tools) and returns an object shaped like an OpenAI response.

    `thread_id` is carried for later: with a checkpointer, reusing it resumes the
    same conversation, which is how follow-up questions ("how many of *those*
    used Herwig?") will get a referent.
    """
    from hepcoveragekg.query import graph as graph_module

    started = time.perf_counter()
    session, state, config = _prepare(conn, index, question, max_rounds,
                                      max_places, max_rows, minimal_prompt,
                                      chat, thread_id)
    graph_module.build(checkpointer=checkpointer).invoke(state, config=config)

    session.seconds = time.perf_counter() - started
    return session
