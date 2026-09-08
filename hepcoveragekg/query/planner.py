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

# The ceiling on widening, when a critic is present to say the tail is still
# relevant (D-060). Without a critic nothing widens and `SEARCH_BREADTH` is the
# whole story.
#
# A stop is still needed, for two reasons that are not about cost. The
# retriever's tail eventually becomes the corpus -- the broadest concepts run to
# 192 entities and there is no rank at which BM25 declines to answer -- and each
# doubling is another round of critic calls whose verdicts feed the decision to
# double again. 240 is four doublings from 60 and comfortably past the largest
# concept measured (`jet energy scale`, 109 clusters); a search that wants more
# than that is a question about the whole corpus, which `crosstab` and `facets`
# answer better than a widening search ever will.
MAX_SEARCH_BREADTH = int(os.environ.get("SEARCH_BREADTH_MAX", 240))

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
# ...ALL OF WHICH ASSUMES THE MODEL ANSWERS DIRECTLY. A reasoning model does
# not: it emits its chain of thought first, billed as output and invisible in
# the reply, and that comes out of this same allowance. At 800 it is truncated
# mid-thought and returns an empty message with no tool call -- the harness
# records `answered=True` with empty text, `papers` still holds the retrieval
# footprint, and the scorer correctly gives it zero. A plumbing limit then reads
# as a model that cannot answer. Measured on qwen3-32b, same everything except
# this number: free-SQL 0.199 -> 0.592, empty answers 13/24 -> 0/24 (D-084).
#
# So the default is now chosen by what the model IS, not by what the 72B needed.
# The env var still wins, because the cluster's timeout arithmetic above is real
# and an operator serving a slow local model must be able to pull it back down.
MAX_COMPLETION_TOKENS = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", 800))
REASONING_COMPLETION_TOKENS = int(
    os.environ.get("LLM_REASONING_COMPLETION_TOKENS", 4000))



def planner_temperature() -> float:
    """Sampling temperature for the PLANNER's own calls. 0.0 unless asked.

    WHY THIS EXISTS, AND WHY IT DEFAULTS TO ZERO. Every call site hardcoded
    `temperature=0.0`, so "repeats" were four greedy runs that differed only by
    batch-scheduling nondeterminism (D-101) -- accidental floating-point noise,
    not diversification. That made D-102's comparison of rewording against
    "resampling" a comparison against nothing, and its conclusion that
    temperature is dominated unsupported (D-103).

    Read FRESH from the environment on every call, not captured at import, for
    the reason `completion_cap` is: a value exported by a job script after this
    module is imported would otherwise be accepted and then silently ignored.

    Deliberately NOT applied to the critic. The critic is a judge, and a judge
    that answers differently on reruns stops being a fixed yardstick -- the
    whole reason `--critic-seed` shuffles ROW ORDER rather than sampling. Mixing
    a sampled planner with a sampled critic would also confound which half any
    change came from.
    """
    raw = os.environ.get("PLANNER_TEMPERATURE", "").strip()
    if not raw:
        return 0.0
    try:
        t = float(raw)
    except ValueError:
        logger.warning("PLANNER_TEMPERATURE=%r is not a number; using 0.0", raw)
        return 0.0
    if not 0.0 <= t <= 2.0:
        logger.warning("PLANNER_TEMPERATURE=%s out of range [0,2]; using 0.0", t)
        return 0.0
    return t


def completion_cap(model: str) -> int:
    """The output allowance for `model`, reasoning models getting the larger one."""
    # Read fresh, not from the module constant: that is frozen at import, so a
    # value set afterwards (a job script exporting it, a test) would be accepted
    # as "an override is present" and then silently ignored in favour of 800.
    override = os.environ.get("LLM_MAX_COMPLETION_TOKENS")
    if override:
        return int(override)                  # explicit operator override wins
    from hepcoveragekg.query import budget as _b
    name = (model or "").strip().lower()
    if name in _b.REASONING_MODELS or any(
            k in name for k in ("qwq", "qwen3", "thinking", "-r1", "reason")):
        return REASONING_COMPLETION_TOKENS
    return MAX_COMPLETION_TOKENS

# Sent when the model tries to answer having retrieved nothing. Shared with
# graph.py so the two cannot drift apart.
NUDGE = (
    "You have not retrieved anything yet. Search first, then answer. "
    "This applies even when you expect to find nothing: 'no paper here covers that' "
    "is a claim about the literature and has to be checked. If the question is not "
    "about high-energy-physics papers at all, search anyway (it costs nothing) and "
    "then answer with reason='out_of_scope'."
)

# Sent when the model abstains while holding rows it has itself judged relevant.
#
# Deliberately NOT "answer anyway". A false "not in the graph" is a fabricated
# claim about the literature, and a false "here is the answer" is worse -- so the
# only safe challenge is one a correct abstention can pass. Asking for the reason
# does that: "all 13 are Sherpa 2.2.2 and the question asked for 2.2.1" is a
# complete response and gets recorded as the justification.
ABSTENTION_CHALLENGE = (
    "You are about to report that the graph does not contain this, while holding "
    "{held} retrieved rows judged relevant to it. That may well be right -- near "
    "misses are not answers. Say plainly why those rows do NOT answer the "
    "question, then give your answer. If they genuinely do not, keep "
    "reason='not_in_graph' and state the mismatch; that is a useful finding, not "
    "a failure. Do not invent an answer to satisfy this message."
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
        "name": "path",
        "description": (
            "Entities satisfying SEVERAL predicate->object constraints AT ONCE, and optionally "
            "the papers they appear in. Use this the moment a question says AND: 'searches using "
            "b-tagged jets AND missing transverse momentum', 'a ttZ background AND a control "
            "region normalising it'. One call replaces a hop per condition, and the conditions "
            "are applied together rather than one after another -- which is the difference "
            "between 8 papers that satisfy both and 40 that satisfy either. "
            "Each constraint is {predicate, object_ids}; pass a saved set's ids for object_ids. "
            "project='papers' resolves the survivors to arXiv ids in the same call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "constraints": {
                    "type": "array",
                    "description": "one per condition in the question",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predicate": {"type": "string"},
                            "object_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["predicate", "object_ids"],
                    },
                },
                "project": {"type": "string",
                    "description": "'subjects' (default) or 'papers'"},
                "mode": {"type": "string",
                    "description": "'all' (default, intersect -- what AND means) or 'any'"},
            },
            "required": ["constraints"],
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
        "name": "refine",
        "description": (
            "Drop entities you can see do NOT belong, and get back what remains. Saves a new "
            "set you can count or cite. Use it when a search or a hop returned things that "
            "are clearly not what the question asked about -- your own judgement, recorded "
            "with a reason. Nothing is deleted from the original set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_set": {"type": "string", "description": "the set to refine, e.g. set_1"},
                "drop_ids": {"type": "array", "items": {"type": "string"},
                             "description": "entity ids that do not belong"},
                "reason": {"type": "string",
                           "description": "why they do not belong -- recorded, not optional"},
            },
            "required": ["entity_set", "drop_ids", "reason"],
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
                "papers_from": {
                    "type": "string",
                    "description": (
                        "CITE, do not retype. The name of a set whose papers ARE the answer "
                        "(e.g. set_1_kept) -- the paper list is taken from it. Use this "
                        "instead of writing arXiv ids into `text`."
                    ),
                },
                "value_from": {
                    "type": "string",
                    "description": (
                        "CITE, do not retype. For a 'how many' question: which `count` call "
                        "holds the answer -- 'step_3', or just 'count' for the last one. The "
                        "number is lifted from that result, so call `count` first and cite it "
                        "rather than tallying rows by eye. A set cannot be cited here: its "
                        "size is how wide the search was, not the answer."
                    ),
                },
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


# Tools that exist only under the v2 answer contract, and the `answer` fields
# that go with them. Switchable because they are an ABLATION ARM, not a
# refactor: adding a tool changes the tool list the model sees, and changing
# `answer`'s schema changes the prompt, so a run with them differs from one
# without by more than the thing under test. Every comparison this week that
# went wrong went wrong exactly there.
V2_TOOLS = ("refine",)
V2_ANSWER_FIELDS = ("papers_from", "value_from")

# v3 keeps only the part of the contract that MEASURED well and drops the two
# that did not. Evidence, 2026-08-17:
#
#   citing papers     kept -- untested, and the mechanism a citation is for
#   citing a count    DROPPED. It reports `count(predicate, set)` faithfully,
#                     and the set is too broad: 36 papers where the gold is 2.
#                     The model's prose guess beat it, 0.314 to 0.186. Being
#                     more truthful about a wrong set is worse than estimating.
#   `refine`          DROPPED. It competes with the critic for one job, and the
#                     model prefers its own: 67% of counts ran over
#                     `set_N_refined` against 18% over the critic's `set_N_kept`.
#                     Two mechanisms doing the same thing, introduced together,
#                     so neither can be attributed.
#   abstention        kept -- fired 48 times, effect still unmeasured
#
# So v3 is one claim: cite what you found instead of retyping it, and defend an
# abstention you make while holding evidence.
V3_ANSWER_FIELDS = ("papers_from",)


#: The literal-list answer field, replacing `papers_from`'s indirect reference.
#:
#: WHY THIS EXISTS. `papers_from` names a SET created earlier in the run --
#: "set_1_kept" -- so the model must remember which set was saved when, and that
#: the critic's filtered copy is the one holding the answer. That is a reference
#: to internal state. free-SQL's tool just takes the ids.
#:
#: Measured on gpt-5.6-sol, same model and same eight questions on one
#: afternoon: typed 0.287 with named_none 0.917, free-SQL 0.669 with
#: named_none 0.000. The typed run had the BETTER process -- 0.991 of calls
#: productive against 0.990, 3.96 distinct tools against 2.00, never stopping
#: early -- and lost it at the handoff.
#:
#: So this separates two very different conclusions that the data cannot
#: currently tell apart: "typed retrieval is worse than SQL" and "typed
#: retrieval is fine and our answer contract is awkward".
SIMPLE_ANSWER_FIELD = {
    "papers": {
        "type": "array",
        "items": {"type": "string"},
        "description": ("the arXiv ids this answer asserts, e.g. "
                        "['2106.01676', '2001.06899']. Write them out."),
    }
}


#: OFF unless asked for. Adding a tool rewrites the prompt for EVERY question,
#: so an unflagged `path` would move the baseline it is meant to be measured
#: against -- the same reason `--search-sets` keeps its tool hidden when off.
PATH_TOOL = "path"


def tools_for(answer_contract: bool = False, contract: str = "",
              simple_answer: bool = False, path_tool: bool = False,
              name_ids: bool = False) -> list[dict]:
    """The tool schemas for one run.

    `contract` is "v1" (default), "v2" (everything) or "v3" (the lean version).
    `answer_contract=True` still means v2, so existing callers do not move.
    """
    import copy

    contract = contract or ("v2" if answer_contract else "v1")
    keep = {"v1": (), "v2": V2_ANSWER_FIELDS, "v3": V3_ANSWER_FIELDS}[contract]

    specs = []
    for spec in TOOL_SPECS:
        if spec["name"] in V2_TOOLS and contract != "v2":
            continue
        if spec["name"] == PATH_TOOL and not path_tool:
            continue
        if spec["name"] == "answer":
            spec = copy.deepcopy(spec)
            for field_name in V2_ANSWER_FIELDS:
                if field_name not in keep:
                    spec["parameters"]["properties"].pop(field_name, None)
            if simple_answer:
                # Swap the indirect reference for a literal list. `papers_from`
                # goes, so the two cannot both be offered -- a model given both
                # would pick one arbitrarily and the arm would measure nothing.
                spec["parameters"]["properties"].pop("papers_from", None)
                spec["parameters"]["properties"].update(SIMPLE_ANSWER_FIELD)
                spec["parameters"]["properties"]["text"]["description"] = (
                    "the answer, in prose")
            if contract != "v1":
                # An answer with no text is not an answer. The schema never said
                # so -- there was no `required` array at all -- and Qwen filled
                # `text` anyway, so nothing surfaced it. gpt-5.6-luna called
                # `answer` with no arguments on 24 of 24 questions: `answered`
                # went True, `text` stayed empty, and `judged_f1` then scored the
                # 54-paper retrieval footprint at 0.635, which read like a win.
                #
                # Applied to v2 and v3 ONLY. v1 is the frozen control, pinned at
                # fingerprint aa028ae68fd37d83, and a schema change there would
                # silently move the baseline every earlier result is measured
                # against -- the D-062 failure exactly.
                spec["parameters"]["required"] = ["text", "reason"]
                # v2 replaces the wording, because "citing what was retrieved"
                # now means something specific -- naming a set, not listing ids
                # in prose.
                spec["parameters"]["properties"]["text"]["description"] = (
                    "the answer, in prose. Cite the set of papers in `papers_from` "
                    "rather than writing arXiv ids into this text.")
                if name_ids:
                    # THE ARM (D-117). The line above tells the model NOT to
                    # write arXiv ids into `text` -- and `text` is the only
                    # field `set_f1` and `judged_set_f1` read. The system has
                    # been instructed away from the one thing that scores, and
                    # the citation it was pointed at instead resolved on 7% of
                    # answers (D-116).
                    #
                    # An arm and not a default: v3 is what every recent result
                    # was measured on, and rewording it silently is the D-062
                    # failure -- a one-line prompt change that moved the control
                    # while an arm was being read.
                    spec["parameters"]["properties"]["text"]["description"] = (
                        "The answer in prose, AND the arXiv ids of the papers it "
                        "is about, written out: '2004.14060, 2006.05880'. Write "
                        "the ids even when you also cite a set in `papers_from`.")
                    spec["parameters"]["properties"]["papers_from"]["description"] = (
                        "OPTIONAL, in ADDITION to the ids in `text`. The name of a "
                        "set from this run, exactly as reported (e.g. "
                        "'set_1_kept'). Not a description of one: 'the facets "
                        "result' names no set and resolves to nothing.")
                    spec["description"] += (
                        " CALL THIS TOOL. An answer written as prose, or as a tag "
                        "like <answer .../>, is not a tool call.")
        specs.append(spec)
    return specs


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
    # The tool the model actually asked for, when the call was rewritten before
    # it ran. Recorded rather than swallowed: a redirect that leaves no trace
    # would make the planner look as though it had chosen correctly, and tool
    # selection is something this project measures (S-49).
    redirected_from: Optional[str] = None
    # The result itself, for the few tools whose output IS an answer rather than
    # a list to read. Kept so `answer(value_from=...)` can lift the number
    # straight out instead of the model retyping it or the harness re-deriving
    # it -- one row of three integers, not a transcript.
    result: Optional[dict] = None


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

    # What the final answer CITED rather than retyped (S-29, applied to output).
    # The model wrote 11 arXiv ids into prose at the median and got the wrong
    # ones: 100% of the papers it named were genuinely retrieved, and only 22.8%
    # of the gold ones were among them. It is a SELECTION failure, and a citation
    # cannot select wrongly.
    answer_papers: list[str] = field(default_factory=list)
    answer_value: Optional[float] = None
    answer_cited: str = ""

    # Set when an abstention was challenged for a reason, and what came back.
    # The challenge asks the model to JUSTIFY, never to answer differently --
    # a system taught not to abstain would fabricate coverage, which is worse
    # than the false negatives this is aimed at.
    abstention_challenged: bool = False

    # Every non-answer call made in this run -> what it returned, so an exact
    # repeat is answered from the record rather than re-executed. See
    # graph.execute; a retry is a round not spent on a different route.
    calls_made: dict = field(default_factory=dict)

    # Which rungs of the widening ladder have been offered, and whether the
    # model then made a DIFFERENT call or abstained anyway. The second is the
    # measurement that matters: a mechanism the model ignores is a mechanism
    # that does not work, and without this we would be measuring whether the
    # suggestion was made rather than whether it helped.
    widenings_used: set = field(default_factory=set)
    widenings_offered: int = 0
    widenings_taken: int = 0

    # An `answer` call carrying neither text nor a citation was asked to try
    # again, once. Recorded so a run that needed asking is distinguishable from
    # one that answered first time.
    answer_retried: bool = False

    # THE ANSWER GATE (D-107). `answer_gate_kind` is what tripped it -- one of
    # "placeholder", "deferred", "silent", or "" for an answer that named ids
    # first time. `answer_gate_failed` means it was asked and still named
    # nothing, which is the number that says whether the arm works.
    answer_gate_kind: str = ""
    #: Which notation the model used for its answer call (D-116). Recorded so a
    #: NEW form shows up as a number rather than as a week of odd results.
    answer_syntax: str = ""
    answer_gate_retried: bool = False
    answer_gate_failed: bool = False
    #: The answer as written before the gate asked again (D-127). Restored in
    #: `finish` when the retry yields nothing, so a weak answer is not turned
    #: into no answer.
    answer_before_gate: str = ""
    #: The ranked answer (D-128): did it ask, and how many candidates it held.
    ranked_answer_asked: bool = False
    ranked_answer_shown: int = 0

    # THE ANSWER CRITIC (D-106). The per-paper review, or None when the arm is
    # off. Kept whole rather than reduced to a count: which papers it dropped
    # and why is the measurement, and a keep-rate alone hides a judge that is
    # defaulting (D-105).
    answer_review: Any = None
    #: How many entities the kind fallback appended, summed over this run's
    #: searches (D-119). Zero with the arm on and kinded searches made means
    #: the arm did nothing -- the D-105 shape, and armcheck asserts it.
    kind_fallback_added: int = 0
    kinded_searches: int = 0
    #: Enumeration expansion (D-131): concepts the question named that the
    #: first search did not cover, and entities appended for them.
    enum_concepts: int = 0
    enum_added: int = 0
    #: Every `papers_of` reordering this run made (D-113). A list, because a
    #: run calls `papers_of` more than once and each call is a separate
    #: judgement whose spread has to be readable -- an arm whose rankings were
    #: all discarded as unusable is a no-op at double the price, and that must
    #: be in the record rather than inferred from a score that did not move.
    rankings: list = field(default_factory=list)
    #: The answer as WRITTEN, kept when the critic struck ids out of it. The
    #: edit has to be auditable: a harness that silently rewrites an answer and
    #: then scores it is measuring itself.
    answer_before_critic: str = ""

    # THE PLAN REVIEWER (separate from the search critic, which judges retrieved
    # rows). Counted apart from the planner's own calls and tokens: the whole
    # cost question for this arm is what the second model adds, and folding it
    # into llm_calls would hide exactly that. `reviews_rejected` is the metric
    # that says whether the reviewer is doing anything at all -- one that
    # approves everything is a no-op costing double.
    review_calls: int = 0
    reviews_rejected: int = 0
    review_prompt_tokens: int = 0
    review_completion_tokens: int = 0
    review_unparsed: int = 0
    review_ceiling_hit: bool = False

    # Set when a citation was rejected, with the reason. Recorded rather than
    # silently dropped: a refused citation means the answer carries no number,
    # and that has to be visible in the trace.
    citation_refused: str = ""

    # (number written in the prose, number the citation resolved to) when they
    # disagree. An answer that contradicts itself should never be scored as if
    # it had one number.
    citation_disagrees: Optional[tuple] = None
    citation_corrected: bool = False

    # One `critic.Review` per search, when the critic is on. Kept whole rather
    # than reduced to a count: the ablation needs to know WHICH candidates were
    # flagged down and why, and a drop that leaves no trace is the failure the
    # flag-never-filter rule exists to prevent (D-060).
    reviews: list = field(default_factory=list)

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
                 "error": s.error, "seconds": round(s.seconds, 4),
                 **({"redirected_from": s.redirected_from} if s.redirected_from else {})}
                for s in self.steps
            ],
            **({"reviews": [
                {"search_text": r.search_text, "tally": r.tally,
                 "kept": len(r.kept_ids), "candidates": len(r.verdicts),
                 "defaulted": r.defaulted, "calls": r.calls, "errors": r.errors,
                 "tail_keep_rate": r.tail_keep_rate(),
                 "dropped": [{"entity_id": v.entity_id, "why": v.reason}
                             for v in r.verdicts if not v.kept]}
                for r in self.reviews
            ]} if self.reviews else {}),
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

# THE SAME CALL, WRITTEN AS AN XML TAG. Measured 2026-09-07 across three runs:
# of the 150 answers per run that never reached `answer()`, 115 (77%) end in
#
#     <answer text="18 analyses use b-tagged jets ..." papers_from="the facets
#             result" reason="answered" answerable="true"/>
#
# which is an `answer` call in everything but syntax. `_TEXT_TOOL_CALL` wants
# `<tool_call>{json}</tool_call>` and matches none of it, so the whole message
# fell through to `after_plan`, where `last_content` becomes the answer -- raw,
# uncited, with no `answer_papers` and no gate.
#
# That single miss is the root of four separate findings: the gate firing twice
# in 164 questions (D-108), `cited` empty in 59 of 60 answers (D-107), the
# answer-critic reaching 9% of its chances, and the citation mechanism reading
# as dead. The model was calling `answer` about 78% of the time; we recorded 7%.
# THREE SYNTAXES, not two. Sampling the misses after the first fix showed a
# third form, and it is the commonest of all: the tool name as a WRAPPER round
# the JSON arguments.
#
#     <answer>
#     {"text": "23 analyses require missing transverse momentum ...",
#      "papers_from": "the facets response", "reason": "answered"}
#     </answer>
#
# Attribute form second, and note why the naive `[^>]*` regex only caught a
# third of those: attribute VALUES carry physics. `text="H->bb candidate"` has
# a `>` in it, so a character class excluding `>` ends the tag mid-value, the
# attributes fail to parse, and the call is dropped. The extent of the tag has
# to be found with the quoting respected, which is what `_xml_tag_end` does.
_XML_WRAPPED = re.compile(
    r"<([a-z_][a-z0-9_]*)\s*>\s*(\{.*?\})\s*</\1\s*>", re.IGNORECASE | re.DOTALL)
_XML_OPEN = re.compile(r"<([a-z_][a-z0-9_]*)(?=[\s/>])", re.IGNORECASE)
_XML_ATTR = re.compile(r'([a-z_][a-z0-9_]*)\s*=\s*"([^"]*)"', re.IGNORECASE)


def _xml_tag_end(text: str, start: int) -> int:
    """Index just past the `>` that closes the tag opened at `start`.

    Quote-aware, because an attribute value like `text="H->bb candidate"`
    contains the character a naive scan stops on.
    """
    in_quote = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '"':
            in_quote = not in_quote
        elif ch == ">" and not in_quote:
            return i + 1
    return -1


class _RecoveredCall:
    """Shaped like an SDK tool call, so the loop needs no special case."""

    def __init__(self, name: str, arguments: str, index: int):
        self.id = f"recovered-{index}"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)


#: The `answer` tool's arguments, for harvesting them out of prose.
ANSWER_ARGS = ("text", "papers_from", "value_from", "reason", "answerable", "papers")


def answer_syntax(content: str) -> str:
    """WHICH notation the model used to write its answer call.

    Recorded on every run so a new form arrives as a number in the record
    rather than as a week of confusing results. Four have been observed so far
    and two models produced them; the fifth will be someone else's model, and
    the point of this is that `unknown` climbing is the alarm.

        json        `{"text": ...}` anywhere, however wrapped
        tag_per_arg `<papers_from>set_1</papers_from>`
        attrs       `<answer text="..." reason="answered"/>`
        prose       ids in the text, no structure at all -- legitimate
        none        no answer content found
    """
    content = content or ""
    for blob in re.findall(r"\{[^{}]*\}", content, re.DOTALL):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and any(k in parsed for k in ANSWER_ARGS):
            return "json"
    for key in ANSWER_ARGS:
        if re.search(rf"<{key}\s*>", content, re.IGNORECASE):
            return "tag_per_arg"
    for key, _ in _XML_ATTR.findall(content):
        if key.lower() in ANSWER_ARGS:
            return "attrs"
    if re.search(r"\b\d{4}\.\d{4,5}\b", content):
        return "prose"
    return "none"


def harvest_answer_args(content: str) -> dict:
    """The `answer` arguments a model wrote without calling `answer`.

    STOP CHASING SYNTAXES. Three runs of QwQ produced `<answer .../>`,
    `<answer>{json}</answer>` and bare JSON; one run of qwen3-32b added
    `<answerable>{json}</answerable>` -- the tool's ARGUMENT name used as the
    tag -- and `<papers_from>set_1</papers_from><reason>answered</reason>`,
    one tag per argument. Each new model invents another, and a recogniser
    built from a list of observed forms is a recogniser that fails on the next
    model.

    What does NOT vary is the argument NAMES: the model has the schema and
    serialises it wrongly. So harvest by name, in any of:

        <papers_from>set_1</papers_from>          tag per argument
        <anything>{"papers_from": "set_1"}</...>  json in a wrapper
        papers_from="set_1"                       attribute anywhere

    Returns {} when nothing recognisable is there, which is the plain-prose
    case and belongs to the gate, not here.
    """
    content = content or ""
    out: dict = {}

    # JSON anywhere, whatever wraps it. Later blobs win: a model that writes
    # twice is correcting itself.
    for blob in re.findall(r"\{[^{}]*\}", content, re.DOTALL):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            for key in ANSWER_ARGS:
                if key in parsed:
                    out[key] = parsed[key]

    # One tag per argument. A tag whose body is JSON was a WRAPPER, not a
    # value: qwen3-32b writes `<answerable>{"text": ..., "reason": ...}
    # </answerable>`, and reading that body as the value of `answerable`
    # turns it into False -- an abstention the model never made.
    for key in ANSWER_ARGS:
        hit = re.search(rf"<{key}\s*>(.*?)</{key}\s*>", content,
                        re.IGNORECASE | re.DOTALL)
        if not hit or key in out:
            continue
        body = hit.group(1).strip()
        if body.startswith("{"):
            continue
        out[key] = body

    # Attributes, anywhere in the message.
    for key, value in _XML_ATTR.findall(content):
        if key.lower() in ANSWER_ARGS and key.lower() not in out:
            out[key.lower()] = value

    for key in ("answerable",):
        if isinstance(out.get(key), str):
            out[key] = out[key].strip().lower() in ("true", "yes", "1")
    return out


def _recover_tool_calls(content: str, known: Optional[set] = None) -> list:
    """Tool calls the model wrote into the message body, in either syntax.

    `known` is the set of tool names this run offers. It is required for the
    XML form and ignored for the JSON one: `<tool_call>{...}</tool_call>` is
    unambiguous, while `<answer .../>` is only a tool call because `answer` is
    a tool -- without the check, any `<b ...>` or `<sub ...>` in prose would be
    recovered as one.
    """
    content = content or ""
    out: list = []
    for i, blob in enumerate(_TEXT_TOOL_CALL.findall(content)):
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
    if out or not known:
        return out
    # THE XML FORMS, only when the JSON form found nothing -- a message holding
    # both is the model correcting itself, and the explicit call wins.
    #
    # Wrapper form first: `<answer>{json}</answer>` carries real JSON, so it is
    # both commoner and more trustworthy than attributes scraped off a tag.
    seen: set = set()
    for j, (tag, blob) in enumerate(_XML_WRAPPED.findall(content)):
        if tag.lower() not in known:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        # `{"name": ..., "arguments": ...}` inside the wrapper is the JSON form
        # wearing a tag; anything else IS the argument object.
        args = parsed.get("arguments") if "name" in parsed and "arguments" in parsed \
            else parsed
        name = str(parsed.get("name") or tag).lower()
        if name not in known:
            continue
        seen.add(name)
        out.append(_RecoveredCall(
            name, args if isinstance(args, str) else json.dumps(args), 100 + j))
    if out:
        return out

    pos, j = 0, 0
    while True:
        match = _XML_OPEN.search(content, pos)
        if not match:
            break
        tag = match.group(1).lower()
        end = _xml_tag_end(content, match.end())
        if end < 0:
            break
        pos = end
        if tag not in known:
            continue
        args = {}
        for key, value in _XML_ATTR.findall(content[match.end():end]):
            low = value.strip().lower()
            args[key] = (True if low == "true" else
                         False if low == "false" else value)
        if not args:
            continue
        out.append(_RecoveredCall(tag, json.dumps(args), 200 + j))
        j += 1
    return out


# Arguments that must name entities the graph already returned.
_ID_ARGS = ("entity_ids", "object_ids", "subject_a", "subject_b")

# A paper, written either way. The graph holds papers as entities -- 60 of them,
# `hepkg:paper:arxiv:2308.02285`, labelled with the title -- so the planner is
# not wrong to reach for one by id. What it cannot know is that the paper NODE
# is thin: it is the subject of 2 assertions (`paper_reports_result`), while the
# paper itself contains 187. The other 185 hang off `assertion.paper_id`, which
# only `contents_of` reads.
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")

# The count an answer states in prose, for checking it against what it cited.
_CLAIMED_IN_TEXT = re.compile(
    r"\b(\d[\d,]*)\s+(?:distinct\s+|different\s+|unique\s+)?"
    r"(?:papers?|analyses|analysis|studies)\b", re.I)
PAPER_ID_PREFIX = "hepkg:paper:arxiv:"

# Tools whose single row IS an answer, so it is worth keeping. `count` is the
# one that matters: it reports how many PAPERS carry an assertion, which is what
# "how many analyses..." asks. Everything else returns rows to be read.
ANSWER_SHAPED = {"count"}

# Which entity tool means which `contents_of` call when handed a paper.
_PAPER_REDIRECT = {"describe": "entity_ids", "count": "object_ids"}


def _paper_arxiv(value: Any) -> Optional[str]:
    """The arXiv id this argument names, if it names a paper at all."""
    if not isinstance(value, str):
        return None
    if _ARXIV_ID.match(value):
        return value
    if value.startswith(PAPER_ID_PREFIX):
        rest = value[len(PAPER_ID_PREFIX):]
        return rest if _ARXIV_ID.match(rest) else None
    return None


def resolve_paper_calls(tool: str, args: dict) -> tuple[str, dict, Optional[str]]:
    """Send a paper handed to an entity tool where papers actually live.

    Measured on the 1,352 Tier A questions: **192 sessions (14.2%) died here**,
    and 94.8% of them abstained against 0.9% everywhere else. The chain was
    always the same, and the guards drove it:

        describe(entity_ids=["2308.02285"])   -> unknown_entity_id
        UNKNOWN_ID_MESSAGE: "call `search` FIRST ... use the exact entity_id"
        search("2308.02285")                  -> 0 rows
        "The graph does not contain any information about 2308.02285."

    Every step is the model doing as it was told. `_check_ids` is right that the
    id was never returned by a search; the correction it gives is right for
    entities and wrong for papers, and nothing in the chain mentions the one tool
    that takes an arXiv id directly. The result is the worst answer this project
    can produce -- a false claim about coverage, with a clean trace.

    Two rules, and the second matters more than it looks:

      a paper id is not an invented id. It is verifiable by shape and usually
          came from the question itself, so `_check_ids` lets it through. If the
          paper is not in the corpus the tool returns no rows, which is honest.

      `describe`/`count` on a paper become `contents_of`. Without this, making
          papers findable would be WORSE than the bug: the planner would search,
          get the real paper entity, describe it, receive 2 rows, and answer
          "this analysis measures 2 observables" -- confidently, from 2 of 187
          facts. That trades a visible abstention for a silent wrong number.

    Returns the tool and arguments to run, plus a note recording the rewrite when
    one happened. The note is deliberate: the redirect must not hide the
    confusion, because "the planner treats papers as ordinary entities" is a
    finding about tool selection (S-49), not just a bug to paper over.
    """
    id_arg = _PAPER_REDIRECT.get(tool)
    if not id_arg:
        return tool, args, None

    value = args.get(id_arg)
    if value is None or args.get("object_set") or args.get("entity_set"):
        return tool, args, None
    values = [value] if isinstance(value, str) else list(value)
    papers = [_paper_arxiv(v) for v in values]
    if not values or not all(papers):
        return tool, args, None  # mixed or not papers at all -- leave it alone

    rewritten = {"paper_ids": papers}
    if args.get("predicate"):
        rewritten["predicate"] = args["predicate"]
    return "contents_of", rewritten, (
        f"{tool} was called on a paper, so it ran as contents_of("
        f"paper_ids={papers}) -- papers are entities here, but the paper NODE "
        f"holds only its results; everything else in the paper is reached this way."
    )


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

    Papers are exempt: `_paper_arxiv` recognises them by shape, so they need no
    prior search to be legitimate -- see `resolve_paper_calls`.
    """
    unknown: list[str] = []
    for name in _ID_ARGS:
        value = args.get(name)
        if value is None:
            continue
        for candidate in ([value] if isinstance(value, str) else value):
            if not isinstance(candidate, str) or candidate in known:
                continue
            if _paper_arxiv(candidate):
                continue
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
    # A row may carry a LIST of ids (facets: every entity that earned the tag).
    found |= {str(e) for r in rows if isinstance(r, dict)
              for e in (r.get("entity_ids") or []) if e}
    found |= set(re.findall(r"hepkg:[a-z_]+:[A-Za-z0-9_.\-]+", note or ""))
    return found


#: Rung -> sort key. Rows without a verdict sort after `broader` and before
#: `unrelated`: an unjudged candidate is unknown, a judged-unrelated one is not.
_RUNG_ORDER = {"exact": 0, "broader": 1, None: 2, "unrelated": 3}


def order_by_rung(rows: list, kept: Optional[set] = None) -> None:
    """Stable in-place sort of entity rows by relevance, best first.

    Two signals, both free. `bears_on` is the critic's rung and rides on search
    rows. `kept` is the union of every `*_kept` set this run has built: an
    entity the critic already judged relevant for THIS question, when it
    surfaces again from `subjects_of` or `describe` with no verdict of its
    own, is not an unknown -- it is a known-relevant. gf-08's 224-row
    `subjects_of` result had no rungs at all; this is what orders it.

    A no-op when neither signal applies. Stable, so retrieval order (itself a
    weak ranking) breaks ties.
    """
    kept = kept or set()
    def key(r):
        if not isinstance(r, dict):
            return 2
        if "bears_on" in r:
            return _RUNG_ORDER.get(r["bears_on"], 2)
        if kept and r.get("entity_id") in kept:
            return 1                         # known-relevant, ranks with `broader`
        return 2
    if not any(isinstance(r, dict) and ("bears_on" in r or (kept and r.get("entity_id") in kept))
               for r in rows):
        return
    rows.sort(key=key)


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


def _paper_ranker(conn, session, rerank, answer_critic=None):
    """Reorder a `papers_of` result best-first, or None when the arm is off.

    DECOUPLED FROM THE FILTER (D-124). This used to require --answer-critic as
    well, because both use the same judge client -- so the ranker could not be
    had without the filter. The live stack showed why that matters: the filter
    struck seven gold papers in sixteen drops ("uses b-tagged jet veto", which
    the prompt itself says counts as using), while the ranker only ever
    reorders and cannot lose a paper. --rerank now builds its own judge;
    --answer-critic remains the filter and is a separate decision.

    `answer_critic` is accepted and ignored, so existing callers do not move.
    """
    if not rerank:
        return None
    from hepcoveragekg.query import answer_critic as AC

    client, model = _rank_client()
    cap = completion_cap(model)

    def chat(messages):
        # `enable_thinking=false` first, then plain. D-105: a reasoning judge
        # spends its budget thinking and returns empty content, and here that
        # would leave every paper ungraded and the order untouched -- which is
        # at least the safe direction, but it must still be visible.
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=cap,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except Exception:  # noqa: BLE001
            pass
        return client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=cap)

    def rank(question, result):
        papers = []
        seen = set()
        for row in result.rows:
            pid = row.get("paper_id")
            if pid and pid not in seen:
                seen.add(pid)
                papers.append(str(pid))
        if len(papers) < 2:
            return result
        # The ids of the rows being ranked, not only what the session already
        # knows: graph.py adds a result's ids to `known_entity_ids` AFTER the
        # executor returns, and this ranker runs inside it -- so on a
        # facets-first run (gf-01, gf-02, gf-04 every time) the known set was
        # empty, the evidence was empty, and every ranking had zero
        # candidates and zero judge calls, on OpenRouter and on the cluster
        # alike (D-133).
        ids = set(session.known_entity_ids) | _collect_ids(result.rows, result.note)
        try:
            evidence = AC.evidence_by_paper(conn, ids, papers)
            ranking = AC.rank_papers(chat, question, evidence)
        except Exception as exc:  # noqa: BLE001 -- a ranker must not kill a run
            logger.warning("paper rerank failed, order unchanged: %s", exc)
            return result
        session.rankings.append(ranking)
        if not ranking.order or not ranking.usable:
            # An unusable ranking is left alone rather than applied. A judge
            # that puts nothing in the middle grades measured WORSE than not
            # ranking (D-113), so acting on it would be a known regression.
            return result
        rank_of = {p: n for n, p in enumerate(ranking.order)}
        result.rows.sort(key=lambda r: rank_of.get(str(r.get("paper_id")), 10**6))
        return result

    return rank


#: Words that carry no concept. Kept short on purpose: the point is to find the
#: nouns the question enumerates, not to parse it.
_ENUM_STOP = frozenset("""which analyses analysis use uses using that the a an or and their in as
rather than of for to with is are event selection both have has whose they them it its
estimate estimated report reports require requires""".split())


def enumerated_concepts(question: str, limit: int = 4) -> list:
    """The concepts a question ENUMERATES, split on its own conjunctions (D-131).

    "estimate a background using an ABCD method, or an ABCD-style sideband or
    matrix method over independent regions" names three methods; the model
    faceted one (D-118, D-130). Decomposition by the model did not yield the
    other two. Splitting the question on ', or' / ' or ' / ' and ' / commas
    does, deterministically and for free. Parentheticals and a trailing
    "rather than ..." clause are dropped: the first restate, the second negate.

    Measured offline against Gabriel's gold, searches typed by the model plus
    these: gf-02 8/11 -> 11/11, gf-01 4/8 -> 8/8, gf-04 17/18 -> 18/18, no
    question lower.
    """
    t = re.sub(r"\(.*?\)", "", question or "")
    t = re.sub(r"\b(rather than|instead of)\b.*$", "", t, flags=re.IGNORECASE)
    out = []
    for part in re.split(r",\s*or\s+|\s+or\s+|,\s+|\s+and\s+|\bor\b", t):
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-\+→>]*", part)
                 if w.lower() not in _ENUM_STOP]
        if 1 <= len(words) <= 5:
            phrase = " ".join(words)
            if len(phrase) > 3 and phrase not in out:
                out.append(phrase)
    return out[:limit]


#: Nouns that name a KIND of thing, not a thing. "ABCD method" and "matrix
#: method" share one; overlap on it alone must not count as coverage.
_ENUM_GENERIC = frozenset("""method methods region regions level technique techniques
framework background backgrounds selection candidate candidates system""".split())


def _enum_facets_note(conn, index, sets, question_text, covered_text, limit, on_enum):
    """Enumeration expansion when the FIRST retrieval was `facets` (D-131 fix).

    gf-02's first move is a facet filter, so the search-branch trigger never
    fired on it. Here the uncovered concepts are searched, canonical-expanded,
    saved as a set the model can pass to papers_of, and named in the note.
    Returns the note fragment, or "" when nothing was added.
    """
    from hepcoveragekg.query import retrieve, templates   # lazy, like build_executor's
    missing = [c for c in enumerated_concepts(question_text) if not _covers(covered_text, c)]
    have: set = set(); hits = []
    for c in missing:
        for h in retrieve.search(index, c, conn=conn, kind=None, limit=limit):
            if h.entity_id not in have:
                have.add(h.entity_id); hits.append(h)
    if on_enum:
        on_enum(len(missing), len(hits))
    if not hits:
        return ""
    name = f"set_{len(sets) + 1}"
    sets[name] = templates.expand_canonical(conn, [h.entity_id for h in hits])
    return (f" The question also names {len(missing)} other concept(s) ({'; '.join(missing)}); "
            f"{len(hits)} entities matching those were retrieved and saved as {name} "
            f"({len(sets[name])} after canonical expansion) -- pass {name} to papers_of "
            f"to see their papers.")


def _enum_tokens(text: str) -> set[str]:
    """Words of `text`, hyphen-split and singularised, so that a question's
    "b-tagged jets" meets a facet label's "$b$-tagged jet" (D-133 addendum:
    token-exact matching made enumeration re-search every concept a facets
    call had already covered, and poured 60-150 entities into the set)."""
    out: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z\-]*", text):
        for part in w.lower().split("-"):
            if len(part) < 2:
                continue
            if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
                part = part[:-1]
            out.add(part)
    return out


def _covers(search_text: str, concept: str) -> bool:
    """Did the model's own search already name this concept?

    Token overlap on the concept's SPECIFIC words. The first version counted
    "method" as shared between "ABCD method" and "matrix method" and never
    searched the second -- the exact miss this mechanism exists to close.
    """
    a = _enum_tokens(search_text or "")
    b = _enum_tokens(concept) - _ENUM_STOP
    specific = b - _ENUM_GENERIC
    key = specific or b
    return bool(key) and len(a & key) >= max(1, len(key) // 2)


def _enum_counter(session):
    def hook(concepts: int, added: int):
        session.enum_concepts += int(concepts)
        session.enum_added += int(added)
    return hook


def _fallback_counter(session):
    """Counts kinded searches and appended entities onto the session (D-119)."""
    def hook(added: int, kinded: bool = False):
        if kinded:
            session.kinded_searches += 1
        session.kind_fallback_added += int(added)
    return hook


def build_executor(conn, index, sets: Optional[dict] = None,
                   critic: Optional[Callable] = None,
                   answer_contract: bool = False,
                   force_critic_set: bool = False,
                   rerank: bool = False,
                   rank_papers: Optional[Callable] = None,
                   question_of: Optional[Callable] = None,
                   kind_fallback: bool = False,
                   on_fallback: Optional[Callable] = None,
                   enum_expand: bool = False,
                   question_text: str = "",
                   on_enum: Optional[Callable] = None) -> Callable[[str, dict], Any]:
    """Bind the tools to this database, index and set of named results.

    Returned as a closure so the planner never holds a connection itself and
    cannot be handed a writable one by accident.

    `critic` is `(search_text, hits) -> critic.Review` and defaults to OFF, so
    the baseline arm of the ablation is literally this code with nothing passed
    -- there is no second branch to keep in step with the first. When present it
    marks each hit's rung and saves a second handle; it never removes anything
    (D-060).
    """
    from hepcoveragekg.query import critic as critic_mod, retrieve, templates

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
        # THE CRITIC JUDGES, THEN THE PLANNER DECLINES IT.
        #
        # Measured 2026-08-17: of 767 `count` calls in a critic-on arm, 58% used
        # `set_N_kept` and **39% used the raw set** -- four counts in ten threw
        # the critic's verdicts away. That is the likeliest reason its counting
        # gain (+0.063) is so much smaller than its set-F1 gain (x2.3): on set
        # questions the kept handle gets used, on counts it often does not.
        #
        # Under this flag a set with a judged counterpart is silently replaced by
        # it, so "does the critic judge better than the planner's discretion?"
        # becomes a measurement rather than a hope. Off by default: it takes a
        # decision away from the model, and that has to be earned.
        if force_critic_set and name and f"{name}_kept" in sets:
            name = f"{name}_kept"
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
            # `facets` is included on every hit that has one. This is how the
            # model learns which closed-vocabulary key covers a concept -- from
            # an entity that demonstrably exists, rather than from a vocabulary
            # list in the prompt. Shipping the list instead would have put five
            # of the seven Tier 1 gold keys into the system prompt (D-054).
            review = critic(args["text"], hits) if critic else None

            # WIDEN WHILE THE TAIL IS STILL RELEVANT (D-060).
            #
            # `limit` exists only because everything retrieved gets used, so a
            # junk candidate becomes a wrong count. With a relevance step it
            # costs one line of prompt instead, and the ceiling can move.
            #
            # It binds hard: 297 of 458 Tier B searches (64.8%) came back
            # EXACTLY full, measured before any of this existed. `jet energy
            # scale` has 109 clusters against a limit of 60, and the trace of
            # those 49 losses looks like a clean successful search.
            #
            # Widening is not undoable the way flagging is -- a candidate never
            # retrieved cannot be recovered downstream -- so the threshold is
            # generous even though the flagging is not.
            widened = 0
            while (review is not None
                   and critic_mod.should_widen(review, len(hits), limit)
                   and limit < MAX_SEARCH_BREADTH):
                limit = min(limit * 2, MAX_SEARCH_BREADTH)
                wider = retrieve.search(index, args["text"], conn=conn,
                                        kind=args.get("kind"), limit=limit)
                if len(wider) <= len(hits):
                    break            # the retriever had no more to give
                hits = wider
                review = critic(args["text"], hits, refresh=True)
                widened += 1

            # THE KIND FALLBACK (D-119, arm). A `kind` narrows the search to one
            # entity type, and whether that helps depends on how the GRAPH typed
            # the concept, which the model cannot see. Replayed against the DIAS
            # DB on Gabriel's questions: "Higgs" with kind=detector_object
            # reaches 2 of 16 gold papers, without it 13 -- the candidate
            # entities are typed event_region, physics_process, observable, and
            # only 9 of 200-odd are detector_object. Yet the same filter HELPS
            # gf-08 (+2) and gf-01 (+3). So neither keep nor drop: keep the
            # kinded hits first, append the unfiltered ones, and let the critic
            # judge the union. Measured offline over the five live kinded
            # searches: 61 -> 74 of 79 gold papers reached, at one extra local
            # search and zero LLM calls.
            fallback_added = 0
            if args.get("kind") and on_fallback:
                on_fallback(0, kinded=True)          # a kinded search happened
            if kind_fallback and args.get("kind"):
                have = {h.entity_id for h in hits}
                extra = [h for h in retrieve.search(index, args["text"], conn=conn,
                                                    kind=None, limit=limit)
                         if h.entity_id not in have]
                if extra:
                    hits = list(hits) + extra
                    fallback_added = len(extra)
                    if on_fallback:
                        on_fallback(fallback_added)
                    if critic:
                        review = critic(args["text"], hits, refresh=True)
            # ENUMERATION EXPANSION (D-131, arm). The question names its own
            # concepts; the model searches one of them. Once per run, on the
            # first search, every enumerated concept the searched text does
            # not cover is searched too (unfiltered) and appended. Free, and
            # deterministic where model decomposition was not (D-130).
            enum_added = 0
            if enum_expand and question_text and not sets:
                missing = [c for c in enumerated_concepts(question_text)
                           if not _covers(args["text"], c)]
                have = {h.entity_id for h in hits}
                extra = []
                for c in missing:
                    for h in retrieve.search(index, c, conn=conn, kind=None, limit=limit):
                        if h.entity_id not in have:
                            have.add(h.entity_id); extra.append(h)
                if extra:
                    hits = list(hits) + extra
                    enum_added = len(extra)
                    if critic:
                        review = critic(args["text"], hits, refresh=True)
                if on_enum:
                    on_enum(len(missing), enum_added)
            # `retrieve.concept` is `search` + `expand_canonical`, and the hits
            # are already in hand -- calling it here re-ran BM25 and the dense
            # encoder over the whole index a second time for every search, to
            # arrive at the list above. Expanding directly is the same result at
            # half the retrieval cost, which the widening loop above multiplies.
            ids = templates.expand_canonical(conn, [h.entity_id for h in hits])
            name = f"set_{len(sets) + 1}"
            sets[name] = ids
            rungs = ({v.entity_id: v for v in review.verdicts} if review else {})

            rows = []
            for h in hits:
                row = {"entity_id": h.entity_id, "label": h.label, "kind": h.kind}
                if h.facets:
                    row["facets"] = h.facets
                verdict = rungs.get(h.entity_id)
                if verdict is not None:
                    row["bears_on"] = verdict.rung
                rows.append(row)
            tagged = sum(1 for h in hits if h.facets)
            note = (f"saved as {name} ({len(ids)} entities, canonical clusters expanded). "
                    f"Pass {name} as object_set/entity_set -- do not retype the ids.")
            if enum_added:
                note += (f" The question also names {len(missing)} other concept(s) "
                         f"({'; '.join(missing)}); {enum_added} more entities matching "
                         f"those are included.")
            if fallback_added:
                note += (f" kind={args.get('kind')!r} matched {len(hits) - fallback_added}; "
                         f"{fallback_added} more entities of OTHER kinds also match "
                         f"'{args['text']}' and are included -- the graph types this "
                         f"concept several ways.")
            if tagged:
                note += (f" {tagged}/{len(hits)} hits carry facet tags; those keys are"
                         " what the `facets` tool takes.")
            if review is not None:
                # A verdict moves a CLUSTER, not an entity. `concept` expands the
                # seed hits through `expand_canonical`, so `set_N` is larger than
                # the hit list and holds ids the critic never saw. The kept set
                # must therefore be the expansion of the kept SEEDS -- taking a
                # subset of `ids` would keep cluster-mates of dropped seeds and
                # silently undo the judgement. Dedup already ruled those the same
                # thing, so they travel together in both directions.
                # RANK-PRESERVING EXPANSION (D-113). `expand_canonical`
                # returns `sorted(ids)`, so any order handed to it is thrown
                # away alphabetically -- which is why the critic's rung has
                # never reached the answerer despite being computed every run.
                # Expanding each rung separately and concatenating keeps
                # `exact` ahead of `broader` without touching a function every
                # other tool depends on. Membership is identical either way;
                # only the order differs, and the order is what the answerer's
                # own truncation acts on.
                if rerank and review.kept_ids:
                    kept, seen_k = [], set()
                    for rung in (critic_mod.EXACT, critic_mod.BROADER):
                        seeds = review.ids_at(rung)
                        for eid in (templates.expand_canonical(conn, seeds)
                                    if seeds else []):
                            if eid not in seen_k:
                                seen_k.add(eid)
                                kept.append(eid)
                else:
                    kept = templates.expand_canonical(conn, review.kept_ids) \
                        if review.kept_ids else []
                sets[f"{name}_kept"] = kept
                tally = review.tally
                if widened:
                    # Said out loud for the same reason truncation is: a planner
                    # that cannot tell a widened search from a normal one will
                    # reason about breadth it never had.
                    note += (f" The first {SEARCH_BREADTH} candidates were still"
                             f" relevant at the tail, so the search was widened to"
                             f" {limit} and re-judged.")
                note += (
                    f" RELEVANCE: {tally[critic_mod.EXACT]} exact,"
                    f" {tally[critic_mod.BROADER]} broader,"
                    f" {tally[critic_mod.UNRELATED]} unrelated."
                    f" {name} still holds everything; {name}_kept holds the"
                    f" {len(kept)} entities judged to bear on the question."
                )
            return templates.QueryResult(shape="search", rows=rows, note=note)
        if tool == "describe":
            return templates.describe(conn, ids_for(args, "entity_ids", "entity_set"),
                                      args.get("predicate"))
        if tool == "subjects_of":
            return templates.subjects_of(conn, args["predicate"],
                                         ids_for(args, "object_ids", "object_set"))
        if tool == "papers_of":
            result = templates.papers_of(conn, ids_for(args, "entity_ids", "entity_set"))
            # THE TRUNCATION POINT (D-113). `papers_of` orders by paper_id --
            # arXiv id, which is arbitrary -- and `_render_rows` then cuts the
            # list at `max_rows`. So which papers reach the answerer at all is
            # decided alphabetically. Measured against Gabriel's labels that is
            # worth 0.45 precision at every cut-off; ordering the same papers by
            # a graded judge reaches 0.78 at five and 0.62 at ten.
            #
            # NOTHING IS REMOVED. Every row still travels; only the order
            # changes. A set question's answer IS the set, and a count over a
            # trimmed set is a wrong number -- so the cut stays where it already
            # was, at the answerer's own limit, and the ranking decides what
            # falls on the right side of it.
            if rank_papers is not None and result.rows:
                result = rank_papers(question_of(), result)
            return result
        if tool == "contents_of":
            papers = args.get("paper_ids") or args.get("paper_id") or []
            # Same truncation, same fix as papers_of: contents_of rows carry a
            # paper_id and are cut at max_rows in retrieval order.
            if isinstance(papers, str):
                papers = [papers]
            result = templates.contents_of(conn, papers, args.get("predicate"))
            if rank_papers is not None and result.rows and len(result.rows) > 1:
                result = rank_papers(question_of(), result)
            return result
        if tool == "facets":
            values = args.get("values") or args.get("value") or []
            if isinstance(values, str):
                values = [values]
            result = templates.facets(
                conn, args["field"], values,
                mode=args.get("mode", "all"),
                category=args.get("category"),
                experiment=args.get("experiment"),
            )
            # The label reader, NOT a filter. Every paper stays in `result.rows`
            # -- the tag is right for all of them and only the variant differs,
            # and for a coverage map those variants are the finding. What gets
            # added is a reading of how each paper's own words relate to the
            # question, which both answers a narrow question and says where to
            # look next.
            if critic and result.rows:
                reading = critic.read_facets(args["field"], values, result.rows)
                if reading is not None:
                    summary = critic_mod.facet_summary(reading)
                    if summary:
                        result.note = f"{result.note} {summary}".strip()
            # facets found 79 of Gabriel's gold papers to search's 52 (D-118) and
            # is cut at max_rows like everything else; rank its paper rows too.
            if rank_papers is not None and result.rows and len(result.rows) > 1:
                result = rank_papers(question_of(), result)
            # A facets-first run never reaches the search-branch expansion, and
            # gf-02 is facets-first every time (D-131 outcome). Expand here.
            if enum_expand and question_text and not sets:
                vals = values if isinstance(values, list) else [values]
                # Coverage is judged on what the facet call actually matched
                # (the rows' evidence labels), not on its codes: "BJet" does
                # not read as "b-tagged jets", and enumeration re-searched
                # every concept the facet had already found (D-133 addendum).
                covered = [str(v) for v in vals]
                for row in result.rows[:400]:
                    if isinstance(row, dict):
                        covered.extend(str(x) for x in (row.get("evidence_labels") or []))
                        if row.get("matched"):
                            covered.append(str(row["matched"]))
                frag = _enum_facets_note(conn, index, sets, question_text,
                                         " ".join(covered), SEARCH_BREADTH, on_enum)
                if frag:
                    result.note = (result.note or "") + frag
            return result
        if tool == "facet_entities":
            value = args.get("value")
            if isinstance(value, list):      # a values=[...] slip; one key is meant
                value = value[0] if value else None
            if not value:
                raise ValueError("facet_entities needs a single `value`")
            return templates.facet_entities(conn, args["field"], value,
                                            kind=args.get("kind"))
        if tool == "refine":
            # The answerer's own relevance judgement, made explicit.
            #
            # Marking is ADDITIVE and flag-only, exactly as the critic's is: the
            # original set is untouched and the reason is recorded. The critic
            # judges retrieval candidates; this lets the model drop what it can
            # see does not belong once it has looked at the rows -- and the
            # counting then happens in SQL over what remains, instead of by eye.
            #
            # Which is the point. Counting is the measured weak spot (0.50), and
            # `count` alone cannot express an answer aggregated across several
            # calls; a set that the model has curated can.
            source = args["entity_set"]
            if source not in sets:
                raise ValueError(
                    f"no set named '{source}'. Available: {sorted(sets) or 'none yet'}.")
            drop = {str(x) for x in (args.get("drop_ids") or [])}
            kept = [i for i in sets[source] if i not in drop]
            unknown = drop - set(sets[source])
            name = f"{source}_refined"
            sets[name] = kept
            note = (f"saved as {name}: {len(kept)} of {len(sets[source])} kept, "
                    f"{len(sets[source]) - len(kept)} dropped ({args['reason'][:80]}). "
                    f"{source} is unchanged. Pass {name} to count, papers_of, or "
                    f"answer(value_from=...).")
            if unknown:
                note += f" {len(unknown)} of the ids given were not in {source} and were ignored."
            return templates.QueryResult(shape="refine", rows=[], note=note)
        if tool == "count":
            return templates.count(conn, args["predicate"],
                                   ids_for(args, "object_ids", "object_set"))
        if tool == "compare":
            return templates.compare(conn, args["subject_a"], args["subject_b"],
                                     args["predicate"])
        if tool == "path":
            return templates.path(conn, args.get("constraints") or [],
                                  project=args.get("project") or "subjects",
                                  mode=args.get("mode") or "all")
        if tool == "crosstab":
            return templates.crosstab(conn, args["predicate_a"], args["predicate_b"])
        if tool == "quotes":
            return templates.quotes(conn, args["assertion_id"])
        raise ValueError(f"unknown tool: {tool}")

    return run


#: Tool-call markup that some servers leave in `message.content` after parsing
#: the call out of it. Observed on qwen3-32b via OpenRouter: EVERY stated
#: objective came back ending in "ool_call>", and one in six was nothing else --
#: so the reviewer judged a plan against an empty objective and the trace
#: recorded a goal the model never set. It also reaches `session.answer`, which
#: is `last_content` whenever the model answers in prose.
_TOOL_MARKUP = re.compile(
    r"</?\s*tool_call\s*>|<\|?/?tool[_▁]?call\|?>|^[a-z_]*ool_call>|ool_call>",
    re.I | re.M)


def clean_content(text: str) -> str:
    """Message prose with the server's tool-call scaffolding removed."""
    return _TOOL_MARKUP.sub("", text or "").strip()


def _reviewer_chat(planner_chat):
    """The callable the reviewer talks through.

    Falls back to the planner's own client, which is the intended default: a
    strong reasoning model reviewing a strong reasoning model. Set REVIEWER_MODEL
    (optionally with REVIEWER_BASE_URL / REVIEWER_API_KEY) to point it elsewhere.
    """
    model = os.environ.get("REVIEWER_MODEL", "").strip()
    if not model:
        return planner_chat
    import openai
    from hepcoveragekg.query import budget as budget_mod
    client = openai.OpenAI(
        base_url=os.environ.get("REVIEWER_BASE_URL")
                 or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("REVIEWER_API_KEY")
                or os.environ.get("LLM_API_KEY", "dummy"))
    cap = completion_cap(model)

    def chat(msgs, tls=None):
        kwargs = dict(model=model, messages=msgs,
                      temperature=planner_temperature(), max_tokens=cap)
        if tls:
            kwargs["tools"] = tls
        return client.chat.completions.create(**kwargs)
    return budget_mod.guard(chat, budget_mod.from_env(model))


#: Appended when --state-objective is on. Two things in one call, not two calls:
#: the traces show runs stopping at 2-3 rounds with a median of one row per
#: later step, so the model is not noticing that it holds too little. Asking it
#: to say whether the last result met its goal is the cheapest way to make that
#: judgement explicit, and it gives the reviewer something to check the plan
#: against. Kept short on purpose: it is re-sent every round.
OBJECTIVE_BLOCK = """

BEFORE THE TOOL CALLS IN EACH TURN, WRITE TWO SHORT LINES:

  GOT: what the previous result gave you, and whether it met the objective you
       set last round. Say plainly if it did not -- an empty or thin result is
       information, and pretending otherwise wastes the rounds you have left.
       Write "GOT: nothing yet" on the first round.
  AIM: what THIS round is for, in one sentence. Not the whole question -- the
       specific thing these calls are meant to establish.

Then make the calls. Keep both lines to one sentence each."""


def system_prompt(conn, minimal: bool = False, state_objective: bool = False) -> str:
    """Purpose + what the graph contains. Assembled fresh so it cannot go stale."""
    from hepcoveragekg.query import prompts, schema_card

    purpose = prompts.PURPOSE_MINIMAL if minimal else prompts.PURPOSE
    return "\n\n".join([
        purpose,
        schema_card.render(conn),
        "Work by calling tools. You may call several in one turn when they do not depend on "
        "each other -- one round of several calls costs far less than several rounds of one. "
        "Call `answer` when the retrieved rows support an answer, or to say the graph does not "
        "hold what was asked."
        + (OBJECTIVE_BLOCK if state_objective else ""),
    ])



def rows_held(session: "Session") -> int:
    """How many retrieved rows the session is sitting on.

    Used to decide whether an abstention deserves challenging. Counts rows from
    successful steps rather than set membership, because a set can be large while
    every hop off it came back empty -- and an abstention there is correct.
    """
    return sum(s.rows for s in session.steps if not s.error)


#: A set reference inside `papers_from`. Set names are word characters, so
#: anything else is a separator -- "set_1_kept and set_3_kept", "set_1,set_3"
#: and "set_1 + set_3" all yield the same two names.
_SET_REF = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def resolve_citations(session: "Session", args: dict,
                      papers_of: Optional[Callable] = None) -> None:
    """Turn `papers_from` / `value_from` into the answer's data.

    The output half of S-29. Handles solved retyping ids INTO tools -- the model
    once wrote 31 ids into one call, 863 tokens, looping to 6,521 characters on a
    retry. Nothing solved retyping them OUT: measured 2026-08-15, the planner
    named 11 arXiv ids at the median, **100% genuinely retrieved**, and hit only
    22.8% of the gold papers while retrieval had reached 91.7%. Selection, not
    hallucination -- and a citation cannot select wrongly.

    **CITE A RESULT, NEVER AN INPUT.** The first version of this took the number
    from a SET, by counting the papers its entities appear in. That was wrong
    twice over, measured on 2026-08-16 for a question whose gold is 2:

        count(result_has_systematic, set) ->  36 papers   the question asked
        papers_of(set)                    ->  77 papers   what was resolved

    `papers_of` is predicate-blind -- it answers "where do these entities appear
    at all", not "which papers record this assertion" (the D-043 discrepancy, in
    a new place). So the harness took the number away from the model and
    re-derived it by a route that ignored the question. v1, where the model reads
    the figure off `count` and types it, scored 0.220 against v2's 0.179: the
    mechanism was losing to the transcription it replaced.

    A value now cites a STEP whose result is an answer -- in practice a `count`
    call, which reports papers, facts and assertions for a predicate over a set.
    A paper LIST still cites a set, because there the set genuinely is the answer.
    """
    cited = []

    # ONE NAME OR SEVERAL. The model routinely cites more than one set --
    # "set_1_kept and set_3_kept" was the single most common unresolved value in
    # runs 54201-06 -- and the old exact-match lookup dropped every one of them
    # in silence. Splitting on anything that is not a set-name character and
    # keeping the parts that ARE sets resolves the compound form without
    # inventing a new syntax for the model to get wrong.
    raw_name = str(args.get("papers_from") or "").strip()
    wanted = [w for w in _SET_REF.findall(raw_name)]
    known = [w for w in wanted if w in session.sets]
    if raw_name and not known:
        # AND IT MUST NOT BE SILENT. `value_from` has always refused an
        # unresolvable reference and got a correction round out of it; the paper
        # list took the same mistake and threw the answer away instead. D-107:
        # 23 of 60 Gabriel answers scored a hard 0 while holding the right
        # papers, and this is one of the two ways it happened.
        session.citation_refused = (
            f"papers_from={raw_name!r} does not name a set from this run. "
            f"The sets you can cite are: {', '.join(sorted(session.sets)) or '(none)'}. "
            f"Cite one of those, or several separated by spaces, or write the "
            f"arXiv ids into the answer text yourself."
        )
    elif known:
        # A set holds ENTITY ids while the answer is about PAPERS -- the graph
        # keeps 56 spellings of Pythia, and no question is asking about
        # spellings. Resolving through `papers_of` is right HERE, where the
        # question is "which analyses", and wrong for a count.
        entities: list = []
        seen_e = set()
        for w in known:
            for e in session.sets[w]:
                if e not in seen_e:
                    seen_e.add(e)
                    entities.append(e)
        papers = papers_of(entities) if papers_of else None
        if papers is not None:
            session.answer_papers = list(papers)
            cited.append("papers=" + "+".join(known))

    ref = str(args.get("value_from") or "").strip()
    if ref:
        step = _step_by_ref(session, ref)
        if step is None or not step.result:
            session.citation_refused = (
                f"value_from={ref} does not name a result that holds a number. "
                f"Cite the `count` call that answers the question -- counting is "
                f"what it is for, and it respects the predicate. A set cannot be "
                f"cited for a number: its size is how wide the search was."
            )
        else:
            session.answer_value = float(step.result.get("papers", 0))
            cited.append(f"value={ref}")

    session.answer_cited = ", ".join(cited)

    # If the prose states a number AND a citation resolved to a different one,
    # the answer disagrees with itself. Observed live: value=24 beside the text
    # "There are 22 analyses that define the region 2l SS Signal Region."
    # Neither is preferred silently -- the disagreement is the finding.
    if session.answer_value is not None:
        stated = _CLAIMED_IN_TEXT.search(session.answer or "")
        if stated:
            spoken = float(stated.group(1).replace(",", ""))
            if spoken != session.answer_value:
                session.citation_disagrees = (spoken, session.answer_value)


def _step_by_ref(session: "Session", ref: str):
    """The step a citation names: `step_3`, `3`, or the tool's own name.

    Lenient about the spelling because the alternative is refusing a citation
    over punctuation and sending the model back to typing numbers. When a bare
    tool name matches several steps the LAST is taken -- a planner that counts
    twice has refined its answer, not changed the subject.
    """
    token = ref.strip().lower().removeprefix("step_").removeprefix("step ")
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(session.steps):
            return session.steps[index]
        return None
    matches = [s for s in session.steps if s.tool == token and s.result]
    return matches[-1] if matches else None


#: Mined disagreements appended to the small critic's prompt, computed once.
#: Reading the run files on every judged chunk would re-parse megabytes 48% of
#: the time -- the critic's share of all calls.
_CRITIC_EXAMPLES: Optional[str] = None


def _critic_prompt(small: bool) -> Optional[str]:
    """The small critic's prompt, with mined examples when the arm is on.

    Only the SMALL one gets them. They are the 72B's own judgements, so giving
    them to the 72B would be showing it its own answers -- which measures
    nothing and would quietly contaminate the arm it is meant to be compared
    against.
    """
    global _CRITIC_EXAMPLES
    from hepcoveragekg.query import critic as critic_mod

    if not small:
        return None
    if not os.environ.get("CRITIC_EXAMPLES"):
        return critic_mod.SMALL_PROMPT
    if _CRITIC_EXAMPLES is None:
        import glob

        from hepcoveragekg.query import critic_examples as ce
        small_runs = glob.glob(os.environ.get("CRITIC_EXAMPLES_SMALL", "") or "")
        big_runs = glob.glob(os.environ.get("CRITIC_EXAMPLES_BIG", "") or "")
        try:
            _CRITIC_EXAMPLES = ce.render(ce.disagreements(small_runs, big_runs))
        except Exception:              # a missing run file must not kill a run
            _CRITIC_EXAMPLES = ""
    return critic_mod.SMALL_PROMPT + (_CRITIC_EXAMPLES or "")


def _critic_client():
    """The endpoint the CRITIC talks to, which need not be the planner's.

    Separate because the critic is 48% of all LLM calls and its task is narrow
    -- short prompt, fifteen candidates, structured JSON back -- so it is the
    obvious place to put a small model and leave the 72B to answer. Splitting
    the endpoints is what makes that measurable rather than hypothetical.

    Falls back to the planner's client when unset, so every existing run and
    every test keeps its current behaviour and the split is an ablation arm
    rather than a change of default.
    """
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    base = os.environ.get("CRITIC_BASE_URL")
    if not base:
        return _client()
    # ITS OWN KEY. The critic reused LLM_API_KEY, which is fine while both
    # endpoints are the same local vLLM and wrong the moment the planner moves
    # to a hosted model: the OpenRouter key went to the local server, every
    # chunk came back 401, and `judge_candidates` did what it is designed to do
    # with a failure -- default every candidate to KEPT. The run then carried on
    # labelled critic-on while running critic-off. That is the mislabelling
    # D-070 exists to prevent, arriving through a different door.
    return OpenAI(
        base_url=base,
        api_key=os.environ.get("CRITIC_API_KEY")
                 or os.environ.get("LLM_API_KEY", "dummy"),
        timeout=float(os.environ.get("LLM_TIMEOUT", 120)),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", 3)),
    ), os.environ.get("CRITIC_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")


def _rank_client():
    """The judge the RANKER uses: RANK_* env, else the critic's client (D-129).

    Candidate filtering is easy enough for an 8B (D-089); ordering papers by
    how well they satisfy a question is not -- llama-8b and qwen3-14b ranked at
    random in both pointwise and listwise form, qwen3-32b ranked (F1 0.645).
    Separating the endpoints lets the cluster keep its local search critic and
    send only the ranking to a 32B, hosted, for cents.
    """
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    base = os.environ.get("RANK_BASE_URL")
    model = os.environ.get("RANK_MODEL")
    if not base or not model:
        return _critic_client()
    return OpenAI(
        base_url=base,
        api_key=os.environ.get("RANK_API_KEY") or os.environ.get("CRITIC_API_KEY")
                or os.environ.get("LLM_API_KEY", "dummy"),
        timeout=float(os.environ.get("LLM_TIMEOUT", 120)),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", 3)),
    ), model


def _client():
    """OpenAI-compatible client, configured exactly as the aliases layer's.

    OpenRouter works through this unchanged -- it speaks the same protocol -- so
    a hosted frontier model is a matter of three environment variables and not
    of a second code path:

        LLM_BASE_URL=https://openrouter.ai/api/v1
        LLM_MODEL_NAME=openai/gpt-4o
        LLM_API_KEY=<key, from .env, never an argument>

    `max_retries` drops to 1 off-cluster. Against a free local server three
    retries of an 18,899-token prompt cost a minute; against a metered one they
    cost three times the money for a request that already failed.
    """
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    base = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    remote = "openrouter" in base or "api.openai" in base
    headers = {}
    if "openrouter" in base:
        # OpenRouter attributes usage to these; harmless elsewhere.
        headers = {"HTTP-Referer": "https://github.com/Rauljo/HEPCoverageKG",
                   "X-Title": "HEPCoverageKG"}
    return OpenAI(
        base_url=base,
        api_key=os.environ.get("LLM_API_KEY", "dummy"),
        timeout=float(os.environ.get("LLM_TIMEOUT", 120)),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", 1 if remote else 3)),
        default_headers=headers or None,
    ), os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")


def _build_critic(question: str, session: Session, use_critic, seed=None):
    """The candidate critic, bound to this question -- or None, which is OFF.

    OFF is the default everywhere, so the ablation's control arm is this code
    path with nothing switched on rather than a second implementation that has
    to be kept in step (D-060).

    `use_critic` may be a ready-made callable (a stub, or a different-family
    model as the S-37 ablation), or True to build one on the planner's own
    endpoint. Verdicts are cached per search text: a planner that searches the
    same words twice in one session should not pay twice.
    """
    if not use_critic:
        return None

    from hepcoveragekg.query import critic as critic_mod

    if callable(use_critic):
        judge = use_critic
    else:
        client, model = _critic_client()

        def judge(search_text, hits):
            def chat(messages):
                # THINKING OFF FOR THE JUDGE, and this is why the critic was
                # silently inert for weeks.
                #
                # Qwen3.5-9B is served with `--reasoning-parser qwen3`, so its
                # chain of thought goes to `reasoning_content` and `content`
                # stays empty until it stops thinking. On this task it never
                # stops: measured 2026-09-05, finish_reason=length,
                # completion_tokens=4000, content EMPTY. `_parse` then finds no
                # JSON, returns {}, and every candidate DEFAULTS TO KEPT by the
                # deliberate asymmetry in critic.py -- silently, because a
                # default is not an error. Across four 164-question runs:
                # 33,836 candidates, 100% defaulted, ZERO drops.
                #
                # Raising the cap does not help; it thinks for whatever it is
                # given. Judging a candidate against a question is a
                # CLASSIFICATION, not a reasoning problem -- with thinking off
                # the same model returns correct verdicts in under 800 tokens
                # ("b-tagged jet" -> exact, "Muon" -> unrelated).
                #
                # Sent as extra_body because it is a vLLM/Qwen template switch,
                # not an OpenAI field; a server that rejects it gets a plain
                # retry, so a non-Qwen judge is unaffected.
                try:
                    return client.chat.completions.create(
                        model=model, messages=messages, temperature=0.0,
                        max_tokens=completion_cap(model),
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                except Exception:  # noqa: BLE001 -- judge must not kill the run
                    pass
                # THE JUDGE NEEDS ITS OWN CAP, and this is D-084 repeating.
                # `MAX_COMPLETION_TOKENS` is 800 -- generous for the verdict
                # JSON, and nothing at all for a REASONING judge, which spends
                # the allowance on its chain of thought and is cut off before it
                # emits the block. `_parse` then finds no JSON and returns {},
                # and every candidate DEFAULTS TO KEPT by the asymmetry a few
                # lines below in critic.py -- silently, because a default is not
                # an error.
                #
                # Measured 2026-09-05 with CRITIC_MODEL=Qwen/Qwen3.5-9B across
                # four 164-question runs: 33,836 candidates, 100% defaulted,
                # ZERO drops. The critic had judged nothing at all, while every
                # arm was reported as running with a critic. `completion_cap`
                # already knows this model needs 4000; it was simply never
                # applied on this path.
                return client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    max_tokens=completion_cap(model))

            # A small judge needs the demonstration-led prompt: on 2026-08-17
            # the 8B went from 38.9%/86.7% kept on two all-drop controls to
            # 3.3%/3.3% when the comparison was made structural and the examples
            # balanced. Chosen by which endpoint the critic is on, so a run says
            # what it used rather than depending on an operator remembering.
            small = bool(os.environ.get("CRITIC_BASE_URL"))
            return critic_mod.judge_candidates(
                chat, question, search_text, hits, seed=seed,
                prompt=_critic_prompt(small))

    def _chat(messages):
        return client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            max_tokens=MAX_COMPLETION_TOKENS)

    seen: dict[str, Any] = {}

    def review(search_text: str, hits, refresh: bool = False) -> Any:
        """`refresh` re-judges the same words over a WIDER candidate list.

        The cache is keyed on the search text alone, which is right for a
        planner that searches the same words twice -- and wrong for the widening
        loop, where the words are identical and the candidates are not. Without
        the flag a widened search would be handed back the verdicts for the
        narrower list it has just replaced, and `should_widen` would read a tail
        that no longer exists.
        """
        if not refresh and search_text in seen:
            return seen[search_text]
        result = judge(search_text, hits)
        if search_text in seen:
            session.reviews.remove(seen[search_text])
        seen[search_text] = result
        session.reviews.append(result)
        return result

    def read_facets(field: str, values, rows) -> Any:
        """The facets site. Reads labels; removes nothing (D-060 addendum)."""
        if callable(use_critic) and not hasattr(use_critic, "read_facets"):
            return None          # a stub supplied for the search site only
        reader = getattr(use_critic, "read_facets", None)
        result = (reader(field, values, rows) if reader
                  else critic_mod.read_facet_labels(_chat, question, field, values, rows))
        if result is not None:
            session.reviews.append(result)
        return result

    review.read_facets = read_facets
    return review


def _tool_example_block(answer_contract, contract, simple_answer) -> str:
    from hepcoveragekg.query import tool_examples as te
    return te.render(tools_for(answer_contract, contract, simple_answer))


def _prepare(conn, index, question, max_rounds, max_places, max_rows,
             minimal_prompt, chat, thread_id, use_critic=None, critic_seed=None,
             answer_contract=False, contract="", force_critic_set=False,
             persist=False, push_further=False, simple_answer=False,
             fewshot="", tool_examples=False, reviewer=False,
             state_objective=False, subgoals=False, subgoal_status=False,
             path_tool=False, answer_gate=False, answer_critic=False,
             rerank=False, name_ids=False, kind_fallback=False,
             ranked_answer=False, enum_expand=False):
    """The state and runtime config a run needs. Shared by answer() and stream()."""
    session = Session(question=question)
    if chat is None:
        from hepcoveragekg.query import budget as budget_mod
        client, model = _client()

        cap = completion_cap(model)

        def chat(msgs, tls):  # noqa: E306
            kwargs = dict(model=model, messages=msgs,
                          temperature=planner_temperature(),
                          max_tokens=cap)
            if tls:
                kwargs["tools"] = tls
            return client.chat.completions.create(**kwargs)

        # A hard stop when LLM_BUDGET_USD is set, and nothing at all when it is
        # not. The cap belongs here rather than in the runner because this is
        # the only place every planner call passes through -- including the
        # critic's, which is 48% of them and would otherwise spend uncounted.
        chat = budget_mod.guard(chat, budget_mod.from_env(model))

    state = {
        "question": question,
        "messages": [
            {"role": "system",
             "content": (system_prompt(conn, minimal_prompt, state_objective)
                         + (_tool_example_block(answer_contract, contract,
                                                simple_answer)
                            if tool_examples else "")
                         + (fewshot or ""))},
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
        "execute": build_executor(conn, index, session.sets,
                                  critic=_build_critic(question, session, use_critic,
                                                       critic_seed),
                                  answer_contract=answer_contract,
                                  force_critic_set=force_critic_set,
                                  rerank=rerank,
                                  rank_papers=_paper_ranker(conn, session, rerank,
                                                            answer_critic),
                                  question_of=lambda: question,
                                  kind_fallback=kind_fallback,
                                  on_fallback=_fallback_counter(session),
                                  enum_expand=enum_expand,
                                  question_text=question,
                                  on_enum=_enum_counter(session)),
        "tools": [{"type": "function", "function": spec}
                  for spec in tools_for(answer_contract, contract, simple_answer,
                                        path_tool, name_ids)],
    }
    # WHICH CONTRACT, not "was the v2 flag passed". These are not the same
    # thing and the difference silently disabled half of v3.
    #
    # `tools_for` has always resolved v1/v2/v3 properly, but the abstention
    # challenge in graph.py was gated on `answer_contract` -- true only for v2.
    # So `--contract v3`, whose whole definition is "cite the paper set AND
    # defend an abstention held against evidence", ran with the second mechanism
    # dead. Every v3 measurement to date, including the +0.059 set F1 that
    # justified making it the proposed default, was made with one of its two
    # parts switched off. Measured on the 2026-08-28 arms: 105 abstentions, 0
    # challenged -- while holding 59 retrieved entities across 30 papers each.
    resolved = contract or ("v2" if answer_contract else "v1")
    runtime["contract"] = resolved
    runtime["answer_contract"] = bool(answer_contract)      # v2-only tools
    runtime["challenge_abstention"] = resolved in ("v2", "v3")
    # An ARM, never a default, until it is measured against the failure that
    # would matter: an abstention rate collapsing toward zero while precision
    # falls is a system fabricating coverage, not finding it.
    # THE STATED OBJECTIVE (arm). Folded into the planner's existing call rather
    # than asked in a second one: a separate "what were you trying to do?" call
    # costs a round and could disagree with the first. The prose the model
    # writes alongside its tool calls IS the objective; this only makes writing
    # it compulsory and gives it a shape the reviewer can read.
    runtime["state_objective"] = bool(state_objective)
    # ONE CALL, BEFORE THE FIRST ROUND, and it does not consume one. The cost of
    # the arm is this call plus a longer prompt; the round budget is unchanged
    # so decomposition cannot starve retrieval by itself.
    runtime["subgoal_status"] = bool(subgoal_status)
    if subgoals or subgoal_status:
        from hepcoveragekg.query import subgoals as _sg
        state["sub_objectives"] = _sg.decompose(chat, question)
        state["subgoal_status"] = ""

    # THE PLAN REVIEWER (arm). Defaults to the PLANNER'S OWN model: the point is
    # a strong reasoning model judging the plan, and the 8B that judges search
    # candidates cannot assess whether a route answers a question. REVIEWER_MODEL
    # overrides it so the two can be separated in an ablation -- without that,
    # "the reviewer helped" and "a second look from a big model helped" are the
    # same measurement.
    runtime["reviewer"] = bool(reviewer)
    if reviewer:
        runtime["review_chat"] = _reviewer_chat(chat)
        from hepcoveragekg.query import schema_card
        runtime["review_schema"] = schema_card.render(conn)
    # THE ANSWER GATE (D-107). Costs no call at all -- it reads the answer the
    # model already wrote -- so the only thing an arm buys is the retry round.
    runtime["rerank"] = bool(rerank)
    runtime["kind_fallback"] = bool(kind_fallback)
    runtime["ranked_answer"] = bool(ranked_answer)
    runtime["enum_expand"] = bool(enum_expand)
    runtime["answer_gate"] = bool(answer_gate)

    # THE ANSWER CRITIC (D-106). Uses the SEARCH critic's endpoint, because the
    # job is the same shape (a small judge, many short verdicts) and because
    # putting it on the planner's model would confound "a judge helped" with
    # "a second look from the big model helped" -- the mistake `REVIEWER_MODEL`
    # exists to avoid on the plan reviewer.
    runtime["answer_critic"] = bool(answer_critic)
    if answer_critic:
        runtime["conn"] = conn
        _ac_client, _ac_model = _critic_client()
        _ac_cap = completion_cap(_ac_model)

        def _answer_critic_chat(messages):  # noqa: E306
            # `enable_thinking=False` FIRST, then the plain call. This is D-105
            # exactly: a reasoning judge spends its whole allowance thinking,
            # returns empty content, and every verdict defaults to KEEP while
            # the run is labelled critic-on.
            try:
                return _ac_client.chat.completions.create(
                    model=_ac_model, messages=messages, temperature=0.0,
                    max_tokens=_ac_cap,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            except Exception:  # noqa: BLE001 -- judge must not kill the run
                pass
            return _ac_client.chat.completions.create(
                model=_ac_model, messages=messages, temperature=0.0,
                max_tokens=_ac_cap)

        runtime["answer_critic_chat"] = _answer_critic_chat
    runtime["persist"] = bool(persist)
    runtime["push_further"] = bool(push_further)
    runtime["simple_answer"] = bool(simple_answer)
    # HEADROOM FOR BOUNCE-BACKS. A rejected plan re-enters `plan` without
    # spending a round, so the graph can legitimately traverse many more nodes
    # than rounds. Sized from the reviewer's own ceiling rather than guessed, or
    # LangGraph aborts a healthy run as a suspected loop.
    from hepcoveragekg.query.reviewer import MAX_REVIEW_CYCLES
    budget_nodes = max_rounds * 3 + 6
    if reviewer:
        budget_nodes += 2 * MAX_REVIEW_CYCLES + max_rounds
    config = {"configurable": runtime, "recursion_limit": budget_nodes}
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
    use_critic: Any = None,
    critic_seed: Optional[int] = None,
    answer_contract: bool = False,
    contract: str = "",
    force_critic_set: bool = False,
    persist: bool = False,
    push_further: bool = False,
    simple_answer: bool = False,
    fewshot: str = "",
    tool_examples: bool = False,
    reviewer: bool = False,
    state_objective: bool = False,
    subgoals: bool = False,
    subgoal_status: bool = False,
    path_tool: bool = False,
    answer_gate: bool = False,
    answer_critic: bool = False,
    rerank: bool = False,
    name_ids: bool = False,
    kind_fallback: bool = False,
    ranked_answer: bool = False,
    enum_expand: bool = False,
):
    """Yield `(node_name, session)` after each node completes.

    The same run as `answer()`, surfaced step by step. Exists so a UI can show
    what the system is doing while it does it -- which for a non-technical
    reader is the difference between a spinner and an explanation.
    """
    from hepcoveragekg.query import graph as graph_module

    session, state, config = _prepare(conn, index, question, max_rounds,
                                      max_places, max_rows, minimal_prompt,
                                      chat, thread_id, use_critic, critic_seed,
                                      answer_contract, contract, force_critic_set,
                                      persist, push_further, simple_answer,
                                      fewshot, tool_examples, reviewer,
                                      state_objective, subgoals, subgoal_status,
                                      path_tool, answer_gate, answer_critic,
                                      rerank, name_ids, kind_fallback,
                                      ranked_answer, enum_expand)
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
    use_critic: Any = None,
    critic_seed: Optional[int] = None,
    answer_contract: bool = False,
    contract: str = "",
    force_critic_set: bool = False,
    persist: bool = False,
    push_further: bool = False,
    simple_answer: bool = False,
    fewshot: str = "",
    tool_examples: bool = False,
    reviewer: bool = False,
    state_objective: bool = False,
    subgoals: bool = False,
    subgoal_status: bool = False,
    path_tool: bool = False,
    answer_gate: bool = False,
    answer_critic: bool = False,
    rerank: bool = False,
    name_ids: bool = False,
    kind_fallback: bool = False,
    ranked_answer: bool = False,
    enum_expand: bool = False,
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
                                      chat, thread_id, use_critic, critic_seed,
                                      answer_contract, contract, force_critic_set,
                                      persist, push_further, simple_answer,
                                      fewshot, tool_examples, reviewer,
                                      state_objective, subgoals, subgoal_status,
                                      path_tool, answer_gate, answer_critic,
                                      rerank, name_ids, kind_fallback,
                                      ranked_answer, enum_expand)
    graph_module.build(checkpointer=checkpointer).invoke(state, config=config)

    session.seconds = time.perf_counter() - started
    return session
