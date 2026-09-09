"""
Evaluation harness: the socket every system plugs into.

The structural decision (ideas/eval-harness-design.md): the harness does not
run *our* system, it runs **any** system that can answer a question. The
planner, plain RAG, BM25 alone, the no-corpus LLM, GraphRAG, later chATLAS --
all of them are adapters behind one protocol.

Two consequences, and they are the reason this file exists at all:

  the baselines share the runner.  Otherwise each arm grows its own harness,
      the numbers are not comparable, and the comparison tables get assembled
      by hand at 2am before a deadline.

  an ablation is just a System with a different config.  No separate
      machinery: the ablation study (S-36) becomes a loop over configs.

`Answer` is deliberately richer than a string, because the per-layer scorers
need the trace. But a baseline has no tool calls and no named sets, and that is
fine -- the fields stay empty and the scorers that depend on them **abstain**
rather than scoring zero. Scoring plain RAG 0 on plan validity would invent a
difference that does not exist.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .questions import Question


@dataclass
class Answer:
    """What a system returns, and everything the scorers can look at.

    Everything below `answered` is optional trace. A system that cannot supply
    a field leaves it at its default and the scorers notice.
    """
    text: str = ""
    answered: bool = True            # False = declined / abstained (S-13)
    papers: list[str] = field(default_factory=list)   # the set the answer is about
    value: Any = None                # the number, for counting questions

    # Trace -- present for the planner, absent for flat baselines.
    steps: list[dict] = field(default_factory=list)
    sets: dict[str, list[str]] = field(default_factory=dict)
    entity_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    seen_values: list[str] = field(default_factory=list)
    verification_score: Optional[float] = None
    unsupported_claims: list[str] = field(default_factory=list)

    # Cost -- every system can fill these in, so they are never absent.
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    seconds: float = 0.0

    # Why an arm behaved as it did. Without these an ablation reports a score
    # difference and no evidence of what produced it -- the D-059 failure shape,
    # where the CLI stripped `provenance` and the unit test bypassed the CLI.
    reviews: list[dict] = field(default_factory=list)   # what the critic judged
    # The plan reviewer, kept apart from `llm_calls`/`prompt_tokens` so the
    # arm's cost is measurable rather than buried in the planner's.
    review_calls: int = 0
    reviews_rejected: int = 0
    review_prompt_tokens: int = 0
    review_completion_tokens: int = 0
    review_unparsed: int = 0
    review_ceiling_hit: bool = False
    recovered_calls: int = 0   # tool calls the SERVER's parser missed and we recovered
    cited: str = ""            # which handles the answer pointed at, if any
    abstention_challenged: bool = False
    citation_corrected: bool = False
    citation_disagrees: list = field(default_factory=list)
    invented_ids: list[str] = field(default_factory=list)
    nudged: bool = False

    # THE ANSWER GATE (D-107). `named_ids` is the count of arXiv ids in the
    # written text -- the single number that separates "this arm answers
    # better" from "this arm prints ids more often", which is what runs
    # 54201-06 could not distinguish. `gate_kind` says which shape tripped it,
    # `gate_failed` whether it stayed broken after being asked once.
    named_ids: int = 0
    gate_kind: str = ""
    #: Which notation the answer call was written in (D-116).
    answer_syntax: str = ""
    gate_retried: bool = False
    gate_failed: bool = False
    stopped_because: str = ""

    # THE ANSWER CRITIC (D-106). Same shape as `reviews`: the tally is the
    # measurement and `defaulted` is the alarm -- a judge defaulting every
    # verdict to KEEP looks identical to a judge that agreed with everything.
    answer_review: dict = field(default_factory=dict)

    # THE RERANK (D-113). One entry per `papers_of` reordering, each carrying
    # its grade spread and whether it was applied. An arm whose rankings were
    # all refused as unusable is a no-op at double the price, and that has to be
    # readable in the run rather than guessed at from a score that did not move.
    rankings: list = field(default_factory=list)
    #: The kind fallback (D-119): kinded searches made, entities it appended.
    kinded_searches: int = 0
    kind_fallback_added: int = 0
    #: The ranked answer (D-128).
    ranked_answer_asked: bool = False
    ranked_answer_shown: int = 0
    #: The grade strike (D-142): ids removed from the named answer.
    grade_struck: list = field(default_factory=list)
    constrained_candidates: int = 0
    constrained_ids: list = field(default_factory=list)
    constrained_mode: str = ""
    #: Enumeration expansion (D-131).
    enum_concepts: int = 0
    enum_added: int = 0

    error: str = ""                  # a crash, recorded rather than raised

    @property
    def has_trace(self) -> bool:
        """Whether the per-layer scorers apply at all."""
        return bool(self.steps)


@runtime_checkable
class System(Protocol):
    name: str
    config: dict

    def answer(self, q: Question) -> Answer: ...


def config_hash(config: dict) -> str:
    """Identity of a configuration.

    Ablation results are unattributable without it: two runs a week apart look
    identical in the report and differ in what produced them. Sorted keys, so
    the hash depends on the values and not on dict ordering.
    """
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# effective configuration
# --------------------------------------------------------------------------
# Arguments that are plumbing rather than configuration: they carry no setting,
# and two of them (`chat`, `checkpointer`) are live objects whose repr changes
# per process, which would give every run a different config hash.
_NOT_CONFIG = frozenset({"conn", "index", "question", "chat", "checkpointer",
                         "thread_id"})

# Environment variables that change what the system DOES. These never passed
# through `planner.answer`, so recording only its arguments would still leave a
# run unable to say which judge model it used or how wide it was allowed to
# search. Value is (variable, default-when-unset).
_ENV_CONFIG: tuple[tuple[str, str], ...] = (
    ("LLM_MODEL_NAME", ""),
    ("SEARCH_BREADTH_MAX", "240"),
    ("LLM_MAX_COMPLETION_TOKENS", "800"),
    ("CRITIC_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    ("CRITIC_BASE_URL", ""),
    ("CRITIC_API_KEY_SET", ""),
    # THE ENCODER. Absent until 2026-09-02, so a run file could not say which
    # embedding model produced it and four encoder arms could not be told apart
    # from their outputs afterwards -- the same gap that made an `index_values`
    # run indistinguishable from its baseline and produced a spurious
    # r = -0.945 (D-089).
    ("ALIASES_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
    # The schema-card arm. Recorded because it changes the prompt, and because
    # it once changed it invisibly.
    ("KIND_SEMANTICS", "0"),
    # Sampling temperature for the planner. Recorded because every arm measured
    # before 2026-09-04 assumed greedy decoding, and a run at 0.7 is not
    # comparable to one at 0.0 (D-103).
    ("PLANNER_TEMPERATURE", "0.0"),
)


def effective_config(func: Callable, overrides: dict) -> dict:
    """Every knob's ACTUAL value, defaults included.

    The bug this replaces: `config = {**planner_kwargs}` recorded only what the
    caller happened to override, so a run using defaults recorded nothing about
    them. 25 of 44 stored runs name no arm at all, and the three knobs the
    ablation study turns on are exactly the ones that were invisible --
    `critic_seed` (None is ranked, an int is shuffled), `contract`, and the
    `CRITIC_BASE_URL` that decides whether the 8B judge or the 72B one runs.
    Reconstructing which arm produced a run meant reading job scripts.

    Derived from the signature rather than a hand-kept list, so a knob added to
    `planner.answer` next month is recorded without anyone remembering to.
    """
    config: dict = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in _NOT_CONFIG or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        value = overrides.get(name, param.default)
        config[name] = None if value is inspect.Parameter.empty else value

    config.update(env_config())

    # A derived flag, because it is the arm name a human uses and reading it off
    # `critic_seed is None` at analysis time is how it gets misread.
    config["critic_order"] = "ranked" if config.get("critic_seed") is None else "shuffled"
    return config


def env_config() -> dict:
    """The environment half of a run's provenance, for any system.

    SPLIT OUT 2026-09-02 because it was reachable only through
    `effective_config`, which inspects `planner.answer`'s signature -- so the
    free-SQL agent, which builds its config dict by hand, recorded NO
    environment at all. Not the judge model, not the search breadth, not the
    encoder. Every free-SQL run in the project to that date is unable to say
    which critic or which embedding model produced it, which is how four
    encoder arms became impossible to tell apart afterwards (D-089).

    One function, both callers, so they cannot drift again.
    """
    out: dict = {}
    for var, fallback in _ENV_CONFIG:
        if var == "CRITIC_API_KEY_SET":
            # PRESENCE, never the value. A key in a run file is a key in git.
            out["env.CRITIC_API_KEY_SET"] = bool(os.environ.get("CRITIC_API_KEY"))
            continue
        # EMPTY IS UNSET, matching how the readers resolve these. Slurm's
        # `--export=ALL,VAR=` sets a variable to blank, and recording "" when
        # the code actually used its default makes the run file lie about the
        # arm -- the precise thing this function exists to prevent.
        out[f"env.{var}"] = (os.environ.get(var) or "").strip() or fallback
    return out


# --------------------------------------------------------------------------
# the stub -- S-22's point made concrete
# --------------------------------------------------------------------------

class StubSystem:
    """Answers nothing, always.

    Its whole purpose is to be run on day one, before a question set exists, so
    the harness produces a number that can be watched moving. It is also the
    floor: any real system that fails to beat "I don't know" on every question
    has a plumbing problem, not a quality problem.
    """

    def __init__(self, name: str = "stub") -> None:
        self.name = name
        self.config: dict = {"kind": "stub"}

    def answer(self, q: Question) -> Answer:
        return Answer(text="I don't know.", answered=False)


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------

class PlannerSystem:
    """The real thing. Thin, because `planner.run` already returns a Session
    carrying everything the scorers want.

    The connection and retrieval index are built once and reused across
    questions -- rebuilding the index per question would cost ~20s each and
    would dominate the wall-clock numbers being reported.
    """

    def __init__(self, conn, index, *, name: str = "hepkg",
                 index_values: bool = False, index_quotes: bool = False,
                 **planner_kwargs) -> None:
        self.name = name
        self._conn = conn
        self._index = index
        self._kwargs = planner_kwargs
        from ..query import planner
        # The EFFECTIVE configuration -- every knob, defaults included, plus the
        # environment variables that change behaviour. Recording only
        # `planner_kwargs` left a run unable to name its own arm; see
        # `effective_config`.
        self.config = {
            "kind": "planner",
            "model": os.environ.get("LLM_MODEL_NAME", ""),
            # RECORDED BECAUSE IT CHANGES THE ANSWER. `index_values` is chosen
            # in the CLI when the index is built, so it never passed through
            # `planner.answer` and `effective_config` could not see it -- the
            # 2026-08-31 run scoring 0.438 recorded nothing to distinguish it
            # from the 0.352 baseline. A knob outside the signature has to be
            # named explicitly or the comparison is unreproducible.
            "index_values": bool(index_values),
            "index_quotes": bool(index_quotes),
            # The ranker's judge, when it differs from the critic's (D-129).
            "env.RANK_MODEL": os.environ.get("RANK_MODEL", ""),
            "env.RANK_BASE_URL": os.environ.get("RANK_BASE_URL", ""),
            # How many ranked candidates the ranked answer shows (D-135).
            "env.RANKED_TOP_N": os.environ.get("RANKED_TOP_N", ""),
            # Hits per enumerated concept (D-131 addendum 4).
            "env.ENUM_LIMIT": os.environ.get("ENUM_LIMIT", ""),
            "env.RANKED_ASK_MIN_MISSING": os.environ.get("RANKED_ASK_MIN_MISSING", ""),
            "env.STRIKE_GRADE_MAX": os.environ.get("STRIKE_GRADE_MAX", ""),
            "env.CONSTRAINED_IDS": os.environ.get("CONSTRAINED_IDS", ""),
            **effective_config(planner.answer, planner_kwargs),
        }

    def answer(self, q: Question) -> Answer:
        from ..query import planner

        started = time.time()
        try:
            session = planner.answer(self._conn, self._index, q.text, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 -- one dead question must not end the run
            return Answer(text="", answered=False, error=f"{type(exc).__name__}: {exc}",
                          seconds=time.time() - started)
        return from_session(session, conn=self._conn)


def _papers_of(conn, entity_ids: list[str]) -> list[str]:
    """Which papers the retrieved entities occur in.

    This is what the system FOUND, as opposed to what it wrote down. The two
    diverge badly and the difference is not a detail: asked which papers used
    Pythia, the planner retrieved most of the corpus and listed 13 of them,
    because no English answer lists 58 papers. Scoring the prose alone measured
    its writing rather than its searching.
    """
    if not conn or not entity_ids:
        return []
    marks = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT DISTINCT paper_id FROM entity_occurrence WHERE entity_id IN ({marks})",
        tuple(entity_ids),
    ).fetchall()
    return sorted({r["paper_id"] for r in rows if r["paper_id"]})


def from_session(session, conn=None) -> Answer:
    """Session -> Answer. Kept separate so tests can build one without a model."""
    verification = getattr(session, "verification", None)
    # Everything the graph handed back, not only what `search` saved into a named
    # set. Reading `sets` alone scored `contents_of` at zero retrieval while the
    # ANSWER correctly named 83% of the entities -- an impossible pair, and the
    # giveaway that the metric was measuring one code path rather than the system.
    entity_ids = sorted(
        set(getattr(session, "known_entity_ids", set()))
        | {i for ids in session.sets.values() for i in ids}
    )
    # A CITED answer wins over the reconstructed one. `_papers_of(entity_ids)` is
    # every paper any retrieved entity appears in -- a retrieval footprint, 39
    # papers at the median against a gold of 2 -- whereas `answer_papers` is the
    # set the model actually pointed at. Falling back to the footprint keeps old
    # runs and uncited answers scoreable.
    cited_papers = list(getattr(session, "answer_papers", []) or [])
    return Answer(
        text=session.answer,
        answered=bool(session.answerable),
        papers=cited_papers or _papers_of(conn, entity_ids),
        value=getattr(session, "answer_value", None),
        # `preview` TRAVELS. It was recorded on the Session (graph.py sets
        # body[:200]) and dropped here, so no run file could say what any tool
        # returned; diagnosing Gabriel's questions on 2026-09-08 needed a full
        # replay against the DIAS database because 406 stored steps carried
        # 406 empty previews.
        steps=[{"round": s.round, "tool": s.tool, "args": s.args, "rows": s.rows,
                "error": s.error, "seconds": round(s.seconds, 3),
                "preview": s.preview,
                **({"redirected_from": s.redirected_from} if s.redirected_from else {})}
               for s in session.steps],
        sets={k: list(v) for k, v in session.sets.items()},
        entity_ids=entity_ids,
        evidence_ids=list(session.evidence_ids),
        seen_values=sorted(session.seen_values)[:5000],
        verification_score=getattr(verification, "score", None),
        unsupported_claims=[c.text for c in getattr(verification, "unsupported", [])],
        llm_calls=session.llm_calls,
        prompt_tokens=session.prompt_tokens,
        completion_tokens=session.completion_tokens,
        rounds=session.rounds,
        seconds=session.seconds,
        reviews=[{"search_text": r.search_text, "tally": r.tally,
                  "kept": len(r.kept_ids), "candidates": len(r.verdicts),
                  "defaulted": r.defaulted, "calls": r.calls, "errors": r.errors,
                  "tail_keep_rate": r.tail_keep_rate(),
                  "dropped": [{"id": v.entity_id, "why": v.reason}
                              for v in r.verdicts if not v.kept]}
                 for r in getattr(session, "reviews", [])],
        # THE PLAN REVIEWER, counted apart from the planner. `review_calls` and
        # its tokens are the cost of the arm; `reviews_rejected` is whether it
        # did anything. A reviewer approving everything is a no-op at double the
        # price, and that has to be readable in the run, not inferred later.
        review_calls=getattr(session, "review_calls", 0),
        reviews_rejected=getattr(session, "reviews_rejected", 0),
        review_prompt_tokens=getattr(session, "review_prompt_tokens", 0),
        review_completion_tokens=getattr(session, "review_completion_tokens", 0),
        review_unparsed=getattr(session, "review_unparsed", 0),
        review_ceiling_hit=bool(getattr(session, "review_ceiling_hit", False)),
        recovered_calls=getattr(session, "recovered_calls", 0),
        cited=getattr(session, "answer_cited", ""),
        abstention_challenged=bool(getattr(session, "abstention_challenged", False)),
        citation_corrected=bool(getattr(session, "citation_corrected", False)),
        citation_disagrees=list(getattr(session, "citation_disagrees", None) or []),
        invented_ids=list(getattr(session, "invented_ids", [])),
        nudged=bool(getattr(session, "nudged", False)),
        named_ids=len(set(_ARXIV_IN_TEXT.findall(session.answer or ""))),
        gate_kind=getattr(session, "answer_gate_kind", "") or "",
        answer_syntax=getattr(session, "answer_syntax", "") or "",
        # HOW THE RUN ENDED. A Session field that never reached the record, so
        # "the model answered without calling answer()" was invisible in every
        # run to date and had to be inferred from the shape of the prose.
        stopped_because=getattr(session, "stopped_because", "") or "",
        gate_retried=bool(getattr(session, "answer_gate_retried", False)),
        gate_failed=bool(getattr(session, "answer_gate_failed", False)),
        answer_review=_answer_review_dict(getattr(session, "answer_review", None)),
        rankings=[r.to_dict() for r in getattr(session, "rankings", [])],
        kinded_searches=getattr(session, "kinded_searches", 0),
        kind_fallback_added=getattr(session, "kind_fallback_added", 0),
        ranked_answer_asked=bool(getattr(session, "ranked_answer_asked", False)),
        ranked_answer_shown=getattr(session, "ranked_answer_shown", 0),
        grade_struck=list(getattr(session, "grade_struck", []) or []),
        constrained_candidates=int(getattr(session, "constrained_candidates", 0) or 0),
        constrained_ids=list(getattr(session, "constrained_ids", []) or []),
        constrained_mode=getattr(session, "constrained_mode", "") or "",
        enum_concepts=getattr(session, "enum_concepts", 0),
        enum_added=getattr(session, "enum_added", 0),
    )


#: Same pattern the scorer reads ids with, so `named_ids` can never disagree
#: with whether `judged_set_f1` found any.
_ARXIV_IN_TEXT = re.compile(r"\b\d{4}\.\d{4,5}\b")


def _answer_review_dict(review) -> dict:
    """The per-paper answer review, flattened for the run record."""
    if review is None or not getattr(review, "verdicts", None):
        return {}
    out = review.to_dict()
    out["dropped_papers"] = [{"id": v.paper_id, "why": v.why}
                             for v in review.verdicts if not v.keep]
    return out
