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
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

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
    recovered_calls: int = 0   # tool calls the SERVER's parser missed and we recovered
    cited: str = ""            # which handles the answer pointed at, if any
    abstention_challenged: bool = False
    citation_corrected: bool = False
    citation_disagrees: list = field(default_factory=list)
    invented_ids: list[str] = field(default_factory=list)
    nudged: bool = False

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

    def __init__(self, conn, index, *, name: str = "hepkg", **planner_kwargs) -> None:
        self.name = name
        self._conn = conn
        self._index = index
        self._kwargs = planner_kwargs
        self.config = {
            "kind": "planner",
            "model": os.environ.get("LLM_MODEL_NAME", ""),
            # The prompt is part of the configuration: PURPOSE full vs minimal
            # is an ablation axis, so a run that changes it must hash differently.
            **planner_kwargs,
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
        steps=[{"round": s.round, "tool": s.tool, "args": s.args, "rows": s.rows,
                "error": s.error, "seconds": round(s.seconds, 3),
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
        recovered_calls=getattr(session, "recovered_calls", 0),
        cited=getattr(session, "answer_cited", ""),
        abstention_challenged=bool(getattr(session, "abstention_challenged", False)),
        citation_corrected=bool(getattr(session, "citation_corrected", False)),
        citation_disagrees=list(getattr(session, "citation_disagrees", None) or []),
        invented_ids=list(getattr(session, "invented_ids", [])),
        nudged=bool(getattr(session, "nudged", False)),
    )
