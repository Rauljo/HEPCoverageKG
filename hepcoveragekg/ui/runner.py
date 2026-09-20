"""Run one question the way the chapter's kept configurations run it, and
carry a conversation across questions.

Three ways to answer, all of them arms the results chapter kept:

  typed, standard   constrained decoding + the answer critic selecting the list
                    (the configuration every later mechanism was measured against)
  typed, high       the same, as chained sub-goals: decompose, one short run per
                    step, judge the union once (mirrors `eval.chain`, with the
                    legs surfaced so the page can show them)
  free-SQL          the query-writing agent, with the judge selecting its list
                    (the configuration that scored best on Qwen3.8-flash)

A conversation is a list of turns. A follow-up first goes to a one-call
router that either answers it from what earlier turns established -- no
retrieval, no cost beyond the call -- or says RETRIEVE, in which case the
full run is given the earlier turns as context, the way a chain leg is given
its earlier steps. The context is the last three turns, each answer cut to
600 characters, and the papers established so far; that is the same budget
`eval.chain._leg_question` uses, and it keeps the prompt from growing with
the conversation.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..eval.questions import Question, Truth
from ..eval.systems import Answer, from_session
from . import models as M
from .provenance import PaperEvidence, provenance

_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")
CHAIN_LEG_ROUNDS = 4          # the chained arms' legs (Table 4.11)
CHAIN_MAX_GOALS = 3
CONTEXT_TURNS = 3
CONTEXT_ANSWER_CHARS = 600
CONTEXT_PAPERS = 40


@dataclass
class Turn:
    question: str
    text: str = ""
    answered: bool = True
    papers: list[str] = field(default_factory=list)     # the list the answer ships
    entity_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)   # paper -> judge's reason
    dropped: list[str] = field(default_factory=list)
    provenance: list[PaperEvidence] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    thoughts: list = field(default_factory=list)
    goals: list[str] = field(default_factory=list)       # chain sub-goals, when chained
    rounds: int = 0
    llm_calls: int = 0
    judge_calls: int = 0
    judge_candidates: int = 0
    judge_kept: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost: float = 0.0
    model: str = ""
    agent: str = ""
    effort: str = ""
    from_memory: bool = False
    verification: Any = None
    error: str = ""


# --------------------------------------------------------------------------
# one question
# --------------------------------------------------------------------------

def _chat():
    """A plain chat callable on the planner's endpoint, for the decomposition
    and the follow-up router (the shape `subgoals.decompose` expects)."""
    from ..query import planner as _p
    client, model = _p._client()
    cap = _p.completion_cap(model)

    def chat(messages, tools=None):
        kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=cap)
        if tools:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs)
    return chat


def _question(text: str) -> Question:
    return Question(qid="app", text=text, shape="set", truth=Truth())


def _ship(answer: Answer) -> list[str]:
    """The papers an answer commits to: the judge's list when there is one,
    else the ids written in the text."""
    picked = list(getattr(answer, "constrained_ids", None) or [])
    if picked:
        return picked
    return list(dict.fromkeys(_ARXIV.findall(answer.text or "")))


def _verdicts(review) -> tuple[dict[str, str], list[str]]:
    """paper -> reason, and the dropped ids, from an AnswerReview or its dict."""
    reasons: dict[str, str] = {}
    dropped: list[str] = []
    if review is None:
        return reasons, dropped
    verdicts = getattr(review, "verdicts", None)
    if verdicts:
        for v in verdicts:
            reasons[v.paper_id] = v.why or ""
            if not v.keep:
                dropped.append(v.paper_id)
        return reasons, dropped
    if isinstance(review, dict):
        for d in review.get("dropped_papers", []) or []:
            reasons[str(d.get("id"))] = str(d.get("why", ""))
            dropped.append(str(d.get("id")))
    return reasons, dropped


def run_typed(conn, index, question: str, *, max_rounds: int,
              on_node: Optional[Callable] = None) -> tuple[Answer, Any]:
    """The typed agent with constrained decoding and the answer critic.
    Streams so the page can show each round; returns the run record and
    the session (which carries the judge's verdict objects)."""
    from ..query import planner
    session = None
    for node, session in planner.stream(conn, index, question, max_rounds=max_rounds,
                                        answer_critic=True, question_shape="papers"):
        if on_node:
            on_node(node, session)
    answer = from_session(session, conn=conn)
    return answer, session


def run_freesql(conn, index, question: str, *, max_rounds: int) -> Answer:
    """The free-SQL agent with the judge selecting its list (CRITIC_SELECTS=1
    and CONSTRAINED_IDS=1 are set by `models.apply_env`)."""
    from ..eval.free_sql import FreeSQLSystem
    system = FreeSQLSystem(conn, index, max_rounds=max_rounds)
    return system.answer(_question(question))


def run_chained(conn, index, question: str, *, max_rounds: int = 6,
                on_goals: Optional[Callable] = None, on_leg: Optional[Callable] = None,
                on_node: Optional[Callable] = None):
    """Chained sub-goals, mirroring `eval.chain.ChainedSubgoalSystem` step for
    step, with the legs exposed. Returns (answer, goals, judge review).
    `max_rounds` is only for the fail-open path (no decomposition); the legs
    run at the chained arms' four rounds."""
    from ..eval.chain import _leg_question
    from ..eval.free_sql import critic_selects_papers
    from ..query import subgoals as sg

    chat = _chat()
    goals = sg.decompose(chat, question, CHAIN_MAX_GOALS)
    if on_goals:
        on_goals(goals)
    if not goals:                                   # fail open, as the chain does
        answer, session = run_typed(conn, index, question, max_rounds=max_rounds, on_node=on_node)
        return answer, [], getattr(session, "answer_review", None)

    prior: list = []
    papers: set = set()
    entities: set = set()
    evidence: set = set()
    steps: list = []
    calls = rounds = ptok = ctok = 0
    last: Optional[Answer] = None
    for i, goal in enumerate(goals):
        is_last = i == len(goals) - 1
        if on_leg:
            on_leg(i, goal, is_last)
        text = _leg_question(question, goal, prior, sorted(papers), is_last)
        a, _ = run_typed(conn, index, text, max_rounds=CHAIN_LEG_ROUNDS, on_node=on_node)
        steps += list(a.steps or [])
        entities |= set(a.entity_ids or [])
        evidence |= set(a.evidence_ids or [])
        papers |= set(a.papers or [])
        calls += a.llm_calls or 0
        rounds += a.rounds or 0
        ptok += a.prompt_tokens or 0
        ctok += a.completion_tokens or 0
        prior.append((goal, a.text))
        last = a
        if a.error:
            break

    review = None
    if papers and last is not None and not last.error:
        picked, review = critic_selects_papers(conn, question, set(papers))
        if picked:
            last.constrained_ids = list(picked)
            last.constrained_mode = "chain-critic"
            last.text = (last.text or "").rstrip() + "\n\nPapers: " + ", ".join(picked)
    assert last is not None
    last.entity_ids = sorted(entities)
    last.evidence_ids = sorted(evidence)
    last.papers = sorted(papers)
    last.steps = steps
    last.llm_calls, last.rounds = calls, rounds
    last.prompt_tokens, last.completion_tokens = ptok, ctok
    return last, goals, review


def to_turn(conn, question: str, answer: Answer, *, spec: M.ModelSpec, agent: str,
            effort: str, review=None, goals: list[str] = (), session=None,
            seconds: float = 0.0) -> Turn:
    """The run record, plus provenance for the papers it ships."""
    reasons, dropped = _verdicts(review if review is not None
                                 else getattr(session, "answer_review", None) or answer.answer_review)
    papers = _ship(answer)
    summary = review if isinstance(review, dict) else (answer.answer_review if isinstance(answer.answer_review, dict) else {})
    judge_calls = int((summary or {}).get("calls") or 0)
    judge_candidates = int((summary or {}).get("candidates") or 0)
    judge_kept = int((summary or {}).get("kept") or 0)
    # The chain-level and free-SQL judges return counts without per-paper
    # verdicts; the dropped list is then the candidate pool minus what shipped.
    if not dropped and getattr(answer, "constrained_ids", None):
        shipped = set(papers)
        dropped = [p for p in (answer.papers or []) if p not in shipped]
    t = Turn(
        question=question, text=answer.text or "", answered=bool(answer.answered),
        papers=papers, entity_ids=list(answer.entity_ids or []),
        evidence_ids=list(answer.evidence_ids or []), verdicts=reasons, dropped=dropped,
        steps=list(answer.steps or []), thoughts=list(getattr(session, "thoughts", []) or []),
        goals=list(goals), rounds=answer.rounds or 0, llm_calls=answer.llm_calls or 0,
        judge_calls=judge_calls, judge_candidates=judge_candidates, judge_kept=judge_kept,
        prompt_tokens=answer.prompt_tokens or 0,
        completion_tokens=answer.completion_tokens or 0,
        seconds=seconds or answer.seconds or 0.0, model=spec.label, agent=agent,
        effort=effort, verification=getattr(session, "verification", None),
        error=answer.error or "")
    t.cost = M.cost_usd(spec, t.prompt_tokens, t.completion_tokens, t.judge_calls)
    t.provenance = provenance(conn, question, papers)
    for pe in t.provenance:
        pe.verdict = reasons.get(pe.paper_id, "")
    return t


# --------------------------------------------------------------------------
# the conversation
# --------------------------------------------------------------------------

def answer_excerpt(text: str, limit: int = CONTEXT_ANSWER_CHARS) -> str:
    """The part of an answer worth carrying forward: the conclusion, not the
    reasoning. A reasoning model's answer text often opens with its chain of
    thought, so the excerpt is taken from the end, keeping the `Papers:` line
    whole when there is one."""
    text = (text or "(no answer)").strip()
    if "Papers:" in text:
        head, _, tail = text.rpartition("Papers:")
        papers_line = "Papers: " + " ".join(tail.split())[:limit]
        head = head.strip()
        room = max(limit - len(papers_line), 0)
        return (("…" + head[-room:] + "\n") if room and head else "") + papers_line
    return ("…" + text[-limit:]) if len(text) > limit else text


def context_block(turns: list[Turn]) -> str:
    """What a follow-up is given: the last turns and the papers so far."""
    recent = [t for t in turns if not t.error][-CONTEXT_TURNS:]
    if not recent:
        return ""
    lines = ["CONVERSATION SO FAR"]
    for i, t in enumerate(recent, 1):
        lines.append(f"  Q{i}. {t.question}")
        lines.append(f"     -> {answer_excerpt(t.text)}")
    established: list[str] = []
    for t in reversed(turns):
        for p in t.papers:
            if p not in established:
                established.append(p)
    if established:
        lines.append("")
        lines.append("PAPERS ESTABLISHED SO FAR: " + ", ".join(established[:CONTEXT_PAPERS]))
    return "\n".join(lines)


ROUTER_PROMPT = """\
You are continuing a conversation about what a set of ATLAS and CMS papers cover. \
Below are the earlier questions, the answers given, and the papers established. \
Then comes a NEW QUESTION.

If the new question can be answered COMPLETELY from the material below, write the \
answer now, in prose, naming only arXiv ids that appear in the material. \
If answering it needs anything not in the material -- a paper not listed, a count \
not given, a detail the answers do not contain -- reply with the single word RETRIEVE \
and nothing else. When in doubt, reply RETRIEVE.

{context}

NEW QUESTION
{question}"""


def route_followup(turns: list[Turn], question: str) -> Optional[str]:
    """Answer from memory, or None meaning: retrieve."""
    ctx = context_block(turns)
    if not ctx:
        return None
    chat = _chat()
    response = chat([{"role": "user", "content": ROUTER_PROMPT.format(context=ctx, question=question)}])
    text = (response.choices[0].message.content or "").strip()
    if not text or text.upper().startswith("RETRIEVE") or "RETRIEVE" in text.upper()[:40]:
        return None
    return text


def followup_question(turns: list[Turn], question: str) -> str:
    """The full-run version of a follow-up: the context, then the question,
    the way a chain leg is asked."""
    ctx = context_block(turns)
    if not ctx:
        return question
    return (f"{ctx}\n\nNEW QUESTION\n{question}\n\nAnswer the NEW QUESTION. You may use "
            "what is established above together with whatever you retrieve now; "
            "retrieve when the question needs anything the material above does not hold.")


def memory_turn(conn, question: str, text: str, *, spec: M.ModelSpec, seconds: float) -> Turn:
    """A follow-up answered from earlier turns: no retrieval, provenance
    looked up for the papers it names."""
    papers = list(dict.fromkeys(_ARXIV.findall(text)))
    t = Turn(question=question, text=text, papers=papers, model=spec.label,
             agent="memory", effort="", from_memory=True, seconds=seconds, llm_calls=1)
    t.provenance = provenance(conn, question, papers)
    return t
