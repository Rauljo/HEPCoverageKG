"""Decompose a question into sub-objectives, and carry their status.

WHY, WITH THE FAILURE IT TARGETS. gf-01 asks for searches using b-tagged jets
AND missing transverse momentum. Precision is 0.90 on single-condition questions
and **0.30** on that one, and batch 2 contains 35 hand-decomposed per-condition
rows because the system could not hold three conditions at once.

Plan-on-Graph (NeurIPS 2024) names the same failure independently -- their
limitation 3, "forgetting partial conditions": the model remembered the song was
Taylor Swift's and forgot it had to have won an AMA. Their ablation makes the
fix their most valuable mechanism: removing Memory costs 4.3 points on CWQ, more
than removing Reflection (3.8) or Guidance (3.1).

TWO ARMS, NESTED, because the paper separates them and the separation is the
interesting part:

    --subgoals         the list, written once, never updated.
    --subgoal-status   the list, plus what is known about each, rewritten every
                       round. Contains the first.

E minus D isolates whether the value is in SPLITTING the question or in
REMEMBERING the split. Running D alone is expected to do little -- that is
PoG's `w/o Memory` variant, their second-worst result -- and it exists to be
subtracted.

THREE, NOT "AS FEW AS POSSIBLE". PoG asks for the minimum; we cap at three for a
harder reason: 93% of our runs finish in four rounds or fewer, so four
sub-objectives leaves one round each and decomposition starves retrieval. This
is also the first mechanism for which the round budget genuinely binds -- today
only 2.8% of runs reach `max_rounds`, so raising it alone does nothing.

STATUS IS CARRIED VERBATIM, NOT PARSED. The model rewrites the whole block each
turn and we re-inject its own text. Parsing per-objective state out of prose is
a second failure surface for no gain, and a status the model wrote is a status
it can read.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MAX_SUBGOALS = 3

DECOMPOSE_PROMPT = """\
Break this question into the sub-objectives needed to answer it from a knowledge
graph of high-energy-physics papers.

Rules:
  - AT MOST {n}. Fewer is better. One is fine if the question is simple.
  - Order them. A later one may depend on an earlier one's result.
  - Each is a retrieval or reasoning step, not a restatement of the question.
  - Every CONDITION in the question must appear in some sub-objective. A
    question asking for analyses that use X AND Y has a sub-objective for each,
    and one for combining them.

Reply with a numbered list and nothing else.

QUESTION: {question}"""

#: What the planner is told to maintain. Re-sent every round, REPLACED not
#: appended -- the message list already grows each turn, and five stale copies
#: of a status block would leave the model working out which is current.
STATUS_INSTRUCTION = """

SUB-OBJECTIVES FOR THIS QUESTION
{goals}

EVERY TURN, BEFORE YOUR TOOL CALLS, REWRITE THIS BLOCK:

STATUS:
{blank}

Say what you now know for each -- "not started", "18 papers found, set_1_kept",
"done: 2001.06899, 2004.14060". Carry forward what you already established; the
point is that nothing is dropped between rounds. Then make your calls."""

_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$", re.M)
_STATUS = re.compile(r"STATUS:\s*(.+?)(?=\n\s*\n|\Z)", re.S | re.I)


def decompose(chat: Callable, question: str, max_n: int = MAX_SUBGOALS) -> list[str]:
    """Ask once for the sub-objectives. Empty list on any failure.

    Fails open: a run without sub-objectives is the baseline, which is a worse
    arm but a working one. A decomposition step that could abort the question
    would make the arm's cost a crash rather than a score.
    """
    try:
        response = chat([{"role": "user",
                          "content": DECOMPOSE_PROMPT.format(n=max_n, question=question)}],
                        None)
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"decomposition failed, continuing without: {exc}")
        return []
    goals = [m.group(2).strip() for m in _NUMBERED.finditer(text)]
    goals = [g for g in goals if len(g) > 3][:max_n]
    if not goals:
        logger.info("decomposition returned nothing parseable: %r", text[:120])
    return goals


def render(goals: list[str], status: str = "") -> str:
    """The block that goes in the prompt."""
    if not goals:
        return ""
    numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(goals, 1))
    if not status:
        blank = "\n".join(f"  {i}. (not started)" for i in range(1, len(goals) + 1))
        return STATUS_INSTRUCTION.format(goals=numbered, blank=blank)
    return STATUS_INSTRUCTION.format(goals=numbered, blank=status.strip())


def extract_status(text: str) -> str:
    """The STATUS block the model just wrote, to be handed back next round."""
    m = _STATUS.search(text or "")
    return m.group(1).strip() if m else ""


#: What the dedicated status call is asked (SUBGOAL_STATUS_CALL=1, D-173).
#:
#: The status block is normally written by the planner inside its own turn,
#: which means one call does two jobs: judge what the last results gave, and
#: decide what to do next. This splits them -- a call that only reads results
#: and reports completeness, then a planning call that reads its verdict. The
#: cost is one extra call per round; the hoped-for gain is that neither job is
#: done while the other is being thought about.
STATUS_CALL_PROMPT = """\
You are keeping track of progress on one question. You do NOT plan, you do NOT
call tools, and you do NOT answer the question.

QUESTION
{question}

SUB-OBJECTIVES
{goals}

STATUS AS OF LAST ROUND
{previous}

WHAT THE LAST ROUND'S CALLS RETURNED
{results}

Rewrite the status, one line per sub-objective, numbered. For each say what is
now established and where it came from -- "not started", "18 papers found,
set_1_kept", "done: 2001.06899, 2004.14060". Carry forward anything already
established; nothing may be dropped. If a call came back empty, say so: an
empty result is information about that sub-objective, not a blank.

Reply with the numbered lines and nothing else."""


def describe_results(steps, limit: int = 6) -> str:
    """The last few calls and what they returned, for the status call."""
    out = []
    for s in list(steps or [])[-limit:]:
        tool = getattr(s, "tool", None) or (s.get("tool") if isinstance(s, dict) else "")
        args = getattr(s, "args", None) if not isinstance(s, dict) else s.get("args")
        rows = getattr(s, "rows", 0) if not isinstance(s, dict) else s.get("rows", 0)
        err = getattr(s, "error", "") if not isinstance(s, dict) else s.get("error", "")
        preview = getattr(s, "preview", "") if not isinstance(s, dict) else s.get("preview", "")
        head = f"  {tool}({args}) -> {rows} rows"
        if err:
            head += f" [error: {err}]"
        out.append(head + (f"\n     {str(preview)[:200]}" if preview and not err else ""))
    return "\n".join(out) or "  (nothing run yet)"


def status_call(chat: Callable, question: str, goals: list[str],
                previous: str, steps) -> str:
    """One call that only updates the status. Empty string on any failure.

    Fails open like `decompose`: a round without a refreshed status is the
    ordinary arm, which is a worse configuration but a working one.
    """
    if not goals:
        return ""
    numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(goals, 1))
    body = STATUS_CALL_PROMPT.format(
        question=question, goals=numbered,
        previous=(previous.strip() or "  (nothing established yet)"),
        results=describe_results(steps))
    try:
        response = chat([{"role": "user", "content": body}], None)
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- a status call must not kill a run
        logger.warning(f"status call failed, keeping the previous status: {exc}")
        return ""
    lines = [m.group(0).strip() for m in _NUMBERED.finditer(text)]
    return "\n".join(f"  {l}" for l in lines) if lines else ""


def render_readonly(goals: list[str], status: str = "") -> str:
    """The block when a separate call maintains the status: the planner READS
    it and is not asked to rewrite it."""
    if not goals:
        return ""
    numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(goals, 1))
    body = status.strip() or "\n".join(f"  {i}. (not started)" for i in range(1, len(goals) + 1))
    return ("\n\nSUB-OBJECTIVES FOR THIS QUESTION -- every condition in the question "
            "appears in one of these, and all of them must be satisfied before you "
            "answer:\n" + numbered
            + "\n\nPROGRESS SO FAR (kept for you; do not rewrite it):\n" + body
            + "\n\nMake the calls that advance the sub-objectives still outstanding.")


def goals_only(goals: list[str]) -> str:
    """`--subgoals` without the status arm: the list, and no upkeep."""
    if not goals:
        return ""
    numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(goals, 1))
    return ("\n\nSUB-OBJECTIVES FOR THIS QUESTION -- every condition in the "
            "question appears in one of these, and all of them must be satisfied "
            "before you answer:\n" + numbered)
