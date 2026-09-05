"""A second model that checks the plan BEFORE it is executed.

WHAT IT IS FOR. The trace anatomy says the planner does follow the graph -- 52%
of its step transitions are "named something, then walked an edge from it" --
but it commits to a route in one shot and stops after two or three rounds with a
median of one row per later step. Nothing in the loop ever asks whether the
route serves the question. The critic we already have judges *retrieved rows*
for relevance, after the fact; this judges the *plan*, before it costs a round.

WHY IT MIGHT WORK. The critic ablation is the largest single effect measured on
this project: +0.231 judged F1 (0.411 with, 0.180 without). A second model
judging something is not a new idea here, only a new target.

WHY IT MIGHT NOT. `force-critic-set` changed 417 of 426 traces and tied on 148
of 150 counts -- a mechanism can alter everything and matter nowhere. So the
accept/reject rate is recorded from the first run: **a reviewer that approves
everything is a no-op that doubles the bill**, and that has to be visible in the
metrics rather than inferred later.

FAIL OPEN, AND SAY SO. An unparseable verdict approves. A reviewer that failed
closed would block a run on its own bad output, and an infinite bounce-back is
a worse failure than an unreviewed plan -- 19 hours were lost to a retry storm
on 2026-08-30 and the lesson is cheap to apply here. Parse failures are counted
so "the reviewer is broken" cannot masquerade as "the reviewer approves a lot".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Absolute ceiling on bounce-backs for one question. Not a design limit -- the
#: arm runs with retries uncapped on purpose, to measure how many are used --
#: but a runaway guard, so a stubborn planner and a strict reviewer cannot spin
#: overnight. Hitting it is logged loudly and recorded on the session.
MAX_REVIEW_CYCLES = 20

_VERDICT = re.compile(r"^\s*(APPROVE|REVISE)\b", re.I | re.M)


@dataclass
class Review:
    approved: bool
    feedback: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parsed: bool = True


_COMMON = """\
You are reviewing another agent's plan BEFORE it runs. You do not answer the
question yourself and you do not write the plan. You decide one thing: would
running this plan serve the objective it states, and does that objective serve
the question?

Reply with APPROVE or REVISE on the first line, then one or two sentences.

APPROVE unless something is concretely wrong. A plan does not have to be the
plan you would have written, and it does not have to answer the whole question
in one round -- a step that makes real progress is a good step. Reject only for
a definite fault: it looks for something the schema does not hold, it references
an id or a set that does not exist yet, it repeats a call that has already been
made, or the objective does not bear on the question.

If you REVISE, say what to do instead in one concrete sentence. "Be more
specific" is not usable feedback; "search for the region entity first, then use
its set with subjects_of" is."""

_TYPED = _COMMON + """

The agent works by calling typed tools over a knowledge graph. Sets returned by
a search are referenced by name (set_1, set_1_kept) rather than by retyping ids.
"""

_SQL = _COMMON + """

The agent works by writing SQLite SELECT statements against the schema below.
As well as the objective, check the SQL itself: that the tables and columns
exist, that joins connect on the right keys, that a count is counting the right
thing, and that it is a SELECT. Flag SQL that would error or return the wrong
shape, even when the intent is right.
"""


def _ask(chat: Callable, system: str, user: str) -> tuple[str, int, int]:
    response = chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}], None)
    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    return ((message.content or "").strip(),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0)


def _parse(text: str) -> tuple[bool, bool]:
    """(approved, parsed). Unrecognised verdicts approve -- see module docstring."""
    m = _VERDICT.search(text or "")
    if not m:
        return True, False
    return m.group(1).upper() == "APPROVE", True


def review_plan(chat: Callable, question: str, schema: str, objective: str,
                plan: str, kind: str = "typed",
                history: str = "") -> Review:
    """Judge one round's intent and the calls proposed to serve it.

    `objective` is what the planner said it was trying to do this round. It may
    be empty -- the reviewer arm can run without the stated-objective arm, and
    then this falls back to judging the plan against the question alone, which
    is a weaker but still meaningful check.
    """
    system = _SQL if kind == "sql" else _TYPED
    parts = [f"QUESTION\n{question}", f"\nSCHEMA\n{schema}"]
    if history:
        parts.append(f"\nALREADY DONE THIS RUN\n{history}")
    parts.append("\nTHE AGENT'S STATED OBJECTIVE FOR THIS ROUND\n"
                 + (objective.strip() or "(none stated)"))
    parts.append(f"\nWHAT IT PROPOSES TO RUN\n{plan}")
    try:
        text, pin, pout = _ask(chat, system, "\n".join(parts))
    except Exception as exc:  # noqa: BLE001 -- a broken reviewer must not kill the run
        logger.warning(f"reviewer call failed, approving by default: {exc}")
        return Review(True, "", 0, 0, parsed=False)
    approved, parsed = _parse(text)
    if not parsed:
        logger.info("reviewer verdict unparseable, approving: %r", text[:120])
    return Review(approved=approved, feedback=text, prompt_tokens=pin,
                  completion_tokens=pout, parsed=parsed)


def describe_calls(calls: list[dict]) -> str:
    """The proposed tool calls, as the reviewer sees them."""
    out = []
    for c in calls:
        out.append(f"  {c.get('name')}({c.get('arguments') or '{}'})")
    return "\n".join(out) or "  (no calls)"


def describe_history(session: Any, limit: int = 8) -> str:
    """What has already been run, so the reviewer can catch a repeat."""
    out = []
    for s in (getattr(session, "steps", None) or [])[-limit:]:
        args = s.args if hasattr(s, "args") else (s.get("args") or {})
        tool = s.tool if hasattr(s, "tool") else s.get("tool")
        rows = s.rows if hasattr(s, "rows") else s.get("rows")
        out.append(f"  {tool}({args}) -> {rows} rows")
    return "\n".join(out)
