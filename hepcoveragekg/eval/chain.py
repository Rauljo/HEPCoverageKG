"""Sequential sub-goal CHAINING: N short runs, not one long one.

WHY THIS EXISTS, AND WHY IT IS NOT `SUBGOAL_SEQUENTIAL`.

`SUBGOAL_SEQUENTIAL=1` walks one agent through a list of objectives inside a
single run. On qwen3.8-flash that works (judged F1 0.665, its best result on
the supervisor's nine). On QwQ it collapses to 0.134, and the traces say why:
QwQ answers after two to five rounds no matter how large the budget is -- 12
rounds available, 2-5 used, every record ending by answering rather than
hitting the cap. It never reached the final objective, so it never saw the
instruction that permits answering, and it stopped anyway naming 1.2 papers.

So the mechanism asked QwQ for persistence it does not have. This module asks
for the opposite: each sub-objective is its own SHORT run -- three or four
rounds, exactly what QwQ does unprompted -- and the chain carries what earlier
legs established into the next one. The agent never has to sustain anything;
the orchestration sustains it.

WHAT CARRIES FORWARD, which is the whole difficulty. A leg that only sees its
own sub-objective cannot answer "intersect the two sets", and the final leg
must answer the ORIGINAL question, not the last sub-objective. So each leg is
given: the original question, every earlier sub-objective with the answer it
produced, and the paper ids established so far. The final leg is told to
answer the original question using all of it.

WHAT IS MERGED. The record must describe the CHAIN, not its last leg --
otherwise the scorer sees one short run's retrieval and reports that as the
system's. Entities, papers, evidence, steps, calls, rounds and seconds are
unioned or summed across legs; the text and paper set come from the final leg,
which is the one that answered.
"""
from __future__ import annotations

import os
import time

from .systems import Answer, Question


def _leg_question(original: str, goal: str, prior: list, papers: list,
                  last: bool) -> str:
    """What one leg is asked. The original question is always present."""
    lines = [f"OVERALL QUESTION\n{original}", ""]
    if prior:
        lines.append("ALREADY ESTABLISHED BY EARLIER STEPS")
        for i, (g, a) in enumerate(prior, 1):
            lines.append(f"  {i}. {g}")
            lines.append(f"     -> {(a or '(nothing)').strip()[:600]}")
        lines.append("")
    if papers:
        lines.append("PAPERS ESTABLISHED SO FAR: " + ", ".join(sorted(papers)[:40]))
        lines.append("")
    if last:
        lines.append("THIS IS THE FINAL STEP. Answer the OVERALL QUESTION above, "
                     "using everything already established together with whatever "
                     "you retrieve now.")
        if prior:
            lines.append("Do not discard the earlier findings -- they are part of "
                         "the answer.")
    else:
        lines.append(f"THIS STEP ONLY\n{goal}\n")
        lines.append("Answer THIS STEP as fully and concretely as you can, naming "
                     "the papers and entities it establishes. Do not try to answer "
                     "the overall question yet.")
    return "\n".join(lines)


class ChainedSubgoalSystem:
    """Wraps any system with `.answer(Question) -> Answer`, one leg per goal."""

    def __init__(self, inner, chat, *, max_goals: int = 3, name: str = "chain") -> None:
        self._inner = inner
        self._chat = chat
        self._max_goals = max_goals
        self.name = name
        self.config = dict(getattr(inner, "config", {}) or {})
        self.config.update({"kind": "chain", "chain_max_goals": max_goals,
                            "inner": getattr(inner, "name", "?")})

    def answer(self, q: Question) -> Answer:
        from ..query import subgoals as sg

        started = time.time()
        goals = sg.decompose(self._chat, q.text, self._max_goals)
        if not goals:
            # Fails open to the ordinary system, which is the baseline: a
            # decomposition that returns nothing must not cost the question.
            out = self._inner.answer(q)
            out.chain_legs = 0
            return out

        prior: list = []
        papers: set = set()
        entities: set = set()
        evidence: set = set()
        steps: list = []
        calls = rounds = tools = 0
        last_answer = None
        for i, goal in enumerate(goals):
            last = i == len(goals) - 1
            text = _leg_question(q.text, goal, prior, sorted(papers), last)
            leg_q = Question(**{**q.__dict__, "text": text}) if hasattr(q, "__dict__") else q
            a = self._inner.answer(leg_q)
            steps += list(a.steps or [])
            entities |= set(a.entity_ids or [])
            evidence |= set(a.evidence_ids or [])
            papers |= set(a.papers or [])
            calls += a.llm_calls or 0
            rounds += a.rounds or 0
            tools += len(a.steps or [])
            prior.append((goal, a.text))
            last_answer = a
            if a.error:
                break

        out = last_answer
        out.entity_ids = sorted(entities)
        out.evidence_ids = sorted(evidence)
        out.papers = sorted(papers)
        out.steps = steps
        out.llm_calls = calls + 1          # +1 for the decomposition call
        out.rounds = rounds
        out.seconds = time.time() - started
        out.chain_legs = len(goals)
        out.chain_goals = list(goals)
        return out
