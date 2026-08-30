"""Worked examples for the planner prompt, harvested from runs that scored well.

The idea: a strong model solves questions we can check, and its traces become
exemplars for weaker or differently-tuned models. gpt-5.6-sol scores 0.669 on
free-SQL where Qwen scores 0.457; if the gap is partly "knows what a good plan
looks like", exemplars transfer it.

THE CONTAMINATION RULE, which is the whole reason this file is careful.

An exemplar must come from a question the model will NOT be scored on. Every sol
run to date is on the supervisor's eight questions, which are exactly the
evaluation set -- so none of them can be used, and the harvest has to happen on
paperA or conceptB questions first. `select` REFUSES any exemplar whose qid
appears in the evaluation set rather than trusting the caller to have checked,
because this is the kind of leak that is invisible in the results and inflates
them.

WHAT THIS IS, SAID PLAINLY. It is distillation. It changes the claim from "can
model X do this" to "can model X imitate sol", and that has to be declared
wherever the numbers appear. It also means a fair comparison applies the SAME
exemplars to every model, including the one they came from -- otherwise the
sweep measures who got helped rather than who is good.

WHY TRACES AND NOT ANSWERS. An exemplar showing only the final answer teaches
the shape of an answer. What actually differs between a 0.669 run and a 0.457
one is the PLAN -- which tools, in which order, with what widening after a thin
result. So an exemplar carries the tool calls and their row counts, and the
answer only as the thing they led to.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


class Contaminated(ValueError):
    """An exemplar drawn from a question that will be scored."""


MAX_EXAMPLES = 3
MIN_SCORE = 0.6


def _steps(answer: dict, limit: int = 6) -> list[str]:
    out = []
    for s in (answer.get("steps") or [])[:limit]:
        args = s.get("args") or {}
        # The query text and the kind are what a reader learns from; ids and
        # set names are run-specific noise.
        shown = {k: v for k, v in args.items()
                 if k in ("text", "kind", "predicate", "field", "values",
                          "mode", "query", "paper_ids")}
        rendered = json.dumps(shown, ensure_ascii=False)
        if len(rendered) > 120:
            rendered = rendered[:117] + "..."
        out.append(f"  {s.get('tool')}({rendered}) -> {s.get('rows', 0)} rows")
    return out


def candidates(run_path: Path | str, metric: str = "judged_f1",
               min_score: float = MIN_SCORE) -> list[dict]:
    """Records from one run that scored well enough to be worth imitating."""
    got = []
    for line in Path(run_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "_meta" in rec:
            continue
        scores = rec.get("rescored") or rec.get("scores") or {}
        value = scores.get(metric)
        if value is None or value < min_score:
            continue
        answer = rec.get("answer") or {}
        if not (answer.get("text") or "").strip():
            continue
        got.append({"qid": rec["qid"], "question": rec.get("question", ""),
                    "score": value, "answer": answer})
    return sorted(got, key=lambda r: -r["score"])


def select(pool: Iterable[dict], evaluation_qids: set,
           limit: int = MAX_EXAMPLES) -> list[dict]:
    """The best exemplars that are NOT in the evaluation set.

    Refuses rather than filters. A caller who passes a contaminated pool has
    made a mistake about what is being measured, and silently dropping the
    offending item would let the same mistake through next time in a form that
    does not raise.
    """
    chosen = []
    for item in pool:
        if item["qid"] in evaluation_qids:
            raise Contaminated(
                f"{item['qid']} is in the evaluation set. An exemplar drawn "
                f"from a scored question leaks the answer, and the leak is "
                f"invisible in the results. Harvest on a disjoint set.")
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def render(examples: Iterable[dict], with_plan: bool = False) -> str:
    """The block appended to the system prompt.

    TWO VARIANTS, because they teach different things and it is an open question
    which transfers.

      answer only  what a good answer LOOKS like -- the shape, the citation, the
                   fact that ids get written down. Cheap in tokens.
      with plan    the tool calls that produced it. What actually separates a
                   0.669 run from a 0.457 one is which tools were used in what
                   order, so this is the one with a mechanism behind it -- but
                   it is several times the tokens, and every token is re-sent
                   every round.
    """
    examples = list(examples)
    if not examples:
        return ""
    parts = ["", "WORKED EXAMPLES -- from questions whose answers were checked.",
             "Follow the SHAPE, not the content: these are not the question you",
             "are being asked, and their papers are not your papers."]
    for i, ex in enumerate(examples, 1):
        parts.append(f"\nExample {i}. {ex['question']}")
        if with_plan:
            parts.extend(_steps(ex["answer"]))
        text = " ".join((ex["answer"].get("text") or "").split())
        parts.append(f"  answer: {text[:240]}")
    return "\n".join(parts)


def from_run(run_path: Path | str, evaluation_qids: set,
             limit: int = MAX_EXAMPLES, metric: str = "judged_f1",
             min_score: float = MIN_SCORE, with_plan: bool = False) -> str:
    """Harvest, check, render -- the whole path in one call."""
    pool = candidates(run_path, metric=metric, min_score=min_score)
    return render(select(pool, evaluation_qids, limit), with_plan=with_plan)
