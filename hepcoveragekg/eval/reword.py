"""
Evaluation harness: rewriting generated questions with a model.

The generator (S-33) produces questions from templates, deterministically, so
the truth always comes from SQL and never from a model. The cost is that the
wording is robotic -- *"In how many papers is Pythia 8 recorded as generator?"*
is not how anyone speaks, and a system that handles that phrasing might still
stumble on how a physicist actually asks.

This adds natural wording **without ever letting a model near the answer**. Two
modes, with opposite goals and therefore opposite guards:

  FAITHFUL (Tier B).  Say the same thing in better English. The entity must be
      named verbatim and no constraint may be added or dropped. The truth is
      inherited unchanged, so a rewrite that shifts the meaning silently
      corrupts the label -- which is why the guard checks the entity name
      survived.

  VAGUE (retrieval).  Ask the question a physicist would ask when they do NOT
      know the name. The entity must NOT appear, so retrieval cannot succeed by
      string matching and the dense half of the index has to earn its place.
      Only "was the target entity retrieved" is scored, so ambiguity costs
      nothing here -- it is the point.

**Both guards are mechanical**, and that matters: "keep the name" and "do not
name it" are exactly the instructions a model half-follows, and a hoped-for
constraint is not a constraint. A rewrite failing its guard is discarded rather
than repaired, because a silently-drifted question is worse than a clumsy one.

Which model should write these is a separate question from whether the rewriting
works. For measuring whether rewording adds ambiguity, any model does -- the
signal is how much MORE the same system retrieves. For claiming the system
handles natural phrasing, the rewriter must be a different family from the
answerer (S-37), or the questions are phrased in the vocabulary the answerer
already finds easiest.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional

from .generate import _normalise, _qid
from .questions import Question

logger = logging.getLogger(__name__)

FAITHFUL_PROMPT = """Rewrite this question about a high-energy-physics paper database so it
sounds like a physicist asking a colleague. Keep the meaning EXACTLY the same.

Rules:
- Keep the name "{label}" exactly as written. Do not shorten, expand or translate it.
- Do not add any condition that is not already there (no energies, experiments, years).
- Do not remove any condition that is there.
- Do not make it vaguer. It must have the same single answer.
- One sentence. No preamble.

Question: {text}

Rewritten question:"""

VAGUE_PROMPT = """A physicist wants to find work related to "{label}" ({kind}) in a database of
high-energy-physics papers, but does NOT know that exact name.

Write the question they would actually ask. It should be natural and a little
loose, the way people really speak.

Rules:
- Do NOT use the words "{label}" or any obvious substring of it.
- Describe it instead: what it does, what it is for, where it appears.
- Stay in high-energy physics. Do not invent numbers, experiments or paper names.
- One sentence, phrased as a question. No preamble.

Question:"""


def _clean(text: str) -> str:
    """One sentence, no quotes, no leading label."""
    text = (text or "").strip().strip('"').strip()
    text = re.sub(r"^(rewritten question|question)\s*:\s*", "", text, flags=re.I)
    first = text.split("\n")[0].strip()
    return first


def keeps_the_name(question: str, label: str) -> bool:
    """FAITHFUL guard: the entity survived the rewrite.

    Checked on the normalised forms, so `Pythia 8.230` still matches
    `pythia 8.230` -- punctuation and case are not the meaning.
    """
    return _normalise(label) in _normalise(question)


def hides_the_name(question: str, label: str, min_token: int = 4) -> bool:
    """VAGUE guard: no meaningful word of the entity leaked into the question.

    Whole-word matching on tokens of four characters or more. Shorter tokens
    ("of", "b", "jet") appear in ordinary English and would reject almost
    everything, while a leaked "pythia" or "systematic" is exactly what turns
    this into a string-matching exercise.
    """
    text = _normalise(question)
    words = set(text.split())
    return not any(tok in words for tok in _normalise(label).split()
                   if len(tok) >= min_token)


def reword(questions: Iterable[Question], mode: str = "faithful",
           chat: Optional[Callable] = None, model: str = "",
           temperature: float = 0.7, attempts: int = 3) -> tuple[list[Question], dict]:
    """Rewrite questions, keeping the truth. Returns (questions, stats).

    `chat(prompt) -> str` is injectable so the whole path is testable with no
    model, and so switching rewriter (Qwen now, Mistral later) is a caller
    concern rather than an edit here.

    `attempts` retries a rewrite that fails its guard. The VAGUE guard rejects
    often -- measured 5 of 6 on the first pass, because "describe this without
    naming it" is genuinely hard and models reach for the name. Retrying is much
    cheaper than lowering the bar, and lowering it would defeat the purpose:
    a leaked name turns the retrieval test into string matching.
    """
    if chat is None:
        chat = _default_chat(model, temperature)

    out: list[Question] = []
    stats = {"asked": 0, "kept": 0, "failed_guard": 0, "errored": 0}

    for q in questions:
        label = q.provenance.get("label") or q.provenance.get("entity_id", "")
        if not label:
            continue
        stats["asked"] += 1
        prompt = (FAITHFUL_PROMPT.format(label=label, text=q.text) if mode == "faithful"
                  else VAGUE_PROMPT.format(label=label, kind=q.provenance.get("kind", "concept")))

        text = ""
        for _ in range(max(1, attempts)):
            try:
                candidate = _clean(chat(prompt))
            except Exception as exc:  # noqa: BLE001 -- one bad rewrite is not a run
                logger.debug("rewrite failed for %s: %s", q.qid, exc)
                stats["errored"] += 1
                break
            if not candidate or not candidate.endswith("?"):
                continue
            ok = (keeps_the_name(candidate, label) if mode == "faithful"
                  else hides_the_name(candidate, label))
            if ok:
                text = candidate
                break
        if not text:
            # Discarded, never repaired: a question that drifted from its truth
            # is worse than no question, and a leaked name makes the retrieval
            # test measure string matching instead of retrieval.
            stats["failed_guard"] += 1
            continue

        stats["kept"] += 1
        out.append(Question(
            qid=_qid(f"rw{mode}", q.provenance.get("predicate", ""), q.qid, 0),
            text=text,
            source="generated", split=q.split, shape=q.shape,
            needs=list(q.needs), difficulty=q.difficulty,
            truth=q.truth, truth_source=q.truth_source,
            provenance={**q.provenance, "rewritten_from": q.qid,
                        "rewrite_mode": mode, "rewriter": model or "injected",
                        "original_text": q.text},
            # Same group as the original, so the report can compare them directly
            # -- the retrieval-breadth difference between a template question and
            # its rewrite is the ambiguity signal.
            group=q.group or f"rw-{q.qid}",
            relation=f"{mode}_rewrite_of:{q.qid}",
        ))
    return out, stats


def _default_chat(model: str, temperature: float) -> Callable[[str], str]:
    """OpenAI-compatible endpoint, configured like the rest of the project."""
    import os

    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "dummy"),
        timeout=float(os.environ.get("LLM_TIMEOUT", 60)),
    )
    name = model or os.environ.get("LLM_MODEL_NAME", "")

    def chat(prompt: str) -> str:
        response = client.chat.completions.create(
            model=name, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=120,
        )
        return response.choices[0].message.content or ""

    return chat
