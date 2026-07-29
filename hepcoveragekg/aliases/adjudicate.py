"""
HEPCoverageKG aliases: Phase C - LLM Adjudication

Asks an LLM whether two candidate terms denote the same entity.

A failed call is NOT a "no". This module used to return is_match=False on any
exception, which made a rate-limit or a dropped connection indistinguishable
from a genuine negative verdict -- and the negatives are exactly what the
downstream contradiction analysis reads. Every result now carries an explicit
`status` ("ok" or "error") plus an `error_type`, and callers must route on
`status` before trusting `is_match`.

Provider-agnostic: any OpenAI-compatible endpoint (self-hosted vLLM, Groq,
Together, OpenAI, ...) works by setting LLM_BASE_URL / LLM_API_KEY /
LLM_MODEL_NAME. Retries and timeout are configurable because the settings that
suit a dedicated GPU server are wrong for a shared, rate-limited API.

Environment:
    LLM_BASE_URL      OpenAI-compatible endpoint  (default: local vLLM)
    LLM_API_KEY       credential                  (default: "dummy", fine for local vLLM)
    LLM_MODEL_NAME    model id
    LLM_TIMEOUT       per-request seconds         (default: 60)
    LLM_MAX_RETRIES   SDK-level retries           (default: 5)

Retries use the OpenAI SDK's own backoff, which honours Retry-After headers on
429s; layering tenacity on top would double-retry and ignore that signal.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
_model_name: str = ""

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 5

# Failures that will not fix themselves on a retry -- surfaced loudly rather
# than absorbed into the results, because they mean the run is misconfigured.
_FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


def _get_llm_client() -> tuple[AsyncOpenAI, str]:
    global _client, _model_name
    if _client is None:
        load_dotenv()
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("LLM_API_KEY", "dummy")
        _model_name = os.environ.get("LLM_MODEL_NAME", "NousResearch/Meta-Llama-3.1-8B-Instruct")
        timeout = float(os.environ.get("LLM_TIMEOUT", DEFAULT_TIMEOUT))
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        _client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        logger.info(
            f"LLM client: {base_url} model={_model_name} "
            f"timeout={timeout}s max_retries={max_retries}"
        )
    return _client, _model_name


def reset_client() -> None:
    """Drop the cached client so changed environment variables take effect."""
    global _client, _model_name
    _client, _model_name = None, ""


def _classify(exc: Exception) -> str:
    """Bucket an exception so a run's failures can be diagnosed after the fact."""
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    if isinstance(exc, openai.AuthenticationError):
        return "auth"
    if isinstance(exc, openai.BadRequestError):
        return "bad_request"
    if isinstance(exc, openai.APIStatusError):
        return f"http_{exc.status_code}"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "parse"
    return type(exc).__name__


# A model that answers the question correctly but names the key differently has
# not failed. Qwen2.5-72B occasionally replies "same" or "is_same" instead of
# "is_match" (2 of 2,200 on the first real run); rejecting those as parse errors
# discards a perfectly good verdict.
_VERDICT_KEYS = ("is_match", "same", "is_same", "match", "are_same", "identical")


def _verdict(result: Any) -> Optional[bool]:
    """The boolean answer under whichever key the model used, or None."""
    if not isinstance(result, dict):
        return None
    for key in _VERDICT_KEYS:
        if key in result:
            value = result[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "yes", "same"):
                return True
            if isinstance(value, str) and value.strip().lower() in ("false", "no", "different"):
                return False
            return bool(value)
    return None


_WORD_CONFIDENCE = {"certain": 1.0, "high": 0.9, "probable": 0.75, "likely": 0.75,
                    "medium": 0.6, "unsure": 0.4, "low": 0.3, "guess": 0.2}


def _confidence(value: Any) -> float:
    """The model's confidence, kept as a RAW float in [0, 1].

    Not bucketed. The 8B produced a useless distribution -- all 2,712 matches
    scored >= 0.9 -- but that is a fact about that model, and bucketing here
    would destroy the evidence needed to check whether a better model
    calibrates. evaluate.py measures calibration directly (accuracy per
    confidence band); if a model turns out to be uninformative, we learn it from
    the numbers rather than assuming it in the parser.

    Word answers are mapped to a number so a model ignoring the format still
    yields something comparable.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _WORD_CONFIDENCE:
            return _WORD_CONFIDENCE[text]
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num > 1.0:  # some models answer on a 0-100 scale
        num = num / 100.0
    return max(0.0, min(1.0, num))


def _ok(is_match: bool, confidence: float, explanation: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "is_match": is_match,
        "confidence": confidence,
        "explanation": explanation,
        "error_type": None,
    }


def _error(exc: Exception) -> dict[str, Any]:
    """An unusable verdict. is_match is None -- never False -- so that a caller
    which forgets to check `status` fails loudly instead of silently counting a
    failure as a rejection."""
    return {
        "status": "error",
        "is_match": None,
        "confidence": 0.0,
        "explanation": f"{_classify(exc)}: {exc}",
        "error_type": _classify(exc),
    }


def _get_theory_sync(term: str) -> str:
    """Fetch a 2-sentence Wikipedia summary, but only for short terms.

    Long strings ('VV diboson background MC sample') are already descriptive and
    will not resolve to a page. Currently unused: the pipeline runs on compute
    nodes with no outbound network. The import is local so this module never
    hard-depends on `wikipedia`.
    """
    words = term.strip().split()
    if len(words) > 3 or len(words) == 0:
        return ""

    try:
        import wikipedia  # optional dependency; only needed if Theory RAG is re-enabled

        # auto_suggest=False prevents aggressive fuzzy matching to unrelated concepts
        return wikipedia.summary(term, sentences=2, auto_suggest=False)
    except Exception as e:
        # PageError, DisambiguationError, ImportError, or network failure
        logger.debug(f"Wikipedia RAG failed for '{term}': {e}")
        return ""


# Deliberately GENERIC. No worked HEP examples: the failure cases (particle
# swaps, lepton flavours, region ids, generator versions) are exactly what the
# trial set measures, so putting them here would be training on the test set and
# we would learn nothing about whether the model understands identity. Add
# examples later only if the numbers demand it, and use different ones.
_TASK = """You are an expert particle physicist curating a knowledge graph of published analyses.

You will see two entries extracted from different papers. Decide whether they denote THE SAME
entity, so that the graph should hold ONE node for both.

The test is substitutability, not topical similarity:
  SAME       - the entries name one and the same thing, merely worded differently. A physicist
               reading either would picture the identical object, and every statement true of one
               is true of the other.
  DIFFERENT  - anything a physicist would need to keep apart. If merging them would lose a
               distinction that changes what was measured, how it was selected, or which thing
               was used, they are DIFFERENT.

Being closely related, belonging to the same family, serving the same purpose, or appearing in
similar analyses does NOT make two entries the same. Near-identical wording does not make them the
same either. When the evidence does not settle it, answer with a LOW confidence rather than
guessing "same".

Calibrate the confidence honestly: use it to express how sure you actually are, so that among the
answers you give 0.9 to, about nine in ten should turn out correct. Reserve values above 0.9 for
cases where the evidence is decisive."""

_SCHEMA = """
Answer ONLY with a JSON object:
{
  "is_match": true or false,
  "confidence": <number between 0.0 and 1.0>,
  "explanation": "<one sentence: the specific thing that makes them the same or different>"
}"""


def _build_prompt(term_a: str, term_b: str, kind: str, theory_a: str, theory_b: str,
                  context_a: str, context_b: str) -> str:
    """Assemble the adjudication prompt.

    context_a / context_b are rendered blocks from aliases.context (every wording
    the papers used, aliases, attributes, evidence quotes). When absent, the
    prompt degrades to the bare labels.
    """
    parts = [_TASK, "", f"Entity kind: {kind}", ""]

    if context_a or context_b:
        parts += [context_a or f'ENTITY A:\n  primary name: "{term_a}"', ""]
        parts += [context_b or f'ENTITY B:\n  primary name: "{term_b}"', ""]
    else:
        parts += [f'ENTITY A: "{term_a}"', f'ENTITY B: "{term_b}"', ""]

    if theory_a:
        parts += [f'Reference definition for A: "{theory_a}"']
    if theory_b:
        parts += [f'Reference definition for B: "{theory_b}"']

    parts += [_SCHEMA]
    return "\n".join(parts)


async def adjudicate_pair(
    term_a: str,
    term_b: str,
    kind: str = "concept",
    context_a: str = "",
    context_b: str = "",
) -> dict[str, Any]:
    """Ask the LLM whether term_a and term_b are the same entity.

    Returns a dict with `status` ("ok" | "error"), `is_match` (bool, or None when
    status is "error"), `confidence`, `explanation` and `error_type`.

    Check `status` before reading `is_match`. An error is not a negative verdict.
    """
    client, model_name = _get_llm_client()

    # Theory RAG is disabled: compute nodes have no outbound network.
    theory_a, theory_b = "", ""
    prompt = _build_prompt(term_a, term_b, kind, theory_a, theory_b, context_a, context_b)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")

        result = json.loads(content)
        verdict = _verdict(result)
        if verdict is None:
            raise ValueError(f"Response has no verdict key: {content[:200]}")

        return _ok(
            verdict,
            _confidence(result.get("confidence")),
            str(result.get("explanation", "")),
        )
    except _FATAL_ERRORS:
        # Misconfiguration (bad key, wrong model id): every pair would fail the
        # same way, so stop rather than write 100k identical error rows.
        raise
    except Exception as e:
        logger.warning(f"Adjudication failed for '{term_a}' vs '{term_b}': {_classify(e)}: {e}")
        return _error(e)
