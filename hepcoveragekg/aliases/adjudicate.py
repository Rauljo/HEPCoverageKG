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


def _build_prompt(term_a: str, term_b: str, kind: str, theory_a: str, theory_b: str,
                  context_a: str, context_b: str) -> str:
    prompt = f"""You are an expert physicist data-steward for a high energy physics knowledge graph.
Your task is to determine if two terms refer to the exact same {kind}.

Term A: "{term_a}"
Term B: "{term_b}"

"""
    if theory_a:
        prompt += f'Physics Definition for A: "{theory_a}"\n'
    if theory_b:
        prompt += f'Physics Definition for B: "{theory_b}"\n'

    if context_a or context_b:
        prompt += "\nGraph Context (what these terms connect to in the literature):\n"
        prompt += f"Context A: {context_a}\nContext B: {context_b}\n"

    prompt += """
Are these two terms exact synonyms that should be merged into a single canonical entity?
Note: Variations in version numbers, isotopologues, or precise particle energies signify distinct entities (e.g. 13 TeV and 14 TeV are different).

Respond ONLY with a valid JSON object matching this schema:
{
  "is_match": true/false,
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<your short reasoning>"
}
"""
    return prompt


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
        if "is_match" not in result:
            raise ValueError(f"Response missing 'is_match': {content[:200]}")

        return _ok(
            bool(result["is_match"]),
            float(result.get("confidence", 0.0)),
            str(result.get("explanation", "")),
        )
    except _FATAL_ERRORS:
        # Misconfiguration (bad key, wrong model id): every pair would fail the
        # same way, so stop rather than write 100k identical error rows.
        raise
    except Exception as e:
        logger.warning(f"Adjudication failed for '{term_a}' vs '{term_b}': {_classify(e)}: {e}")
        return _error(e)
