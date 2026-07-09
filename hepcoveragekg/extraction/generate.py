# =============================================================================
# HEPCoverageKG: plain LLM call, backend swappable via env vars only
#
# Talks to any OpenAI-compatible chat completions endpoint - vLLM (self-
# hosted, e.g. on the HPC cluster's A100s), Groq (its API is itself OpenAI-
# compatible via https://api.groq.com/openai/v1), or any other provider that
# speaks the same protocol. Switching backends is a .env change (LLM_BASE_URL
# / LLM_MODEL_NAME / LLM_API_KEY), never a code change - this function's
# signature is the only thing rag_engine.py depends on.
# =============================================================================
import os
from typing import Optional

from openai import OpenAI

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Self-hosted vLLM doesn't check the API key, but the client library
        # requires some non-empty value; hosted providers (Groq etc.) do.
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("LLM_API_KEY", "not-needed"))
    return _client


def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Sends one prompt to the configured LLM backend and returns the raw text response."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
