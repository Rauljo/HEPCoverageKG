"""
HEPCoverageKG aliases: Phase C - LLM Adjudication

Invokes the LLM to adjudicate candidate pairs with Theory RAG.
"""
from __future__ import annotations

import json
import logging
import os
import asyncio
from typing import Optional


from openai import AsyncOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
_model_name: str = ""


def _get_llm_client() -> tuple[AsyncOpenAI, str]:
    global _client, _model_name
    if _client is None:
        load_dotenv()
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("LLM_API_KEY", "dummy")
        _model_name = os.environ.get("LLM_MODEL_NAME", "NousResearch/Meta-Llama-3.1-8B-Instruct")
        _client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _client, _model_name


def _get_theory_sync(term: str) -> str:
    """
    Fetch 2-sentence Wikipedia summary ONLY if the term is <= 3 words.
    Long strings (e.g. 'VV diboson background MC sample') are already descriptive
    and will fail on Wikipedia.
    """
    words = term.strip().split()
    if len(words) > 3 or len(words) == 0:
        return ""
        
    try:
        # auto_suggest=False prevents aggressive fuzzy matching to unrelated concepts
        summary = wikipedia.summary(term, sentences=2, auto_suggest=False)
        return summary
    except Exception as e:
        # Catch PageError, DisambiguationError, or network errors gracefully
        logger.debug(f"Wikipedia RAG failed for '{term}': {e}")
        return ""


async def adjudicate_pair(
    term_a: str, 
    term_b: str, 
    kind: str = "concept",
    context_a: str = "",
    context_b: str = ""
) -> dict:
    """
    Asks the LLM if term_a and term_b represent the exact same scientific entity.
    Returns: {"is_match": bool, "confidence": float, "explanation": str}
    """
    client, model_name = _get_llm_client()
    
    # 1. Theory RAG (Temporarily disabled for speed and HPC compatibility)
    # theory_a, theory_b = await asyncio.gather(
    #     asyncio.to_thread(_get_theory_sync, term_a),
    #     asyncio.to_thread(_get_theory_sync, term_b)
    # )
    theory_a, theory_b = "", ""
    
    # 2. Construct Prompt
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

    # 3. Invoke LLM
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
            
        result = json.loads(content)
        # Ensure schema compliance
        return {
            "is_match": bool(result.get("is_match", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "explanation": str(result.get("explanation", ""))
        }
    except Exception as e:
        logger.error(f"LLM Adjudication failed for '{term_a}' vs '{term_b}': {e}")
        return {"is_match": False, "confidence": 0.0, "explanation": f"Error: {e}"}

