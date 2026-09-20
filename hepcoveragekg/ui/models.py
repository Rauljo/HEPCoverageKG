"""Which model answers, which judges, and what a call costs.

The three planners are the ones the chapter measured. Two are hosted on
OpenRouter; QwQ is self-hosted (vLLM on the cluster, reached through the SSH
tunnel `.env` points LLM_BASE_URL at), because OpenRouter no longer serves it.
The judge is the 8B model every measured arm used on the hosted lane: cheap,
and the chapter found the judge's size barely matters.

Prices are OpenRouter list prices per million tokens on 2026-09-20 -- the
numbers the efficiency table uses -- so the cost shown per answer is the
table's estimate, not a bill. QwQ is priced as Qwen3-32B, the same size and
family, as the chapter does.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

OPENROUTER = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str
    base_url: str
    key_env: str                 # which .env variable holds the key
    price_in: float              # $ per million prompt tokens
    price_out: float             # $ per million completion tokens
    max_rounds: int              # the rounds the chapter's runs allowed it
    efforts: tuple               # which effort levels make sense for it
    default_agent: str           # "typed" | "freesql", from the chapter's results
    note: str


MODELS: dict[str, ModelSpec] = {
    "Qwen3.8-flash": ModelSpec(
        label="Qwen3.8-flash", model="qwen/qwen3.8-flash", base_url=OPENROUTER,
        key_env="OPENROUTER_API_KEY", price_in=0.15, price_out=0.47, max_rounds=12,
        efforts=("standard",), default_agent="freesql",
        note="Trained for agentic work: walks the graph paper by paper (10 to 12 "
             "rounds, about 200k tokens a question). Chaining adds nothing to it, so "
             "there is one effort level. The free-SQL agent with the judge scored best "
             "on it (0.74 against 0.66 typed)."),
    "Qwen3-32B": ModelSpec(
        label="Qwen3-32B", model="qwen/qwen3-32b", base_url=OPENROUTER,
        key_env="OPENROUTER_API_KEY", price_in=0.08, price_out=0.28, max_rounds=12,
        efforts=("standard", "high"), default_agent="typed",
        note="Clean tool use, three-round walk. Standard effort adds the answer critic; "
             "high effort chains sub-goals, the largest gain measured (+0.13 F1) at "
             "three times the rounds."),
    "QwQ-32B (self-hosted)": ModelSpec(
        label="QwQ-32B", model=os.environ.get("QWQ_MODEL_NAME", "Qwen/QwQ-32B-AWQ"),
        base_url=os.environ.get("QWQ_BASE_URL", os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")),
        key_env="LLM_API_KEY", price_in=0.08, price_out=0.28, max_rounds=6,
        efforts=("standard", "high"), default_agent="typed",
        note="The cluster's reasoning model, through the SSH tunnel. A third of its "
             "tool calls fail without the status block; chaining helps it most "
             "(+0.11 F1). Priced as Qwen3-32B: OpenRouter no longer serves it."),
}

JUDGE = ModelSpec(
    label="Llama-3.1-8B (judge)", model="meta-llama/llama-3.1-8b-instruct",
    base_url=OPENROUTER, key_env="OPENROUTER_API_KEY", price_in=0.05, price_out=0.08,
    max_rounds=0, efforts=(), default_agent="", note="")

# One judge call, measured on 24 calls of the production prompt (D-216).
JUDGE_CALL_PROMPT_TOKENS = 4850
JUDGE_CALL_COMPLETION_TOKENS = 190

EFFORT_HELP = {
    "standard": "Constrained decoding with the answer critic selecting the list: the "
                "configuration every later mechanism was measured against.",
    "high": "The same, run as chained sub-goals: the question is split into up to "
            "three steps, each retrieved on its own, and the union judged once. Helps "
            "questions with several conditions; costs three times the rounds.",
}


def apply_env(spec: ModelSpec, max_rounds: int | None = None) -> None:
    """Point the planner, the free-SQL agent and the judge at their endpoints.

    The library reads these at call time (`planner._client`, `_critic_client`),
    so setting them before a question is enough, and switching model between
    questions needs nothing else. The two CONSTRAINED/CRITIC switches are the
    chapter's working configuration: the judge selects the list the answer
    ships, and the list is drawn from retrieved papers only.
    """
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["LLM_BASE_URL"] = spec.base_url
    os.environ["LLM_MODEL_NAME"] = spec.model
    os.environ["LLM_API_KEY"] = os.environ.get(spec.key_env, "") or os.environ.get("LLM_API_KEY", "dummy")
    os.environ["CRITIC_BASE_URL"] = JUDGE.base_url
    os.environ["CRITIC_MODEL"] = JUDGE.model
    os.environ["CRITIC_API_KEY"] = os.environ.get(JUDGE.key_env, "")
    os.environ["CONSTRAINED_IDS"] = "1"
    os.environ["CRITIC_SELECTS"] = "1"
    os.environ["SEARCH_BREADTH_MAX"] = os.environ.get("SEARCH_BREADTH_MAX", "240")
    os.environ["KIND_SEMANTICS"] = "0"
    os.environ.setdefault("LLM_TIMEOUT", "300")
    os.environ.pop("LLM_MAX_COMPLETION_TOKENS", None)   # let the model's class decide
    if max_rounds:
        os.environ["APP_MAX_ROUNDS"] = str(max_rounds)


def endpoint_ok(spec: ModelSpec, timeout: float = 4.0) -> tuple[bool, str]:
    """Does the endpoint answer, and does it serve the model we will ask for?"""
    try:
        import httpx
        key = os.environ.get(spec.key_env, "") or "dummy"
        r = httpx.get(f"{spec.base_url.rstrip('/')}/models",
                      headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        ids = [m.get("id", "") for m in (r.json().get("data") or [])]
        if spec.base_url == OPENROUTER:
            return True, "reachable"           # the list is thousands long; trust it
        if spec.model in ids:
            return True, f"serving {spec.model}"
        return False, f"serves {', '.join(ids[:3]) or 'nothing'}, not {spec.model}"
    except Exception as exc:  # noqa: BLE001 -- a dead tunnel is the normal case
        return False, f"{type(exc).__name__}"


def cost_usd(spec: ModelSpec, prompt_tokens: int, completion_tokens: int,
             judge_calls: int = 0) -> float:
    """What this answer would cost at list prices, planner and judge included."""
    planner = (prompt_tokens * spec.price_in + completion_tokens * spec.price_out) / 1e6
    judge = judge_calls * (JUDGE_CALL_PROMPT_TOKENS * JUDGE.price_in
                           + JUDGE_CALL_COMPLETION_TOKENS * JUDGE.price_out) / 1e6
    return planner + judge
