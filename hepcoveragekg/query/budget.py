"""A hard spend cap, because the account has $20 in it and a loop does not know that.

Every run so far has been free: a local vLLM on a cluster card, where a runaway
retry costs a minute of GPU nobody was using. A metered endpoint changes the
failure mode completely. The planner's own measurements say why:

  18,899 prompt tokens per record, and the whole prompt is re-sent every round.
  A 3-round question is ~57k input tokens. Gabriel's 8 questions x 3 repeats is
  466k input tokens -- $1.28 on GPT-4o, $7.86 on Opus.

So the cap is not a nicety. One malformed loop that retries a 19k-token prompt
two hundred times is the whole budget, and it would happen unattended overnight.

WHAT THIS IS AND IS NOT. It is a stop, not an estimate. It counts what the API
says it billed -- `usage.prompt_tokens` and `usage.completion_tokens` -- and
refuses the next call once the total passes the cap. It does not predict, it
does not sample, and it does not trust a token estimate made from character
counts. The first call always goes through, because a cap that refuses before
measuring anything cannot tell an expensive model from a cheap one.

It is also deliberately NOT a retry-limiter or a rate-limiter. Those are
different failures with different fixes; conflating them produces a guard that
half-does three jobs.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised in place of the call that would have gone over."""


#: $ per MILLION tokens, (input, output). Only needed for the model actually
#: being run; anything unlisted must supply LLM_PRICE_IN / LLM_PRICE_OUT, and
#: is refused rather than guessed at -- a wrong price silently disables the cap.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-opus-4": (15.00, 75.00),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "qwen/qwen-2.5-72b-instruct": (0.12, 0.39),
    # Fetched from OpenRouter's /models on 2026-08-29 rather than remembered.
    # These postdate the assistant's training data, and a guessed price would
    # leave the cap inert.
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-5.6-luna-pro": (0.20, 1.20),
    "openai/gpt-5.6-sol": (2.00, 10.00),
    "openai/gpt-5.6-sol-pro": (2.00, 10.00),
    "openai/gpt-5.6-terra": (2.00, 12.00),
    "openai/gpt-5.6-terra-pro": (2.00, 12.00),
}

#: Models that emit REASONING tokens, billed as output and invisible in the
#: reply. The 419 completion tokens measured on Qwen is not a safe estimate for
#: these: reasoning can be several thousand per call, so a preflight built from
#: Qwen's numbers understates the bill, and `max_tokens` set for Qwen can leave
#: no room for an answer after the reasoning is spent.
REASONING_MODELS = frozenset({
    "openai/gpt-5.6-luna", "openai/gpt-5.6-luna-pro",
    "openai/gpt-5.6-sol", "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-terra", "openai/gpt-5.6-terra-pro",
})


def price_for(model: str) -> Optional[tuple[float, float]]:
    """(input, output) $/M, from the table or the environment, else None."""
    override = (os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT"))
    if all(override):
        return float(override[0]), float(override[1])
    return PRICES.get(model.strip().lower())


@dataclass
class Budget:
    """What has been spent, and what may still be."""
    limit_usd: float
    price_in: float                      # $ per million input tokens
    price_out: float                     # $ per million output tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def spent_usd(self) -> float:
        return (self.prompt_tokens / 1e6 * self.price_in
                + self.completion_tokens / 1e6 * self.price_out)

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd

    def check(self) -> None:
        """Refuse the NEXT call if the last one took us over.

        Checked before rather than after, so the overshoot is bounded by one
        call instead of by however many were in flight. The first call is always
        allowed: a cap that refuses before it has measured anything cannot tell
        an expensive model from a cheap one, and would make the guard itself the
        reason a run produced no data.
        """
        if self.calls and self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.2f} of ${self.limit_usd:.2f} after "
                f"{self.calls} calls ({self.prompt_tokens:,} in, "
                f"{self.completion_tokens:,} out). Stopping before the next one.")

    def record(self, usage) -> None:
        """Count what the API says it billed, not what we guessed."""
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def summary(self) -> dict:
        return {"spent_usd": round(self.spent_usd, 4),
                "limit_usd": self.limit_usd,
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


_ACTIVE: Optional[Budget] = None


def active() -> Optional[Budget]:
    return _ACTIVE


def reset(budget: Optional[Budget]) -> None:
    global _ACTIVE
    _ACTIVE = budget


def from_env(model: str) -> Optional[Budget]:
    """A Budget when `LLM_BUDGET_USD` is set, else None.

    Absent the variable there is no cap, which is right for the local vLLM: a
    cap on a free endpoint is pure risk of stopping a good run for nothing.

    A metered run with an UNKNOWN price is refused outright rather than run
    uncapped. Guessing the price would leave the cap in place and silently
    inert, which is worse than not having one -- the operator would believe they
    were protected.
    """
    raw = os.environ.get("LLM_BUDGET_USD")
    if not raw:
        return None
    limit = float(raw)
    price = price_for(model)
    if price is None:
        raise BudgetExceeded(
            f"LLM_BUDGET_USD is set but the price of {model!r} is unknown. "
            f"Add it to budget.PRICES, or set LLM_PRICE_IN and LLM_PRICE_OUT "
            f"($ per million tokens). Refusing to run uncapped.")
    return Budget(limit_usd=limit, price_in=price[0], price_out=price[1])


def guard(chat, budget: Optional[Budget]):
    """Wrap a chat callable so it checks before and records after."""
    if budget is None:
        return chat

    def guarded(*args, **kwargs):
        budget.check()
        reply = chat(*args, **kwargs)
        budget.record(getattr(reply, "usage", None))
        return reply
    return guarded
