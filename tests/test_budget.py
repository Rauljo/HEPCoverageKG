"""The spend cap. There is $20 in the account and a loop does not know that."""
from __future__ import annotations

import pytest

from hepcoveragekg.query import budget as B


class Usage:
    def __init__(self, p, c): self.prompt_tokens = p; self.completion_tokens = c


class Reply:
    def __init__(self, p=1000, c=100): self.usage = Usage(p, c)


def test_it_stops_before_the_call_that_would_go_over():
    """Checked BEFORE, so the overshoot is bounded by one call rather than by
    however many were in flight."""
    b = B.Budget(limit_usd=0.01, price_in=2.50, price_out=10.00)
    calls = {"n": 0}

    def chat(*a, **k):
        calls["n"] += 1
        return Reply(p=2_000_000, c=0)          # $5 in one call

    guarded = B.guard(chat, b)
    guarded()                                    # the first is always allowed
    with pytest.raises(B.BudgetExceeded):
        guarded()
    assert calls["n"] == 1, "the second call must not have been made"


def test_the_first_call_always_goes_through():
    """A cap that refuses before measuring anything cannot tell an expensive
    model from a cheap one, and would itself be why a run produced no data."""
    b = B.Budget(limit_usd=0.0, price_in=15.0, price_out=75.0)
    b.check()                                    # no calls yet -> allowed


def test_it_counts_what_the_api_billed_not_an_estimate():
    b = B.Budget(limit_usd=10.0, price_in=2.50, price_out=10.00)
    b.record(Usage(1_000_000, 100_000))
    assert b.spent_usd == pytest.approx(2.50 + 1.00)
    assert b.calls == 1
    b.record(None)                               # a reply without usage
    assert b.calls == 1, "a missing usage block must not be counted as spend"


def test_no_budget_variable_means_no_cap(monkeypatch):
    """Right for the local vLLM: a cap on a free endpoint is pure risk of
    stopping a good run for nothing."""
    monkeypatch.delenv("LLM_BUDGET_USD", raising=False)
    assert B.from_env("openai/gpt-4o") is None
    chat = lambda *a, **k: Reply()
    assert B.guard(chat, None) is chat


def test_an_unknown_price_is_refused_not_guessed(monkeypatch):
    """Guessing would leave the cap in place and silently inert, which is worse
    than not having one -- the operator would believe they were protected."""
    monkeypatch.setenv("LLM_BUDGET_USD", "5")
    monkeypatch.delenv("LLM_PRICE_IN", raising=False)
    monkeypatch.delenv("LLM_PRICE_OUT", raising=False)
    with pytest.raises(B.BudgetExceeded, match="price of"):
        B.from_env("some/model-nobody-priced")


def test_a_price_can_be_supplied_for_an_unlisted_model(monkeypatch):
    monkeypatch.setenv("LLM_BUDGET_USD", "5")
    monkeypatch.setenv("LLM_PRICE_IN", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUT", "3.0")
    b = B.from_env("some/model-nobody-priced")
    assert (b.price_in, b.price_out) == (1.0, 3.0)


def test_the_summary_is_what_a_run_should_record():
    b = B.Budget(limit_usd=2.0, price_in=2.50, price_out=10.00)
    b.record(Usage(500_000, 10_000))
    s = b.summary()
    assert s["spent_usd"] == pytest.approx(1.35) and s["calls"] == 1
    assert s["limit_usd"] == 2.0
