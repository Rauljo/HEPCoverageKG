# =============================================================================
# Tier 3 adjudication: a failed call must never be recorded as a "no".
#
# The original code returned is_match=False on any exception, so a rate-limit or
# a dropped connection was indistinguishable from a genuine negative verdict --
# and the negatives are what the contradiction analysis reads.
# =============================================================================
from __future__ import annotations

import asyncio
import json

import httpx
import openai
import pytest

from hepcoveragekg.aliases import adjudicate


class _FakeCompletions:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def create(self, **kwargs):
        return self._behaviour(kwargs)


class _FakeClient:
    def __init__(self, behaviour):
        self.chat = type("chat", (), {"completions": _FakeCompletions(behaviour)})()


def _reply(payload: str):
    """A well-formed OpenAI-shaped response carrying `payload` as the content."""
    message = type("msg", (), {"content": payload})()
    choice = type("choice", (), {"message": message})()
    return type("resp", (), {"choices": [choice]})()


def _run(behaviour, **kwargs):
    adjudicate._client = _FakeClient(behaviour)
    adjudicate._model_name = "test-model"
    try:
        return asyncio.run(adjudicate.adjudicate_pair("a", "b", **kwargs))
    finally:
        adjudicate.reset_client()


def _request():
    return httpx.Request("POST", "http://test/v1/chat/completions")


# --------------------------------------------------------------------------- #
# Success path
# --------------------------------------------------------------------------- #


def test_positive_verdict():
    out = _run(lambda kw: _reply(json.dumps(
        {"is_match": True, "confidence": 0.9, "explanation": "same thing"}
    )))
    assert out["status"] == "ok"
    assert out["is_match"] is True
    assert out["confidence"] == 0.9
    assert out["error_type"] is None


def test_negative_verdict_is_a_real_false():
    out = _run(lambda kw: _reply(json.dumps(
        {"is_match": False, "confidence": 0.8, "explanation": "different"}
    )))
    assert out["status"] == "ok"
    assert out["is_match"] is False  # a genuine "no", distinguishable from an error


# --------------------------------------------------------------------------- #
# Failure path -- the point of this module
# --------------------------------------------------------------------------- #


def _raise(exc):
    def behaviour(kwargs):
        raise exc
    return behaviour


@pytest.mark.parametrize(
    "exc, expected",
    [
        (openai.RateLimitError("429", response=httpx.Response(429, request=_request()), body=None),
         "rate_limit"),
        (openai.APITimeoutError(request=_request()), "timeout"),
        (openai.APIConnectionError(request=_request()), "connection"),
    ],
)
def test_api_failures_are_errors_not_rejections(exc, expected):
    out = _run(_raise(exc))
    assert out["status"] == "error"
    assert out["error_type"] == expected
    # The regression that mattered: is_match must NOT be False.
    assert out["is_match"] is None
    assert out["is_match"] is not False


def test_malformed_json_is_an_error():
    out = _run(lambda kw: _reply("not json at all"))
    assert out["status"] == "error"
    assert out["error_type"] == "parse"
    assert out["is_match"] is None


def test_empty_response_is_an_error():
    out = _run(lambda kw: _reply(""))
    assert out["status"] == "error"
    assert out["is_match"] is None


def test_response_missing_is_match_is_an_error():
    """A reply that parses but omits the verdict must not default to False."""
    out = _run(lambda kw: _reply(json.dumps({"confidence": 0.5, "explanation": "hmm"})))
    assert out["status"] == "error"
    assert out["error_type"] == "parse"
    assert out["is_match"] is None


def test_auth_failure_raises_rather_than_returning():
    """Misconfiguration would fail identically on every pair, so it must stop the
    run instead of writing thousands of identical error rows."""
    exc = openai.AuthenticationError(
        "401", response=httpx.Response(401, request=_request()), body=None
    )
    with pytest.raises(openai.AuthenticationError):
        _run(_raise(exc))


# --------------------------------------------------------------------------- #
# Client configuration
# --------------------------------------------------------------------------- #


def test_client_honours_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL_NAME", "some-model")
    monkeypatch.setenv("LLM_TIMEOUT", "12")
    monkeypatch.setenv("LLM_MAX_RETRIES", "7")
    adjudicate.reset_client()
    try:
        client, model = adjudicate._get_llm_client()
        assert model == "some-model"
        assert str(client.base_url).rstrip("/") == "http://example.invalid/v1"
        assert client.timeout == 12
        assert client.max_retries == 7  # survives sustained rate limiting
    finally:
        adjudicate.reset_client()


# --------------------------------------------------------------------------- #
# Orchestrator routing: errors must not reach `rejected`
# --------------------------------------------------------------------------- #


def test_deep_pass_routes_errors_out_of_the_verdict_counts(tmp_path, monkeypatch):
    """One match, one real rejection, one failed call -> 1/1/1, not 1/2/0.

    Counting the failure as a rejection is the bug this whole change exists to
    prevent: it silently inflates the negatives that downstream analysis reads.
    """
    from tests.test_aliases import _db_with
    from hepcoveragekg.aliases import run as aliases_run

    conn = _db_with([
        ("hepkg:object:alpha", "detector_object", ["1"]),
        ("hepkg:object:beta", "detector_object", ["1"]),
        ("hepkg:object:gamma", "detector_object", ["1"]),
        ("hepkg:object:delta", "detector_object", ["1"]),
    ])

    # Phase A stub: three candidate pairs, no embedding model involved.
    import hepcoveragekg.aliases.semantics as semantics
    monkeypatch.setattr(
        semantics, "generate_candidates",
        lambda items, **kw: [
            ("hepkg:object:alpha", "hepkg:object:beta"),
            ("hepkg:object:beta", "hepkg:object:gamma"),
            ("hepkg:object:gamma", "hepkg:object:delta"),
        ],
    )

    async def fake_adjudicate(a, b, kind="concept", **kw):
        if b.endswith("beta"):
            return {"status": "ok", "is_match": True, "confidence": 0.9,
                    "explanation": "same", "error_type": None}
        if b.endswith("gamma"):
            return {"status": "ok", "is_match": False, "confidence": 0.9,
                    "explanation": "different", "error_type": None}
        return {"status": "error", "is_match": None, "confidence": 0.0,
                "explanation": "rate_limit: 429", "error_type": "rate_limit"}

    monkeypatch.setattr(adjudicate, "adjudicate_pair", fake_adjudicate)

    out = tmp_path / "proposals.json"
    stats = aliases_run.propose_deep_semantics(conn, out, concurrency=2)

    assert stats["matched"] == 1
    assert stats["rejected"] == 1        # the failure is NOT counted here
    assert stats["errors"] == 1
    assert stats["error_rate_limit"] == 1
    assert stats["written"] == 3         # every pair is still recorded

    written = json.loads(out.read_text())
    errored = [e for e in written if e["status"] == "error"]
    assert len(errored) == 1
    assert errored[0]["is_match"] is None
    assert errored[0]["error_type"] == "rate_limit"


def test_deep_pass_writes_only_where_told(tmp_path, monkeypatch):
    """The output path is honoured exactly -- no hardcoded fallback location."""
    from tests.test_aliases import _db_with
    from hepcoveragekg.aliases import run as aliases_run
    import hepcoveragekg.aliases.semantics as semantics

    conn = _db_with([("hepkg:object:alpha", "detector_object", ["1"])])
    monkeypatch.setattr(semantics, "generate_candidates", lambda items, **kw: [])

    target = tmp_path / "nested" / "here.json"
    aliases_run.propose_deep_semantics(conn, target, concurrency=1)
    assert target.exists()
    assert not (tmp_path / aliases_run.DEFAULT_DEEP_OUT.name).exists()


def test_dry_run_makes_no_llm_calls_and_writes_nothing(tmp_path, monkeypatch):
    """The pre-flight check: a threshold change can move the candidate count by
    an order of magnitude, so it must be possible to see that number without
    paying for it."""
    from tests.test_aliases import _db_with
    from hepcoveragekg.aliases import run as aliases_run
    import hepcoveragekg.aliases.semantics as semantics

    conn = _db_with([
        ("hepkg:object:alpha", "detector_object", ["1"]),
        ("hepkg:object:beta", "detector_object", ["1"]),
        ("hepkg:generator:pythia8.212", "generator", ["1"]),
        ("hepkg:generator:pythia8.230", "generator", ["1"]),
    ])
    monkeypatch.setattr(
        semantics, "generate_candidates", lambda items, **kw: [(items[0], items[1])]
    )

    def explode(*a, **k):
        raise AssertionError("dry run must not contact the LLM")

    monkeypatch.setattr(adjudicate, "adjudicate_pair", explode)

    out = tmp_path / "should_not_exist.json"
    stats = aliases_run.propose_deep_semantics(conn, out, dry_run=True)

    assert stats["dry_run"] == 1
    assert stats["candidates"] == 2
    # the generator pair carries differing versions -> vetoed before any call
    assert stats["after_guards"] == 1
    assert stats["vetoed"] == 1
    assert not out.exists()
