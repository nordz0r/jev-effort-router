"""Backends: TypeSafe native (default) and OpenRouter. Same body, different endpoint, model id,
key and headers; 429/529 get one retry inside the budget; the key never reaches a log."""

from __future__ import annotations

import logging

import pytest

from client import REASON_NO_API_KEY, REASON_UPSTREAM_ERROR, JevClient
from config import load_settings
from grid import DEFAULT_GRID
from stubs import StubResponse, StubTransport, decision_payload

MESSAGES = [{"role": "user", "content": "fix the flaky test"}]
SECRET = "ts-secret-value-123456"


def make(transport, config=None):
    config = dict(config or {})
    settings = load_settings(lambda key, default=None: config.get(key, default))
    return JevClient(settings, transport=transport), settings


def test_typesafe_is_the_default_backend(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    transport = StubTransport([StubResponse(decision_payload())])
    client, settings = make(transport)

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    call = transport.calls[0]
    assert settings.backend == "typesafe"
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["json"]["model"] == "jev-1.13.0"
    assert set(call["json"]) == {"state", "model", "questions"}
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert "X-Title" not in call["headers"]
    assert reason is None and decision.model == DEFAULT_GRID[0].model_id


def test_openrouter_model_id_maps_to_the_native_one():
    settings = load_settings(lambda k, d=None: {"jev_model": "typesafe/jev-1.13"}.get(k, d))
    assert settings.jev_model == "jev-1.13.0"


def test_openrouter_backend_request_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload())])
    client, _ = make(transport, {"backend": "openrouter"})

    client.decide(MESSAGES, DEFAULT_GRID)

    call = transport.calls[0]
    assert call["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert call["json"]["model"] == "typesafe/jev-1.13"
    assert call["headers"]["Authorization"] == "Bearer sk-or-test"
    assert call["headers"]["X-Title"] == "hermes-jev-effort-router"


def test_endpoint_and_key_variable_are_configurable(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("NORD_JEV_KEY", SECRET)
    transport = StubTransport([StubResponse(decision_payload())])
    client, _ = make(transport, {"api_key_env": "NORD_JEV_KEY", "endpoint": "http://127.0.0.1:1/v1/systemone"})

    client.decide(MESSAGES, DEFAULT_GRID)

    assert transport.calls[0]["url"] == "http://127.0.0.1:1/v1/systemone"
    assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_missing_key_makes_no_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    transport = StubTransport([])
    client, _ = make(transport)

    assert client.decide(MESSAGES, DEFAULT_GRID) == (None, REASON_NO_API_KEY)
    assert transport.call_count == 0


@pytest.mark.parametrize("status", [429, 529])
def test_one_retry_on_rate_limit_or_overload(monkeypatch, status):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    transport = StubTransport([StubResponse(status_code=status, text="busy"), StubResponse(decision_payload())])
    client, _ = make(transport)

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert transport.call_count == 2
    assert reason is None and decision is not None


def test_second_overload_fails_open(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    transport = StubTransport([StubResponse(status_code=529, text="x"), StubResponse(status_code=529, text="x")])
    client, _ = make(transport)

    assert client.decide(MESSAGES, DEFAULT_GRID) == (None, REASON_UPSTREAM_ERROR)
    assert transport.call_count == 2


def test_no_retry_when_the_budget_is_spent(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    transport = StubTransport([StubResponse(status_code=429, text="x")])
    client, _ = make(transport, {"timeout_s": 0.1})

    assert client.decide(MESSAGES, DEFAULT_GRID) == (None, REASON_UPSTREAM_ERROR)
    assert transport.call_count == 1


def test_missing_usage_cost_is_fine(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    body = {**decision_payload(), "usage": {"input_tokens": 10, "output_tokens": 2}}
    client, _ = make(StubTransport([StubResponse(body)]))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert reason is None and decision is not None


def test_the_key_never_reaches_the_log(monkeypatch, caplog):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    caplog.set_level(logging.DEBUG)

    def echo(url, body, headers, timeout):  # an upstream that echoes the auth header back
        return StubResponse(status_code=401, text=f"bad token {headers['Authorization']}")

    client, _ = make(StubTransport(handler=echo))
    assert client.decide(MESSAGES, DEFAULT_GRID) == (None, REASON_UPSTREAM_ERROR)

    def boom(url, body, headers, timeout):
        return RuntimeError(f"connect failed with {headers['Authorization']}")

    client, _ = make(StubTransport(handler=boom))
    assert client.decide(MESSAGES, DEFAULT_GRID) == (None, REASON_UPSTREAM_ERROR)

    assert caplog.records
    assert SECRET not in caplog.text


@pytest.mark.parametrize("score, level", [(0, "low"), (0.49, "low"), (0.6, "medium"), (1.4, "medium"), (1.6, "high"), (2, "high")])
def test_effort_score_is_rounded_to_a_level(monkeypatch, score, level):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    body = decision_payload("2", 0.9, "medium", 0.9)
    body["answers"]["reasoning_effort"]["score"] = score
    client, _ = make(StubTransport([StubResponse(body)]))

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID, effort_families=False)

    assert decision.effort_requested == level


def test_unclear_resolves_to_the_default_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    client, _ = make(StubTransport([StubResponse(decision_payload("unclear", 0.9))]),
                     {"default_model": DEFAULT_GRID[2].model_id})

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision.model == DEFAULT_GRID[2].model_id
    assert "unclear" in decision.fallback_reasons
