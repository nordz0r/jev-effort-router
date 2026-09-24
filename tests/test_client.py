"""Client wire contract: payload shape, answer interpretation, failure containment."""

from __future__ import annotations

import pytest

from client import (
    REASON_LOW_CONFIDENCE,
    REASON_MALFORMED,
    REASON_NO_API_KEY,
    REASON_TIMEOUT,
    REASON_UNKNOWN_CHOICE,
    REASON_UPSTREAM_ERROR,
    JevClient,
)
from stubs import ReadTimeout, StubResponse, StubTransport, decision_payload
from config import load_settings
from grid import DEFAULT_GRID

MESSAGES = [
    {"role": "user", "content": "première question"},
    {"role": "assistant", "content": "première réponse"},
    {"role": "user", "content": "refactore le module de paiement"},
]


def make(transport, config=None):
    # These are the OpenRouter wire tests (the pre-0.3 default backend); TypeSafe's native
    # backend is covered in test_backends.py.
    config = {"backend": "openrouter", **(config or {})}
    settings = load_settings(lambda key, default=None: config.get(key, default))
    return JevClient(settings, transport=transport), settings


def test_payload_matches_the_documented_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload())])
    client, settings = make(transport)

    client.decide(MESSAGES, DEFAULT_GRID, platform="cli", provider="ollama-cloud")

    payload = transport.last_payload
    assert payload["model"] == settings.jev_model == "typesafe/jev-1.13"
    assert transport.calls[0]["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer sk-or-test"

    state = payload["state"]
    assert state["user_message"] == "refactore le module de paiement"
    assert "première question" in state["recent_context"]
    # 0.3: surface and provider carry nothing about the task and are no longer sent.
    assert "surface" not in state and "provider" not in state
    assert state["user_message_chars"] == len("refactore le module de paiement")
    # The configured model is deliberately NOT sent: it anchors the decision on itself.
    assert "currently_configured_model" not in state

    questions = payload["questions"]
    assert questions["model_route"]["type"] == "choice"
    assert set(questions["model_route"]["criteria"]) == {"1", "2", "3", "4", "5", "6", "unclear"}
    # 0.3: options are task descriptions only; the model id is not sent to Jev.
    assert "deepseek" not in questions["model_route"]["criteria"]["1"]
    assert questions["reasoning_effort"]["type"] == "score"
    assert len(questions["reasoning_effort"]["criteria"]) == 3


def test_payload_is_stated_in_one_language(monkeypatch):
    """The whole payload must be English — instructions, criteria and effort levels alike.

    A half-translated payload is a half-measured one: the grid profiles, the two question
    instructions and the effort criteria all travel in the same request and are all weighed.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload())])
    client, _ = make(transport)

    client.decide(MESSAGES, DEFAULT_GRID, platform="cli", provider="ollama-cloud")

    french = set("àâäçéèêëîïôöùûüœ")
    questions = transport.last_payload["questions"]
    for question in questions.values():
        criteria = question["criteria"]
        strings = [question["instructions"], *(criteria.values() if isinstance(criteria, dict) else criteria)]
        for text in strings:
            offenders = sorted(french & set(text.lower()))
            assert not offenders, f"French text in the Jev payload: {offenders} in {text!r}"


def test_context_window_is_bounded(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload())])
    client, _ = make(transport, config={"context_turns": 1})
    long_messages = [
        {"role": "user", "content": f"tour {index} " + "x" * 5000} for index in range(6)
    ]

    client.decide(long_messages, DEFAULT_GRID)

    state = transport.last_payload["state"]
    assert "tour 0" not in state["recent_context"]
    assert "tour 5" not in state["recent_context"]  # the last user turn is `user_message`
    assert len(state["recent_context"]) <= 12001
    assert len(state["user_message"]) <= 4001


def test_decision_interpretation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload("2", 0.91, "high", 0.66))])
    client, _ = make(transport)

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert reason is None
    assert decision.model == "kimi-k3"
    assert decision.effort == "high"
    assert decision.model_confidence == pytest.approx(0.91)
    assert decision.effort_confidence == pytest.approx(0.66)
    assert decision.model_choice == "2"
    assert decision.latency_ms >= 0

    # The grid position is offered to Jev; `choice` returns that key, and `probabilities`
    # is the full distribution over the criteria we sent (docs/jev-decisions-api.md).
    assert set(decision.model_probabilities) == {"1", "2", "3", "4", "5", "6"}
    assert decision.model_probabilities["2"] == pytest.approx(0.91)


def test_kimi_effort_override_is_applied(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "medium", 0.9))])
    client, _ = make(transport)

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision.effort_requested == "medium"
    assert decision.effort == "high"


def test_missing_api_key_short_circuits(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    transport = StubTransport([])
    client, _ = make(transport)

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision is None
    assert reason == REASON_NO_API_KEY
    assert transport.call_count == 0


def test_timeout(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    def handler(*_args, **_kwargs):
        raise ReadTimeout("slow")

    client, _ = make(StubTransport(handler=handler))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)
    assert decision is None
    assert reason == REASON_TIMEOUT


def test_upstream_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(StubTransport([StubResponse(status_code=500, text="boom")]))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)
    assert decision is None
    assert reason == REASON_UPSTREAM_ERROR


def test_non_json_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(StubTransport([StubResponse(payload=None, status_code=200, text="<html>")]))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)
    assert decision is None
    assert reason == REASON_MALFORMED


def test_missing_answers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(StubTransport([StubResponse({"answers": {}})]))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)
    assert decision is None
    assert reason == REASON_MALFORMED


def test_unknown_model_choice_degrades_to_the_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(StubTransport([StubResponse(decision_payload("99", 0.99))]), config={"default_model": "deepseek-v4.1-flash"})

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert reason is None
    assert decision.model == "deepseek-v4.1-flash"
    assert REASON_UNKNOWN_CHOICE in decision.fallback_reasons


def test_low_confidence_without_a_fallback_keeps_the_configured_model(monkeypatch):
    # 0.3: `default_model` is empty by default, so a distrusted answer never downgrades the turn.
    client, _ = make(StubTransport([StubResponse(decision_payload("2", 0.2, "high", 0.2))]))

    decision, reason = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision is None
    assert reason == REASON_LOW_CONFIDENCE


def test_low_confidence_degrades(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # Below threshold the more capable of (choice, fallback) by grid position wins; here the
    # fallback (glm-5.3, position 3) sits above the distrusted choice (kimi-k3, position 2).
    client, _ = make(StubTransport([StubResponse(decision_payload("2", 0.2, "high", 0.2))]), config={"default_model": "glm-5.3"})

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision.model == "glm-5.3"
    assert decision.effort == "medium"
    assert decision.fallback_reasons.count(REASON_LOW_CONFIDENCE) == 2


def test_a_confident_model_with_an_unconfident_effort_only_degrades_the_effort(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(StubTransport([StubResponse(decision_payload("2", 0.95, "high", 0.10))]))

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID)

    assert decision.model == "kimi-k3"
    # Degraded to the configured default ("medium"), then translated for the model family:
    # Kimi K3's documented middle is `high`, so that is what goes on the wire.
    assert decision.effort_requested == "medium"
    assert decision.effort == "high"
    assert decision.fallback_reasons == (REASON_LOW_CONFIDENCE,)


def test_non_choice_answer_types_are_treated_as_malformed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client, _ = make(
        StubTransport(
            [StubResponse(decision_payload("1", 0.9, "medium", 0.9, model_type="noul", effort_type="noul"))]
        ),
        config={"default_model": "deepseek-v4.1-flash"},
    )

    decision, _ = client.decide(MESSAGES, DEFAULT_GRID)

    # `choice` is empty on a wrongly-typed answer, so the model is off-grid and the turn
    # falls back rather than reading a probability as a model name.
    assert decision.model == "deepseek-v4.1-flash"
    assert REASON_UNKNOWN_CHOICE in decision.fallback_reasons


def test_custom_endpoint_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    transport = StubTransport([StubResponse(decision_payload())])
    client, _ = make(transport, config={"endpoint": "https://example.invalid/decisions"})

    client.decide(MESSAGES, DEFAULT_GRID)

    assert transport.calls[0]["url"] == "https://example.invalid/decisions"
