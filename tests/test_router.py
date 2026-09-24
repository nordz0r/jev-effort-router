"""The router end to end: routing, replay, degradation, and the skip gates.

These exercise the real middleware callback — the same callable Hermes invokes — against a
stub Decisions API, so the contract under test is the one that ships.
"""

from __future__ import annotations

import json

import pytest

from stubs import StubContext, StubResponse, StubState, StubTransport, decision_payload, ollama_request
from config import load_settings
from router import Router


def build(tmp_path, transport, config=None, state=None, catalog=None):
    """A real router wired the way ``register(ctx)`` wires it."""
    settings = load_settings(lambda key, default=None: (config or {}).get(key, default))
    audit = StubState(tmp_path)
    router = Router(
        lambda: settings,
        get_state=lambda: audit,
        client_factory=lambda _settings: _ClientWith(transport, settings),
        # No provider catalog: these tests are about routing, and an absent catalog is the
        # "no evidence" answer, so a chosen model is never refused for a missing entry.
        catalog=_NoCatalog() if catalog is None else catalog,
    )
    return router, settings


class _NoCatalog:
    def is_known(self, model_id, provider_prefixes=()):
        return None


def _ClientWith(transport, settings):
    from client import JevClient

    return JevClient(settings, transport=transport)


def route(router, request=None, **overrides):
    kwargs = dict(
        turn_id="turn-1",
        session_id="session-1",
        platform="cli",
        model="deepseek-v4.1-flash",
        provider="ollama-cloud",
        base_url="https://ollama.com/v1",
        api_mode="chat_completions",
        api_call_count=0,
        api_request_id="turn-1:api:0",
    )
    kwargs.update(overrides)
    request = request if request is not None else ollama_request()
    return router.on_llm_request(request, dict(request), **kwargs)


# -- routing -----------------------------------------------------------------------


def test_routes_model_and_effort(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "high", 0.8))])
    router, _ = build(tmp_path, transport)

    result = route(router)

    assert result is not None
    assert result["request"]["model"] == "kimi-k3"
    assert result["request"]["reasoning_effort"] == "high"
    # `reasoning_config` must NOT be in the payload: it is a params-level input the provider
    # profile consumes to derive the top-level field, and Ollama rejects the whole call if it
    # reaches the wire.
    assert "reasoning_config" not in result["request"]
    assert result["source"] == "jev-effort-router"


def test_request_is_built_from_the_pre_middleware_payload(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("1", 0.9))])
    router, _ = build(tmp_path, transport)
    original = ollama_request(model="glm-5.3")
    modified = dict(original, injected_by_an_earlier_middleware=True)

    result = router.on_llm_request(
        modified,
        original,
        turn_id="t",
        session_id="s",
        model="glm-5.3",
        provider="ollama-cloud",
        api_mode="chat_completions",
        api_call_count=0,
    )

    assert result["request"]["model"] == "deepseek-v4.1-flash"
    assert result["request"].get("injected_by_an_earlier_middleware") is None
    assert result["request"]["messages"] == original["messages"]


def test_effort_is_omitted_when_the_decision_carries_none(tmp_path):
    transport = StubTransport(
        [StubResponse(decision_payload("1", 0.9, "medium", 0.9, include_effort=False))]
    )
    router, _ = build(tmp_path, transport)
    request = ollama_request(reasoning_config={"enabled": True, "effort": "low"})

    result = route(router, request, )

    assert result["request"]["model"] == "deepseek-v4.1-flash"
    # The model changed, so a stale effort from the old model must not survive.
    assert "reasoning_effort" not in result["request"]


# -- one decision per turn ---------------------------------------------------------


def test_follow_up_requests_replay_the_turns_first_decision(tmp_path):
    transport = StubTransport(
        [
            StubResponse(decision_payload("2", 0.9, "high", 0.9)),
            StubResponse(decision_payload("5", 0.9, "low", 0.9)),
        ]
    )
    router, _ = build(tmp_path, transport)

    first = route(router, api_call_count=0, api_request_id="turn-1:api:0")
    second = route(router, api_call_count=1, api_request_id="turn-1:api:1")

    assert first["request"]["model"] == "kimi-k3"
    assert second["request"]["model"] == "kimi-k3"
    assert second["request"]["reasoning_effort"] == "high"
    assert transport.call_count == 1


def test_replay_works_with_the_call_counter_hermes_really_sends(tmp_path):
    """Hermes passes the *incremented* counter, so a turn's first call arrives as api_call_count=1.

    Keying the memo store off `api_call_count == 0` therefore never stored anything and every
    call of a turn re-decided. This reproduces the real values.
    """
    transport = StubTransport(
        [
            StubResponse(decision_payload("2", 0.9, "high", 0.9)),
            StubResponse(decision_payload("5", 0.9, "low", 0.9)),
        ]
    )
    router, _ = build(tmp_path, transport)

    first = route(router, api_call_count=1, api_request_id="turn-1:api:1")
    second = route(router, api_call_count=2, api_request_id="turn-1:api:2")
    third = route(router, api_call_count=3, api_request_id="turn-1:api:3")

    assert first["request"]["model"] == "kimi-k3"
    assert [second["request"]["model"], third["request"]["model"]] == ["kimi-k3", "kimi-k3"]
    assert transport.call_count == 1


def test_a_new_turn_routes_again(tmp_path):
    transport = StubTransport(
        [
            StubResponse(decision_payload("2", 0.9, "high", 0.9)),
            StubResponse(decision_payload("5", 0.9, "low", 0.9)),
        ]
    )
    router, _ = build(tmp_path, transport)

    first = route(router, turn_id="turn-1")
    second = route(router, turn_id="turn-2", api_request_id="turn-2:api:0")

    assert first["request"]["model"] == "kimi-k3"
    assert second["request"]["model"] == "minimax-m3"
    assert transport.call_count == 2


def test_route_per_session_decides_once(tmp_path):
    transport = StubTransport(
        [
            StubResponse(decision_payload("2", 0.9, "high", 0.9)),
            StubResponse(decision_payload("5", 0.9, "low", 0.9)),
        ]
    )
    router, _ = build(tmp_path, transport, config={"route_per_turn": False})

    first = route(router, turn_id="turn-1")
    second = route(router, turn_id="turn-2", api_request_id="turn-2:api:0")

    assert first["request"]["model"] == "kimi-k3"
    assert second["request"]["model"] == "kimi-k3"
    assert transport.call_count == 1


# -- fail open ---------------------------------------------------------------------


def test_timeout_leaves_the_request_untouched(tmp_path):
    transport = StubTransport(handler=lambda *args: _raise_read_timeout())
    router, _ = build(tmp_path, transport)
    request = ollama_request()

    assert route(router, request, ) is None


def test_http_error_leaves_the_request_untouched(tmp_path):

    transport = StubTransport([StubResponse(status_code=402, text='{"error":"insufficient credits"}')])
    router, _ = build(tmp_path, transport)

    assert route(router, ollama_request()) is None


def test_malformed_body_leaves_the_request_untouched(tmp_path):

    transport = StubTransport([StubResponse(payload=None, status_code=200, text="not json")])
    router, _ = build(tmp_path, transport)

    assert route(router, ollama_request()) is None


def test_low_confidence_falls_back_to_the_configured_defaults(tmp_path):

    transport = StubTransport([StubResponse(decision_payload("2", 0.10, "high", 0.12))])
    router, _ = build(tmp_path, transport, config={"default_model": "glm-5.3", "default_effort": "low"})

    result = route(router, ollama_request())

    # The turn still goes out routed — on the fallback (more capable than the distrusted
    # choice by grid position), and flagged as degraded.
    assert result["request"]["model"] == "glm-5.3"
    assert result["request"]["reasoning_effort"] == "low"
    record = json.loads((tmp_path / "routes.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    # The model and the effort degrade independently, so a weak answer on both is recorded
    # twice — the audit trail has to show how many of the two decisions were distrusted.
    assert record["fallback_reasons"] == ["low_confidence", "low_confidence"]


def test_unknown_choice_falls_back(tmp_path):

    transport = StubTransport([StubResponse(decision_payload("99", 0.99, "medium", 0.99))])
    router, _ = build(tmp_path, transport, config={"default_model": "deepseek-v4.1-flash"})

    result = route(router, ollama_request())

    assert result["request"]["model"] == "deepseek-v4.1-flash"
    record = json.loads((tmp_path / "routes.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert "unknown_choice" in record["fallback_reasons"]


def test_missing_api_key_is_inert(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    transport = StubTransport([])
    router, _ = build(tmp_path, transport)

    assert route(router, ollama_request()) is None
    assert transport.call_count == 0


def test_a_raising_client_never_breaks_the_turn(tmp_path):
    class Exploding:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def decide(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    settings = load_settings(lambda _key, default=None: default)
    router = Router(
        lambda: settings,
        get_state=lambda: StubState(tmp_path),
        client_factory=lambda _settings: Exploding(),
        catalog=_NoCatalog(),
    )

    assert route(router, ollama_request()) is None


# -- skip gates --------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "openrouter"},
        {"provider": "anthropic"},
        {"api_mode": "anthropic_messages"},
        {"model": "gpt-9-ultra"},
    ],
)
def test_untouched_on_surfaces_the_router_does_not_own(tmp_path, overrides):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport)

    assert route(router, ollama_request(), **overrides) is None
    assert transport.call_count == 0


def test_disabled_switch_is_inert(tmp_path):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport, config={"enabled": False})

    assert route(router, ollama_request()) is None
    assert transport.call_count == 0


def test_a_turn_without_a_user_message_is_skipped(tmp_path):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport)
    request = ollama_request(messages=[{"role": "system", "content": "only a system message"}])

    assert route(router, request, ) is None
    assert transport.call_count == 0


def _raise_read_timeout():

    raise ReadTimeout("timed out")
