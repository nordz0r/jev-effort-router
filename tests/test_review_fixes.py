"""Re-review 5310863550 fixes: backend-aware key status and hints, per-turn memo of failed
turns in session mode, the no-space grid rule, half-up Score rounding."""

from __future__ import annotations

import json
import logging

import pytest

import host
import tools
from config import load_settings
from grid import Entry, parse_entry
from state import normalise_effort_choice
from stubs import ReadTimeout, StubResponse, StubTransport, decision_payload
from test_ocx import OCX, _custom_providers, ocx_route, records  # noqa: F401 - autouse fixture
from test_router import build


def settings_for(**config):
    return load_settings(lambda key, default=None: config.get(key, default))


def status(settings, tmp_path_factory=None):
    import tempfile
    from pathlib import Path

    router, _ = build(Path(tempfile.mkdtemp()), StubTransport([]))
    return json.loads(tools._status(router, settings))


# -- key status and hints -----------------------------------------------------------


def test_status_checks_the_openrouter_key_on_the_openrouter_backend(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    payload = status(settings_for(backend="openrouter"))
    assert payload["api_key_present"] is True
    assert payload["api_key_env"] == "OPENROUTER_API_KEY" and payload["backend"] == "openrouter"

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert status(settings_for(backend="openrouter"))["api_key_present"] is False


def test_status_checks_a_custom_api_key_env(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("NORD_JEV_KEY", "k")
    assert status(settings_for(api_key_env="NORD_JEV_KEY"))["api_key_present"] is True
    assert status(settings_for())["api_key_present"] is False


def test_status_reads_the_key_through_the_profile_secret_scope(monkeypatch):
    # The same path the router uses: host.secret -> agent.secret_scope.get_secret.
    seen = []
    monkeypatch.setattr(host, "secret", lambda name: seen.append(name) or ("scoped" if name == "NORD_JEV_KEY" else ""))
    assert status(settings_for(api_key_env="NORD_JEV_KEY"))["api_key_present"] is True
    assert "NORD_JEV_KEY" in seen


def test_hints_name_the_configured_variable():
    from client import REASON_NO_API_KEY, REASON_UPSTREAM_ERROR

    assert "TYPESAFE_API_KEY" in tools._hint_for(REASON_NO_API_KEY, settings_for())
    assert "OPENROUTER_API_KEY" not in tools._hint_for(REASON_NO_API_KEY, settings_for())
    assert "OPENROUTER_API_KEY" in tools._hint_for(REASON_NO_API_KEY, settings_for(backend="openrouter"))
    assert "NORD_JEV_KEY" in tools._hint_for(REASON_NO_API_KEY, settings_for(api_key_env="NORD_JEV_KEY"))
    assert "typesafe" in tools._hint_for(REASON_UPSTREAM_ERROR, settings_for())


def test_registration_warning_names_the_configured_variable(monkeypatch, caplog):
    from test_registration import _load_plugin

    plugin = _load_plugin()
    monkeypatch.delenv("NORD_JEV_KEY", raising=False)
    caplog.set_level(logging.INFO)

    plugin._warn_missing_key_once(settings_for(api_key_env="NORD_JEV_KEY"))

    assert "NORD_JEV_KEY" in caplog.text and "OPENROUTER_API_KEY" not in caplog.text

    caplog.clear()
    plugin._warn_missing_key_once(settings_for())  # TYPESAFE_API_KEY is set by conftest
    assert "not visible" not in caplog.text


# -- route_per_turn: false ----------------------------------------------------------


def test_session_mode_remembers_a_failed_turn_for_that_turn(tmp_path):
    transport = StubTransport([ReadTimeout("slow")])
    router, _ = build(tmp_path, transport, config={**OCX, "route_per_turn": False})

    assert ocx_route(router) is None
    assert ocx_route(router, api_call_count=2) is None
    assert transport.call_count == 1
    assert records(tmp_path)[-1]["reason"] == "replayed_no_decision"

    transport.responses.append(StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3)))
    assert ocx_route(router, turn_id="turn-2")["request"]["model"] == "xai/grok-4.7"


def test_session_mode_keeps_an_outage_fallback_to_its_turn(tmp_path):
    transport = StubTransport([ReadTimeout("slow")])
    router, _ = build(tmp_path, transport, config={**OCX, "route_per_turn": False, "default_model": "gldf-flash"})

    assert ocx_route(router)["request"]["model"] == "gldf-flash"
    assert ocx_route(router, api_call_count=2)["request"]["model"] == "gldf-flash"
    assert transport.call_count == 1

    # Next turn: Jev is asked again instead of the session being pinned to the fallback.
    transport.responses.append(StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3)))
    assert ocx_route(router, turn_id="turn-2")["request"]["model"] == "xai/grok-4.7"
    assert transport.call_count == 2
    # ...and that real decision is the one the session keeps.
    assert ocx_route(router, turn_id="turn-3")["request"]["model"] == "xai/grok-4.7"
    assert transport.call_count == 2


# -- grid entries without ": " ------------------------------------------------------


def test_colon_without_a_space_is_one_id(caplog):
    caplog.set_level(logging.DEBUG)
    assert parse_entry("a-model:does-a-thing") == Entry("a-model:does-a-thing", "")
    assert "no ': ' separator" in caplog.text
    caplog.clear()
    assert parse_entry("nemotron-3-nano:30b") == Entry("nemotron-3-nano:30b", "")
    assert parse_entry("a-model: does-a-thing") == Entry("a-model", "does-a-thing")


# -- half-up rounding ---------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [(0.49, "low"), (0.5, "medium"), (1.49, "medium"), (1.5, "high"), (2.0, "high"), (2.5, None), (-0.6, None)],
)
def test_effort_score_rounds_half_up(score, expected):
    assert normalise_effort_choice(score) == expected
