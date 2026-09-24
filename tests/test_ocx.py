"""Routing on a single OpenAI-compatible endpoint (OpenCodex / ocx): ``provider/model`` and
``combo/<id>`` ids, a configurable routed provider, and the same fail-open contract."""

from __future__ import annotations

from catalog import ModelCatalog
from config import load_settings
from effort import family_for, resolve_effort
from grid import Entry, parse_entry
from stubs import ReadTimeout, StubResponse, StubTransport, decision_payload, ollama_request
from test_router import build, route

OCX_GRID = [
    "deepseek/deepseek-v4.1-flash: general work",
    "anthropic/claude-sonnet-4.5: complex code",
    "combo/fast: trivial requests",
]
OCX = {"routed_providers": ["ocx"], "grid": OCX_GRID, "default_model": "deepseek/deepseek-v4.1-flash"}
BASE_URL = "https://ocx.example.invalid/v1"


def ocx_route(router, request=None, **overrides):
    kwargs = dict(provider="ocx", base_url=BASE_URL, model="deepseek/deepseek-v4.1-flash")
    kwargs.update(overrides)
    if request is None:
        request = ollama_request(model=kwargs["model"])
        request["reasoning_effort"] = "medium"
    return route(router, request, **kwargs)


# -- ids and effort families ---------------------------------------------------------


def test_effort_family_strips_the_provider_prefix():
    assert family_for("deepseek/deepseek-v4.1-flash").name == "deepseek"
    assert family_for("moonshot/kimi-k3").name == "kimi-k3"
    assert family_for("anthropic/claude-sonnet-4.5") is None
    assert family_for("combo/fast") is None
    assert family_for("combo/deepseek-v4.1-flash") is None  # a combo can be any model
    assert family_for("mystery-model").name == "ollama-cloud"  # bare ids keep the old contract


def test_unknown_family_effort_modes():
    assert resolve_effort("combo/fast", "high") is None
    assert resolve_effort("combo/fast", "high", "omit") is None
    assert resolve_effort("anthropic/claude-sonnet-4.5", "High", "pass") == "high"
    assert resolve_effort("deepseek/deepseek-v4.1-flash", "high", "pass") == "high"


def test_grid_entries_accept_prefixed_and_combo_ids():
    assert parse_entry("anthropic/claude-sonnet-4.5: complex code") == Entry("anthropic/claude-sonnet-4.5", "complex code")
    assert parse_entry("combo/fast: trivial") == Entry("combo/fast", "trivial")
    assert parse_entry("nemotron-3-nano:30b: tiny") == Entry("nemotron-3-nano:30b", "tiny")


# -- which requests are routed -------------------------------------------------------


def test_routed_providers_and_base_urls_are_configurable():
    default = load_settings()
    assert default.routes("ollama-cloud") and not default.routes("ocx", BASE_URL)

    by_name = load_settings(lambda key, d=None: {"routed_providers": "ocx, other"}.get(key, d))
    assert by_name.routes("OCX") and by_name.routes("other") and not by_name.routes("ollama-cloud")

    by_url = load_settings(lambda key, d=None: {"routed_base_urls": [BASE_URL + "/"]}.get(key, d))
    assert by_url.routes("custom:ocx", BASE_URL) and not by_url.routes("custom:ocx", "https://elsewhere.invalid/v1")


def test_default_config_leaves_an_ocx_request_untouched(tmp_path):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport)

    assert ocx_route(router, model="deepseek-v4.1-flash") is None
    assert transport.call_count == 0


def test_routes_to_a_prefixed_model_and_replays_it_for_the_turn(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("1", 0.9, "high", 0.8, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX)

    first = ocx_route(router)
    second = ocx_route(router, api_call_count=2, api_request_id="turn-1:api:2")

    assert first["request"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert first["request"]["reasoning_effort"] == "high"  # known family behind the prefix
    assert second["request"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert transport.call_count == 1


def test_matches_by_base_url_when_the_provider_name_is_deployment_specific(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "high", 0.8, grid_size=3))])
    config = {**OCX, "routed_providers": [], "routed_base_urls": [BASE_URL]}
    router, _ = build(tmp_path, transport, config=config)

    result = ocx_route(router, provider="custom:my-ocx")

    assert result["request"]["model"] == "anthropic/claude-sonnet-4.5"


def test_unknown_family_keeps_the_hosts_effort_by_default(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "high", 0.8, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX)

    result = ocx_route(router)

    assert result["request"]["model"] == "anthropic/claude-sonnet-4.5"
    assert result["request"]["reasoning_effort"] == "medium"  # untouched


def test_combo_effort_follows_unknown_effort(tmp_path):
    for mode, expected in (("omit", None), ("pass", "high")):
        transport = StubTransport([StubResponse(decision_payload("3", 0.9, "high", 0.8, grid_size=3))])
        router, _ = build(tmp_path / mode, transport, config={**OCX, "unknown_effort": mode})

        result = ocx_route(router)

        assert result["request"]["model"] == "combo/fast"
        assert result["request"].get("reasoning_effort") == expected


def test_configured_model_outside_the_grid_is_skipped(tmp_path):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router, model="deepseek-v4.1-flash") is None  # bare id, grid is prefixed
    assert transport.call_count == 0


def test_ollama_catalog_does_not_gate_ocx(tmp_path):
    cache = tmp_path / "ollama_cloud_models_cache.json"
    cache.write_text('{"models": ["deepseek-v4.1-flash"]}', encoding="utf-8")
    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "high", 0.8, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX, catalog=ModelCatalog(cache))

    result = ocx_route(router)

    assert result["request"]["model"] == "anthropic/claude-sonnet-4.5"
    assert router.grid_report(load_settings(lambda key, d=None: OCX.get(key, d))) == []


# -- fail-open -----------------------------------------------------------------------


def test_fails_open_on_a_transport_error(tmp_path):
    transport = StubTransport([ReadTimeout("slow")])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router) is None


def test_low_confidence_with_a_fallback_outside_the_grid_keeps_the_configured_model(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("2", 0.1, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config={**OCX, "default_model": "deepseek-v4.1-flash"})

    assert ocx_route(router) is None
