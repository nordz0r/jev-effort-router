"""Routing on one OpenAI-compatible endpoint (OpenCodex / ocx) reached as a Hermes
``custom_providers`` entry: ``provider/model`` and ``combo/<id>`` ids, provider matching,
effort handling off Ollama, the fallback rules, shadow mode, and fail-open."""

from __future__ import annotations

import json

import pytest

import router as router_module
from catalog import ModelCatalog
from config import load_settings, norm_provider
from effort import family_for, resolve_effort
from grid import Entry, parse_entry, parse_grid
from stubs import ReadTimeout, StubResponse, StubTransport, decision_payload, ollama_request
from test_router import build, route

# Least capable first: the grid order is the tier order used below threshold.
OCX_GRID = [
    "combo/fast: trivial request answered in one short step",
    "gldf-flash: bounded task needing some judgment",
    "xai/grok-4.7: hard or open-ended task where a mistake is costly",
]
OCX = {"routed_providers": ["custom:ocx"], "grid": OCX_GRID}
OCX_URL = "https://ocx.example.invalid/v1"
ZAI_URL = "https://zai.example.invalid/v1"


@pytest.fixture(autouse=True)
def _custom_providers(monkeypatch):
    """Hermes passes provider="custom" for every custom_providers entry; the router recovers
    the entry name from base_url through the host config (stubbed here)."""
    names = {OCX_URL: "ocx", ZAI_URL: "zai"}
    monkeypatch.setattr(router_module, "custom_provider_name", lambda url: names.get(url, ""))


def ocx_route(router, request=None, **overrides):
    kwargs = dict(provider="custom", base_url=OCX_URL, model="gldf-flash")
    kwargs.update(overrides)
    if request is None:
        request = ollama_request(model=kwargs["model"])
        request["reasoning_effort"] = "medium"
    return route(router, request, **kwargs)


def records(tmp_path):
    path = tmp_path / "routes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# -- ids -----------------------------------------------------------------------------


def test_grid_ids_keep_colons_prefixes_and_case():
    assert parse_entry("openrouter/qwen/qwen3-coder:free") == Entry("openrouter/qwen/qwen3-coder:free", "")
    assert parse_entry("nemotron-3-nano:30b: tiny") == Entry("nemotron-3-nano:30b", "tiny")
    assert parse_entry("a-model:does a thing") == Entry("a-model", "does a thing")
    assert parse_entry("openrouter/moonshotai/kimi-k3: code") == Entry("openrouter/moonshotai/kimi-k3", "code")
    assert parse_entry("Combo/Fast: trivial") == Entry("Combo/Fast", "trivial")


def test_one_key_yaml_mappings_are_entries():
    assert parse_entry({"combo/fast": "trivial"}) == Entry("combo/fast", "trivial")
    assert parse_grid([{"combo/fast": "trivial"}, {"xai/grok-4.7": "hard"}])[1].model_id == "xai/grok-4.7"


@pytest.mark.parametrize("raw", ["combo/", "moonshot/", "/x", "a//b", {"combo/": "x"}, ": desc"])
def test_ids_with_an_empty_segment_are_rejected(raw):
    assert parse_entry(raw) is None


# -- effort --------------------------------------------------------------------------


def test_effort_families_apply_only_on_ollama():
    assert family_for("moonshot/kimi-k3").name == "kimi-k3"
    assert family_for("combo/fast") is None
    assert family_for("Combo/Fast") is None
    # Off Ollama the table is skipped: kimi's medium->high override does not apply.
    assert resolve_effort("moonshot/kimi-k3", "medium", "pass", families=False) == "medium"
    assert resolve_effort("moonshot/kimi-k3", "medium", "omit", families=False) is None
    assert resolve_effort("moonshot/kimi-k3", "medium", "pass", families=True) == "high"


@pytest.mark.parametrize(
    "mode, host, jev_effort, expected",
    [
        ("omit", "medium", True, None),        # default: the host's value is dropped
        ("keep", "medium", True, "medium"),    # plain level kept
        ("keep", "xhigh", True, None),         # host ladder value never forwarded
        ("pass", "medium", True, "high"),      # Jev's level
        ("pass", "medium", False, None),       # no Jev level: host value dropped, not kept
        ("bogus", "medium", True, None),       # invalid setting -> omit
    ],
)
def test_unknown_effort_modes_on_ocx(tmp_path, mode, host, jev_effort, expected):
    payload = decision_payload("3", 0.9, "high", 0.9, grid_size=3, include_effort=jev_effort)
    router, _ = build(tmp_path, StubTransport([StubResponse(payload)]), config={**OCX, "unknown_effort": mode})
    request = ollama_request(model="gldf-flash")
    request["reasoning_effort"] = host

    result = ocx_route(router, request)

    assert result["request"]["model"] == "xai/grok-4.7"
    assert result["request"].get("reasoning_effort") == expected


# -- which requests are routed -------------------------------------------------------


def test_custom_prefix_is_ignored_on_both_sides():
    assert norm_provider("custom:ocx") == norm_provider("ocx") == "ocx"
    assert load_settings(lambda k, d=None: {"routed_providers": ["ocx"]}.get(k, d)).match("custom:ocx")
    assert load_settings(lambda k, d=None: {"routed_providers": ["custom:ocx"]}.get(k, d)).match("ocx")
    assert not load_settings(lambda k, d=None: {"routed_providers": ["custom:ocx"]}.get(k, d)).match("custom")


def test_explicit_empty_routed_providers_means_none():
    settings = load_settings(lambda k, d=None: {"routed_providers": []}.get(k, d))
    assert settings.match("ollama-cloud") is None
    assert load_settings().match("ollama-cloud") == "provider"


def test_base_url_matches_only_at_a_boundary():
    settings = load_settings(lambda k, d=None: {"routed_providers": [], "routed_base_urls": [OCX_URL]}.get(k, d))
    assert settings.match("custom", OCX_URL + "/") == "base_url"
    assert settings.match("custom", OCX_URL + "/chat") == "base_url"
    assert settings.match("custom", "https://ocx.example.invalid/v1evil") is None
    assert settings.match("custom", "https://ocx.example.invalid.evil.com/v1") is None
    assert settings.match("custom", "https://ocx.example.invalid:8443/v1") is None


def test_empty_provider_is_never_routed(tmp_path):
    cache = tmp_path / "ollama_cloud_models_cache.json"
    cache.write_text('{"models": ["deepseek-v4.1-flash"]}', encoding="utf-8")
    transport = StubTransport([])
    config = {**OCX, "routed_base_urls": [OCX_URL]}
    router, _ = build(tmp_path, transport, config=config, catalog=ModelCatalog(cache))

    assert ocx_route(router, provider="") is None
    assert transport.call_count == 0


def test_custom_ocx_is_routed_and_custom_zai_is_untouched(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router)["request"]["model"] == "xai/grok-4.7"
    assert ocx_route(router, base_url=ZAI_URL, turn_id="turn-2") is None
    assert ocx_route(router, provider="custom:zai", turn_id="turn-3") is None
    assert transport.call_count == 1


def test_base_url_match_skips_the_ollama_catalog(tmp_path):
    cache = tmp_path / "ollama_cloud_models_cache.json"
    cache.write_text('{"models": ["deepseek-v4.1-flash"]}', encoding="utf-8")
    transport = StubTransport([StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3))])
    config = {**OCX, "routed_providers": [], "routed_base_urls": [OCX_URL]}
    router, settings = build(tmp_path, transport, config=config, catalog=ModelCatalog(cache))

    result = ocx_route(router, base_url=OCX_URL, provider="custom:unnamed")

    assert result["request"]["model"] == "xai/grok-4.7"
    assert router.grid_report(settings) == []


def test_configured_model_must_match_a_grid_id_exactly(tmp_path):
    transport = StubTransport([])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router, model="GLDF-flash") is None
    assert ocx_route(router, model="ocx/gldf-flash") is None
    assert transport.call_count == 0


# -- decisions and fallbacks ---------------------------------------------------------


def test_routes_and_replays_for_the_turn(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX)

    first = ocx_route(router)
    second = ocx_route(router, api_call_count=2, api_request_id="turn-1:api:2")

    assert first["request"]["model"] == second["request"]["model"] == "xai/grok-4.7"
    assert transport.call_count == 1
    assert [r["replayed"] for r in records(tmp_path)] == [False, True]


def test_below_threshold_takes_the_more_capable_of_choice_and_default(tmp_path):
    # choice grok (position 3) at 0.3 vs default gldf-flash (position 2): grok stays.
    transport = StubTransport([StubResponse(decision_payload("3", 0.3, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config={**OCX, "default_model": "gldf-flash"})
    assert ocx_route(router)["request"]["model"] == "xai/grok-4.7"

    # choice combo/fast (position 1) at 0.3 vs default gldf-flash: never downgrade -> gldf-flash.
    transport = StubTransport([StubResponse(decision_payload("1", 0.3, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path / "b", transport, config={**OCX, "default_model": "gldf-flash"})
    assert ocx_route(router, model="xai/grok-4.7")["request"]["model"] == "gldf-flash"


def test_below_threshold_without_default_keeps_the_configured_model(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("1", 0.3, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router) is None


COMBO_GRID = OCX_GRID + ["combo/glm-grok-failover: default: long agentic work"]


@pytest.mark.parametrize(
    "payload",
    [
        decision_payload("1", 0.2, "high", 0.9, grid_size=4),       # below threshold -> max(choice, default)
        decision_payload("unclear", 0.9, "high", 0.9, grid_size=4),  # unclear -> default
        None,                                                        # Jev unavailable -> default
    ],
)
def test_combo_default_model_serves_every_fallback_path(tmp_path, payload):
    response = StubResponse(payload) if payload is not None else ReadTimeout("slow")
    config = {**OCX, "grid": COMBO_GRID, "default_model": "combo/glm-grok-failover", "unknown_effort": "pass"}
    router, _ = build(tmp_path, StubTransport([response]), config=config)

    result = ocx_route(router)

    assert result["request"]["model"] == "combo/glm-grok-failover"
    # Combo: no effort family; `pass` sends Jev's level, or default_effort when Jev was down.
    assert result["request"]["reasoning_effort"] == ("high" if payload is not None else "medium")


# -- shadow --------------------------------------------------------------------------


def test_shadow_decides_and_records_but_never_rewrites(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config={**OCX, "mode": "shadow"})

    assert ocx_route(router) is None
    assert ocx_route(router, api_call_count=2) is None
    assert ocx_route(router, base_url=ZAI_URL, turn_id="turn-2") is None  # gates still apply

    shadow = [r for r in records(tmp_path) if r["event"] == "shadow"]
    assert transport.call_count == 1
    assert [r["replayed"] for r in shadow] == [False, True]
    assert shadow[0]["model"] == "xai/grok-4.7" and shadow[0]["configured_model"] == "gldf-flash"
    assert shadow[0]["model_confidence"] == 0.9
    assert any(r["event"] == "skip" and r["reason"] == "provider_not_routed" for r in records(tmp_path))


def test_shadow_records_the_fallback_reason(tmp_path):
    router, _ = build(tmp_path, StubTransport([ReadTimeout("slow")]),
                      config={**OCX, "mode": "shadow", "default_model": "gldf-flash"})

    assert ocx_route(router) is None
    assert records(tmp_path)[-1]["fallback_reasons"] == ["timeout"]


# -- fail-open -----------------------------------------------------------------------


def test_transport_error_without_default_leaves_the_request_untouched(tmp_path):
    router, _ = build(tmp_path, StubTransport([ReadTimeout("slow")]), config=OCX)

    assert ocx_route(router) is None


def test_a_hostile_provider_value_never_raises(tmp_path):
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")

    router, _ = build(tmp_path, StubTransport([]), config=OCX)

    assert ocx_route(router, provider=Hostile()) is None


def test_readme_nord_example_is_a_valid_configuration():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = text[text.index("plugins:\n  enabled"):]
    block = block[: block.index("```")]
    settings_map = yaml.safe_load(block)["plugins"]["entries"]["jev-effort-router"]["settings"]
    settings = load_settings(lambda key, default=None: settings_map.get(key, default))

    assert settings.mode == "shadow" and settings.backend == "typesafe"
    assert settings.api_key_env == "TYPESAFE_API_KEY" and settings.jev_model == "jev-1.13.0"
    assert [e.model_id for e in settings.grid] == ["combo/fast", "gldf-flash", "combo/glm-grok-failover"]
    assert settings.entry_for(settings.default_model) is not None
    assert settings.match("custom:ocx") and not settings.match("custom:zai")


def test_a_failed_turn_is_not_retried_on_every_tool_call(tmp_path):
    transport = StubTransport([ReadTimeout("slow")])
    router, _ = build(tmp_path, transport, config=OCX)

    assert ocx_route(router) is None
    assert ocx_route(router, api_call_count=2) is None
    assert transport.call_count == 1
    assert records(tmp_path)[-1]["reason"] == "replayed_no_decision"

    # The next turn asks again.
    transport.responses.append(StubResponse(decision_payload("3", 0.9, "high", 0.9, grid_size=3)))
    assert ocx_route(router, turn_id="turn-2")["request"]["model"] == "xai/grok-4.7"
