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


COMBO_GRID = OCX_GRID + ["gldf-hermes: default: long agentic work"]


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
    config = {**OCX, "grid": COMBO_GRID, "default_model": "gldf-hermes", "unknown_effort": "pass"}
    router, _ = build(tmp_path, StubTransport([response]), config=config)

    result = ocx_route(router)

    assert result["request"]["model"] == "gldf-hermes"
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
    assert settings.default_model == "zai/glm-5.3" and settings.confidence_threshold == 0.5
    assert [e.model_id for e in settings.grid] == [
        "google-antigravity/gemini-3.8-flash",
        "zai/glm-5.3-flash",
        "zai/glm-5.3",
        "gpt-6-sol",
        "gpt-6-astra",
    ]
    flash = settings.entry_for("google-antigravity/gemini-3.8-flash")
    assert flash.context == 1_048_576
    assert flash.efforts == ("low", "medium", "high")
    assert "trivial request answered in one short step" in flash.description
    assert settings.entry_for("zai/glm-5.3-flash").context == 1_000_000
    assert settings.entry_for("zai/glm-5.3-flash").efforts == ("low", "high", "max")
    assert "short bounded task still needing a little judgment" in settings.entry_for("zai/glm-5.3-flash").description
    assert settings.entry_for("zai/glm-5.3").context == 1_000_000
    assert settings.entry_for("zai/glm-5.3").efforts == ("low", "high", "max")
    assert settings.entry_for("gpt-6-sol").context == 272000
    assert settings.entry_for("gpt-6-sol").efforts == ("low", "medium", "high", "xhigh", "max")
    assert settings.entry_for("gpt-6-astra").context == 272000
    long_ids = [e.model_id for e in settings.long_context_models]
    assert long_ids == [
        "google-antigravity/gemini-3.1-pro",
        "zai/glm-5.3",
        "google-antigravity/gemini-3.8-flash",
    ]
    assert settings.long_context_models[0].context == 1_048_576
    assert settings.long_context_models[0].efforts == ("low", "high")
    assert settings.long_context_models[1].efforts == ("low", "high", "max")
    assert settings.long_context_models[2].efforts == ("low", "medium", "high")
    assert settings.context_reserve_tokens == 32000
    assert "gldf-hermes" not in [e.model_id for e in settings.grid]
    assert "xai/grok-4.7" not in [e.model_id for e in settings.grid]
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


# -- per-model effort levels ---------------------------------------------------------

GLM = {"id": "zai/glm-5.3", "description": "hard task", "efforts": ["low", "high", "max", "ultra"], "context": 1000000}


def test_map_form_entries_carry_efforts_and_context():
    entry = parse_entry(GLM)
    assert entry.efforts == ("low", "high", "max", "ultra") and entry.context == 1_000_000
    nested = parse_entry({"zai/glm-5.3": {"description": "hard", "efforts": "high, low", "context": "1e6"}})
    assert nested.efforts == ("low", "high") and nested.context == 1_000_000
    # String form keeps working and declares nothing.
    assert parse_entry("zai/glm-5.3: hard") == Entry("zai/glm-5.3", "hard")
    assert parse_entry({"id": "x", "efforts": ["bogus"], "context": -1}) == Entry("x", "")


@pytest.mark.parametrize(
    "requested, supported, expected",
    [
        ("medium", ["low", "high", "max", "ultra"], "high"),  # glm: medium rounds UP to high
        ("low", ["low", "high", "max", "ultra"], "low"),
        ("high", ["low", "medium"], "medium"),                # nothing stronger: strongest supported
        ("medium", ["none", "minimal"], "minimal"),           # never lands on none
        (None, ["low", "high"], None),                        # no level requested: field omitted
    ],
)
def test_round_up_onto_declared_levels(requested, supported, expected):
    from effort import round_up

    assert round_up(requested, supported) == expected


def _glm_router(tmp_path, payload, **config):
    grid = ["gldf-flash: bounded task", GLM]
    return build(tmp_path, StubTransport([StubResponse(payload)]), config={**OCX, "grid": grid, **config})


def test_declared_efforts_round_the_decided_level_up(tmp_path):
    router, _ = _glm_router(tmp_path, decision_payload("2", 0.9, "medium", 0.9, grid_size=2))

    result = ocx_route(router)

    assert result["request"]["model"] == "zai/glm-5.3"
    assert result["request"]["reasoning_effort"] == "high"


def test_declared_efforts_win_over_keep(tmp_path):
    router, _ = _glm_router(tmp_path, decision_payload("2", 0.9, "low", 0.9, grid_size=2), unknown_effort="keep")
    request = ollama_request(model="gldf-flash")
    request["reasoning_effort"] = "medium"

    assert ocx_route(router, request)["request"]["reasoning_effort"] == "low"


def test_absent_efforts_keep_the_unknown_effort_rules(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("3", 0.9, "medium", 0.9, grid_size=3))]),
                      config=OCX)
    result = ocx_route(router)
    assert result["request"]["model"] == "xai/grok-4.7"
    assert "reasoning_effort" not in result["request"]  # omit (default)


def test_fallback_default_effort_is_rounded_up_too(tmp_path):
    grid = ["gldf-flash: bounded task", GLM]
    router, _ = build(tmp_path, StubTransport([ReadTimeout("slow")]),
                      config={**OCX, "grid": grid, "default_model": "zai/glm-5.3", "default_effort": "medium"})

    result = ocx_route(router)

    assert result["request"]["model"] == "zai/glm-5.3" and result["request"]["reasoning_effort"] == "high"


# -- context fit -----------------------------------------------------------------------

FIT_GRID = [
    {"id": "small", "description": "trivial", "context": 40000, "efforts": ["low", "high"]},
    {"id": "mid", "description": "bounded", "context": 100000},
    {"id": "big", "description": "hard", "context": 300000, "efforts": ["low", "high", "max"]},
]
FIT = {"routed_providers": ["custom:ocx"], "grid": FIT_GRID}


def big_request(tokens, model="small"):
    request = ollama_request(model=model)
    request["messages"] = [{"role": "user", "content": "x" * (tokens * 4)}]
    return request


def test_estimate_is_chars_over_four_across_messages_and_tools():
    from fit import estimate_tokens

    request = {"messages": [{"role": "user", "content": "a" * 400}], "tools": [{"name": "t"}]}
    import json as _json
    chars = len(_json.dumps(request["messages"])) + len(_json.dumps(request["tools"]))
    assert estimate_tokens(request) == -(-chars // 4)


def test_estimate_does_not_count_base64_image_payloads_as_text():
    from fit import IMAGE_TOKEN_COST, estimate_tokens

    huge = "A" * 1_000_000  # ~1MB base64-ish payload
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
                ],
            }
        ]
    }
    estimate = estimate_tokens(request)
    # Must stay near text size + one fixed image cost — never ~250k+ from the base64.
    assert estimate < 5_000, estimate
    assert estimate >= IMAGE_TOKEN_COST
    text_only = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "what is in this image?"}]}]
    }
    # Scrubbed stub adds a few dozen chars; base64 must not dominate.
    assert estimate - estimate_tokens(text_only) < IMAGE_TOKEN_COST + 100
    bare = {"messages": [{"role": "user", "content": "x" + huge}]}
    assert estimate_tokens(bare) > 200_000


def test_a_model_that_does_not_fit_escalates_up_the_grid(tmp_path):
    # Jev picks "small" (40k); a ~20k prompt + 32k reserve does not fit it -> "mid" (100k).
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config={**FIT, "unknown_effort": "pass"})

    result = ocx_route(router, big_request(20000), model="small")

    assert result["request"]["model"] == "mid"
    assert result["request"]["reasoning_effort"] == "medium"  # re-derived for mid (no efforts: pass)
    record = records(tmp_path)[-1]
    assert record["fallback_reasons"][-1] == "context_escalated"
    assert record["context_from"] == "small" and record["context_estimate"] > 20000


def test_escalation_skips_tiers_that_still_do_not_fit(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config=FIT)

    result = ocx_route(router, big_request(150000), model="small")

    assert result["request"]["model"] == "big"
    assert result["request"]["reasoning_effort"] == "high"  # big declares efforts: medium -> high


def test_long_context_models_when_no_grid_tier_fits(tmp_path):
    config = {**FIT, "long_context_models": [{"id": "zai/glm-5.3", "context": 1000000, "efforts": ["low", "high", "max", "ultra"]}]}
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("3", 0.9, "medium", 0.9, grid_size=3))]),
                      config=config)

    result = ocx_route(router, big_request(400000), model="small")

    assert result["request"]["model"] == "zai/glm-5.3"
    assert result["request"]["reasoning_effort"] == "high"
    assert records(tmp_path)[-1]["fallback_reasons"][-1] == "context_long_model"


def test_long_context_ids_take_their_window_from_the_grid(tmp_path):
    router, settings = build(tmp_path, StubTransport([]), config={**FIT, "long_context_models": ["big"]})
    assert settings.long_context_entry(settings.long_context_models[0]).context == 300000


def test_nothing_fits_leaves_the_request_untouched(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config={**FIT, "long_context_models": [{"id": "zai/glm-5.3", "context": 200000}]})

    assert ocx_route(router, big_request(400000), model="small") is None
    record = records(tmp_path)[-1]
    assert record["event"] == "skip" and record["reason"] == "context_no_fit"
    assert record["context_estimate"] > 400000 and record["chosen_model"] == "small"


def test_models_without_a_window_always_fit(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("3", 0.9, "medium", 0.9, grid_size=3))]),
                      config=OCX)
    assert ocx_route(router, big_request(5_000_000, model="gldf-flash"))["request"]["model"] == "xai/grok-4.7"


def test_reserve_is_configurable(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config={**FIT, "context_reserve_tokens": 0})
    assert ocx_route(router, big_request(30000), model="small")["request"]["model"] == "small"


def test_fallback_path_is_context_checked(tmp_path):
    router, _ = build(tmp_path, StubTransport([ReadTimeout("slow")]), config={**FIT, "default_model": "small"})

    result = ocx_route(router, big_request(20000), model="small")

    assert result["request"]["model"] == "mid"
    assert records(tmp_path)[-1]["fallback_reasons"] == ["timeout", "context_escalated"]


def test_context_is_rechecked_on_replay_as_the_prompt_grows(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))])
    router, _ = build(tmp_path, transport, config=FIT)

    assert ocx_route(router, big_request(10), model="small")["request"]["model"] == "small"
    assert ocx_route(router, big_request(20000), model="small", api_call_count=2)["request"]["model"] == "mid"
    assert transport.call_count == 1


def test_shadow_audits_the_escalation_without_rewriting(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config={**FIT, "mode": "shadow"})

    assert ocx_route(router, big_request(20000), model="small") is None
    record = records(tmp_path)[-1]
    assert record["event"] == "shadow" and record["model"] == "mid"
    assert record["context_from"] == "small" and "context_escalated" in record["fallback_reasons"]


def test_shadow_audits_no_fit(tmp_path):
    router, _ = build(tmp_path, StubTransport([StubResponse(decision_payload("1", 0.9, "medium", 0.9, grid_size=3))]),
                      config={**FIT, "mode": "shadow"})

    assert ocx_route(router, big_request(400000), model="small") is None
    record = records(tmp_path)[-1]
    assert record["reason"] == "context_no_fit" and record["mode"] == "shadow"
