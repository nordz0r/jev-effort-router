"""entry_only: session entry + default_model fallback, omitted from Jev Choice."""

from __future__ import annotations

import pytest

import router as router_module
from client import _more_capable
from config import load_settings
from grid import Entry, choice_grid, criteria, parse_entry, parse_grid
from stubs import ReadTimeout, StubResponse, StubTransport, decision_payload, ollama_request
from test_router import build, route

ENTRY_GRID = [
    {"id": "gldf-hermes", "context": 500000, "entry_only": True},
    {"id": "flash", "description": "trivial", "context": 1000000},
    {"id": "strong", "description": "hard", "context": 272000},
]
ENTRY_CFG = {
    "routed_providers": ["custom:ocx"],
    "grid": ENTRY_GRID,
    "default_model": "gldf-hermes",
}
OCX_URL = "https://ocx.example.invalid/v1"


@pytest.fixture(autouse=True)
def _custom_providers(monkeypatch):
    monkeypatch.setattr(router_module, "custom_provider_name", lambda url: "ocx" if url == OCX_URL else "")


def _route(router, **overrides):
    kwargs = dict(provider="custom", base_url=OCX_URL, model="gldf-hermes")
    kwargs.update(overrides)
    request = ollama_request(model=kwargs["model"])
    request["reasoning_effort"] = "medium"
    return route(router, request, **kwargs)


def test_entry_only_parses_from_yaml_map_and_truthy_strings():
    assert parse_entry({"id": "m", "entry_only": True}).entry_only is True
    assert parse_entry({"id": "m", "entry_only": "true"}).entry_only is True
    assert parse_entry({"id": "m", "entry_only": "yes"}).entry_only is True
    assert parse_entry({"id": "m", "entry_only": "1"}).entry_only is True
    assert parse_entry({"id": "m", "entry_only": False}).entry_only is False
    assert parse_entry({"id": "m", "entry_only": "no"}).entry_only is False
    assert parse_entry({"id": "m"}).entry_only is False
    assert parse_entry("m: desc").entry_only is False

    nested = parse_entry({"gldf-hermes": {"context": 500000, "entry_only": "on"}})
    assert nested.model_id == "gldf-hermes" and nested.entry_only is True and nested.context == 500000

    grid = parse_grid(ENTRY_GRID)
    assert [e.entry_only for e in grid] == [True, False, False]


def test_session_model_equal_default_entry_only_passes_the_gate(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("1", 0.9, "high", 0.9, grid_size=2))])
    router, _ = build(tmp_path, transport, config=ENTRY_CFG)

    result = _route(router)

    assert result is not None
    assert result["request"]["model"] == "flash"
    assert transport.call_count == 1


def test_criteria_and_choice_options_exclude_entry_only():
    grid = parse_grid(ENTRY_GRID)
    chosen = choice_grid(grid)
    assert [e.model_id for e in chosen] == ["flash", "strong"]
    mapping = criteria(chosen)
    assert list(mapping) == ["1", "2"]
    assert mapping["1"] == "trivial" and mapping["2"] == "hard"

    settings = load_settings(lambda key, default=None: ENTRY_CFG.get(key, default))
    assert [e.model_id for e in settings.choice_grid] == ["flash", "strong"]
    assert settings.entry_for("gldf-hermes").entry_only is True


def test_unclear_unavailable_unknown_still_fall_back_to_entry_only_default(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("unclear", 0.9, "high", 0.9, grid_size=2))])
    router, _ = build(tmp_path, transport, config=ENTRY_CFG)
    assert _route(router)["request"]["model"] == "gldf-hermes"

    transport = StubTransport([ReadTimeout("slow")])
    router, _ = build(tmp_path / "b", transport, config=ENTRY_CFG)
    assert _route(router)["request"]["model"] == "gldf-hermes"

    transport = StubTransport([StubResponse(decision_payload("99", 0.9, "high", 0.9, grid_size=2))])
    router, _ = build(tmp_path / "c", transport, config=ENTRY_CFG)
    assert _route(router)["request"]["model"] == "gldf-hermes"


def test_low_conf_does_not_escalate_a_later_choice_up_to_entry_only_hermes(tmp_path):
    transport = StubTransport([StubResponse(decision_payload("2", 0.3, "high", 0.9, grid_size=2))])
    router, _ = build(tmp_path, transport, config=ENTRY_CFG)

    assert _route(router)["request"]["model"] == "strong"

    hermes = Entry("gldf-hermes", "", entry_only=True)
    flash = Entry("flash", "trivial")
    strong = Entry("strong", "hard")
    dest = (flash, strong)
    assert _more_capable(dest, strong, hermes) is strong
    assert _more_capable(dest, flash, hermes) is flash
    assert _more_capable(dest, flash, strong) is strong
