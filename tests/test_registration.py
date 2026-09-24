"""The plugin's `register(ctx)` surface, loaded from the repository root.

Hermes loads a directory plugin as the package ``hermes_plugins.<slug>`` and calls the
package's ``register(ctx)``. The plugin body must not import anything from this test tree, so
these tests import the plugin's sibling modules (``config``, ``grid``, ...) through the same
path insertion the plugin relies on: a directory plugin's modules are top-level siblings.

For the tests, the simplest faithful arrangement is to import each module by path with the
package prefix Hermes uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Import the plugin the way Hermes' loader does: a package rooted at the repo directory,
#: under ``hermes_plugins.<slug>``. `sys.path` gets the repo root so the plugin's own
#: ``from .config import ...`` relatives resolve inside the package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_NAME = "hermes_plugins_jev_effort_router_registration_test"


def _load_plugin():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def plugin():
    yield _load_plugin()


class Args:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_register_wires_the_declared_surface(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))

    plugin.register(ctx)

    assert "llm_request" in ctx.middleware
    assert set(ctx.tools) == {"jev_effort_router_status", "jev_effort_router_route"}
    assert all(entry["toolset"] == "jev-effort-router" for entry in ctx.tools.values())
    assert set(ctx.hooks) == {"on_session_end", "post_llm_call"}
    assert "jev-effort-router" in ctx.cli_commands
    assert "jev-effort-router" in ctx.slash_commands


def test_manifest_declares_exactly_what_is_registered(plugin, tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    ctx = StubContext(config={}, state=StubState(tmp_path))

    plugin.register(ctx)

    # Middleware IS declared, deliberately, even though the host's manifest schema has no such field
    # (declaring it produces one "unknown manifest field(s) ignored" warning per load). The reason is
    # the admission gate: `hermes plugins validate` fails an entry whose registered middleware is not
    # in `provides_middleware`, so leaving it undeclared means no catalog entry. See `test_manifest_
    # fields_are_known_to_the_host` below for the counterweight it used to enforce.
    assert sorted(manifest["provides_tools"]) == sorted(ctx.tools)
    assert sorted(manifest["provides_hooks"]) == sorted(ctx.hooks)
    assert sorted(manifest["provides_middleware"]) == sorted(ctx.middleware)


def test_manifest_fields_are_known_to_the_host():
    """Every key in plugin.yaml must be one the installed Hermes understands — with one exception.

    An unknown key is not fatal — the host warns and loads anyway.

    ``provides_middleware`` is knowingly outside the host's ``_KNOWN_MANIFEST_FIELDS`` and is
    tolerated here because the catalog admission gate requires it: ``hermes plugins validate``
    diffs the manifest against what ``register(ctx)`` wires and FAILS a plugin that registers
    middleware without declaring it. Declaring it therefore buys admission at the price of one
    load-time warning; not declaring it costs the catalog entry. When upstream adds the key to
    the schema, this exception becomes dead and the test tightens back to an empty set.
    """
    yaml = pytest.importorskip("yaml")
    manifest_module = pytest.importorskip("hermes_cli.plugins_manifest")

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    unknown = sorted(set(manifest) - set(manifest_module._KNOWN_MANIFEST_FIELDS))

    assert unknown == ["provides_middleware"]


def test_register_does_no_network_io(plugin, tmp_path, monkeypatch):
    """`hermes plugins doctor` blocks sockets during register(ctx); registration must be lazy."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import httpx

    def forbidden(*_args, **_kwargs):
        raise AssertionError("registration must not open a socket")

    monkeypatch.setattr(httpx, "Client", forbidden)
    ctx = StubContext(config={}, state=StubState(tmp_path))

    plugin.register(ctx)  # must not raise


def test_registration_without_an_api_key_still_succeeds(plugin, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ctx = StubContext(config={}, state=StubState(tmp_path))

    plugin.register(ctx)

    assert "llm_request" in ctx.middleware


def test_config_schema_matches_the_settings_the_plugin_reads(tmp_path):
    yaml = pytest.importorskip("yaml")
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    declared = set(manifest["config_schema"])

    from config import Settings

    # 0.3: `endpoint` is exposed as a per-backend override (empty = the backend's default).
    assert declared == set(Settings.__dataclass_fields__)


def test_status_tool_reports_the_grid_and_the_audit(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    payload = json.loads(ctx.tools["jev_effort_router_status"]["handler"]({"recent": 3}))

    assert payload["enabled"] is True
    assert payload["api_key_present"] is True
    assert len(payload["grid"]) == 6
    assert payload["grid"][0]["model"] == "deepseek-v4.1-flash"
    assert payload["audit"]["records"] == 0


class _FakeEntry:
    """Enough of ``tools.registry.ToolEntry`` for a host-shaped dispatch."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.is_async = False


def _dispatch_like_the_host(handler, args):
    """Call a tool handler the way ``tools/registry.py::dispatch`` does.

    ``kwargs = _kwargs_accepted_by(entry.handler, kwargs)`` filters the injected context down to
    what the signature accepts, then the handler runs as ``handler(args, **kwargs)``. Reproduced
    here without the host so the suite stays runnable outside a Hermes install — the bug this
    covers was a handler signature the no-argument path happened to tolerate.
    """
    import inspect

    context = {
        "task_id": None,
        "session_id": "session-1",
        "user_task": "u",
        "parent_agent": None,
        "turn_id": "turn-1",
    }
    signature = inspect.signature(handler)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        context
        if accepts_kwargs
        else {key: value for key, value in context.items() if key in signature.parameters}
    )
    return handler(args, **kwargs)


@pytest.mark.parametrize("args", [{}, {"recent": 3}, {"recent": None}])
def test_status_tool_accepts_the_arguments_dict_the_host_passes(plugin, tmp_path, monkeypatch, args):
    """The host hands the model's arguments over as one positional dict.

    A handler declaring ``handler(recent=5)`` binds that dict to ``recent`` and raises
    ``TypeError: int() argument must be ... not 'dict'`` — on every call that carried a parameter,
    while the empty-arguments call kept working.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    payload = json.loads(_dispatch_like_the_host(ctx.tools["jev_effort_router_status"]["handler"], args))

    assert payload["enabled"] is True
    assert len(payload["grid"]) == 6


@pytest.mark.parametrize(
    "args",
    [
        {"task": "debug this crash", "context": "a stack trace"},
        {"task": "debug this crash"},
    ],
)
def test_route_tool_accepts_the_arguments_dict_the_host_passes(plugin, tmp_path, monkeypatch, args):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    from client import JevClient
    from stubs import StubResponse, StubTransport, decision_payload

    transport = StubTransport([StubResponse(decision_payload("2", 0.9, "high", 0.9))])
    # The registered middleware is the router's bound method; its ``__self__`` is the router.
    router = ctx.middleware["llm_request"].__self__
    router.client = lambda settings: JevClient(settings, transport=transport)

    payload = json.loads(_dispatch_like_the_host(ctx.tools["jev_effort_router_route"]["handler"], args))

    assert payload["routed"] is True
    assert payload["model"] == "kimi-k3"
    assert payload["effort"] == "high"
    # The task reached Jev; the optional context did not break the call either way.
    assert transport.call_count == 1


def test_route_tool_without_a_task_is_a_clean_error(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    payload = json.loads(_dispatch_like_the_host(ctx.tools["jev_effort_router_route"]["handler"], {}))

    assert payload == {"error": "task must not be empty"}


def test_slash_command_status_grid_and_usage(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    handler = ctx.slash_commands["jev-effort-router"]["handler"]

    assert "jev-effort-router" in handler("")
    assert "Grid" in handler("grid")
    assert "Usage" in handler("nonsense")


def test_cli_status_shows_the_grid_coverage_block(tmp_path, monkeypatch, capsys):
    """The text surface has to carry the finding, not only the tool's JSON.

    `hermes jev-effort-router status` is what an operator actually reads; a report that only
    exists in the agent tool would leave the under-use invisible from the shell.
    """
    from commands import _status_text
    from config import load_settings
    import importlib.util
    import json
    import sys

    name = "jev_effort_router_coverage_cli_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    ctx = StubContext(config={}, state=StubState(tmp_path))
    sys.modules[name].register(ctx)
    router = ctx.middleware["llm_request"].__self__

    path = tmp_path / "routes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"event": "route", "model": "deepseek-v4.1-flash", "replayed": False,
                 "fallback_reasons": [], "model_probabilities": {}},
                {"event": "route", "model": "deepseek-v4.1-flash", "replayed": False,
                 "fallback_reasons": ["low_confidence"], "model_probabilities": {"3": 0.44, "1": 0.30}},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    text = _status_text(router, load_settings(lambda _key, default=None: default))

    assert "Grid coverage (last 2 routed turns):" in text
    assert "deepseek-v4.1-flash x1" in text
    assert "glm-5.3" in text.split("never chosen:")[1].split("\n")[0] + text.split("picked, below threshold:")[1]


def test_cli_command_status_and_missing_task(plugin, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    handler = ctx.cli_commands["jev-effort-router"]["handler_fn"]

    assert handler(Args(jev_effort_router_command="status")) == 0
    assert "jev-effort-router" in capsys.readouterr().out

    assert handler(Args(jev_effort_router_command="route", task=[])) == 2


def test_cli_reset_drops_the_memo(plugin, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    assert ctx.cli_commands["jev-effort-router"]["handler_fn"](Args(jev_effort_router_command="reset")) == 0
    assert "dropped" in capsys.readouterr().out


def test_cli_tail_prints_json_records(plugin, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    ctx = StubContext(config={}, state=StubState(tmp_path))
    plugin.register(ctx)

    handler = ctx.cli_commands["jev-effort-router"]["handler_fn"]
    assert handler(Args(jev_effort_router_command="tail", count=5)) == 0
    assert capsys.readouterr().out.strip() == ""


# -- ctx/state stubs ---------------------------------------------------------------


class Registration:
    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name


class StubState:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class StubContext:
    """The subset of ``PluginContext`` the plugin uses, with a recording surface."""

    def __init__(self, config=None, state=None) -> None:
        self._config = dict(config or {})
        self.state = state
        self.middleware = {}
        self.hooks = {}
        self.tools = {}
        self.cli_commands = {}
        self.slash_commands = {}

    def get_config(self, key, default=None):
        return self._config.get(key, default)

    def set_config(self, key, value):
        self._config[key] = value

    def register_middleware(self, kind, callback):
        self.middleware[kind] = callback
        return Registration("middleware", kind)

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)
        return Registration("hook", name)

    def register_tool(self, *, name, toolset, schema, handler):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}
        return Registration("tool", name)

    def register_cli_command(self, *, name, help, setup_fn, handler_fn):  # noqa: A002
        self.cli_commands[name] = {"help": help, "setup_fn": setup_fn, "handler_fn": handler_fn}
        return Registration("cli", name)

    def register_command(self, name, *, handler, description="", args_hint=""):
        self.slash_commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}
        return Registration("command", name)
