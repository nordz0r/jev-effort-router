"""Test doubles: a stub Decisions API, a fake ``register(ctx)`` surface, payload builders.

No test touches the network. ``StubTransport`` mimics the shape ``httpx.Client.post`` returns for the
fields the client reads (``status_code``, ``json()``, ``text``), and ``StubContext`` records what the
plugin registers so the wiring can be asserted without a Hermes runtime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class StubResponse:
    def __init__(self, payload: Any = None, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers: Dict[str, str] = {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class StubTransport:
    """Records every request and answers with a queued or callable response."""

    def __init__(self, responses: Optional[List[Any]] = None, handler=None) -> None:
        self.responses = list(responses or [])
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, *, json: Any = None, headers: Any = None, timeout: Any = None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.handler is not None:
            result = self.handler(url, json, headers, timeout)
            if isinstance(result, BaseException):
                raise result
            return result
        if self.responses:
            result = self.responses.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        raise AssertionError("StubTransport received an unexpected request")

    @property
    def last_payload(self) -> Dict[str, Any]:
        return self.calls[-1]["json"]

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ReadTimeout(Exception):
    """Named like an httpx timeout so the client's timeout branch is exercised."""


def _distribution(winner: str, confidence: float, grid_size: int) -> Dict[str, float]:
    """A probability map over ``grid_size`` criteria with ``winner`` on top."""
    keys = [str(index) for index in range(1, max(1, grid_size) + 1)]
    if winner not in keys:
        return {winner: confidence}
    others = [key for key in keys if key != winner]
    share = ((1.0 - confidence) / len(others)) if others else 0.0
    return {key: (confidence if key == winner else share) for key in keys}


def decision_payload(
    model_choice: str = "1",
    model_confidence: float = 0.87,
    effort_choice: str = "medium",
    effort_confidence: float = 0.79,
    *,
    model_probabilities: Optional[Dict[str, float]] = None,
    effort_probabilities: Optional[Dict[str, float]] = None,
    include_effort: bool = True,
    model_type: str = "choice",
    effort_type: str = "score",
    grid_size: int = 6,
) -> Dict[str, Any]:
    """A well-formed Decisions API body, as documented in ``docs/jev-decisions-api.md``.

    The endpoint returns the *full* distribution over the criteria it was sent, so the stub does too:
    the winning key carries ``model_confidence`` and the rest share the remainder.
    """
    if model_probabilities is None:
        model_probabilities = _distribution(model_choice, model_confidence, grid_size)
    answers: Dict[str, Any] = {
        "model_route": {
            "type": model_type,
            "choice": model_choice,
            "probabilities": model_probabilities,
            "confidence": model_confidence,
        }
    }
    if include_effort:
        # The effort question is a Score over low/medium/high: the answer carries the
        # probability-weighted level index, rounded by the client.
        levels = ["low", "medium", "high"]
        score = levels.index(effort_choice) if effort_choice in levels else effort_choice
        answers["reasoning_effort"] = {
            "type": effort_type,
            "score": score,
            "legend": {str(index): level for index, level in enumerate(levels)},
            "probabilities": effort_probabilities or {str(score): effort_confidence},
            "confidence": effort_confidence,
        }
    return {"answers": answers}


def ollama_request(
    *,
    model: str = "deepseek-v4.1-flash",
    messages: Optional[List[Dict[str, Any]]] = None,
    reasoning_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A provider kwargs payload as the ``llm_request`` middleware would receive it."""
    request: Dict[str, Any] = {
        "model": model,
        "messages": messages if messages is not None else [{"role": "user", "content": "hello"}],
        "tools": [],
        "max_tokens": 4096,
        "timeout": 60,
    }
    if reasoning_config is not None:
        request["reasoning_config"] = reasoning_config
    return request


class Registration:
    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name


class Args:
    """A stand-in for the parsed argparse namespace the CLI handler receives."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class StubState:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.values: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


class StubContext:
    """The subset of ``PluginContext`` the plugin uses, with a recording surface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, state: Any = None) -> None:
        self._config = dict(config or {})
        self.state = state
        self.middleware: Dict[str, Any] = {}
        self.hooks: Dict[str, Any] = {}
        self.tools: Dict[str, Any] = {}
        self.cli_commands: Dict[str, Any] = {}
        self.slash_commands: Dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        self._config[key] = value

    def register_middleware(self, kind: str, callback):
        self.middleware[kind] = callback
        return Registration("middleware", kind)

    def register_hook(self, name: str, callback):
        self.hooks.setdefault(name, []).append(callback)
        return Registration("hook", name)

    def register_tool(self, *, name: str, toolset: str, schema: Any, handler):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}
        return Registration("tool", name)

    def register_cli_command(self, *, name: str, help: str, setup_fn, handler_fn):  # noqa: A002
        self.cli_commands[name] = {"help": help, "setup_fn": setup_fn, "handler_fn": handler_fn}
        return Registration("cli", name)

    def register_command(self, name: str, *, handler, description: str = "", args_hint: str = ""):
        self.slash_commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}
        return Registration("command", name)
