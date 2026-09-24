"""Resolved plugin settings.

Every tunable comes from one place so a callback, a tool handler, a slash command and the
CLI all read the same values. ``ctx.get_config`` already resolves
``plugins.entries.jev-effort-router.settings.<key>`` over the manifest default, so this module
only adds coercion, validation and the "value is missing or nonsense" backstops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .grid import DEFAULT_GRID, Entry, parse_entry, parse_grid

#: Hermes provider names of Ollama:cloud. Routed by default, and the only provider whose model
#: cache (``catalog.py``) and per-family effort table (``effort.py``) apply.
OLLAMA_PROVIDERS: Tuple[str, ...] = ("ollama-cloud", "ollama_cloud")

#: Providers routed when ``routed_providers`` is not configured. Anything else is skipped untouched.
DEFAULT_ROUTED_PROVIDERS: Tuple[str, ...] = OLLAMA_PROVIDERS

#: API modes this router understands. The Ollama:cloud profile is chat_completions;
#: a Responses or Anthropic-Messages route builds its payload elsewhere and is left alone.
ROUTED_API_MODES = ("chat_completions",)

EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high")

#: Decision backends: TypeSafe's own System One API (default) or OpenRouter's Decisions API.
#: Same request/answer shape (``docs/jev-decisions-api.md``); endpoint, model id and key differ.
BACKENDS: Dict[str, Dict[str, str]] = {
    "typesafe": {
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "jev_model": "jev-1.13.0",
        "api_key_env": "TYPESAFE_API_KEY",
    },
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/alpha/decisions",
        "jev_model": "typesafe/jev-1.13",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}
DEFAULT_BACKEND = "typesafe"
DEFAULT_JEV_MODEL = BACKENDS[DEFAULT_BACKEND]["jev_model"]
DEFAULT_ENDPOINT = BACKENDS[DEFAULT_BACKEND]["endpoint"]
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
#: Empty: a below-threshold decision keeps the configured model (never a silent downgrade).
DEFAULT_MODEL = ""
DEFAULT_EFFORT = "medium"
DEFAULT_CONTEXT_TURNS = 4
#: Tokens kept free for the answer (and estimate error) when checking a context window.
DEFAULT_CONTEXT_RESERVE = 32000

#: What to do with ``reasoning_effort`` when no effort family applies (every non-Ollama route,
#: and ``combo/<id>``): ``omit`` drops the field, ``keep`` keeps the host's value only when it is
#: low/medium/high (anything else is dropped), ``pass`` sends Jev's level verbatim (and drops
#: the host's value when Jev gave none).
UNKNOWN_EFFORT_MODES: Tuple[str, ...] = ("omit", "keep", "pass")
DEFAULT_UNKNOWN_EFFORT = "omit"


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_float(value: Any, default: float, *, minimum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    if minimum is not None and result < minimum:
        return default
    return result


def _as_int(value: Any, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and result < minimum:
        return default
    if maximum is not None and result > maximum:
        return maximum
    return result


def _as_choice(value: Any, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _as_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _as_list(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """A list setting, also accepted as a comma-separated string. Unset (``None``) or garbage
    means ``default``; an explicit empty list means empty."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return default
    return tuple(text for text in (str(item).strip() for item in value) if text)


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def norm_provider(name: Any) -> str:
    """Canonical provider name: ``custom:ocx`` and ``ocx`` compare equal (Hermes' own
    custom-provider normalisation: lower-case, spaces to dashes)."""
    text = str(name or "").strip().lower().replace(" ", "-")
    return text[len("custom:"):] if text.startswith("custom:") else text


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the plugin's effective settings."""

    enabled: bool = True
    jev_model: str = DEFAULT_JEV_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    timeout_s: float = DEFAULT_TIMEOUT_S
    default_model: str = DEFAULT_MODEL
    default_effort: str = DEFAULT_EFFORT
    context_turns: int = DEFAULT_CONTEXT_TURNS
    route_per_turn: bool = True
    audit_enabled: bool = True
    log_skips: bool = True
    include_user_message_in_audit: bool = False
    grid: Tuple[Entry, ...] = field(default_factory=lambda: tuple(DEFAULT_GRID))
    routed_providers: Tuple[str, ...] = DEFAULT_ROUTED_PROVIDERS
    routed_base_urls: Tuple[str, ...] = ()
    catalog_check: bool = True
    unknown_effort: str = DEFAULT_UNKNOWN_EFFORT
    backend: str = DEFAULT_BACKEND
    mode: str = "route"
    api_key_env: str = BACKENDS[DEFAULT_BACKEND]["api_key_env"]
    context_reserve_tokens: int = DEFAULT_CONTEXT_RESERVE
    long_context_models: Tuple[Entry, ...] = ()

    def match(self, provider: Any, base_url: Any = "") -> Optional[str]:
        """How a request is the router's to rewrite: ``"provider"`` (its provider name is in
        ``routed_providers``, ``custom:`` prefix ignored on both sides), ``"base_url"`` (its
        base_url equals a ``routed_base_urls`` entry or sits under it at a path boundary), or
        ``None`` (not routed; the request goes out untouched). An empty provider is never routed."""
        name = norm_provider(provider)
        if not name:
            return None  # a request that does not say where it goes is never rewritten
        if name in {norm_provider(item) for item in self.routed_providers}:
            return "provider"
        url = _norm_url(base_url)
        for item in self.routed_base_urls:
            prefix = _norm_url(item)
            if url and prefix and (url == prefix or url.startswith(prefix + "/")):
                return "base_url"
        return None

    def is_ollama(self, provider: Any, how: Optional[str]) -> bool:
        """Ollama:cloud semantics (model-cache check, effort families) apply only to a request
        matched by an Ollama:cloud provider name — never to one matched by base_url."""
        return how == "provider" and norm_provider(provider) in OLLAMA_PROVIDERS

    @property
    def grid_ids(self) -> Tuple[str, ...]:
        return tuple(entry.model_id for entry in self.grid)

    def long_context_entry(self, item: Entry) -> Entry:
        """A ``long_context_models`` item, completed from its grid entry when it only names an id."""
        grid_entry = self.entry_for(item.model_id)
        if grid_entry is None:
            return item
        return Entry(
            item.model_id,
            item.description or grid_entry.description,
            item.efforts or grid_entry.efforts,
            item.context or grid_entry.context,
        )

    def entry_for(self, model_id: str) -> Optional[Entry]:
        wanted = (model_id or "").strip()
        for entry in self.grid:
            if entry.model_id == wanted:
                return entry
        return None


def load_settings(get_config: Optional[Callable[..., Any]] = None) -> Settings:
    """Build a :class:`Settings` from ``ctx.get_config`` (or from no config at all)."""
    read = get_config if callable(get_config) else (lambda _key, default=None: default)

    backend = _as_choice(read("backend", DEFAULT_BACKEND), tuple(BACKENDS), DEFAULT_BACKEND)
    defaults = BACKENDS[backend]
    raw_grid = read("grid", None)
    grid = parse_grid(raw_grid) if raw_grid else tuple(DEFAULT_GRID)

    settings = Settings(
        enabled=_as_bool(read("enabled", True), True),
        jev_model=_jev_model(backend, _as_text(read("jev_model", None), defaults["jev_model"])),
        endpoint=_as_text(read("endpoint", None), defaults["endpoint"]),
        confidence_threshold=_as_float(
            read("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            DEFAULT_CONFIDENCE_THRESHOLD,
            minimum=0.0,
        ),
        timeout_s=_as_float(read("timeout_s", DEFAULT_TIMEOUT_S), DEFAULT_TIMEOUT_S, minimum=0.05),
        default_model=_as_text(read("default_model", DEFAULT_MODEL), DEFAULT_MODEL),
        default_effort=_as_choice(read("default_effort", DEFAULT_EFFORT), EFFORT_LEVELS, DEFAULT_EFFORT),
        context_turns=_as_int(
            read("context_turns", DEFAULT_CONTEXT_TURNS), DEFAULT_CONTEXT_TURNS, minimum=0, maximum=50
        ),
        route_per_turn=_as_bool(read("route_per_turn", True), True),
        audit_enabled=_as_bool(read("audit_enabled", True), True),
        log_skips=_as_bool(read("log_skips", True), True),
        include_user_message_in_audit=_as_bool(read("include_user_message_in_audit", False), False),
        grid=grid,
        routed_providers=_as_list(read("routed_providers", None), DEFAULT_ROUTED_PROVIDERS),
        routed_base_urls=_as_list(read("routed_base_urls", None), ()),
        catalog_check=_as_bool(read("catalog_check", True), True),
        unknown_effort=_as_choice(read("unknown_effort", DEFAULT_UNKNOWN_EFFORT), UNKNOWN_EFFORT_MODES,
                                  DEFAULT_UNKNOWN_EFFORT),
        backend=backend,
        mode=_as_choice(read("mode", "route"), ("route", "shadow"), "route"),
        api_key_env=_as_text(read("api_key_env", None), defaults["api_key_env"]),
        context_reserve_tokens=_as_int(read("context_reserve_tokens", DEFAULT_CONTEXT_RESERVE),
                                       DEFAULT_CONTEXT_RESERVE, minimum=0, maximum=10_000_000),
        long_context_models=_entries(read("long_context_models", None)),
    )
    return settings


def _entries(raw: Any) -> Tuple[Entry, ...]:
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    parsed = (parse_entry(item) for item in raw)
    return tuple(entry for entry in parsed if entry is not None)


def _jev_model(backend: str, model: str) -> str:
    """OpenRouter names Jev ``typesafe/jev-1.13``; TypeSafe's own API names the same model
    ``jev-1.13.0`` (and takes aliases such as ``jev-latest`` as-is)."""
    if backend == "typesafe" and model.startswith("typesafe/"):
        model = model[len("typesafe/"):]
        return "jev-1.13.0" if model == "jev-1.13" else model
    return model


def api_key(settings: Optional["Settings"] = None) -> str:
    """The decision backend's key, by the env-var name ``api_key_env`` names.

    Read through Hermes' profile-scoped secret lookup (``host.secret``), so a multiplexed
    gateway serves each profile its own ``<profile>/.env`` value. Never read from
    ``config.yaml`` and never returned anywhere it could be logged.
    """
    from .host import secret

    name = settings.api_key_env if settings is not None else BACKENDS[DEFAULT_BACKEND]["api_key_env"]
    return secret(name)


def redact(value: Any, limit: int = 200, settings: Optional["Settings"] = None) -> str:
    """Render a value for a log line with any credential-looking substring removed."""
    text = str(value)
    key = api_key(settings)
    if key:
        text = text.replace(key, "[REDACTED]")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text
