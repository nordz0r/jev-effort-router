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

from .grid import DEFAULT_GRID, Entry, parse_grid

#: Hermes provider names of Ollama:cloud. Routed by default, and the only provider whose model
#: cache (``catalog.py``) is consulted before a decision is acted on.
OLLAMA_PROVIDERS: Tuple[str, ...] = ("ollama-cloud", "ollama_cloud")

#: Providers routed when ``routed_providers`` is not configured. Anything else is skipped untouched.
DEFAULT_ROUTED_PROVIDERS: Tuple[str, ...] = OLLAMA_PROVIDERS

#: API modes this router understands. The Ollama:cloud profile is chat_completions;
#: a Responses or Anthropic-Messages route builds its payload elsewhere and is left alone.
ROUTED_API_MODES = ("chat_completions",)

EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high")

DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_EFFORT = "medium"
DEFAULT_CONTEXT_TURNS = 4

#: What to do with ``reasoning_effort`` when the chosen model has no known effort family
#: (``combo/<id>``, or a ``provider/model`` id outside ``effort.py``'s table):
#: ``keep`` leaves the request's value as Hermes built it, ``omit`` drops the field, ``pass``
#: sends Jev's low/medium/high verbatim.
UNKNOWN_EFFORT_MODES: Tuple[str, ...] = ("keep", "omit", "pass")
DEFAULT_UNKNOWN_EFFORT = "keep"


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
    """A list setting, also accepted as a comma-separated string; empty means ``default``."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return default
    items = tuple(text for text in (str(item).strip() for item in value) if text)
    return items or default


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


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

    def routes(self, provider: str, base_url: str = "") -> bool:
        """Whether a request to this provider / endpoint is the router's to rewrite.

        An empty provider name counts as routed (the host did not say). ``routed_base_urls``
        entries match the request's ``base_url`` as a prefix, trailing slash and case ignored.
        """
        name = (provider or "").strip().lower()
        if not name or name in {item.lower() for item in self.routed_providers}:
            return True
        url = _norm_url(base_url)
        return bool(url) and any(url.startswith(_norm_url(item)) for item in self.routed_base_urls)

    def checks_catalog(self, provider: str) -> bool:
        """The Ollama:cloud model cache only speaks for Ollama:cloud; for any other provider
        (ocx included) the grid itself is the allowlist. An unnamed provider counts as
        Ollama:cloud only while Ollama:cloud is among the routed providers."""
        name = (provider or "").strip().lower()
        if not name:
            return self.catalog_check and any(item.lower() in OLLAMA_PROVIDERS for item in self.routed_providers)
        return self.catalog_check and name in OLLAMA_PROVIDERS

    @property
    def grid_ids(self) -> Tuple[str, ...]:
        return tuple(entry.model_id for entry in self.grid)

    def entry_for(self, model_id: str) -> Optional[Entry]:
        wanted = (model_id or "").strip()
        for entry in self.grid:
            if entry.model_id == wanted:
                return entry
        return None


def load_settings(get_config: Optional[Callable[..., Any]] = None) -> Settings:
    """Build a :class:`Settings` from ``ctx.get_config`` (or from no config at all)."""
    read = get_config if callable(get_config) else (lambda _key, default=None: default)

    raw_grid = read("grid", None)
    grid = parse_grid(raw_grid) if raw_grid else tuple(DEFAULT_GRID)

    settings = Settings(
        enabled=_as_bool(read("enabled", True), True),
        jev_model=_as_text(read("jev_model", DEFAULT_JEV_MODEL), DEFAULT_JEV_MODEL),
        endpoint=_as_text(read("endpoint", DEFAULT_ENDPOINT), DEFAULT_ENDPOINT),
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
    )
    return settings


def api_key() -> str:
    """The OpenRouter key, from the environment only.

    Never read from ``config.yaml`` and never returned anywhere it could be logged.
    """
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


def redact(value: Any, limit: int = 200) -> str:
    """Render a value for a log line with any credential-looking substring removed."""
    text = str(value)
    key = api_key()
    if key:
        text = text.replace(key, "[REDACTED]")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text
