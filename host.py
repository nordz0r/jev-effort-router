"""The two host lookups the router needs, each failing soft when Hermes is absent or refuses.

* ``secret(name)`` — Hermes' profile-scoped secret read (``agent.secret_scope.get_secret``, the
  same call in-tree provider plugins use). Under a multiplexed gateway the active profile's
  ``.env`` is installed as a context-local scope and ``os.environ`` is deliberately *not* a
  fallback; outside multiplexing it reads ``os.environ``. An unscoped read under multiplexing
  raises in Hermes — here it is "no key", so the router stays inert (fail closed, fail open).
* ``custom_provider_name(base_url)`` — Hermes hands middleware ``provider="custom"`` for every
  ``custom_providers`` entry; the entry's name is recovered from the request's base_url.
"""

from __future__ import annotations

import os
from typing import Any


def secret(name: str) -> str:
    if not name:
        return ""
    try:
        from agent.secret_scope import get_secret  # type: ignore
    except Exception:  # noqa: BLE001 - no Hermes (tests, CI): plain environment
        return (os.environ.get(name) or "").strip()
    try:
        return (get_secret(name) or "").strip()
    except Exception:  # noqa: BLE001 - unscoped read under multiplexing: no key, never a leak
        return ""


def custom_provider_name(base_url: Any) -> str:
    """Name of the ``custom_providers`` entry serving ``base_url``, or ``""``."""
    url = str(base_url or "").strip().rstrip("/").lower()
    if not url:
        return ""
    try:
        from hermes_cli.config import get_compatible_custom_providers  # type: ignore

        entries = get_compatible_custom_providers() or []
    except Exception:  # noqa: BLE001 - no Hermes / unreadable config: no name
        return ""
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("base_url") or "").strip().rstrip("/").lower() == url:
            return str(entry.get("name") or "").strip()
    return ""
