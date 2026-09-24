"""jev-effort-router — route each Hermes turn through TypeSafe Jev's decision endpoint.

What it does: on the first provider request of every user turn, ask Jev (`typesafe/jev-1.13`
over OpenRouter's Decisions API) which grid model (Ollama:cloud, or ocx ``provider/model`` /
``combo/<id>``) and which reasoning-effort level fit the task, then rewrite the outgoing
request accordingly. Jev generates nothing; the selected model still does all the reasoning
and all the writing.

Where it plugs in: the ``llm_request`` middleware kind
(``hermes_cli/middleware.py``, invoked from ``agent/turn_api_request.py::build_api_request``),
which is the documented chokepoint just before the provider call. Nothing in Hermes core is
touched.

How it fails: never loudly. A timeout, an HTTP error, a malformed answer, a confidence below
the threshold, an unknown provider, or an off-grid model all leave the request exactly as it
was, so a broken router is indistinguishable from a router that is not installed.

Everything here is lazy: no socket, no file, and no HTTP client is touched while
``register(ctx)`` runs, because ``hermes plugins doctor`` blocks network access during
registration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .commands import register_commands
from .config import api_key, load_settings
from .router import Router
from .tools import build_tool_registrations

__all__ = ["register", "register_cli"]

logger = logging.getLogger(__name__)

PLUGIN_ID = "jev-effort-router"


def _warn_missing_key_once() -> None:
    """Say it once at registration, then stay quiet: the per-turn path must not spam."""
    if api_key():
        return
    logger.info(
        "jev-effort-router: OPENROUTER_API_KEY is not set — the router is inert until a key is "
        "available (Jev is reached through OpenRouter)."
    )


def register(ctx: Any) -> None:
    """Wire the plugin into Hermes.

    Registration is deliberately cheap and side-effect free: it builds the settings reader, the
    router, and the registrations, and touches nothing else.
    """
    settings_cache: dict = {"settings": None}

    def get_settings():
        cached = settings_cache["settings"]
        if cached is None:
            cached = load_settings(ctx.get_config)
            settings_cache["settings"] = cached
        return cached

    router = Router(get_settings, get_state=lambda: getattr(ctx, "state", None))

    # -- the routing decision itself ---------------------------------------------
    ctx.register_middleware("llm_request", router.on_llm_request)

    # -- observability surface ----------------------------------------------------
    for schema, handler in build_tool_registrations(router, get_settings):
        ctx.register_tool(
            name=schema["name"],
            toolset="jev-effort-router",
            schema=schema,
            handler=handler,
        )

    ctx.register_hook("on_session_end", _make_observer(router, "on_session_end"))
    ctx.register_hook("post_llm_call", _make_observer(router, "post_llm_call"))

    register_commands(ctx, router, get_settings)

    _warn_missing_key_once()
    logger.debug("jev-effort-router: registered llm_request middleware, tools, hooks and commands")


def _make_observer(router: Router, hook_name: str):
    """Turn/session observers: they keep the memo honest."""

    def observer(**kwargs: Any) -> None:
        try:
            if hook_name == "on_session_end":
                # A finished session must not leave a stale decision replayable.
                router.forget()
        except Exception:  # noqa: BLE001 - observers never affect a turn
            logger.debug("jev-effort-router: %s observer failed", hook_name)

    observer.__name__ = f"jev_effort_router_{hook_name}"
    return observer


def register_cli(subparser: Any) -> None:
    """Reserved for the future pip-distributed form; the directory plugin registers
    ``hermes jev-effort-router`` through :func:`register` via ``ctx.register_cli_command``."""
    from .commands import setup_argparse

    setup_argparse(subparser)
