"""Human-facing surfaces: the ``/jev-effort-router`` slash command and ``hermes jev-effort-router``."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, List, Optional

from .config import Settings
from .tools import _hint_for, _status

USAGE = (
    "Usage: hermes jev-effort-router <status|route|grid|tail|reset>\n"
    "  status            routing state, settings, grid and the most recent decisions\n"
    "  route <task...>   ask Jev what it would pick for a task (no session change)\n"
    "  grid              list the models Jev may choose between\n"
    "  tail [n]          print the last n audit records as JSON (default 10)\n"
    "  reset             drop memoized decisions so the next turn re-routes\n"
)

#: Slash-command name (no leading slash). Kept distinct from the CLI family name so a
#: `/jev-effort-router` inside a session and `hermes jev-effort-router` on the shell are both natural.
SLASH_NAME = "jev-effort-router"


def _grid_text(settings: Settings) -> str:
    lines = [f"Grid ({len(settings.grid)} models):"]
    for index, entry in enumerate(settings.grid, start=1):
        default = " (fallback)" if entry.model_id == settings.default_model else ""
        lines.append(f"  {index}. {entry.model_id}{default}")
        lines.append(f"     {entry.description}")
    return "\n".join(lines)


def _status_text(router, settings: Settings, recent: int = 5) -> str:
    """Plain-text status, for a shell or a chat reply."""
    payload = json.loads(_status(router, settings, recent=recent))
    lines = [
        "jev-effort-router",
        f"  enabled:        {payload['enabled']}",
        f"  api key:        {'present' if payload['api_key_present'] else 'MISSING (router is inert)'}",
        f"  jev model:      {payload['jev_model']}",
        f"  endpoint:       {payload['endpoint']}",
        f"  threshold:      {payload['confidence_threshold']}",
        f"  timeout:        {payload['timeout_s']}s",
        f"  per turn:       {payload['route_per_turn']}",
        f"  fallback:       {payload['fallback']['model']} / {payload['fallback']['effort']}",
        f"  audit records:  {payload['audit']['records']} (enabled: {payload['audit']['enabled']})",
        "",
        _grid_text(settings),
    ]
    if payload.get("grid_unavailable"):
        lines.append(
            "  ⚠ not in the provider's catalog (a decision naming one of these is refused): "
            + ", ".join(payload["grid_unavailable"])
        )
    coverage = payload.get("grid_coverage")
    if coverage:
        lines.append("")
        lines.extend(_coverage_text(coverage))
    records = payload.get("recent") or []
    lines.append("")
    if not records:
        lines.append("Recent decisions: none recorded yet.")
    else:
        lines.append(f"Recent decisions ({len(records)}):")
        for record in records:
            if record.get("event") == "route":
                lines.append(
                    "  {ts}  route  {model}  effort={effort}  conf={mc}/{ec}  {lat}ms{deg}{rep}".format(
                        ts=record.get("ts", "?"),
                        model=record.get("model", "?"),
                        effort=record.get("effort"),
                        mc=record.get("model_confidence"),
                        ec=record.get("effort_confidence"),
                        lat=record.get("latency_ms", 0),
                        deg=(
                            "  degraded:" + ",".join(record.get("fallback_reasons") or [])
                            if record.get("fallback_reasons")
                            else ""
                        ),
                        rep="  (replayed)" if record.get("replayed") else "",
                    )
                )
            else:
                lines.append(
                    "  {ts}  skip   {reason}  (configured: {model})".format(
                        ts=record.get("ts", "?"),
                        reason=record.get("reason", "?"),
                        model=record.get("configured_model", "?"),
                    )
                )
    return "\n".join(lines)


def _coverage_text(coverage: dict) -> list:
    """The grid-coverage block: what is being applied, and what is being lost.

    Two different findings, because the fixes differ: a model under ``never chosen`` needs its
    criterion rewritten, a model under ``picked, below threshold`` needs the criterion or the
    threshold revisited. Only first-call records are counted, so a long tool loop does not weight
    the turn it replays.
    """
    lines = [f"Grid coverage (last {coverage.get('window', 0)} routed turns):"]
    applied = coverage.get("applied") or {}
    if applied:
        ranked = sorted(applied.items(), key=lambda item: item[1], reverse=True)
        lines.append("  applied:        " + ", ".join(f"{model} x{count}" for model, count in ranked))
    else:
        lines.append("  applied:        none")
    never = coverage.get("never_chosen") or []
    if never:
        lines.append("  never chosen:   " + ", ".join(never) + "   ← Jev never picks these on this workload")
    below = coverage.get("below_threshold") or {}
    if below:
        ranked = sorted(below.items(), key=lambda item: item[1], reverse=True)
        lines.append(
            "  picked, below threshold: "
            + ", ".join(f"{model} x{count}" for model, count in ranked)
            + "   ← discarded, served by the fallback"
        )
    return lines


def _route_text(router, settings: Settings, task: str) -> str:
    decision, reason = router.client(settings).decide(
        [{"role": "user", "content": task}], settings.choice_grid
    )
    if decision is None:
        return f"No route ({reason}). {_hint_for(reason, settings)}"
    lines = [
        f"model:  {decision.model}",
        f"effort: {decision.effort}  (requested {decision.effort_requested})",
        f"choice: {decision.model_choice}  confidence {decision.model_confidence:.3f}",
        f"effort: {decision.effort_choice}  confidence {decision.effort_confidence:.3f}",
        f"latency: {decision.latency_ms} ms",
    ]
    probable = sorted(
        (decision.model_probabilities or {}).items(), key=lambda item: item[1], reverse=True
    )[:3]
    if probable:
        lines.append("top probabilities: " + ", ".join(f"{key}={value:.3f}" for key, value in probable))
    if decision.fallback_reasons:
        lines.append("degradation: " + ",".join(decision.fallback_reasons))
    return "\n".join(lines)


def _handle(router, get_settings, args) -> int:
    settings = get_settings()
    sub = getattr(args, "jev_effort_router_command", None) or "status"

    if sub == "status":
        print(_status_text(router, settings))
        return 0
    if sub == "grid":
        print(_grid_text(settings))
        return 0
    if sub == "route":
        task = " ".join(getattr(args, "task", []) or []).strip()
        if not task:
            print(USAGE, file=sys.stderr)
            return 2
        text = _route_text(router, settings, task)
        print(text)
        return 0 if text.startswith("model:") else 1
    if sub == "tail":
        limit = max(1, min(int(getattr(args, "count", 10) or 10), 200))
        for record in router.tail(settings, limit=limit):
            print(json.dumps(record, ensure_ascii=False))
        return 0
    if sub == "reset":
        router.forget()
        print("Memoized decisions dropped; the next turn re-routes.")
        return 0

    print(USAGE, file=sys.stderr)
    return 2


def setup_argparse(subparser: Any) -> None:
    """Build the ``hermes jev-effort-router`` argparse tree."""
    sub = subparser.add_subparsers(dest="jev_effort_router_command")
    sub.add_parser("status", help="Show routing state, grid and recent decisions")
    route = sub.add_parser("route", help="Ask Jev what it would pick for a task")
    route.add_argument("task", nargs="*", help="Task description to route")
    sub.add_parser("grid", help="List the models Jev may choose between")
    tail = sub.add_parser("tail", help="Print the last audit records as JSON")
    tail.add_argument("count", nargs="?", type=int, default=10, help="How many records (default 10)")
    sub.add_parser("reset", help="Drop memoized decisions")


def _command_handler(router, get_settings):
    def handler(args) -> int:
        try:
            return _handle(router, get_settings, args)
        except Exception as exc:  # noqa: BLE001 - a CLI must fail with a message, not a traceback
            print(f"jev-effort-router: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    return handler


def register_commands(ctx: Any, router, get_settings: Callable[[], Settings]) -> None:
    """Attach both human surfaces to the plugin context."""
    handler = _command_handler(router, get_settings)

    ctx.register_cli_command(
        name=SLASH_NAME,
        help="Inspect and test Jev-based model routing",
        setup_fn=setup_argparse,
        handler_fn=handler,
    )

    def slash(raw_args: str) -> str:
        """In-session ``/jev-effort-router [status|grid|route <task>|tail [n]|reset]``."""
        try:
            tokens = (raw_args or "").split()
            action = tokens[0].lower() if tokens else "status"
            if action == "status":
                return _status_text(router, get_settings())
            if action == "grid":
                return _grid_text(get_settings())
            if action == "route":
                task = " ".join(tokens[1:]).strip()
                if not task:
                    return "Usage: /jev-effort-router route <task description>"
                return _route_text(router, get_settings(), task)
            if action == "tail":
                limit = 10
                if len(tokens) > 1:
                    try:
                        limit = max(1, min(int(tokens[1]), 200))
                    except ValueError:
                        return "Usage: /jev-effort-router tail [n]"
                records = router.tail(get_settings(), limit=limit)
                if not records:
                    return "No audit records yet."
                return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            if action == "reset":
                router.forget()
                return "Memoized decisions dropped; the next turn re-routes."
            return (
                "Usage: /jev-effort-router [status|grid|tail [n]|reset|route <task>]"
            )
        except Exception as exc:  # noqa: BLE001
            return f"jev-effort-router: {type(exc).__name__}: {exc}"

    ctx.register_command(
        SLASH_NAME,
        handler=slash,
        description="Jev router: routing state, grid, recent decisions",
        args_hint="[status|grid|tail [n]|reset|route <task>]",
    )
