"""Agent-facing tools: ask the router what it sees and what it would do.

The host's tool registry invokes a registered handler as ``handler(args, **context)`` — the
model's arguments arrive as ONE dict in the first positional slot and the per-call context
(``task_id``, ``session_id``, ...) as signature-inspected keyword arguments. A handler shaped
``handler(recent: int = 5, **_)`` therefore binds the arguments dict to ``recent`` and raises
``TypeError: int() argument must be ... not 'dict'`` on every parameterised call, while the
no-argument form (``{}``, or kwargs only) silently works — which is how the bug hid. Every
handler below takes ``args`` first.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .client import JevClient
from .config import Settings, api_key
from .state import build_questions, build_state

STATUS_SCHEMA = {
    "name": "jev_effort_router_status",
    "description": (
        "Report the Jev router's state: whether routing is enabled, which decision model and "
        "endpoint it uses, the confidence threshold, the models it may choose between, and the "
        "most recent routing decisions. Use it when the user asks how model routing is set up, "
        "which model was picked recently, or why a turn was not routed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recent": {
                "type": "integer",
                "description": "How many recent routing records to include (default 5, max 50).",
            }
        },
    },
}

ROUTE_SCHEMA = {
    "name": "jev_effort_router_route",
    "description": (
        "Ask the Jev decision model which grid model and reasoning effort it would "
        "choose for a given task description, without changing the current session. Use it to "
        "test the routing grid, or to show the user what the router would pick. Returns the "
        "choice, its probabilities, the confidence, and the alternatives."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task or user request to route, in natural language.",
            },
            "context": {
                "type": "string",
                "description": "Optional recent context to help the decision.",
            },
        },
        "required": ["task"],
    },
}


def _grid_unavailable(router, settings: Settings) -> list:
    """Whatever the router's catalog knows about the grid, or an empty list.

    A router double without the hook is not a failure: the status view has to work against any
    router object, and "no report" is the same answer as "no evidence".
    """
    report = getattr(router, "grid_report", None)
    if not callable(report):
        return []
    try:
        return [row["model"] for row in report(settings) if not row.get("available")]
    except Exception:  # noqa: BLE001 - an observability surface never raises
        return []


#: How many routed turns the grid-coverage report summarises. This is the window in which a
#: "rarely chosen" model shows up as a finding rather than as noise.
GRID_COVERAGE_SAMPLE = 200


def _grid_coverage(router, settings: Settings, sample: int = GRID_COVERAGE_SAMPLE) -> dict:
    """Which grid entries are actually being chosen, and which never survive the threshold.

    A model that is never routed on is invisible from a single turn: the failure mode that kept
    ``glm-5.3`` and ``glm-5.3-flash`` off the route showed up only as a share of the audit trail.
    Only first-call records count — a turn replays its decision across a whole tool loop, so
    counting replayed records would weight long tool turns as if they were extra decisions.

    Two distinct failures are reported, because they have different fixes:

    * ``never_chosen`` — the model is on no criterion's winning side, so Jev never picks it on
      this workload. Fix the grid wording, not the threshold.
    * ``below_threshold`` — Jev picks it but the answer is not confident enough to apply, so the
      turn silently falls back. Fix the criterion text or the threshold.
    """
    tail = getattr(router, "tail", None)
    if not callable(tail):
        return {}
    try:
        records = tail(settings, limit=max(1, int(sample)))
    except Exception:  # noqa: BLE001 - an observability surface never raises
        return {}
    counted = [r for r in records if r.get("event") == "route" and not r.get("replayed")]
    if not counted:
        return {}

    # The probabilities are keyed by position ("1".."6"), so the grid gives the mapping back.
    by_position = {str(i): entry.model_id for i, entry in enumerate(settings.grid, start=1)}

    applied: Dict[str, int] = {}
    below: Dict[str, int] = {}
    for record in counted:
        degraded = "low_confidence" in (record.get("fallback_reasons") or [])
        if degraded:
            probabilities = record.get("model_probabilities") or {}
            best_key, best_value = "", 0.0
            if isinstance(probabilities, dict):
                for key, value in probabilities.items():
                    if isinstance(value, (int, float)) and value > best_value:
                        best_key, best_value = str(key), float(value)
            lead = by_position.get(best_key) or str(record.get("chosen_model") or "")
            if lead:
                below[lead] = below.get(lead, 0) + 1
        else:
            model = str(record.get("model") or "")
            if model:
                applied[model] = applied.get(model, 0) + 1

    report: Dict[str, Any] = {
        "window": len(counted),
        "applied": applied,
    }
    never = [
        entry.model_id
        for entry in settings.grid
        if entry.model_id not in applied and entry.model_id not in below
    ]
    if never:
        report["never_chosen"] = never
    if below:
        report["below_threshold"] = below
    return report


def _status(router, settings: Settings, recent: int = 5) -> str:
    records = router.tail(settings, limit=max(1, min(int(recent or 5), 50)))
    payload = {
        "enabled": settings.enabled,
        "api_key_present": bool(api_key(settings)),
        "api_key_env": settings.api_key_env,
        "backend": settings.backend,
        "mode": settings.mode,
        "endpoint": settings.endpoint,
        "jev_model": settings.jev_model,
        "confidence_threshold": settings.confidence_threshold,
        "timeout_s": settings.timeout_s,
        "route_per_turn": settings.route_per_turn,
        "fallback": {"model": settings.default_model, "effort": settings.default_effort},
        "grid": [
            {"model": entry.model_id, "profile": entry.description,
             **({"efforts": list(entry.efforts)} if entry.efforts else {}),
             **({"context": entry.context} if entry.context else {})}
            for entry in settings.grid
        ],
        "audit": {
            "enabled": settings.audit_enabled,
            "records": router.count(settings),
        },
        "recent": records,
    }
    coverage = _grid_coverage(router, settings)
    if coverage:
        payload["grid_coverage"] = coverage
    # Only present when the provider catalog could actually be read: an empty list is omitted
    # rather than reported as "every model is missing".
    unavailable = _grid_unavailable(router, settings)
    if unavailable:
        payload["grid_unavailable"] = unavailable
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _route(router, settings: Settings, task: str, context: str = "") -> str:
    if not str(task or "").strip():
        return json.dumps({"error": "task must not be empty"})
    messages = [{"role": "user", "content": str(task)}]
    if str(context).strip():
        messages = [
            {"role": "user", "content": str(context).strip()},
            {"role": "assistant", "content": "(context)"},
            {"role": "user", "content": str(task)},
        ]

    decision, reason = router.client(settings).decide(
        messages, settings.grid
    )
    if decision is None:
        return json.dumps(
            {
                "routed": False,
                "reason": reason,
                "hint": _hint_for(reason, settings),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "routed": True,
            "model": decision.model,
            "effort": decision.effort,
            "effort_requested": decision.effort_requested,
            "model_choice": decision.model_choice,
            "model_confidence": round(decision.model_confidence, 4),
            "model_probabilities": decision.model_probabilities,
            "effort_confidence": round(decision.effort_confidence, 4),
            "effort_probabilities": decision.effort_probabilities,
            "alternatives": list(decision.alternatives),
            "latency_ms": decision.latency_ms,
            "degradation": list(decision.fallback_reasons),
        },
        ensure_ascii=False,
        indent=2,
    )


def _hint_for(reason: Any, settings: Optional[Settings] = None) -> str:
    from .client import REASON_NO_API_KEY, REASON_TIMEOUT, REASON_UPSTREAM_ERROR
    from .config import key_hint

    if reason == REASON_NO_API_KEY:
        return key_hint(settings)
    if reason == REASON_TIMEOUT:
        return "Jev did not answer inside the configured budget; raise timeout_s or retry."
    if reason == REASON_UPSTREAM_ERROR:
        backend = getattr(settings, "backend", "the decision")
        return f"The {backend} backend returned an error — check the key, credits and network access."
    return "The decision was unusable; the turn would keep the configured model."


def build_tool_registrations(
    router, get_settings: Callable[[], Settings]
) -> List[Tuple[Dict[str, Any], Callable[..., str]]]:
    """``(schema, handler)`` pairs for ``ctx.register_tool``.

    The handlers are called exactly as ``tools/registry.py::dispatch`` calls them —
    ``handler(args, **context)``, where ``args`` is the model's argument dict — so they must
    accept that dict positionally. ``**context`` swallows the per-call keys the dispatcher
    injects (``task_id``, ``session_id``, ``user_task``, ``parent_agent``, ...).
    """

    def _arg(args: Any, key: str, default: Any = None) -> Any:
        """One value out of the arguments dict, whichever way the host hands it over."""
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default) if args is not None else default

    def status_handler(args: Any = None, **_: Any) -> str:
        return _status(router, get_settings(), recent=_arg(args, "recent", 5))

    def route_handler(args: Any = None, **_: Any) -> str:
        return _route(
            router,
            get_settings(),
            task=_arg(args, "task", ""),
            context=_arg(args, "context", ""),
        )

    return [(STATUS_SCHEMA, status_handler), (ROUTE_SCHEMA, route_handler)]
