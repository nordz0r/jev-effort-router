"""The routing engine: decide, apply, record — and fail open at every step.

Called from the ``llm_request`` middleware with the request kwargs Hermes is about to send.
The contract is narrow on purpose:

* the payload is returned **unchanged** unless a confident decision was obtained;
* nothing here raises — the host already isolates middleware exceptions, but the router also
  enforces its own budget and swallows its own errors so a slow Jev cannot stall a turn;
* one decision per user turn, replayed for the rest of that turn's tool loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .audit import AuditLog
from .catalog import ModelCatalog
from .client import (
    REASON_DISABLED,
    REASON_EXCEPTION,
    REASON_NO_API_KEY,
    REASON_SKIPPED_API_MODE,
    REASON_SKIPPED_MODEL,
    REASON_SKIPPED_NO_MESSAGE,
    REASON_SKIPPED_PROVIDER,
    REASON_SKIPPED_UNKNOWN_MODEL,
    REASON_USER_DATA,
    Decision,
    JevClient,
)
from .config import ROUTED_API_MODES, Settings
from .effort import family_for
from .memo import Memo, TurnMemo
from .state import last_user_message

logger = logging.getLogger(__name__)

#: Where the routing decision puts the effort on the wire.
EFFORT_WIRE_KEY = "reasoning_effort"


class Router:
    """Stateful routing engine: one instance per loaded plugin."""

    def __init__(self, get_settings, get_state=None, *, client_factory=None, catalog=None) -> None:
        self._get_settings = get_settings
        self._get_state = get_state
        self._client_factory = client_factory or (lambda settings: JevClient(settings))
        self._memo = TurnMemo()
        self._audit: Optional[AuditLog] = None
        self._audit_key: Optional[tuple] = None
        self._client = None
        self._client_key: Optional[tuple] = None
        self._catalog = catalog if catalog is not None else ModelCatalog()

    # -- wiring ------------------------------------------------------------------

    def _data_dir(self) -> Optional[str]:
        """The plugin data directory, when the host exposed one."""
        try:
            state = self._get_state() if callable(self._get_state) else None
            data_dir = getattr(state, "data_dir", None)
            return str(data_dir) if data_dir else None
        except Exception:  # noqa: BLE001 - auditing is optional, never load-bearing
            return None

    def audit(self, settings: Settings) -> AuditLog:
        key = (self._data_dir(), bool(settings.audit_enabled))
        if self._audit is None or self._audit_key != key:
            self._audit = AuditLog(key[0], enabled=key[1])
            self._audit_key = key
        return self._audit

    def client(self, settings: Settings):
        key = (settings.endpoint, settings.jev_model, settings.timeout_s)
        if self._client is None or self._client_key != key:
            self._client = self._client_factory(settings)
            self._client_key = key
        return self._client

    # -- the middleware body -----------------------------------------------------

    def on_llm_request(
        self,
        request: Any = None,
        original_request: Any = None,
        *,
        turn_id: str = "",
        session_id: str = "",
        task_id: str = "",
        platform: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_mode: str = "",
        api_call_count: Any = 0,
        api_request_id: str = "",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """``llm_request`` middleware entry point.

        Returns ``None`` to leave the request untouched, or ``{"request": {...}}`` with the
        rewritten provider kwargs.
        """
        try:
            settings = self._get_settings()
        except Exception as exc:  # noqa: BLE001
            logger.debug("jev-effort-router: settings unavailable (%s); leaving request untouched", exc)
            return None

        where = _Where(
            turn_id=turn_id,
            session_id=session_id,
            api_request_id=api_request_id,
            platform=platform,
            provider=provider,
            model=model,
        )

        if not settings.enabled:
            self._record_skip(settings, REASON_DISABLED, where)
            return None

        # -- skip gates: anything the router does not own goes out untouched ---------
        if not settings.routes(provider, base_url):
            self._record_skip(settings, REASON_SKIPPED_PROVIDER, where)
            return None
        if api_mode and api_mode not in ROUTED_API_MODES:
            self._record_skip(settings, REASON_SKIPPED_API_MODE, where, extra={"api_mode": api_mode})
            return None
        if settings.entry_for(model) is None:
            self._record_skip(settings, REASON_SKIPPED_MODEL, where)
            return None
        if not isinstance(request, dict) or not isinstance(original_request, dict):
            self._record_skip(settings, REASON_USER_DATA, where)
            return None

        try:
            memo, replayed = self._lookup(settings, turn_id, session_id)

            decision: Optional[Decision] = None
            if memo is not None:
                decision = _decision_from_memo(memo)
            else:
                fetched = self._decide(settings, request, where)
                if fetched is None:
                    return None
                decision = fetched
                # A decision naming a model the provider does not have is worse than no decision:
                # asking for it turns a degraded turn into a dead one (HTTP 404, no response).
                # Leave the request exactly as the operator configured it.
                if settings.checks_catalog(provider) and not self._provider_has(decision.model):
                    logger.warning(
                        "jev-effort-router: Jev chose %r, which is not in the provider's catalog; "
                        "leaving the turn on the configured model",
                        decision.model,
                    )
                    self._record_skip(settings, REASON_SKIPPED_UNKNOWN_MODEL, where,
                                      extra={"chosen_model": decision.model})
                    return None
                memo = Memo(
                    model=decision.model,
                    effort=decision.effort,
                    effort_requested=decision.effort_requested,
                    confidence=max(decision.model_confidence, decision.effort_confidence),
                    choice=decision.model_choice,
                    fallback_reasons=decision.fallback_reasons,
                )
                # Storing on "no memo found" rather than on an api_call_count test is deliberate:
                # Hermes passes the *incremented* counter, so the first provider call of a turn
                # arrives with api_call_count == 1 (see agent/turn_iteration_prep.py). Keying the
                # store off that value silently never stored anything, and every call inside a
                # turn's tool loop re-decided. Absence of a memo is the reliable "first call of
                # this turn" signal.
                if settings.route_per_turn:
                    self._memo.put_turn(turn_id, memo)
                else:
                    self._memo.put_session(session_id, memo)

            routed = self._apply(original_request, decision, settings.unknown_effort)
            self._record_route(settings, decision, where, api_call_count=api_call_count, replayed=replayed)
            return {"request": routed, "source": "jev-effort-router", "reason": decision.model}
        except Exception as exc:  # noqa: BLE001 - a router must never break a turn
            logger.warning(
                "jev-effort-router: routing failed (%s: %s); leaving request untouched",
                type(exc).__name__,
                exc,
            )
            self._record_skip(settings, REASON_EXCEPTION, where)
            return None

    # -- decision plumbing -------------------------------------------------------

    def _provider_has(self, model_id: str) -> bool:
        """Whether the chosen model is safe to put on the wire, according to the provider catalog.

        Answers ``True`` whenever the catalog has nothing to say — an absent cache, an unreadable
        file or a host that exposes none of it all mean "no evidence", and the router must not
        start refusing decisions on missing evidence. Only a catalog that lists models and does
        not contain this one is a No.
        """
        try:
            known = self._catalog.is_known(model_id)
        except Exception as exc:  # noqa: BLE001 - a catalog probe never blocks a turn
            logger.debug("jev-effort-router: catalog check failed (%s: %s)", type(exc).__name__, exc)
            return True
        return known is not False

    def _lookup(
        self, settings: Settings, turn_id: str, session_id: str
    ) -> Tuple[Optional[Memo], bool]:
        """The memoized decision for this request, and whether it is a replay.

        ``route_per_turn: true`` (the default) memoizes per turn, so the first request of a
        turn decides and the rest of the turn's tool loop replays it. ``false`` memoizes per
        session instead, which routes once per conversation.
        """
        if settings.route_per_turn:
            memo = self._memo.get_turn(turn_id)
            return memo, memo is not None
        memo = self._memo.get_session(session_id)
        if memo is not None:
            return memo, True
        # No session decision yet, but this turn may already have decided one: don't re-decide
        # inside the same turn's tool loop.
        memo = self._memo.get_turn(turn_id)
        return memo, memo is not None

    def _decide(self, settings: Settings, request: Dict[str, Any], where: "_Where") -> Optional[Decision]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages or not last_user_message(messages).strip():
            self._record_skip(settings, REASON_SKIPPED_NO_MESSAGE, where)
            return None

        try:
            decision, reason = self.client(settings).decide(
                messages,
                settings.grid,
                platform=where.platform,
                provider=where.provider,
            )
        except Exception as exc:  # noqa: BLE001 - belt and braces around the client
            logger.warning("jev-effort-router: decision call failed (%s); leaving request untouched", exc)
            self._record_skip(settings, REASON_EXCEPTION, where)
            return None

        if decision is None:
            if reason == REASON_NO_API_KEY:
                logger.info("jev-effort-router: OPENROUTER_API_KEY is not set; turns keep the configured model")
            self._record_skip(settings, reason or REASON_EXCEPTION, where)
            return None
        return decision

    def _apply(
        self, original_request: Dict[str, Any], decision: Decision, unknown_effort: str = "keep"
    ) -> Dict[str, Any]:
        """Build the rewritten request from the pre-middleware payload.

        Working from ``original_request`` keeps this router idempotent when more than one
        middleware rewrites the same call, and makes "the request is unchanged" literally true
        on the no-route path.

        Only the top-level ``reasoning_effort`` is written. ``reasoning_config`` is a
        *params-level* input that the provider profile consumes to derive that top-level field
        (``agent/transports/chat_completions.py::_build_kwargs_from_profile``, then
        ``plugins/model-providers/ollama-cloud/__init__.py`` which returns only
        ``reasoning_effort``). It is never a wire kwarg, and leaving it in the final payload
        makes Ollama reject the entire call with "Completions.create() got an unexpected keyword
        argument 'reasoning_config'" — a hard turn failure, not a degraded route.
        """
        routed = dict(original_request)
        routed["model"] = decision.model
        if decision.effort:
            routed[EFFORT_WIRE_KEY] = decision.effort
        elif family_for(decision.model) is not None or unknown_effort == "omit":
            routed.pop(EFFORT_WIRE_KEY, None)
        # else: no known effort family and ``unknown_effort: keep`` — the host's value stays.
        return routed

    # -- audit -------------------------------------------------------------------

    def _record_route(
        self,
        settings: Settings,
        decision: Decision,
        where: "_Where",
        *,
        api_call_count: Any = 0,
        replayed: bool = False,
    ) -> None:
        record: Dict[str, Any] = {
            "event": "route",
            "model": decision.model,
            "effort": decision.effort,
            "effort_requested": decision.effort_requested,
            "model_choice": decision.model_choice,
            "effort_choice": decision.effort_choice,
            "model_confidence": round(decision.model_confidence, 4),
            "effort_confidence": round(decision.effort_confidence, 4),
            "model_probabilities": decision.model_probabilities,
            "effort_probabilities": decision.effort_probabilities,
            "alternatives": list(decision.alternatives),
            "latency_ms": decision.latency_ms,
            "fallback_reasons": list(decision.fallback_reasons),
            "jev_model": settings.jev_model,
            "replayed": bool(replayed),
        }
        record.update(where.fields())
        record.update({key: value for key, value in {"api_call_count": api_call_count}.items() if value not in (None, "")})
        self.audit(settings).append(record)
        if decision.degraded:
            logger.info(
                "jev-effort-router: routed %s (effort %s) with degradation %s",
                decision.model,
                decision.effort,
                ",".join(decision.fallback_reasons),
            )
        else:
            logger.debug(
                "jev-effort-router: routed %s (effort %s) in %d ms",
                decision.model,
                decision.effort,
                decision.latency_ms,
            )

    def _record_skip(self, settings: Settings, reason: str, where: "_Where", *, extra: Optional[Dict[str, Any]] = None) -> None:
        if not settings.log_skips:
            return
        record: Dict[str, Any] = {
            "event": "skip",
            "reason": reason,
            "provider": where.provider,
            "configured_model": where.model,
        }
        record.update({key: value for key, value in (extra or {}).items() if value not in (None, "")})
        record.update(where.fields())
        self.audit(settings).append(record)

    # -- read-only surfaces used by the tools and commands -----------------------

    def tail(self, settings: Settings, limit: int = 10):
        return self.audit(settings).tail(limit)

    def count(self, settings: Settings) -> int:
        return self.audit(settings).count()

    def grid_report(self, settings: Settings) -> List[Dict[str, Any]]:
        """The grid annotated with what the provider catalog says about each entry.

        Empty when there is no catalog evidence, so a caller can tell "nothing to report" from
        "every entry is fine".
        """
        report: List[Dict[str, Any]] = []
        if not settings.checks_catalog(""):
            return report
        for entry in settings.grid:
            try:
                known = self._catalog.is_known(entry.model_id)
            except Exception:  # noqa: BLE001 - a status view never raises
                known = None
            if known is None:
                return []
            report.append({"model": entry.model_id, "available": bool(known)})
        return report

    def forget(self) -> None:
        self._memo.clear()


class _Where:
    """The identifiers a routing record is stamped with."""

    __slots__ = ("turn_id", "session_id", "api_request_id", "platform", "provider", "model")

    def __init__(self, *, turn_id, session_id, api_request_id, platform, provider, model) -> None:
        self.turn_id = turn_id
        self.session_id = session_id
        self.api_request_id = api_request_id
        self.platform = platform
        self.provider = provider
        self.model = model

    def fields(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "turn_id": self.turn_id,
                "session_id": self.session_id,
                "api_request_id": self.api_request_id,
                "platform": self.platform,
            }.items()
            if value
        }


def _decision_from_memo(memo: Memo) -> Decision:
    """Present a replayed memo through the same shape as a fresh decision."""
    return Decision(
        model=memo.model,
        effort=memo.effort,
        effort_requested=memo.effort_requested,
        model_confidence=memo.confidence,
        effort_confidence=memo.confidence,
        model_choice=memo.choice,
        effort_choice="",
        alternatives=(),
        latency_ms=0,
        fallback_reasons=memo.fallback_reasons,
    )
