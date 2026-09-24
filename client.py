"""The Jev client: one HTTP round-trip, no parsing of prose, total failure containment.

Every method returns a :class:`Decision` or ``None``. It never raises: a timeout, a 500, a
402, a malformed body, an unknown choice or a low-confidence answer all end as ``None`` plus
a reason string, so the caller's request goes out byte-identical to a plugin-disabled run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings, api_key, redact
from .effort import resolve_effort
from .grid import Entry, resolve
from .state import (
    EFFORT_QUESTION_ID,
    MODEL_QUESTION_ID,
    UNCLEAR_KEY,
    build_payload,
    normalise_effort_choice,
)

logger = logging.getLogger(__name__)

#: TypeSafe documents 429 (rate limit) and 529 (overloaded) as retry-after-backoff statuses.
RETRY_STATUSES = (429, 529)
RETRY_BACKOFF_S = 0.2

#: Reasons a turn was left unrouted. Stable strings — they are what the audit file records.
REASON_DISABLED = "disabled"
REASON_NO_API_KEY = "no_api_key"
REASON_SKIPPED_PROVIDER = "provider_not_routed"
REASON_SKIPPED_API_MODE = "api_mode_not_routed"
REASON_SKIPPED_MODEL = "model_not_in_grid"
REASON_SKIPPED_UNKNOWN_MODEL = "model_not_in_provider_catalog"
REASON_SKIPPED_NO_MESSAGE = "no_user_message"
REASON_SKIPPED_REPLAYED = "turn_already_routed"
REASON_UPSTREAM_ERROR = "upstream_error"
REASON_TIMEOUT = "timeout"
REASON_MALFORMED = "malformed_answer"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_UNKNOWN_CHOICE = "unknown_choice"
REASON_UNCLEAR = "unclear"
REASON_USER_DATA = "user_data"
REASON_EXCEPTION = "exception"


@dataclass(frozen=True)
class Decision:
    """A routed turn: the model and the effort that will be applied, plus the audit trail."""

    model: str
    effort: Optional[str]
    effort_requested: Optional[str]
    model_confidence: float
    effort_confidence: float
    model_choice: str
    effort_choice: str
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    effort_probabilities: Dict[str, float] = field(default_factory=dict)
    alternatives: Tuple[str, ...] = ()
    latency_ms: int = 0
    fallback_reasons: Tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return bool(self.fallback_reasons)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _as_probabilities(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): _as_float(value) for key, value in raw.items()}


def _answer(answers: Any, question_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(answers, dict):
        return None
    value = answers.get(question_id)
    return value if isinstance(value, dict) else None


def _choice_text(answer: Dict[str, Any]) -> str:
    """The chosen option, or ``""`` when the answer is not a Choice answer.

    The endpoint echoes the question type back. Anything other than ``choice`` (``noul``,
    ``score``) carries no ``choice`` field, and reading one anyway would put a probability or
    an empty string on the wire as if it were a model name. ``""`` falls into the off-grid
    path, which degrades to the configured defaults.
    """
    if answer_type(answer) != "choice":
        return ""
    return str(answer.get("choice") or "")


def answer_type(answer: Dict[str, Any]) -> str:
    """The question type the endpoint echoed on this answer, lower-cased.

    A missing ``type`` is treated as ``choice``: older/proxied responses may omit it, and a
    Choice answer is the only shape these two questions ever ask for.
    """
    return str(answer.get("type") or "choice").strip().lower() or "choice"


class JevClient:
    """Decision-endpoint client.

    ``transport`` is an injection seam for tests: anything exposing
    ``post(url, *, json, headers, timeout)`` returning an object with ``status_code``,
    ``json()`` and ``text``. When it is ``None``, ``httpx`` is imported lazily inside the
    call — never at import or registration time, because ``hermes plugins doctor`` blocks
    socket access while ``register(ctx)`` runs.
    """

    def __init__(self, settings: Settings, transport: Any = None) -> None:
        self._settings = settings
        self._transport = transport

    # -- transport ---------------------------------------------------------------

    def _post(self, payload: Dict[str, Any], key: str, timeout: float) -> Tuple[Optional[Any], Optional[str]]:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self._settings.backend == "openrouter":
            # Ranking header; optional, it makes the traffic identifiable on OpenRouter.
            headers["X-Title"] = "hermes-jev-effort-router"
        if self._transport is not None:
            response = self._transport.post(self._settings.endpoint, json=payload, headers=headers, timeout=timeout)
            return response, None

        import httpx  # lazy: no socket machinery while the plugin registers

        with httpx.Client(timeout=timeout) as client:
            return client.post(self._settings.endpoint, json=payload, headers=headers, timeout=timeout), None

    # -- routing -----------------------------------------------------------------

    def decide(
        self,
        messages: Sequence[Dict[str, Any]],
        grid: Sequence[Entry],
        *,
        platform: str = "",
        provider: str = "",
        effort_families: bool = True,
    ) -> Tuple[Optional[Decision], Optional[str]]:
        """Ask Jev for a model and an effort level.

        Returns ``(decision, None)`` on a usable answer and ``(None, reason)`` otherwise. A
        partially usable answer still yields a decision: a confident model with an unconfident
        effort degrades only the effort, and vice versa.

        Note the absence of a ``current_model`` argument. It used to be forwarded into the
        request `state`, and measurement against the live endpoint showed it anchors Jev on the
        model the operator already has configured — a code-debugging task switched from
        ``kimi-k3`` (4/4 calls without it) to ``deepseek-v4.1-flash`` (4/4 calls with it).
        Telling Jev what is configured defeats the purpose of asking it.
        """
        settings = self._settings
        key = api_key(settings)
        if not key:
            return None, REASON_NO_API_KEY

        payload = build_payload(
            messages,
            grid,
            jev_model=settings.jev_model,
            context_turns=settings.context_turns,
        )

        started = time.monotonic()
        response = None
        for attempt in (0, 1):
            remaining = settings.timeout_s - (time.monotonic() - started)
            try:
                response, _ = self._post(payload, key, max(remaining, 0.05))
            except Exception as exc:  # noqa: BLE001 - a router must never break a turn
                latency_ms = int((time.monotonic() - started) * 1000)
                name = type(exc).__name__
                timeout_types = ("Timeout", "ReadTimeout", "ConnectTimeout", "PoolTimeout", "TimeoutException")
                if any(token in name for token in timeout_types):
                    logger.warning("jev-effort-router: Jev timed out after %d ms (%s)", latency_ms, name)
                    return None, REASON_TIMEOUT
                logger.warning("jev-effort-router: Jev call failed: %s: %s", name, redact(exc, 300, settings))
                return None, REASON_UPSTREAM_ERROR
            # 429 rate limit / 529 overloaded: one short back-off retry, only inside the budget.
            remaining = settings.timeout_s - (time.monotonic() - started)
            if attempt == 0 and getattr(response, "status_code", None) in RETRY_STATUSES and remaining > 2 * RETRY_BACKOFF_S:
                time.sleep(RETRY_BACKOFF_S)
                continue
            break
        latency_ms = int((time.monotonic() - started) * 1000)

        status = getattr(response, "status_code", None)
        if status is None or not (200 <= int(status) < 300):
            body = redact(getattr(response, "text", ""), 300, settings)
            logger.warning("jev-effort-router: Jev returned HTTP %s: %s", status, body)
            return None, REASON_UPSTREAM_ERROR

        try:
            data = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON body is just another failure
            logger.warning("jev-effort-router: Jev returned a non-JSON body: %s",
                           redact(getattr(response, "text", ""), 300, settings))
            return None, REASON_MALFORMED

        return self._interpret(data, grid, latency_ms=latency_ms, effort_families=effort_families)

    def fallback(self, reason: str, *, effort_families: bool = True) -> Optional[Decision]:
        """The decision when Jev is unavailable (timeout, HTTP error, malformed body): the
        configured ``default_model`` at ``default_effort``, or ``None`` (request untouched)
        when no fallback is on the grid."""
        settings = self._settings
        chosen = settings.entry_for(settings.default_model)
        if chosen is None:
            return None
        return Decision(
            model=chosen.model_id,
            effort=resolve_effort(chosen.model_id, settings.default_effort, settings.unknown_effort, effort_families),
            effort_requested=settings.default_effort,
            model_confidence=0.0,
            effort_confidence=0.0,
            model_choice="",
            effort_choice="",
            fallback_reasons=(reason,),
        )

    def _interpret(
        self, data: Any, grid: Sequence[Entry], *, latency_ms: int, effort_families: bool = True
    ) -> Tuple[Optional[Decision], Optional[str]]:
        settings = self._settings
        if not isinstance(data, dict):
            return None, REASON_MALFORMED
        answers = data.get("answers")
        model_answer = _answer(answers, MODEL_QUESTION_ID)
        effort_answer = _answer(answers, EFFORT_QUESTION_ID)
        if model_answer is None and effort_answer is None:
            return None, REASON_MALFORMED

        reasons: List[str] = []

        # -- model -----------------------------------------------------------------
        chosen: Optional[Entry] = None
        model_confidence = 0.0
        model_choice_text = ""
        model_probabilities: Dict[str, float] = {}
        if model_answer is None:
            reasons.append(REASON_MALFORMED)
        else:
            model_choice_text = _choice_text(model_answer)
            model_confidence = _as_float(model_answer.get("confidence"))
            model_probabilities = _as_probabilities(model_answer.get("probabilities"))
            if model_choice_text.strip().lower() == UNCLEAR_KEY:
                reasons.append(REASON_UNCLEAR)
            else:
                chosen = resolve(_criteria_map(model_answer), model_choice_text, grid)
                if chosen is None:
                    reasons.append(REASON_UNKNOWN_CHOICE)
                elif model_confidence < settings.confidence_threshold:
                    # Below threshold: never downgrade. Take the more capable of the choice and
                    # the fallback, by grid position (grids are listed least capable first).
                    reasons.append(REASON_LOW_CONFIDENCE)
                    fallback = settings.entry_for(settings.default_model)
                    chosen = _more_capable(grid, chosen, fallback) if fallback else None

        if chosen is None:
            chosen = settings.entry_for(settings.default_model)
            if chosen is None:
                # No usable model answer and no fallback on the grid (the default): keep the
                # configured model — a distrusted decision must never silently downgrade it.
                return None, reasons[0] if reasons else REASON_LOW_CONFIDENCE
        model = chosen.model_id

        # -- effort ----------------------------------------------------------------
        # A usable-but-distrusted effort answer (low confidence, off-vocabulary choice)
        # degrades to the operator's configured default. An effort answer that is simply
        # absent carries nothing to degrade *to*: inventing a level for a model whose ladder
        # we have no evidence for is exactly what `effort.py` exists to avoid, so the field
        # is omitted and the model's own default applies. The route still happens either way.
        requested: Optional[str] = None
        effort_confidence = 0.0
        effort_choice_text = ""
        effort_probabilities: Dict[str, float] = {}
        answered = False
        if effort_answer is None:
            reasons.append(REASON_MALFORMED)
        elif answer_type(effort_answer) != "score":
            reasons.append(REASON_MALFORMED)
        else:
            answered = True
            score = effort_answer.get("score")
            effort_choice_text = str(score)
            effort_confidence = _as_float(effort_answer.get("confidence"))
            effort_probabilities = _as_probabilities(effort_answer.get("probabilities"))
            candidate = normalise_effort_choice(score if isinstance(score, (int, float)) else None)
            if candidate is None:
                reasons.append(REASON_UNKNOWN_CHOICE)
            elif effort_confidence < settings.confidence_threshold:
                reasons.append(REASON_LOW_CONFIDENCE)
            else:
                requested = candidate

        if requested is None and answered:
            requested = settings.default_effort

        effort = resolve_effort(model, requested, settings.unknown_effort, effort_families)

        alternatives = tuple(
            entry.model_id for entry in grid if entry.model_id != model
        )
        return (
            Decision(
                model=model,
                effort=effort,
                effort_requested=requested,
                model_confidence=model_confidence,
                effort_confidence=effort_confidence,
                model_choice=model_choice_text,
                effort_choice=effort_choice_text,
                model_probabilities=model_probabilities,
                effort_probabilities=effort_probabilities,
                alternatives=alternatives,
                latency_ms=latency_ms,
                # Deliberately not de-duplicated: the model and the effort degrade
                # independently, so "low_confidence" twice means both were degraded.
                fallback_reasons=tuple(reasons),
            ),
            None,
        )


def _more_capable(grid: Sequence[Entry], first: Entry, second: Entry) -> Entry:
    """The later of two entries in grid order (the grid lists tiers least capable first)."""
    order = {entry.model_id: index for index, entry in enumerate(grid)}
    return max((first, second), key=lambda entry: order.get(entry.model_id, -1))


def _criteria_map(model_answer: Dict[str, Any]) -> Dict[str, str]:
    """The criteria the endpoint was given, when it echoes them back with the answer."""
    echoed = model_answer.get("criteria")
    if isinstance(echoed, dict):
        return {str(key): str(value) for key, value in echoed.items()}
    return {}
