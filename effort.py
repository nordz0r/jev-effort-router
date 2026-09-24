"""Per-family reasoning-effort translation.

Hermes carries a wide internal ladder (``none`` … ``ultra``) and clamps it per provider.
The router only ever asks for ``low``/``medium``/``high``, so the job here is narrower:
land the chosen level on the vocabulary the *selected model's* wire really accepts, never
escalate, and omit the field rather than send a level the route would reject with a 400.

The vocabularies below mirror ``agent/reasoning_effort.py`` in hermes-agent, which is the
reference implementation of the clamping rule ("nearest weaker level, never stronger").
`docs/routing-grid.md` records where each row came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

#: Hermes' ladder, low to high. Used only for nearest-weaker arithmetic.
LADDER: Tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


@dataclass(frozen=True)
class Family:
    """One model family's effort contract on its wire."""

    name: str
    accepted: Tuple[str, ...]
    overrides: Dict[str, str]


#: Ollama:cloud /v1/chat/completions. ``max`` is undocumented but real; ``minimal`` 400s.
OLLAMA_CLOUD = Family(
    name="ollama-cloud",
    accepted=("none", "low", "medium", "high", "max"),
    overrides={"xhigh": "max"},
)

FAMILIES: Tuple[Family, ...] = (
    # DeepSeek V4 / V4.1 — graded, xhigh rounds up to max.
    Family("deepseek", ("low", "medium", "high", "max"), {"xhigh": "max"}),
    # Kimi K3 — documented set is low/high/max; `high` is its positional middle, so a
    # `medium` request maps onto `high` rather than down to `low`.
    Family("kimi-k3", ("low", "high", "max"), {"medium": "high", "xhigh": "max"}),
    # GLM-5.3 — graded low/medium/high/max, live-verified monotonic.
    Family("glm-53", ("low", "medium", "high", "max"), {"xhigh": "max"}),
    # GLM-5.2 — the older knob only has high and max.
    Family("glm-52", ("high", "max"), {"xhigh": "max"}),
    # MiniMax M3 — no narrower vocabulary published; the Ollama:cloud set applies.
    Family("minimax", OLLAMA_CLOUD.accepted, dict(OLLAMA_CLOUD.overrides)),
    # Nemotron 3 Nano — same, and `low` is the honest landing for its "simple tasks" profile.
    Family("nemotron", OLLAMA_CLOUD.accepted, dict(OLLAMA_CLOUD.overrides)),
)


def family_for(model_id: Optional[str]) -> Optional[Family]:
    """Pick the family contract for a model id, or ``None`` when there is none to apply.

    Matching is a substring test on the bare slug, so ``kimi-k3``, ``kimi-k3-256k`` and a
    ``vendor/kimi-k3`` prefix all land on the same row. An unknown bare id is an Ollama:cloud
    model and gets that vocabulary, the widest set that provider accepts. An unknown
    ``provider/model`` id and any ``combo/<id>`` (a combo can be any model) have no known
    contract: ``None``, and the caller decides via ``unknown_effort``. Only consulted for
    Ollama:cloud routes (see :func:`resolve_effort`).
    """
    raw = (model_id or "").strip().lower()
    if raw.startswith("combo/"):
        return None
    slug = raw.rsplit("/", 1)[-1]
    if not slug:
        return OLLAMA_CLOUD
    if "kimi" in slug and _token(slug, "k3"):
        return FAMILIES[1]
    if "deepseek" in slug:
        return FAMILIES[0]
    if slug.startswith("glm"):
        return FAMILIES[3] if _token(slug, "5.2") else FAMILIES[2]
    if "minimax" in slug:
        return FAMILIES[4]
    if "nemotron" in slug:
        return FAMILIES[5]
    return None if "/" in raw else OLLAMA_CLOUD


def _token(slug: str, needle: str) -> bool:
    """Boundary-ish containment so ``k2.6`` never matches ``k3`` and ``5.3`` never ``5.2``."""
    return needle in slug.replace("-", " ").replace("_", " ").split() or needle in slug


def clamp(
    effort: Optional[str],
    accepted: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested level onto ``accepted``; ``None`` means "omit the field".

    Order: declared override, verbatim when accepted, else the nearest **weaker** level, and
    the provider's floor when nothing weaker exists. A level that is not a ladder rung at all
    (a bespoke provider vocabulary) is passed through untouched.
    """
    requested = str(effort or "").strip().lower()
    if not requested:
        return None

    normalised = [level for level in (str(item).strip().lower() for item in accepted) if level in LADDER]
    if not normalised:
        return requested
    if requested in normalised:
        return requested
    if overrides and overrides.get(requested) in normalised:
        return overrides[requested]
    if requested not in LADDER:
        return requested

    # `none` disables reasoning outright — never a landing target for an enabled request.
    candidates = [level for level in normalised if level != "none"]
    if not candidates:
        return requested
    request_index = LADDER.index(requested)
    weaker = [level for level in candidates if LADDER.index(level) < request_index]
    if weaker:
        return max(weaker, key=LADDER.index)
    return min(candidates, key=LADDER.index)


def resolve_effort(
    model_id: Optional[str], effort: Optional[str], unknown: str = "omit", families: bool = True
) -> Optional[str]:
    """The exact value to put on the wire for ``model_id``, or ``None`` to not write the field.

    ``families=False`` (every non-Ollama route) skips the table above: its vocabularies were
    taken from Ollama:cloud's wire and say nothing about another upstream. With no family,
    only ``unknown="pass"`` writes a value (the requested level, verbatim); ``omit``/``keep``
    return ``None`` and the router applies the choice to the host's value.
    """
    family = family_for(model_id) if families else None
    if family is None:
        return (str(effort or "").strip().lower() or None) if unknown == "pass" else None
    return clamp(effort, family.accepted, family.overrides)


def is_omitted(model_id: Optional[str], effort: Optional[str], unknown: str = "omit", families: bool = True) -> bool:
    """True when the router would not write a reasoning-effort field."""
    return resolve_effort(model_id, effort, unknown, families) is None
