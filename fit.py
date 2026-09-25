"""Context-window fit: keep a turn off a model whose window the prompt would overflow.

The prompt size is **estimated**, not tokenised: ``ceil(chars / 4)`` over the JSON of the
request's ``messages`` and ``tools`` (4 characters per token is the usual English/code rule of
thumb; no tokenizer is shipped because every upstream counts differently). A model fits when
``context > estimate + context_reserve_tokens``; a model without a declared ``context`` is
treated as fitting (unknown).

Image content parts (OpenAI-style ``{"type": "image_url", ...}`` and similar) are **not**
counted as text: each such part contributes a fixed ``IMAGE_TOKEN_COST`` (1000) instead of
the base64 payload length. A ~1MB data-URL image must not inflate the estimate by hundreds of
thousands of tokens.

If the chosen model does not fit, the next *more capable* grid entry (grid order, least capable
first) that fits is used; then the first fitting ``long_context_models`` entry; else nothing —
and the router leaves the request untouched.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from .grid import Entry

FIT_OK = "fits"
FIT_ESCALATED = "context_escalated"
FIT_LONG_CONTEXT = "context_long_model"
FIT_NONE = "context_no_fit"

# Fixed charge per multimodal image part; base64 payloads are stripped from the char count.
IMAGE_TOKEN_COST = 1000
_IMAGE_TYPES = frozenset({"image_url", "image", "input_image"})


def _is_image_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    kind = str(part.get("type") or "").strip().lower()
    return kind in _IMAGE_TYPES or "image_url" in part


def _scrub_messages(messages: Any) -> Tuple[Any, int]:
    """Return messages safe to JSON-dump for sizing, plus image-part count.

    Replaces each image content part with a tiny stub so base64 data-URLs do not dominate
    ``ceil(chars / 4)``.
    """
    if not isinstance(messages, list):
        return messages, 0
    images = 0
    scrubbed: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            scrubbed.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            scrubbed.append(message)
            continue
        new_parts: List[Any] = []
        changed = False
        for part in content:
            if _is_image_part(part):
                images += 1
                changed = True
                new_parts.append({"type": "image_url", "image_url": {"url": "[image]"}})
            else:
                new_parts.append(part)
        if changed:
            scrubbed.append({**message, "content": new_parts})
        else:
            scrubbed.append(message)
    return scrubbed, images


def estimate_tokens(request: Dict[str, Any]) -> int:
    chars = 0
    images = 0
    for key in ("messages", "tools"):
        value = request.get(key) if isinstance(request, dict) else None
        if not value:
            continue
        if key == "messages":
            value, images = _scrub_messages(value)
        try:
            chars += len(json.dumps(value, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            chars += len(str(value))
    return int(math.ceil(chars / 4)) + images * IMAGE_TOKEN_COST


def fits(entry: Entry, estimate: int, reserve: int) -> bool:
    return entry.context is None or entry.context > estimate + reserve


def choose(settings: Any, model_id: str, estimate: int) -> Tuple[Optional[Entry], str]:
    """The entry to use for ``model_id`` given the estimate, and how it was reached."""
    reserve = settings.context_reserve_tokens
    grid = list(settings.grid)
    index = next((i for i, entry in enumerate(grid) if entry.model_id == model_id), None)
    current = grid[index] if index is not None else None
    if current is None:
        current = next((settings.long_context_entry(item) for item in settings.long_context_models
                        if item.model_id == model_id), None)
    if current is None or fits(current, estimate, reserve):
        return current, FIT_OK
    if index is not None:
        for entry in grid[index + 1:]:
            if fits(entry, estimate, reserve):
                return entry, FIT_ESCALATED
    for item in settings.long_context_models:
        entry = settings.long_context_entry(item)
        if entry.model_id != model_id and fits(entry, estimate, reserve):
            return entry, FIT_LONG_CONTEXT
    return None, FIT_NONE
