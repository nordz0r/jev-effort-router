"""Context-window fit: keep a turn off a model whose window the prompt would overflow.

The prompt size is **estimated**, not tokenised: ``ceil(chars / 4)`` over the JSON of the
request's ``messages`` and ``tools`` (4 characters per token is the usual English/code rule of
thumb; no tokenizer is shipped because every upstream counts differently). A model fits when
``context > estimate + context_reserve_tokens``; a model without a declared ``context`` is
treated as fitting (unknown).

If the chosen model does not fit, the next *more capable* grid entry (grid order, least capable
first) that fits is used; then the first fitting ``long_context_models`` entry; else nothing —
and the router leaves the request untouched.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

from .grid import Entry

FIT_OK = "fits"
FIT_ESCALATED = "context_escalated"
FIT_LONG_CONTEXT = "context_long_model"
FIT_NONE = "context_no_fit"


def estimate_tokens(request: Dict[str, Any]) -> int:
    chars = 0
    for key in ("messages", "tools"):
        value = request.get(key) if isinstance(request, dict) else None
        if value:
            try:
                chars += len(json.dumps(value, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                chars += len(str(value))
    return int(math.ceil(chars / 4))


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
