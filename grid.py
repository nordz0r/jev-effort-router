"""The routing grid: the models Jev may choose between, and the effort vocabulary.

A Choice question's option list is load-bearing. Every extra option measurably dilutes the
decision, so the built-in grid is exactly the six models that were benchmarked for this
profile — see ``docs/routing-grid.md``. Operators can replace it through the ``grid``
setting, which is the supported way to add a model once it has evidence behind it.

**The descriptions below are the criteria strings sent to Jev verbatim, and they are sent in
English.** They are part of the measured payload, not documentation: translating them changes
what Jev weighs, which is why the grid is stated in one language and left alone. Rewording
them (or flipping the language back) is a behaviour change that needs its own measurement.

**A model id here is the string that goes on the wire, verbatim.** ``nemotron-3-nano`` sat in
this table while the provider's catalog named the tier ``nemotron-3-nano:30b``, so a perfectly
confident decision became ``HTTP 404: model ... not found`` and killed the turn. ``catalog.py``
now cross-checks the grid against the provider's own model list before a decision is acted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Entry:
    """One choice on the grid."""

    #: The model id as it must appear in the provider's ``model`` field.
    model_id: str
    #: Human-readable profile line, sent to Jev as the criterion description.
    description: str

    @property
    def criterion(self) -> str:
        """The exact string Jev sees for this option: the task description only. The model id
        is not sent — Jev judges the task, and code maps the chosen option back to the id."""
        return self.description or self.model_id


#: The six benchmarked Ollama:cloud models, in the order they are offered to Jev.
#: The descriptions are data: they are sent to Jev verbatim as the criteria, in English.
#:
#: Every profile names a **task family**, never a superlative free of a task ("excellent value
#: for money", "excellent in real use"). Those lines were why `glm-5.3` and `glm-5.3-flash` were
#: almost never chosen: a superlative with no task attached reads as a safe pick on every prompt,
#: and the model carrying one (deepseek, first on the list) absorbed the decisions that belonged to
#: the GLMs. The measured effect of naming task families instead is in the changelog.
DEFAULT_GRID: Tuple[Entry, ...] = (
    Entry(
        "deepseek-v4.1-flash",
        "the usual choice for general work: everyday writing, explanation, summarising, "
        "ordinary coding and tool use; 1M context; cheap for its size",
    ),
    Entry(
        "kimi-k3",
        "strongest at complex code and long agentic tasks: multi-file refactors, deep "
        "debugging, large repositories; slow and the most expensive",
    ),
    Entry(
        "glm-5.3",
        "strongest at rigorous reasoning: mathematics, logic, science, quantitative and "
        "financial analysis, where a wrong answer is costly",
    ),
    Entry(
        "glm-5.3-flash",
        "best reasoning-per-cost on large text: drafting, summarising, translating and "
        "structured extraction over long documents; fast",
    ),
    Entry(
        "minimax-m3",
        "fast tool calling: long sequences of API/CLI actions, repetitive automation, "
        "high throughput",
    ),
    Entry(
        "nemotron-3-nano:30b",
        "highest throughput and lowest cost: trivial single-step requests only; weak at "
        "reasoning and at long context",
    ),
)


def valid_model_id(model_id: str) -> bool:
    """No empty path segment: ``combo/``, ``moonshot/``, ``/x`` and ``a//b`` are not ids."""
    return bool(model_id) and all(segment.strip() for segment in model_id.split("/"))


def parse_entry(raw: Any) -> Optional[Entry]:
    """Parse ``"model-id: description"``, a ``{model_id, description}`` mapping, or a one-key
    ``{model-id: description}`` mapping (what unquoted YAML ``- combo/fast: trivial`` yields).

    A string splits on ``": "``; failing that, on the first ``":"`` only when the rest reads
    like a description (has a space), so ``openrouter/qwen/qwen3-coder:free`` stays one id.
    """
    if isinstance(raw, dict):
        if any(key in raw for key in ("model_id", "model", "id")):
            model_id = str(raw.get("model_id") or raw.get("model") or raw.get("id") or "").strip()
            description = str(raw.get("description") or raw.get("profile") or "").strip()
        elif len(raw) == 1:
            key, value = next(iter(raw.items()))
            model_id, description = str(key or "").strip(), str(value or "").strip()
        else:
            return None
        return Entry(model_id, description) if valid_model_id(model_id) else None
    text = str(raw or "").strip()
    if not text:
        return None
    if ": " in text:
        model_id, _, description = text.partition(": ")
    else:
        head, _, tail = text.partition(":")
        model_id, description = (head, tail) if " " in tail.strip() else (text, "")
    model_id = model_id.strip()
    if not valid_model_id(model_id):
        return None
    return Entry(model_id, description.strip())


def parse_grid(raw: Any) -> Tuple[Entry, ...]:
    """Resolve the effective grid, falling back to :data:`DEFAULT_GRID` on nonsense input.

    A partial or unparseable override is not a reason to route blindly: entries that do not
    parse are dropped, and an override that yields nothing at all falls back to the default.
    """
    if raw is None:
        return DEFAULT_GRID
    items: Iterable[Any]
    if isinstance(raw, str):
        items = [line for line in raw.splitlines() if line.strip()]
    elif isinstance(raw, Sequence):
        items = raw
    else:
        return DEFAULT_GRID

    parsed: List[Entry] = []
    seen = set()
    for item in items:
        entry = parse_entry(item)
        if entry is None or entry.model_id in seen:
            continue
        seen.add(entry.model_id)
        parsed.append(entry)
    return tuple(parsed) if parsed else DEFAULT_GRID


def criteria(grid: Sequence[Entry]) -> dict:
    """The ``criteria`` map for the model Choice question.

    Keyed by position (``"1"``, ``"2"``, ...) rather than by model id: Jev returns the
    criterion *key*, so a stable key decouples the answer from the description text and a
    description can be rewritten without silently changing which model is selected.
    """
    return {str(index): entry.criterion for index, entry in enumerate(grid, start=1)}


def resolve(criteria_map: dict, choice: Any, grid: Sequence[Entry]) -> Optional[Entry]:
    """Map Jev's ``choice`` back to a grid entry, or ``None`` when it is not on the grid.

    Accepts the positional key (the documented form), a ``"<index>: ..."`` key, or a raw
    model id: a decision endpoint that echoes the option text instead of the key must not
    turn into a silent misroute.
    """
    if choice is None:
        return None
    key = str(choice).strip()
    if not key:
        return None

    by_position = {str(index): entry for index, entry in enumerate(grid, start=1)}
    if key in by_position:
        return by_position[key]

    head = key.split(":", 1)[0].strip()
    if head in by_position:
        return by_position[head]
    for candidate in (key, head):
        for entry in grid:
            if entry.model_id == candidate:
                return entry

    # Last resort: the echoed option string matches a criterion we sent.
    for position, text in (criteria_map or {}).items():
        if text == key:
            return by_position.get(str(position))
    return None
