"""The grid, the keyed criteria and the choice→model mapping."""

from __future__ import annotations

from grid import DEFAULT_GRID, Entry, criteria, parse_entry, parse_grid, resolve


def test_default_grid_is_the_six_benchmarked_models():
    ids = [entry.model_id for entry in DEFAULT_GRID]
    assert ids == [
        "deepseek-v4.1-flash",
        "kimi-k3",
        "glm-5.3",
        "glm-5.3-flash",
        "minimax-m3",
        # The wire id, tag included: the provider's catalog names this tier
        # "nemotron-3-nano:30b" and rejects the bare name with HTTP 404.
        "nemotron-3-nano:30b",
    ]


def test_criteria_are_keyed_by_position_with_the_task_description_only():
    mapping = criteria(DEFAULT_GRID)
    assert set(mapping) == {"1", "2", "3", "4", "5", "6"}
    # 0.3: the model id is not sent to Jev; code maps the positional key back to it.
    assert mapping["1"] == (
        "the usual choice for general work: everyday writing, "
        "explanation, summarising, ordinary coding and tool use; 1M context; cheap for its size"
    )
    assert all(entry.model_id not in mapping[str(i)] for i, entry in enumerate(DEFAULT_GRID, start=1))


def test_no_profile_is_a_task_free_superlative():
    """Every criterion must name a task family, not just praise the model.

    This is the rule the GLM under-routing came from: "excellent value for money" and
    "excellent in real use for everyday tasks" attached no task to the praise, so they read as
    safe picks on every prompt and the first-listed model absorbed the GLMs' decisions. See the
    changelog measurement.
    """
    banned = ("excellent", "top-tier", "best-in-class", "state of the art", "powerful")
    for entry in DEFAULT_GRID:
        lowered = entry.description.lower()
        for word in banned:
            assert word not in lowered, f"{entry.model_id} praises without naming a task: {word}"


def test_resolve_by_positional_key():
    assert resolve({}, "3", DEFAULT_GRID).model_id == "glm-5.3"


def test_resolve_tolerates_an_echoed_option_string():
    # A decision endpoint that echoes "1: deepseek-v4.1-flash: ..." must not misroute.
    assert resolve({}, "2: kimi-k3: top-tier code and agentic work", DEFAULT_GRID).model_id == "kimi-k3"
    assert resolve({}, "kimi-k3", DEFAULT_GRID).model_id == "kimi-k3"


def test_resolve_rejects_an_off_grid_choice():
    assert resolve({}, "9", DEFAULT_GRID) is None
    assert resolve({}, "gpt-9-ultra", DEFAULT_GRID) is None
    assert resolve({}, "", DEFAULT_GRID) is None
    assert resolve({}, None, DEFAULT_GRID) is None


def test_parse_entry_accepts_string_and_mapping():
    assert parse_entry("a-model: does a thing") == Entry("a-model", "does a thing")
    assert parse_entry("bare-model") == Entry("bare-model", "")
    assert parse_entry({"model_id": "m1", "description": "d1"}) == Entry("m1", "d1")
    assert parse_entry("") is None
    assert parse_entry(None) is None


def test_parse_grid_falls_back_to_default_on_nonsense():
    assert parse_grid(None) == DEFAULT_GRID
    assert parse_grid([]) == DEFAULT_GRID
    assert parse_grid([""]) == DEFAULT_GRID
    assert parse_grid("   \n  ") == DEFAULT_GRID
    assert parse_grid(42) == DEFAULT_GRID


def test_parse_grid_honours_an_override_and_dedupes():
    grid = parse_grid(["alpha: first", "beta: second", "alpha: duplicate"])
    assert [entry.model_id for entry in grid] == ["alpha", "beta"]


def test_parse_grid_accepts_a_multiline_string():
    grid = parse_grid("alpha: first\nbeta: second")
    assert [entry.model_id for entry in grid] == ["alpha", "beta"]


def test_infinite_context_does_not_crash():
    """YAML ``context: .inf`` becomes float('inf'); int() would OverflowError without a catch."""
    entry = parse_entry({"id": "m", "description": "d", "context": float("inf")})
    assert entry is not None and entry.context is None
    assert parse_entry({"id": "m", "description": "d", "context": float("-inf")}).context is None
    assert parse_entry({"id": "m", "description": "d", "context": float("nan")}).context is None
