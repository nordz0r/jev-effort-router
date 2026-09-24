# jev-effort-router

**Per-turn model *and* reasoning-effort routing for [Hermes Agent](https://github.com/NousResearch/hermes-agent)** —
both halves decided by [TypeSafe Jev](https://openrouter.ai/typesafe/jev-1.13), a "System One" decision
model, not an LLM.

Hermes lets you set a reasoning effort. It does not let you set a *different* one on the next turn — the
level is a session setting, so a two-line question and a gnarly refactor both pay whichever effort you
last chose. This plugin makes it per-turn: Jev reads the task, picks the model and the effort for that
turn, and the request goes out rewritten.

**That combination is the whole point, and it is why this plugin exists separately from the others.**
Every other routing entry in the catalog picks a **model** and stops there. The only entries that touch
reasoning effort make you move it **by hand** from the status bar. Nothing else decides both
automatically, per turn.

> **ocx fork (`0.3.0+ocx`).** This fork also routes any OpenAI-compatible provider you name — e.g. an
> OpenCodex (ocx) endpoint configured as a Hermes `custom_providers` entry — and calls Jev on TypeSafe's
> native API by default. See [Routing on an OpenCodex (ocx) endpoint](#routing-on-an-opencodex-ocx-endpoint).
> The Ollama:Cloud text below describes the default configuration.

**Scope: this plugin is for Hermes running on Ollama:Cloud.** Its whole routing grid is six Ollama:Cloud
models, the per-family effort table is written for Ollama:Cloud's reasoning-effort vocabulary, and the
middleware routes **only** the `ollama-cloud` provider — every other provider passes through untouched, so
installing it on another provider changes nothing. You need Hermes on `provider: ollama-cloud`, with that
provider's catalog reachable, for this plugin to have any effect.

Hermes normally runs one model at one reasoning-effort for a whole session. `jev-effort-router` asks Jev —
on every user turn, in ~270 ms and at $0.042/M input tokens — which of six benchmarked Ollama:cloud models
and which effort level fit the task, then rewrites the outgoing provider request accordingly.

Jev writes nothing. The selected model still does all the reasoning and all the generation; Jev only steers.

```
User input
   │
   ▼
llm_request middleware  ──►  ask Jev (choice: model_route, choice: reasoning_effort)
   │                              │
   │                              ▼
   │                       confidence guard (default 0.5)
   ▼                              │
provider request rewritten: model + reasoning_effort
   │
   ▼
main model reasons and answers
```

## Install

```bash
hermes plugins install AlphaPerseii3000/jev-effort-router
hermes plugins enable jev-effort-router
```

Hermes scans a community plugin on install and **blocks this one by default** — verified on 0.21.4, not
assumed:

```text
Decision: BLOCKED — Blocked (community source + caution verdict, 4 findings). Use --force to override.
```

There are two layers and they disagree, which is worth knowing before you file a bug:

- `hermes plugins validate <dir>` reports **one** finding: `caution`, `context_exfil`
  (`docs/jev-decisions-api.md:46`).
- `hermes plugins install` reports **four**: that same `HIGH exfiltration` finding, plus three
  `MEDIUM supply_chain` findings for `pip install` lines in `.github/workflows/tests.yml` and the
  README's own install snippet.

Neither the three MEDIUM findings nor the HIGH one are a defect to fix here: the supply-chain ones are
text matches on the words `pip install` in documentation and CI — the install path itself never executes
them — and the exfiltration finding is the plugin's documented purpose, because routing necessarily
sends a bounded slice of conversation context to an external decision API. The scanner does not
distinguish a docstring describing a network call from a hostile one, so the verdict is expected.

Review the findings above and re-run with `--force` to accept them:

```bash
hermes plugins install AlphaPerseii3000/jev-effort-router --force
```

To skip the scan entirely for this repository, set `plugins.scan_on_install` in your Hermes config.

Then set the key Jev is reached with. The default backend is TypeSafe's native API
(`https://api.typesafe.ai/v1/systemone`, model `jev-1.13.0`); `backend: openrouter` uses OpenRouter's
Decisions API (`typesafe/jev-1.13`) instead:

```bash
# in the Hermes .env file (the profile's .env under multiplexing)
TYPESAFE_API_KEY=...          # backend: typesafe (default)
# OPENROUTER_API_KEY=...      # backend: openrouter
```

The key is read through Hermes' `agent.secret_scope.get_secret(name)`. Under gateway multiplexing each
profile's `.env` is installed as a per-request secret scope, so every profile uses **its own** key;
in that mode `os.environ` is deliberately *not* a fallback, and an unscoped read makes the router inert
for that request rather than borrowing another profile's key. `api_key_env` names a different variable.

Restart the session. That is the whole setup: the six-model grid ships as the default.

Requirements: **Hermes Agent on Ollama:Cloud** (`provider: ollama-cloud` — no other provider is routed),
with the `llm_request` middleware kind (verified on 0.21.2 / v2026.9.11; the catalog lists 0.21.4), Python 3.11+, and TypeSafe (or OpenRouter) credits.

## Verify it is working

In a session:

```
/jev-effort-router status
/jev-effort-router route refactor the payment module
```

From the shell:

```bash
hermes jev-effort-router status
hermes jev-effort-router route "summarise this changelog"
hermes jev-effort-router grid
hermes jev-effort-router tail 20      # last audit records as JSON
hermes jev-effort-router reset        # drop memoised decisions
```

`status` reports whether routing is enabled, whether the key is present, the grid, and the most recent
decisions. `route` exercises Jev end-to-end without running a turn, and exits non-zero when routing fails.

It also reports `grid_coverage` over the last 200 routed turns, which is how you notice a model the
router is *not* using:

```
"grid_coverage": {
  "window": 38,
  "applied": { "deepseek-v4.1-flash": 5, "kimi-k3": 3, "glm-5.3-flash": 1 },
  "never_chosen": ["glm-5.3", "minimax-m3", "nemotron-3-nano:30b"],
  "below_threshold": { "kimi-k3": 8, "minimax-m3": 7 }
}
```

`never_chosen` means Jev never picks that model on this workload — fix the criterion wording in the grid.
`below_threshold` means Jev picks it but the answer does not clear `confidence_threshold`, so the turn is
served by the fallback model instead — fix the wording or the threshold. Only first-call records count, so
a turn that replays one decision across a long tool loop is not counted many times.

Two agent-facing tools are registered as well: `jev_effort_router_status` and `jev_effort_router_route`.

## Configuration

All settings live under `plugins.entries.jev-effort-router.settings` and are editable from the Desktop settings
form generated from `plugin.yaml`.

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch. Off registers the plugin but never rewrites a request. |
| `mode` | `route` | `route` rewrites the request. `shadow` asks Jev and writes the decision to the audit log (`event: "shadow"`) but never changes the request. |
| `backend` | `typesafe` | `typesafe` (native API) or `openrouter` (Decisions API). |
| `endpoint` | per backend | `https://api.typesafe.ai/v1/systemone` / `https://openrouter.ai/api/alpha/decisions`. |
| `api_key_env` | per backend | `TYPESAFE_API_KEY` / `OPENROUTER_API_KEY`. Name only — never the key itself. |
| `jev_model` | per backend | `jev-1.13.0` / `typesafe/jev-1.13`. On `typesafe`, `typesafe/jev-1.13` is mapped to `jev-1.13.0`. |
| `confidence_threshold` | `0.5` | Below this, the more capable of Jev's choice and `default_model` (by grid order) is used; with no `default_model` the configured model is kept. |
| `timeout_s` | `2.0` | Budget for the decision call, including one retry on HTTP 429/529. |
| `default_model` | `""` | Grid id used below threshold, on an `unclear` answer, and when Jev is unavailable. Empty keeps the configured model. |
| `default_effort` | `medium` | Effort used with the fallback model. |
| `context_turns` | `4` | Preceding turns sent to Jev as recent context. |
| `route_per_turn` | `true` | Off routes once per session instead of once per user turn. |
| `audit_enabled` | `true` | Append one JSONL record per turn under the plugin data directory. |
| `log_skips` | `true` | Record why a turn was left unrouted. |
| `include_user_message_in_audit` | `false` | Off keeps conversation content out of the audit. |
| `grid` | `null` | Optional list replacing the built-in grid: `"model-id: description"` strings, one-key maps, or `{id, description, efforts, context}` maps. **List it least capable first** — that order is the tier order. |
| `routed_providers` | `[ollama-cloud]` | Hermes provider names whose requests are routed. The `custom:` prefix is ignored (`custom:ocx` = `ocx`). Unset = default; `[]` = none. |
| `routed_base_urls` | `[]` | Also route requests whose `base_url` equals an entry or continues it after `/`. A request with an empty provider name is never routed. |
| `catalog_check` | `true` | Check decisions against Hermes' Ollama:cloud model cache (Ollama providers only). |
| `context_reserve_tokens` | `32000` | A model fits when its `context` exceeds the estimated prompt (chars/4 over messages + tools) plus this. |
| `long_context_models` | `[]` | Tried in order (first that fits) when neither the chosen model nor a more capable grid entry fits. Grid ids or `{id, context, efforts}` maps. |
| `unknown_effort` | `omit` | Effort off Ollama (and for `combo/<id>` or unknown ids): `omit` drops `reasoning_effort`; `keep` keeps the request's value only if it is `low`/`medium`/`high`; `pass` sends Jev's level (and drops the host value when Jev gave none). |

### Routing on an OpenCodex (ocx) endpoint

ocx is one OpenAI-compatible endpoint that takes `provider/model` and `combo/<id>` ids. In Hermes it is a
`custom_providers` entry selected as `provider: custom:ocx`.

**What the middleware actually sees.** For every `custom_providers` entry Hermes resolves the runtime as
`provider="custom"` plus that entry's `base_url` (`hermes_cli/runtime_provider_custom.py`, `_custom_runtime`);
the `custom:ocx` string is not passed to middleware. The router therefore maps a bare `custom` back to the
entry name by matching `base_url` against `get_compatible_custom_providers()`, then compares names with the
`custom:` prefix stripped. `custom:ocx` is routed; `custom:zai` (another entry, another `base_url`) is not.

**Recommended first install: shadow.** Install with `mode: shadow`, collect 20–50 real turns, read the
audit (`hermes jev-effort-router tail 50`: `model`, `effort`, `model_confidence`, `fallback_reasons`,
`configured_model`), then switch to `mode: route`. Shadow respects every skip gate and the per-turn cache,
and never raises.

Example — profile `nord`. Install the plugin into `profiles/nord/plugins/jev-effort-router/` and add it
to `plugins.enabled` in the nord profile config:

```yaml
plugins:
  enabled: [jev-effort-router]
  entries:
    jev-effort-router:
      settings:
        mode: shadow                      # collect 20-50 real turns, then switch to: route
        routed_providers: [custom:ocx]    # custom:zai and every other provider stay untouched
        backend: typesafe
        api_key_env: TYPESAFE_API_KEY     # in profiles/nord/.env; read via the profile secret scope
        default_model: gldf-hermes        # existing ocx combo (failover zai/glm-5.3 -> xai/grok-4.7)
        default_effort: medium
        unknown_effort: omit
        context_reserve_tokens: 32000
        # ILLUSTRATIVE grid - the final tier table comes from the owner. Least capable first:
        # grid order is the tier order (fallback max() and context escalation both use it).
        grid:
          - id: gemini-3.8-flash          # EXAMPLE id: exact ocx id unconfirmed
            context: 1000000
            description: "trivial request answered in one short step with no investigation: greeting or
              thanks, a one-line factual question, translating a sentence, fixing a named typo or renaming
              one string at a given place"
          - id: gldf-flash                # the profile's model.default; window not declared = unknown
            description: "bounded task needing some judgment: write or change code in one or two files,
              explain an error from a short log or traceback, write tests or a script, search or summarise
              a pasted document, run a known sequence of tool commands"
          - id: gldf-hermes
            context: 500000               # the smaller member window: the failover may land on grok-4.7
            description: "hard or open-ended task where a mistake is costly: architecture or migration
              design, refactoring across several modules, root-cause debugging of intermittent or
              production failures, reasoning across several long files, or a short follow-up that
              continues such a task from recent_context"
        # Used (first that fits) only when neither the chosen model nor a more capable grid entry
        # fits the prompt.
        long_context_models:
          - id: zai/glm-5.3
            context: 1000000
            efforts: [low, high, max, ultra]   # medium is rounded up to high
          - id: gemini-3.1-pro            # EXAMPLE id: exact ocx id unconfirmed
            context: 1000000
```

Notes on this example:

- `gldf-hermes` is an existing ocx combo, being converted to a failover `zai/glm-5.3` → `xai/grok-4.7`.
  The direct glm id on ocx is `zai/glm-5.3` (note the `zai/` prefix); to route to it without the
  combo, use `zai/glm-5.3` both as the grid entry and as `default_model`.
- **The grid is illustrative.** Ids marked EXAMPLE are not confirmed against the ocx catalog; replace them
  once the owner's tier table arrives. Descriptions are the tier criteria from Jev's own evaluation
  (`fast` / `general` / `strong`), mapped onto ocx ids, and are not re-measured on this deployment.
- **Exact id match.** The session's configured model (`model.default`, here `gldf-flash`) must equal a grid
  id exactly, or the turn is left alone. Jev's answer is mapped back to the grid by position, never by name.

Context windows known so far (from the deployment, via Jev), for filling in `context`:

| Model | Window | Id on ocx |
|---|---|---|
| gemini-3.8-flash | 1M | example, unconfirmed |
| zai/glm-5.3 | 1M | exact |
| glm-5.3-flash | 1M | example, unconfirmed |
| grok-4.20-reasoning | 1M | example, unconfirmed |
| gemini-3.1-pro | 1M | example, unconfirmed |
| xai/grok-4.7 | 500k | exact |
| gpt-6-* | 272k | family; exact ids unconfirmed |

#### Per-model effort levels (`efforts`)

A grid entry (or `long_context_models` entry) may declare the levels the model accepts:
`efforts: [low, high, max, ultra]`. The decided level (Jev's, or `default_effort` on a fallback) is then
**rounded up** onto that list along the canonical ladder
`none < minimal < low < medium < high < xhigh < max < ultra`; if nothing stronger is declared, the
strongest declared level is used. `none` is never chosen for an enabled request. The value is written
verbatim to `reasoning_effort`, so declare only levels your upstream accepts. Without `efforts` the
earlier rules apply (Ollama family table on Ollama routes, `unknown_effort` elsewhere). Entries accept
the string form `"id: description"` (no `efforts`/`context`), a map `{id, description, efforts, context}`,
or a one-key map `{id: description}` / `{id: {description, efforts, context}}`.

#### Context fit (`context`, `context_reserve_tokens`, `long_context_models`)

The prompt is **estimated** as `ceil(chars / 4)` over the JSON of the request's `messages` and `tools`
(a rule of thumb, not a tokenizer). A model fits when `context > estimate + context_reserve_tokens`
(default 32000); a model without `context` is treated as fitting. If the chosen model does not fit, the
next more capable grid entry that fits is used (`fallback_reasons: [..., "context_escalated"]`), then the
first fitting `long_context_models` entry (`"context_long_model"`); the effort is re-derived for the new
model. If nothing fits, the request is left untouched and a `skip` record with reason `context_no_fit`
and the estimate is written. The check runs on every request of a turn (the prompt grows inside a tool
loop), on the fallback paths, and in shadow mode (the `shadow` record shows the escalated model plus
`context_from` and `context_estimate`).

How the decision is made off Ollama:

- The model question asks for the **least capable tier that still completes the task**; options are the
  descriptions only (no model ids) plus an `unclear` option, which resolves to `default_model`.
- State sent to Jev: `user_message`, `user_message_chars`, `recent_context`.
- Effort is a Score over low/medium/high, rounded; effort families and the Ollama catalog apply only to
  Ollama providers. Elsewhere `unknown_effort` decides (default `omit`).
- Below `confidence_threshold`: the more capable of (choice, `default_model`) by grid order — a
  distrusted answer never downgrades below the fallback. No `default_model` → request untouched.
- Jev unavailable (timeout, HTTP error, malformed answer): `default_model` at `default_effort` if it is on
  the grid, else untouched. A missing key never calls Jev and leaves the request untouched.

## How it behaves when things go wrong

The router is built to be invisible when it fails. An unknown provider, a model outside the grid, a missing
key or any internal exception leave the request **exactly** as it was. A timeout, an HTTP error or a
malformed answer does the same unless `default_model` is set and on the grid, in which case the turn goes
to that fallback (recorded with its `fallback_reasons`).

Notably, it touches only what it owns:

- **Provider** — only `routed_providers` / `routed_base_urls` are routed (default `ollama-cloud`); any other provider passes through untouched.
- **API mode** — only `chat_completions`; a Responses or Anthropic-Messages route is left alone.
- **Model** — a model outside the configured grid is not routed.
- **Auxiliary calls** — titling, compression, MoA and vision calls are not routed, only the main turn.

### One decision per user turn

The decision is taken at the first provider request of a turn and replayed for the rest of that turn's tool
loop, so the system prompt and the prompt cache are never disturbed mid-turn. The next user turn routes
afresh, which is what lets a session move from a one-line question to a code refactor and be served by
different models without anyone reconfiguring anything.

## The routing grid

Six Ollama:Cloud models, in this order. The list is deliberately short: every extra option measurably
dilutes a Choice decision.

The **Profile** column is the criterion string sent to Jev, verbatim, in English. It lives in
[`grid.py`](grid.py) as an English one-liner and is part of the measured payload rather than a display
string: rewriting it changes what Jev is choosing between, so treat it as data. To use your own wording,
override `grid`.

Every profile names a **task family**. A task-free superlative — "excellent value for money",
"excellent in real use for everyday tasks" — reads as a safe pick on every prompt, and the model
carrying one absorbs decisions that belong to the others. That is what kept `glm-5.3` and
`glm-5.3-flash` out of the route; see the changelog for the before/after measurement.

| # | Model | Profile (as sent to Jev) |
|---|---|---|
| 1 | `deepseek-v4.1-flash` | the usual choice for general work: everyday writing, explanation, summarising, ordinary coding and tool use; 1M context; cheap for its size |
| 2 | `kimi-k3` | strongest at complex code and long agentic tasks: multi-file refactors, deep debugging, large repositories; slow and the most expensive |
| 3 | `glm-5.3` | strongest at rigorous reasoning: mathematics, logic, science, quantitative and financial analysis, where a wrong answer is costly |
| 4 | `glm-5.3-flash` | best reasoning-per-cost on large text: drafting, summarising, translating and structured extraction over long documents; fast |
| 5 | `minimax-m3` | fast tool calling: long sequences of API/CLI actions, repetitive automation, high throughput |
| 6 | `nemotron-3-nano:30b` | highest throughput and lowest cost: trivial single-step requests only; weak at reasoning and at long context |

Adding a model is a reviewed change to [`docs/routing-grid.md`](docs/routing-grid.md) with benchmark
evidence behind it — not a config-only act. See that file for the benchmark sources and pricing.

Model ids are checked against Ollama:Cloud's own catalog before they go on the wire: `status` reports any
grid entry the provider no longer serves under `grid_unavailable`, and a decision naming one is refused
rather than sent. A provider catalog naming a model the grid does not offer is not added automatically —
extending the grid stays a reviewed change.

### Reasoning effort per model family

Model families do not accept the same effort vocabulary, so the plugin carries a per-family table rather
than sending a generic parameter. It never escalates a level, never invents one, and omits the field rather
than risk a 400. The notable mapping: Kimi K3's documented set is `low | high | max`, so a `medium` request
lands on `high`.

## Audit trail

One JSONL record per routing attempt, under `<HERMES_HOME>/plugin-data/jev-effort-router/routes.jsonl`:

```json
{
  "ts": "2026-09-22T19:09:21+00:00",
  "event": "route",
  "model": "kimi-k3",
  "effort": "high",
  "effort_requested": "high",
  "model_choice": "2",
  "model_confidence": 0.9,
  "model_probabilities": {"2": 0.9},
  "effort_confidence": 0.8,
  "alternatives": ["deepseek-v4.1-flash", "glm-5.3", "..."],
  "latency_ms": 271,
  "fallback_reasons": [],
  "jev_model": "typesafe/jev-1.13",
  "replayed": false,
  "turn_id": "turn-1",
  "session_id": "s",
  "platform": "cli"
}
```

The file is the point: it is what lets you judge over weeks whether the grid routes well, without
re-running benchmarks by hand. Skipped turns (`event: "skip"`, with a `reason`) are recorded too. The API
key is never written and the user message only when `include_user_message_in_audit` is explicitly on.

## Development

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q tests
```

`pytest` must be pointed at `tests/` explicitly. The plugin's entry point is `__init__.py` at the
repository root (Hermes requires it beside `plugin.yaml`), which makes that root a package; if pytest
takes the root as its rootdir it tries to import that `__init__.py` as the package of the collected test
modules and fails with *"attempted relative import with no known parent package"*. That is why the ini
lives in `tests/pytest.ini` and CI runs `pytest -q tests`.

The suite drives the real middleware callback against a stub Decisions API — no network, no API key
required (an autouse fixture supplies a dummy one; tests asserting the unconfigured path delete it).

```
tests/test_grid.py          the grid and choice→model mapping
tests/test_effort.py        per-family effort translation
tests/test_client.py        wire contract, answer interpretation, failure containment
tests/test_router.py        routing end-to-end, replay, degradation, skip gates
tests/test_catalog.py       provider-catalog check: a model the provider lacks is refused
tests/test_catalog_status.py the grid_unavailable warning on the status surface
tests/test_audit.py         audit trail and the per-turn memo
tests/test_registration.py  register(ctx) surface, manifest drift, tool dispatch shape, no socket I/O
```

`tests/test_registration.py` is the one that matters most for packaging: it loads `__init__.py` the way
Hermes' loader does and asserts the manifest declares exactly what `register(ctx)` registers, and that
registration opens no socket (`hermes plugins doctor` blocks sockets while `register` runs).

## Design notes

- **No Hermes core change.** The plugin rides the documented `llm_request` middleware point in
  `agent/turn_api_request.py::build_api_request`. See
  [`docs/integration-surface.md`](docs/integration-surface.md).
- **Choices are keyed, not text.** Jev returns the criterion *key*, so criteria are built as
  `{"1": "deepseek-v4.1-flash: …", …}` and mapped back by key. Profile text can be rewritten without
  silently changing which model is selected. See [`docs/jev-decisions-api.md`](docs/jev-decisions-api.md).
- **The key question is asked first.** Both questions go in one request; the endpoint evaluates them
  independently and in parallel, so any coupling between them is resolved in code, not on the wire.
- **`httpx` is imported lazily**, inside the callback — never at import or registration time.

The full contract is in [`docs/SPEC.md`](docs/SPEC.md) (capabilities, constraints, non-goals) with its
companions `routing-grid.md`, `jev-decisions-api.md` and `integration-surface.md`.

## License

MIT — see [LICENSE](LICENSE).
