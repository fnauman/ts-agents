# PLAN_REFACTOR.md

## Objective

Refactor `ts-agents` into a **CLI-native time-series workbench for autonomous agents**.

The product should not be “yet another agent framework” and not “a grab-bag of wrappers.”
It should become:

> a stable, machine-runnable time-series substrate with strong domain priors,
> reproducible execution, and inspectable artifacts.

In practice, that means:

- **general-purpose coding agents** (Codex, Claude Code, Aider-style harnesses, custom shells) can call it reliably
- **humans** can still use it directly from the terminal
- **skills** become the policy layer that helps agents make better time-series decisions
- **sandboxes** become a real product capability, not an implementation detail

---

## Product statement

Use this as the working definition during the refactor:

> `ts-agents` is a CLI-first operating layer for time-series analysis in agentic environments.
> It provides stable tool contracts, reusable skills, and reproducible execution backends for inspection, forecasting, segmentation, and labeled-stream / IoT workflows.

### What it is

- a **stable CLI contract** for time-series tasks
- a **skill and policy layer** for agent workflows
- a **sandbox/runtime layer** for reproducible execution
- an **artifact producer** (JSON, plots, reports, logs)
- a **harness-neutral substrate** for external agents

### What it is not

- primarily a Gradio product
- primarily a custom multi-agent framework
- primarily a notebook helper
- primarily a generic forecasting library competing with `statsforecast`, `aeon`, `sktime`, etc.
- a marketing claim about “autonomy” without hard machine-contract guarantees

---

## Key diagnosis from the current repo

The codebase already has the right bones, but the public surface is still split.

### What is already strong

- `ts_agents/cli/main.py` already gives a CLI-first entrypoint
- `ts_agents/tools/registry.py` and `ts_agents/tools/executor.py` already encode tool metadata, execution routing, and structured execution results
- `skills/` is already a real differentiator
- `docs/philosophy.qmd` already articulates the right core idea
- sandbox backends already exist (`local`, `subprocess`, `docker`, `daytona`, `modal`)

### What is currently blocking the vision

1. **The machine contract is not strong enough yet**
   - many `*_with_data` wrappers in `ts_agents/tools/agent_tools.py` return **human strings**, not structured payloads
   - several wrappers swallow exceptions and return strings like `"Error in Theta: ..."`, which makes failures look like success
   - JSON mode is therefore not consistently machine-usable

2. **JSON mode is not always clean**
   - some demo paths print extra text before the JSON payload
   - that breaks pipeability and agent trust

3. **The data model is too repo-specific**
   - much of the CLI assumes `run_id` + `variable`
   - that is convenient for bundled datasets but weak as a general public contract

4. **The command grammar is only partly coherent**
   - `tool list` exists, but there is no `tool show`
   - `run` is top-level instead of `tool run`
   - discovery is broad instead of incremental

5. **Product messaging is still split across CLI / agents / UI**
   - `README.md`, `AGENTS.md`, and `pyproject.toml` still present the UI as a central surface
   - current messaging is broader than the direction in `PLAN_SCOPING.md`

6. **The repo currently looks more like a toolkit than a power tool**
   - too many low-level tools are exposed directly
   - there are too few opinionated “hero workflows” on top

---

## Strategic decisions

These decisions should be treated as fixed for the refactor unless there is a strong reason to revisit them.

### 1. Make the CLI the product center

The CLI is the durable interface.

- agents are adapters around it
- the UI is optional and de-emphasized
- framework wrappers are convenience layers

### 2. Compete on substrate, not on “having an agent”

The moat is not a custom orchestrator.
The moat is:

- better time-series input contracts
- better artifact contracts
- better skills/policies
- better reproducibility
- better sandboxes

### 3. Keep two layers of product surface

#### Layer A: stable low-level contract
For coding agents and automation.

Examples:
- `tool list`
- `tool show`
- `tool run`
- `skill list`
- `sandbox doctor`

#### Layer B: opinionated hero workflows
For attraction and daily repeated value.

Examples:
- inspect a series
- compare forecasts
- analyze a labeled IoT stream

These can be implemented as `workflow run <name>` first, with friendly aliases later.

### 4. Generic input must become first-class

The future public contract cannot depend mainly on bundled `run_id` + `variable` datasets.
Keep those as convenience adapters, but make generic CSV / parquet / JSON input a first-class path.

### 5. Skills must become more actionable than Markdown prose alone

Keep `SKILL.md`, but add more structured policy metadata that both humans and agents can consume.

---

## Refactor goals

A successful refactor should produce all of the following:

1. **Stable machine-readable success payloads**
2. **Typed, structured error payloads + useful exit codes**
3. **No stdout noise in JSON mode**
4. **First-class stdin / pipe / file input**
5. **Predictable CLI discovery**
6. **A small set of hero workflows on top of the registry**
7. **Clear sandbox probing and reproducibility semantics**
8. **Messaging that makes the repo instantly legible**
9. **Compatibility aliases for one release cycle instead of abrupt churn**

---

## Target public CLI

This is the target public grammar.

### Low-level contract

```bash
# discovery
 ts-agents tool list
 ts-agents tool show forecast_theta_with_data
 ts-agents tool search forecast

# execution
 ts-agents tool run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=12
 ts-agents tool run describe_series --input-json '{"series": [1,2,3,4]}'
 cat payload.json | ts-agents tool run describe_series --stdin

# data / inputs
 ts-agents data list
 ts-agents data vars

# skills
 ts-agents skills list
 ts-agents skills show forecasting
 ts-agents skills export --all-agents

# sandboxes
 ts-agents sandbox list
 ts-agents sandbox doctor docker
 ts-agents sandbox doctor modal
```

### Opinionated workflow layer

```bash
 ts-agents workflow list
 ts-agents workflow run inspect-series --input data.csv --time-col ds --value-col y --output-dir outputs/inspect
 ts-agents workflow run forecast-series --input data.csv --time-col ds --value-col y --horizon 48 --output-dir outputs/forecast
 ts-agents workflow run activity-recognition --input stream.csv --label-col label --value-cols x,y,z --output-dir outputs/activity
```

### Compatibility rules

- keep `ts-agents run ...` as an alias to `ts-agents tool run ...` for one release cycle
- keep `demo ...` as an alias to `workflow run ...` for one release cycle
- keep `agent run ...` available, but de-emphasize it in docs and onboarding

---

## Target output contract

### Rule

**Every machine-readable command must emit exactly one clean JSON document on stdout in JSON mode.**

No extra logs, no progress text, no banner text, no print statements from helper scripts.
All noisy diagnostics must go to stderr or to files.

### Success envelope

Use one stable top-level envelope for all machine-readable commands.

```json
{
  "ok": true,
  "command": "tool run",
  "name": "forecast_theta_with_data",
  "input": {
    "params": {"unique_id": "Re200Rm200", "variable_name": "bx001_real", "horizon": 12}
  },
  "result": {
    "kind": "forecast",
    "summary": "Theta forecast completed successfully.",
    "data": {
      "forecast": [0.1, 0.2],
      "horizon": 12,
      "method": "theta"
    },
    "artifacts": [
      {
        "kind": "plot",
        "path": "outputs/.../forecast.png",
        "mime_type": "image/png",
        "description": "History vs forecast plot"
      }
    ],
    "warnings": []
  },
  "execution": {
    "backend_requested": "docker",
    "backend_actual": "docker",
    "duration_ms": 1234.5
  }
}
```

### Error envelope

```json
{
  "ok": false,
  "command": "tool run",
  "name": "forecast_theta_with_data",
  "error": {
    "code": "dependency_error",
    "message": "Statistical forecasting requires optional dependencies.",
    "retryable": false,
    "hint": "Install with: pip install \"ts-agents[forecasting]\"",
    "details": {
      "missing_extra": "forecasting"
    }
  },
  "execution": {
    "backend_requested": "local",
    "backend_actual": "local"
  }
}
```

### Exit codes

Map exit codes consistently.

- `0` success
- `2` validation / bad arguments
- `3` dependency error
- `4` data error / missing data
- `5` backend unavailable
- `6` execution failure
- `7` approval / permission denied
- `8` timeout / resource exhaustion

Keep the current generic `1` only as a temporary compatibility fallback while migrating.

---

## Target internal architecture

### 1. Split compute from rendering

Today many `*_with_data` wrappers combine:

- data loading
- algorithm execution
- text formatting
- plot rendering
- exception handling

That is the biggest architectural smell.

### New model

#### Pure analysis layer
- stays in `ts_agents/core/`
- returns dataclasses / structured Python objects

#### Data adapter layer
- resolves bundled data, CSV, parquet, stdin JSON, etc.
- returns canonical series/dataset objects

#### Tool layer
- calls analysis using canonical inputs
- returns a structured `ToolPayload`
- never swallows failures as success strings

#### Renderer layer
- turns structured payloads into:
  - human-readable text
  - markdown summaries
  - optional plots / artifact files

#### CLI layer
- only handles parsing, routing, serialization, and exit behavior

---

## New shared types to add

Add new shared dataclasses / models.

### `SeriesRef`
Canonical reference for one univariate series.

Suggested fields:
- `source_type`: `bundled_run`, `csv`, `parquet`, `stdin_json`, `inline`, `python`
- `path`
- `run_id`
- `variable`
- `time_col`
- `value_col`
- `id_col`
- `frequency`
- `metadata`

### `DatasetRef`
Canonical reference for panel / multiseries / labeled-stream input.

Suggested fields:
- `path`
- `format`: `wide`, `long`, `labeled_stream`
- `time_col`
- `value_cols`
- `label_col`
- `id_col`
- `group_cols`
- `metadata`

### `ArtifactRef`
Structured artifact description.

Suggested fields:
- `kind`
- `path`
- `mime_type`
- `description`
- `created_by`

### `ToolPayload`
Structured payload returned by tools.

Suggested fields:
- `kind`
- `summary`
- `data`
- `artifacts`
- `warnings`
- `provenance`

### `CLIEnvelope`
Top-level success/error envelope.

---

## Ordered workstreams

Implement these in order.

---

## Workstream 1 — Freeze product identity and messaging

### Goal
Make the repo instantly understandable.

### Changes

Update messaging in:
- `README.md`
- `pyproject.toml`
- `AGENTS.md`
- `docs/index.qmd`
- `docs/quickstart.qmd`
- `docs/philosophy.qmd`
- `ROADMAP.md`

### Required messaging changes

1. Lead with **CLI + skills + sandboxes**
2. Demote UI to optional / secondary
3. Demote built-in agent implementations to adapters / examples
4. Remove language that implies the UI is a primary onboarding path
5. Replace vague “time series toolkit with CLI, agents, and Gradio UI” wording with a sharper tagline

### Recommended tagline options

Preferred:
- `Time-series skills, tool contracts, and sandboxes for agentic workflows.`

Alternative:
- `A CLI-first time-series workbench for autonomous coding agents.`

### Acceptance criteria

- top of `README.md` communicates the new identity in the first screenful
- `pyproject.toml` description matches the new identity
- Quickstart starts with CLI and workflows, not UI
- UI remains documented, but not central

---

## Workstream 2 — Repair the machine contract

### Goal
Make JSON mode truly agent-safe.

### Files

Primary:
- `ts_agents/cli/main.py`
- `ts_agents/cli/output.py`
- `ts_agents/tools/executor.py`
- `ts_agents/tools/results.py`
- `ts_agents/tools/agent_tools.py`

Likely new:
- `ts_agents/contracts.py` or `ts_agents/tools/contracts.py`
- `ts_agents/cli/errors.py`
- `ts_agents/rendering.py` or `ts_agents/tools/renderers.py`

### Tasks

1. Introduce a stable envelope type for CLI JSON output.
2. Stop returning raw strings as the canonical JSON result for `*_with_data` tools.
3. Convert wrappers to return structured payloads.
4. Make plotting produce artifact files / artifact refs, not inline base64 by default.
5. Reserve inline/base64 embedding for an explicit compatibility path only.
6. Ensure any failure in a tool becomes a structured `ToolError`, not a successful string payload.
7. Ensure JSON mode emits exactly one JSON document.

### Specific code changes

#### In `ts_agents/tools/agent_tools.py`
- remove broad `try/except -> return "Error in ..."`
- raise `ToolError` (or plain exceptions that are converted upstream) instead
- return structured objects / dataclasses / dicts
- separate `data loading` from `formatting`

#### In `ts_agents/cli/output.py`
- support rendering envelopes cleanly
- support text rendering from structured payloads
- support artifact refs instead of `[IMAGE_DATA:...]` as the primary path

#### In `ts_agents/cli/main.py`
- JSON mode must use the new stable envelope
- do not emit `{"error": "..."}` as an untyped fallback
- preserve text mode for humans

### Acceptance criteria

- `describe_series_with_data --json` returns structured stats, not a prose string
- `forecast_theta_with_data --json` returns `ok=false` + typed error if dependencies are missing
- `demo ... --json` emits valid clean JSON with no extra stdout text
- no current wrapper can mask a failure as successful text

---

## Workstream 3 — Standardize CLI grammar and discovery

### Goal
Make the CLI guessable and incremental.

### Files
- `ts_agents/cli/main.py`
- docs + tests

### Tasks

1. Add `tool show <name>`
2. Add `tool run <name>`
3. Add `tool search <query>`
4. Keep current top-level `run` as an alias temporarily
5. Make `tool list` compact by default
6. Make `tool show` the detailed inspection path

### `tool show` should include

- name
- category
- cost
- description
- dependencies
- JSON schema / parameters
- examples
- return kind
- artifact behavior
- timeout / memory / disk expectations

### Example JSON for `tool show`

```json
{
  "name": "forecast_theta_with_data",
  "category": "forecasting",
  "cost": "low",
  "description": "Forecast series using Theta with bundled-data adapter.",
  "input_schema": {...},
  "returns": {
    "kind": "forecast",
    "artifacts": ["plot"]
  },
  "examples": [...],
  "dependencies": ["statsforecast"],
  "resources": {
    "timeout_seconds": 300,
    "memory_mb": 512
  }
}
```

### Acceptance criteria

- `ts-agents tool list` is summary-first
- `ts-agents tool show <tool>` is the detailed discovery path
- `ts-agents tool run <tool>` is fully functional
- `ts-agents run <tool>` still works as a compatibility alias

---

## Workstream 4 — Add first-class stdin and generic input adapters

### Goal
Let agents operate on arbitrary user data without writing glue code.

### Files
Likely new:
- `ts_agents/io/` or `ts_agents/inputs/`
- `ts_agents/cli/input_parsing.py`
- `ts_agents/data_access.py`
- `ts_agents/tools/agent_tools.py`
- `ts_agents/tools/registry.py`

### Tasks

1. Add `--input-json <json-or-path>` support
2. Add `--stdin` support
3. Add `-` file sentinel support where appropriate
4. Add first-class CSV / parquet / JSON input specs
5. Keep `--run` / `--var` as convenience adapters for bundled datasets
6. Normalize all data-aware tools onto the same input abstraction

### Input patterns to support

#### Bundled series
```bash
 ts-agents tool run describe_series_with_data --run Re200Rm200 --var bx001_real
```

#### Inline JSON
```bash
 ts-agents tool run describe_series --input-json '{"series": [1,2,3,4]}'
```

#### Piped JSON
```bash
 echo '{"series": [1,2,3,4]}' | ts-agents tool run describe_series --stdin
```

#### CSV
```bash
 ts-agents workflow run inspect-series \
   --input data.csv \
   --time-col timestamp \
   --value-col y
```

#### Labeled stream
```bash
 ts-agents workflow run activity-recognition \
   --input stream.csv \
   --value-cols x,y,z \
   --label-col label
```

### Acceptance criteria

- the CLI can operate on arbitrary series data without requiring repo-specific run IDs
- pipe-in JSON works in a stable way
- tool discovery/help explains input shapes clearly
- current bundled-data shortcuts still work

---

## Workstream 5 — Introduce hero workflows

### Goal
Give the repo a compelling public face.

### Rationale
The low-level registry is necessary, but not sufficient.
Most users will remember and adopt a few opinionated workflows, not 55 raw tools.

### Initial hero workflows

#### 1. `inspect-series`
Purpose:
- quick diagnostics on unknown data
- summary stats
- periodicity
- decomposition suggestion
- changepoint / segmentation hints

Outputs:
- `summary.json`
- optional plots
- markdown summary
- explicit recommended next steps

#### 2. `forecast-series`
Purpose:
- baseline-first forecasting workflow
- seasonality detection / assumption capture
- explicit model comparison
- artifact bundle + recommendation

Outputs:
- metrics JSON
- forecast CSV / JSON
- plot(s)
- report.md

#### 3. `activity-recognition`
Purpose:
- labeled-stream / IoT workflow
- window-size selection
- classifier evaluation
- report artifacts

Outputs:
- selection JSON
- eval JSON
- confusion matrix plot
- report.md

### Files
Likely new:
- `ts_agents/workflows/`
- `ts_agents/workflows/inspect.py`
- `ts_agents/workflows/forecast.py`
- `ts_agents/workflows/activity.py`
- CLI wiring + docs + tests

### Rules

- workflows should call existing tools internally where possible
- workflows should emit the same stable envelope shape as low-level tools
- workflows should be deterministic in `--no-llm` mode
- workflows should be great examples for external coding agents

### Acceptance criteria

- `workflow list` exists
- all three hero workflows work end-to-end
- `demo ...` aliases to matching workflows
- README quickstart uses workflows, not raw tool calls, as the main onboarding path

---

## Workstream 6 — Make sandboxes a visible product feature

### Goal
Turn backend execution from a hidden flag into a trusted capability.

### Files
- `ts_agents/tools/executor.py`
- CLI wiring
- `SANDBOX.md`
- tests

### Problems to solve

- backend fallback behavior is too implicit
- users/agents need a way to probe capabilities before execution
- backend choice should be visible in outputs
- reproducibility requires explicit actual backend reporting

### Tasks

1. Add `sandbox list`
2. Add `sandbox doctor <backend>`
3. Add capability reporting in JSON mode
4. Make fallback behavior explicit
5. Add `--fallback-backend local` or `--allow-fallback`
6. If a requested backend is unavailable and fallback is not allowed, fail with a typed backend error

### Example JSON

```json
{
  "backend": "docker",
  "available": false,
  "reason": "Docker CLI not found",
  "suggested_fix": "Install Docker or run with --sandbox local"
}
```

### Acceptance criteria

- users can discover backend readiness without trial-and-error
- backend actual vs requested is always reported
- no silent reproducibility changes
- backend errors are structured and actionable

---

## Workstream 7 — Upgrade skills from prose to policy

### Goal
Make skills a genuine advantage for agent use.

### Files
- `skills/*/SKILL.md`
- `ts_agents/cli/skills.py`
- possibly new `skills/*.json` export or structured metadata generation
- docs

### Tasks

1. Keep Markdown skills as the human-editable source of truth.
2. Expand frontmatter / metadata so skills encode policy, not just prose.
3. Add `skills show <name>` to inspect a skill in structured form.
4. Add `skills export --json` or generated machine-readable summaries.
5. Link skills to workflows and preferred tools.

### Suggested metadata additions

```yaml
metadata:
  ts_agents:
    preferred_workflow: inspect-series
    preferred_tools: [describe_series_with_data, detect_periodicity_with_data]
    avoid_tools: []
    min_series_length: 50
    artifact_checklist:
      - summary.json
      - report.md
    escalation_rules:
      - if: seasonality_unknown
        then: run detect_periodicity_with_data
      - if: forecast_requested
        then: compare theta + ets + arima before ensemble
```

### Acceptance criteria

- skills can be consumed as both Markdown and structured metadata
- `skills show` returns actionable policy info
- external coding agents can inspect skills without reading the entire repository

---

## Workstream 8 — Add proof that ts-agents improves agent outcomes

### Goal
Back the positioning with evidence.

### Why this matters

“Ultracharges autonomous agents” is only credible if you can show concrete uplift.

### Add an evaluation harness for the repo itself

Compare at least:

1. plain model / no skills / no ts-agents guidance
2. plain tool access with current registry
3. tool access + structured discovery
4. tool access + skills + workflows

### Candidate tasks

- inspect an unknown univariate series
- compare forecasting baselines on a seasonal series
- detect periodicity and choose a decomposition method
- perform labeled-stream window-size selection and evaluation
- handle missing dependency / backend errors gracefully

### Metrics

- task success rate
- parse / schema failure rate
- number of invalid tool calls
- artifact completeness
- latency / cost (roughly)
- number of retries / recovery success

### Files
Likely new:
- `tests/evals/`
- `benchmarks/` or `experiments/`
- docs reporting results

### Acceptance criteria

- there is at least one reproducible internal benchmark that demonstrates why the repo is useful for agents
- the benchmark is easy to rerun

---

## Workstream 9 — Migration and compatibility

### Goal
Improve the product without breaking everyone overnight.

### Rules

1. Add aliases before removing commands.
2. Mark deprecated paths in help text and docs.
3. Keep one release cycle of compatibility for:
   - top-level `run`
   - `demo`
4. Defer any full UI extraction until after the CLI contract is fixed.

### Acceptance criteria

- old commands still work with deprecation messaging
- new docs only recommend the new grammar

---

## Specific implementation notes by file

### `ts_agents/cli/main.py`

Must do:
- add `tool show`, `tool search`, `tool run`
- add `workflow` command family
- add `sandbox` command family
- add `skills show`
- route all JSON responses through the new envelope
- stop using untyped `{"error": ...}` JSON fallback
- ensure no JSON pollution from helper prints

### `ts_agents/cli/output.py`

Must do:
- centralize envelope rendering
- centralize text rendering from structured payloads
- support artifact refs cleanly
- remove base64 image embedding as the canonical default

### `ts_agents/tools/agent_tools.py`

Must do:
- remove swallowed exceptions
- return structured payloads
- stop using human text as the only return type
- separate plotting / formatting from analytical return data

### `ts_agents/tools/executor.py`

Must do:
- expose requested vs actual backend
- expose backend-availability failures as typed errors
- support explicit fallback policy
- preserve and extend existing `ToolError` / `ExecutionResult` instead of reinventing them

### `ts_agents/tools/registry.py`

Must do:
- surface examples / returns / schemas through `tool show`
- potentially add richer metadata about artifacts and input kinds
- ensure metadata is strong enough for discovery without docs

### `ts_agents/cli/skills.py`

Must do:
- add `skills show`
- export structured skill metadata
- validate richer skill frontmatter

### `README.md`

Must do:
- lead with workflows + CLI contract
- include one “why this exists” paragraph
- include one very short quickstart using hero workflows
- demote UI and internal agent implementations

---

## Tests to add or update

### CLI contract tests

- every command depth supports `--help`
- `tool show <name> --json` returns valid JSON
- `tool run ... --json` returns valid JSON with no leading/trailing noise
- `workflow run ... --json` returns valid JSON with no leading/trailing noise
- `skills show <name> --json` returns valid JSON
- `sandbox doctor <backend> --json` returns valid JSON

### Error tests

- missing required params -> typed validation error + exit code 2
- missing optional dependency -> typed dependency error + exit code 3
- backend unavailable without fallback -> typed backend error + exit code 5
- timeout -> typed timeout error + exit code 8

### Compatibility tests

- `run` still works as alias
- `demo` still works as alias

### Artifact tests

- forecast workflow produces expected files
- activity workflow produces expected files
- inspect workflow produces expected files
- JSON envelopes include artifact refs to those files

### Stdin / input tests

- pipe JSON into `tool run`
- pipe JSON into `workflow run`
- CSV input path works for generic workflows

---

## Recommended implementation sequence for coding agents

Use this order exactly.

1. **Messaging pass**
   - README / docs / pyproject identity cleanup
2. **Output contract pass**
   - introduce envelopes + typed JSON output
3. **Wrapper cleanup pass**
   - structured returns + real exceptions in `agent_tools.py`
4. **CLI grammar pass**
   - `tool show`, `tool run`, `tool search`, compatibility aliases
5. **Input adapter pass**
   - stdin / input-json / CSV/parquet abstractions
6. **Workflow pass**
   - `inspect-series`, `forecast-series`, `activity-recognition`
7. **Sandbox pass**
   - `sandbox list`, `sandbox doctor`, explicit fallback semantics
8. **Skills pass**
   - richer skill metadata + `skills show`
9. **Evaluation pass**
   - internal benchmark / proof of uplift

### Planned PR sequence

Keep the refactor in a small number of related PRs.

#### PR 1 — CLI contract and discovery slice

Status:
- merged

Scope:
- update repo messaging just enough to match the new CLI-first direction
- add `tool show`, `tool search`, and `tool run`
- keep top-level `run` as a compatibility alias
- introduce the stable CLI JSON envelope and typed exit-code mapping
- normalize an initial vertical slice of wrappers to structured payloads:
  - `describe_series_with_data`
  - forecast wrappers
  - `compare_forecasts_with_data`

Why this PR exists:
- it fixes the contract boundary first
- later workflow, input, sandbox, and skills work can build on one stable low-level surface

#### PR 2 — Wrapper normalization and rendering cleanup

Status:
- merged

Scope:
- convert the remaining `*_with_data` wrappers away from success-looking error strings
- separate structured payloads from human rendering across the remaining tool families
- make artifact refs the default machine contract and keep inline/base64 only as a compatibility path

#### PR 3 — Generic inputs and hero workflows

Status:
- merged
- issue: `#50`

Scope:
- add first-class `--input-json`, `--stdin`, and generic file-input handling
- add `workflow` commands for `inspect-series` and `forecast-series`
- keep `demo` as a compatibility alias to the new workflow layer where appropriate

#### PR 4 — Activity workflow, sandboxes, and structured skills

Status:
- in progress
- issue: `#52`

Scope:
- add `activity-recognition`
- add `sandbox list` and `sandbox doctor` plus explicit fallback semantics
- add `skills show` and structured skill export / richer skill metadata

#### Optional PR 5 — Evaluation harness

Scope:
- add the internal benchmark / proof-of-uplift harness if it starts to crowd PR 4

Do not add more algorithms or more agent-framework complexity before steps 1–6 are done.

---

## Explicit non-goals during this refactor

Do not do these unless the core contract work is already complete:

- adding more forecasting models just to look broader
- adding foundation-model integrations as a headline feature
- expanding the UI
- building a more complex internal multi-agent stack
- turning the CLI into an interactive wizard
- optimizing for chat demos at the cost of machine readability

---

## Final success criteria

The refactor is successful when all of these statements are true:

1. A coding agent can discover the right command incrementally without reading the whole repo.
2. JSON mode is clean, typed, and trustworthy.
3. Failures are machine-actionable.
4. Arbitrary user data is easy to feed into the CLI.
5. The repo has a small number of memorable hero workflows.
6. Skills materially improve time-series decision quality.
7. Sandboxes are explicit and debuggable.
8. The README makes the repo sound like one thing, not five things.
9. A user can understand within 30 seconds why they should use `ts-agents` instead of directly stitching together `aeon`, `statsforecast`, notebooks, and generic coding agents.

---

## One-sentence north star

Build `ts-agents` into the **time-series operating layer that makes general-purpose coding agents reliably good at inspection, forecasting, and IoT-style sequence workflows**.
