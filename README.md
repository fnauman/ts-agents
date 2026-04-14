# ts-agents

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://fnauman.github.io/ts-agents/)
[![Python](https://img.shields.io/badge/python-3.11--3.13-3776AB)](#installation)
[![License](https://img.shields.io/badge/license-MIT-2EA44F)](https://github.com/fnauman/ts-agents/blob/main/LICENSE)

`ts-agents` is a CLI toolkit for **long-running autonomous agentic workflows**
on time-series data. It gives agent runtimes a stable, machine-readable surface
so a model can bootstrap, discover what's available, execute real work, and
produce inspectable artifacts — without hand-written glue code per project.

It is built around:
- a stable CLI contract for bootstrap, discovery, and execution (`ts-agents capabilities`, `ts-agents workflow ...`, `ts-agents tool ...`)
- strict JSON envelopes with `schema_version`, typed exit codes, top-level `quality_status`/`degraded`/`requires_review`, and nested workflow `result.status`/`result.data.quality_flags`
- run lifecycle metadata: generated run IDs, `run_manifest.json`, non-clobbering default output directories, and `--resume` / `--overwrite` semantics
- inspectable artifacts instead of chat-only outputs (plots as `ArtifactRef` files, JSON payloads, Markdown reports)
- reusable skills that encode time-series workflow guidance as install-agnostic command templates
- optional sandboxes for safer, reproducible execution (`local`, `subprocess`, `docker`, `daytona`, `modal`) with readiness probes and explicit fallback flags
- optional adapters on top, including Gradio and built-in agent entrypoints

It ships with three first-class workflows:
- `inspect-series` (quick diagnostics + summary/report artifacts)
- `forecast-series` (baseline comparison + forecast/report artifacts)
- `activity-recognition` (labeled-stream window-size selection + evaluation)

Legacy compatibility aliases for `ts-agents demo ...` remain available for one
release cycle and emit deprecation warnings.

Source-checkout-only datasets such as `data/wisdm_subset.csv` are documented
separately and are not part of the published wheel.

**Start here:** [Quickstart](#quickstart) | [Choose your path](#choose-your-path) | [Docs site](https://fnauman.github.io/ts-agents/) | [Evaluation harness](https://fnauman.github.io/ts-agents/evaluation.html) | [Workflow walkthroughs](https://fnauman.github.io/ts-agents/walkthroughs.html)

![ts-agents demo](https://raw.githubusercontent.com/fnauman/ts-agents/main/demo/assets/demo.gif)

## Table of Contents

- [Choose your path](#choose-your-path)
- [For autonomous agents](#for-autonomous-agents)
- [Why ts-agents instead of using statsforecast/sktime/aeon directly?](#why-ts-agents-instead-of-using-statsforecastsktimeaeon-directly)
- [Design principles](#design-principles)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [CLI usage](#cli-usage)
- [Gradio app](#gradio-app)
- [Sandbox backends](#sandbox-backends)
- [Guides](#guides)
- [Development](#development)

## Choose Your Path

### 1. Run a workflow in under a minute

Use the workflow layer when you want reproducible CLI commands on bundled or
custom data.

```bash
python -m pip install ts-agents
ts-agents workflow list
ts-agents workflow show forecast-series --json
ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive
```

Base install is guaranteed to support workflow discovery plus `inspect-series`.
It also supports a light `seasonal_naive` forecasting baseline. Install
`ts-agents[recommended]` (or use `uv sync` from a source checkout) for the full
three-workflow experience, including ARIMA/ETS/Theta forecasting and
`activity-recognition`.

### 2. Use the low-level CLI on bundled or custom data

Use the low-level tool registry when you want direct access to individual
analysis functions.

```bash
ts-agents tool list --bundle demo
ts-agents tool show forecast_theta_with_data
ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}'
ts-agents sandbox list
ts-agents skills show forecasting
```

### 3. Launch the UI or prepare a hosted demo

Use the Gradio app for interactive exploration, or the hosted entrypoint for a
manual/public demo deployment. This is optional and secondary to the CLI.

```bash
python -m pip install "ts-agents[ui]"
ts-agents-ui
ts-agents-hosted
```

From a source checkout (`git clone ...` + `uv sync`), use the root wrappers:

```bash
uv run python main.py
HOST=0.0.0.0 PORT=7860 uv run python app.py
```

`ts-agents-hosted` is environment-variable driven rather than flag-driven.
Configure `HOST`, `PORT`, `GRADIO_SHARE`, `TS_AGENTS_ENABLE_AGENT`,
`TS_AGENTS_AGENT_TYPE`, `TS_AGENTS_PERSIST_SESSIONS`, and
`TS_AGENTS_UI_TITLE` before launch if you need non-default behavior.

## For Autonomous Agents

`ts-agents` is designed so an autonomous agent can bootstrap itself, plan, and
run multi-step time-series work against a stable contract — even across long,
multi-turn sessions.

### 1. Bootstrap with one command

```bash
ts-agents capabilities --json
```

Returns the full agent-facing surface: available workflows, tools, sandboxes,
workflow discovery metadata, the current `install_profile` block, and
status-contract guidance. Use this as the first call of any new agent session.

### 2. Discover execution metadata before running anything

```bash
ts-agents workflow show forecast-series --json
ts-agents tool show forecast_theta_with_data --json
```

Both `show` commands return `cli_templates`, `source_options`, `global_options`,
`status_contract`, `default_output_behavior`, required extras, availability in
the current environment, input modes, and artifact behavior. Agents can plan
commands from this metadata rather than guessing flags.

### 3. Strict machine-readable envelopes

Every `--json` response is:

- wrapped in a stable envelope with `schema_version: "1.0"`
- strict JSON (no raw `NaN`/`Infinity`, `allow_nan=False`)
- accompanied by typed exit codes for validation, dependency, permission, and
  timeout errors — so agents can branch on failure mode instead of parsing
  prose
- tagged with top-level `quality_status`, `degraded`, and `requires_review`,
  while workflow payloads expose `result.status` and
  `result.data.quality_flags` for workflow-specific review signals

### 4. Run lifecycle and provenance

Workflow runs produce:

- a generated run ID and run-scoped output directory under `outputs/<workflow>/<run-id>/`
- a `run_manifest.json` capturing inputs, parameters, execution backend
  metadata, and emitted artifacts
- absolute paths on every `ArtifactRef` so an agent can materialize files from
  any working directory
- non-clobbering defaults, plus `--overwrite` / `--resume` semantics for retry
  loops and long sessions

### 5. Artifacts over chat

Tool/workflow outputs are written to real files (PNG plots, JSON, CSV,
Markdown reports) and returned as `ArtifactRef` entries. Chat is the control
plane; the files are the product — they can be inspected, diffed, cached, and
fed into the next step by the agent.

### 6. Sandbox parity and explicit fallback

```bash
ts-agents sandbox list
ts-agents sandbox doctor docker --json
ts-agents workflow run inspect-series \
  --input-json '{"series":[1,2,3,4]}' \
  --sandbox docker --allow-fallback --fallback-backend local
```

`sandbox doctor` probes readiness (including Docker image presence). Docker,
Daytona, and Modal all stage workflow artifacts back to the host output
directory so `result.artifacts[*].path` and `result.data.output_dir` are
always host-accessible. Fallback is explicit — the executor refuses to silently
switch backends unless `--allow-fallback` is passed.

### 7. Skills as install-agnostic command templates

```bash
ts-agents skills list
ts-agents skills show forecasting --json
```

Skill catalogs export normalized `ts-agents ...` command templates with no
checkout-specific prefixes, so agents can copy them verbatim into tool calls.

## Why ts-agents Instead of Using statsforecast/sktime/aeon Directly?

Those libraries are excellent algorithm/toolkit layers, and `ts-agents`
intentionally builds on that ecosystem rather than trying to replace it.

Use the underlying libraries directly when:
- you only need one modeling library inside a notebook or a custom pipeline
- you do not need artifacts, tool routing, or sandboxed execution

Use `ts-agents` when you want:
- a stable CLI contract that works the same across workflows, agents, and automation
- artifact-first outputs (plots, JSON, markdown/report assets) instead of chat-only responses
- reusable skills and tool bundles that encode workflow guidance
- optional sandbox backends for isolation, deployment, and heavier workloads
- swappable front ends: CLI, Gradio UI, or custom agent orchestration

## Design Principles

- **CLI as the stable contract**: `ts-agents` is the primary interface for automation and reproducibility. Autonomous agents plan against `capabilities`, `workflow show`, and `tool show` instead of hardcoded knowledge.
- **Strict machine envelopes**: `--json` output is versioned, strict, and typed — with status, quality flags, and exit codes — so agents can branch on failure mode rather than parsing prose.
- **Framework adapters, not framework lock-in**: LangChain/deep-agent wrappers are convenience layers over the same tool registry.
- **Artifacts over chat**: tools produce inspectable files (plots, JSON, reports), and agents return summaries plus paths.
- **Run lifecycle as first-class metadata**: every workflow run gets a run ID, a `run_manifest.json`, and non-clobbering defaults — so long, multi-turn sessions remain reproducible and resumable.
- **Swappable front-ends**: CLI agents, custom agents, and Gradio are interfaces around the same core tools.
- **Sandboxed execution with explicit fallback**: backends isolate dependencies and scale heavier workloads; the executor never silently downgrades isolation.

Canonical design doc:
- `docs/philosophy.qmd`

## Quickstart

```bash
# Base install: discovery + inspect-series + seasonal baseline forecast
python -m pip install ts-agents
ts-agents workflow list
ts-agents workflow show forecast-series --json
ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive

# Full workflow stack from a source checkout
uv sync
uv run ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive,arima,theta
uv run python data/make_synthetic_labeled_stream.py --scenario gait --seconds 40 --seed 1337 --out data/demo_labeled_stream.csv
uv run ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z
```

LLM-backed agent/report mode requires `OPENAI_API_KEY`. Either export it
directly or add it to `~/.env` (one `KEY=VALUE` per line; the app loads this
file automatically and will not overwrite variables already in your shell):

```bash
# Option A: export in your shell
export OPENAI_API_KEY=your-key

# Option B: store in ~/.env (loaded automatically)
echo 'OPENAI_API_KEY=your-key' >> ~/.env
```

```bash
uv run ts-agents agent run "Use the forecasting skill to compare ARIMA and Theta for a short univariate series"
```

If you pass `--output-dir`, workflow artifacts are written there. If you omit
it, each workflow run creates a unique run directory such as
`outputs/<workflow>/<run-id>/` with `run_manifest.json`, JSON/CSV outputs, and
any generated plots or reports.

Compatibility note: `ts-agents run ...` and `ts-agents demo ...` still work for
one release cycle, but they now emit deprecation warnings. Prefer
`ts-agents tool run ...` and `ts-agents workflow run ...`.

## Installation

Prerequisites:
- Python 3.11, 3.12, or 3.13
- [uv](https://github.com/astral-sh/uv)

Install from PyPI:

```bash
python -m pip install ts-agents
```

The default install is now intentionally CLI-first and lighter weight. Heavier
features are enabled with extras:

```bash
python -m pip install ts-agents
python -m pip install "ts-agents[forecasting]"
python -m pip install "ts-agents[decomposition,patterns]"
python -m pip install "ts-agents[ui,agents]"
python -m pip install "ts-agents[recommended]"
python -m pip install "ts-agents[all]"
```

Feature extras:
- `ui`: Gradio UI and hosted profile (`ts-agents-ui`, `ts-agents-hosted`)
- `agents`: LangChain-backed simple agent support
- `decomposition`: STL, MSTL, Holt-Winters
- `forecasting`: statistical forecasting tools
- `patterns`: matrix profile and changepoint tooling
- `classification`: aeon/scikit-learn classification workflows
- `viz`: plotting-only installs without Gradio
- `recommended`: the demo-friendly install profile
- `all`: the full optional stack

Install profiles:
- `ts-agents`: workflow discovery, `workflow show`, `inspect-series`, and a dependency-light `seasonal_naive` forecast baseline
- `ts-agents[forecasting]`: unlocks ARIMA, ETS, and Theta for `forecast-series`
- `ts-agents[classification]`: unlocks `activity-recognition`
- `ts-agents[recommended]`: the documented three-workflow experience used in walkthroughs and demos

Run the packaged entrypoints:

```bash
ts-agents --help
ts-agents tool list
```

UI entrypoints require the `ui` extra:

```bash
ts-agents-ui --help
ts-agents-hosted
```

If you are running from a source checkout with `uv sync`, prefix the CLI
commands below with `uv run`.

Source checkout setup:

```bash
git clone https://github.com/fnauman/ts-agents.git
cd ts-agents
uv sync
```

Local editable install from a source checkout:

```bash
python -m pip install -e .
```

Publishing setup in this repo targets:
- PyPI package name: `ts-agents`
- sandbox image: `ghcr.io/fnauman/ts-agents-sandbox`

See [Distribution guide](https://fnauman.github.io/ts-agents/distribution.html) for the release, PyPI, and
GHCR publishing flow.

CLI entrypoints:
- Preferred: `ts-agents ...`
- Also supported: `python -m ts_agents ...`
- Gradio UI: `ts-agents-ui`
- Hosted profile: `ts-agents-hosted`
- Source-checkout UI wrapper: `python main.py`
- Source-checkout hosted wrapper: `python app.py`

### Environment variables

All optional. Set them via `export` or in `~/.env`.

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | LLM agent/demo features | _(none — required for LLM mode)_ |
| `OPENAI_MODEL` | Model override | `gpt-5-mini` |
| `TS_AGENTS_DATA_DIR` | Full dataset path | bundled package data (or repo `./data`) |
| `TS_AGENTS_USE_TEST_DATA` | Use bundled test data | `true` |
| `TS_AGENTS_TEST_DATA_FILE` | Override test dataset filename | `short_real.csv` |
| `TS_AGENTS_SANDBOX_MODE` | Default sandbox backend | `local` |

Sandbox-specific environment variables (Docker/Daytona/Modal auth, snapshots,
streaming, and log files) are documented in `SANDBOX.md`.

### Hosted Demo Deployment

The installed package includes a hosted Gradio profile at `ts-agents-hosted`
intended for public demos such as Hugging Face Spaces. Source-checkout
deployments can use the root `app.py` wrapper, which calls the same hosted
entrypoint as `ts-agents-hosted`. It defaults to:
- manual analysis mode (`agent` disabled)
- no session persistence
- a public-safe configuration that does not require `OPENAI_API_KEY`

Launch it with:

```bash
ts-agents-hosted
uv run python app.py
```

Useful environment variables:
- `HOST` / `PORT` for bind address and port
- `GRADIO_SHARE` for Gradio sharing
- `TS_AGENTS_ENABLE_AGENT` to enable agent chat
- `TS_AGENTS_AGENT_TYPE` for `simple` vs `deep`
- `TS_AGENTS_PERSIST_SESSIONS` to enable persistence
- `TS_AGENTS_UI_TITLE` to override the page title

## Distribution

- Package metadata is configured for the `ts-agents` distribution name.
- GitHub Actions includes a PyPI publish workflow for tagged releases.
- GitHub Actions includes a GHCR workflow for publishing the sandbox image built
  from `Dockerfile.sandbox`.
- GitHub release/tag/docs flow is summarized in [Distribution guide](https://fnauman.github.io/ts-agents/distribution.html).

## CLI Usage

### Discover data and tools

```bash
ts-agents data list
ts-agents data vars
ts-agents tool list
ts-agents tool list --bundle demo
```

### Run workflows

```bash
ts-agents workflow list
ts-agents workflow show forecast-series --json
ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive
```

Use `workflow show` before automation to inspect required extras, supported
input modes, artifact outputs, and availability in the current environment.
If you omit `--output-dir`, the workflow creates a run-scoped directory under
`outputs/<workflow>/` and writes `run_manifest.json` plus the generated
artifacts there.

### Run tools directly

```bash
ts-agents tool run stl_decompose_with_data --run Re200Rm200 --var bx001_real
ts-agents tool run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=30 --json
```

### Save output and inspect tool artifacts

```bash
ts-agents tool run stl_decompose_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --json \
  --save outputs/Re200Rm200/stl.json
```

Current low-level plot-producing tools expose PNG paths under
`result.artifacts[*].path` in the saved JSON payload. `--extract-images` remains
available only for legacy saved outputs that still contain embedded
`[IMAGE_DATA:...]` tokens. Forecasting `forecast_*_with_data` tools are now
data-only; use `ts-agents workflow run forecast-series --output-dir ...` when
you want forecast plots, CSVs, and reports written as artifacts. If you omit
`--output-dir`, the workflow creates a unique run directory automatically.

### Agent mode

```bash
ts-agents agent run "Find peaks in bx001_real for Re200Rm200"
ts-agents agent run --type deep "Compare forecasting methods for bx001_real"
```

### Compatibility aliases

`ts-agents run ...` and `ts-agents demo ...` still work for one release cycle
to avoid breaking existing automation, but both surfaces now emit deprecation
warnings and are intentionally omitted from the recommended examples below.

Note: the WISDM example under `data/wisdm_subset.csv` is a source-checkout
workflow and is not bundled into the published wheel.

Example prompt for Claude Code:

```text
Install `ts-agents[recommended]`, then use the `time-series-activity-recognition` skill. Generate a synthetic labeled stream with `uv run python data/make_synthetic_labeled_stream.py --scenario gait --seconds 40 --seed 1337 --out data/demo_labeled_stream.csv`, run `ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z --output-dir outputs/activity-recognition`, and produce `outputs/reports/activity-recognition.qmd` plus `outputs/reports/activity-recognition.pdf`.
```

Example prompt for Codex:

```text
Use the `forecasting` skill. Run `ts-agents workflow show forecast-series --json`, choose the methods available in the current environment, run `ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive,arima,theta --output-dir outputs/forecasting`, summarize the artifacts, and generate `outputs/reports/forecasting-summary.qmd` plus `outputs/reports/forecasting-summary.pdf`.
```

For polished deliverables, generate a Quarto report and render to PDF:

```bash
quarto render outputs/reports/<report-name>.qmd --to pdf
```

### Skills

```bash
ts-agents skills list
ts-agents skills validate
ts-agents skills export --all-agents
ts-agents skills export --all-agents --symlink
```

Canonical skills are intentionally limited to a focused set in `skills/`.
Agent-specific folders are generated on demand via `skills export` and are not
tracked in this repository.

Copy vs symlink guidance:
- Use **copies** (default export mode) for CI, sharing, and cross-platform reliability.
- Use `--symlink` only for local Unix-like development when you want zero-copy edits.

## Gradio App

Run the app:

```bash
ts-agents-ui
uv run python main.py
```

`main.py` is the source-checkout wrapper for the packaged `ts-agents-ui`
entrypoint. Use `app.py` when you want the hosted/manual profile from a source
checkout instead.

Useful options:

```bash
ts-agents-ui --agent-type deep
ts-agents-ui --no-agent
ts-agents-ui --share
ts-agents-ui --port 8080
```

## Sandbox Backends

Tools run inside a sandbox. Pick one with `--sandbox <mode>` or set
`TS_AGENTS_SANDBOX_MODE`.

| Mode | Isolation | Requirements |
|------|-----------|--------------|
| **local** _(default)_ | None (in-process) | — |
| **subprocess** | Separate Python process | — |
| **docker** | Container | Docker running; build image first: `./build_docker_sandbox.sh` |
| **daytona** | Cloud sandbox | `pip install daytona` + `DAYTONA_API_KEY` ([Daytona docs](https://www.daytona.io/docs)); default bootstrap clones this repo + runs `pip install -e` |
| **modal** | Serverless cloud | Source-checkout deployment path: `pip install modal`, run `modal token new` (opens browser auth) or set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, then from the repo root deploy with `modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox` |

If the chosen backend is unavailable at runtime, the executor fails with a
typed error unless you pass `--allow-fallback`. See `SANDBOX.md` for details.

For full details (env vars, resource limits, networking), see `SANDBOX.md`.

## Guides

- Quickstart: `docs/quickstart.qmd`
- Workflow walkthroughs: `docs/walkthroughs.qmd`
- Evaluation harness: `docs/evaluation.qmd`
- Demo scripts: `demo/README.md`
- Data generation and licensing notes: `data/README.md`
- Docs home: `docs/index.qmd`
- Project philosophy: `docs/philosophy.qmd`
- Distribution and release notes: `docs/distribution.qmd`
- Project roadmap and priorities: `ROADMAP.md`
- Design philosophy slides (Quarto source): `docs/talks/ts_agents_talk.qmd`

## Community

- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`

## Repository Layout

- `main.py` - source-checkout wrapper for `ts-agents-ui`
- `app.py` - source-checkout wrapper for `ts-agents-hosted`
- `ts_agents/cli/` - CLI parser, command handlers, input parsing, output helpers
- `ts_agents/contracts.py` - shared data contracts (ArtifactRef, ToolPayload, CLIEnvelope, CLIError)
- `ts_agents/core/` - pure time-series algorithms
- `ts_agents/tools/` - tool registry, wrappers, execution/sandbox routing
- `ts_agents/workflows/` - first-class workflow implementations (inspect, forecast, activity)
- `ts_agents/agents/` - simple and deep agent implementations
- `ts_agents/evals/` - deterministic evaluation harness
- `ts_agents/ui/` - Gradio tabs/components
- `ts_agents/persistence/` - cache/session/experiment logging
- `tests/` - unit and CLI tests
- `benchmarks/` - checked-in benchmark snapshots and results
- `data/` - sample datasets and data generation/download scripts
- `skills/` - canonical skill definitions
- `build_docker_sandbox.sh` + `Dockerfile.sandbox` - Docker sandbox build assets

## Development

Run tests:

```bash
uv run python -m pytest -q
```

Run CLI test suite only:

```bash
uv run python -m pytest -q tests/cli
```

Render docs site locally (Quarto):

```bash
quarto render docs
quarto preview docs
```

## License

[MIT](https://github.com/fnauman/ts-agents/blob/main/LICENSE)
