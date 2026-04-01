# ts-agents

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://fnauman.github.io/ts-agents/)
[![Python](https://img.shields.io/badge/python-3.11--3.13-3776AB)](#installation)
[![License](https://img.shields.io/badge/license-MIT-2EA44F)](https://github.com/fnauman/ts-agents/blob/main/LICENSE)

`ts-agents` provides time-series skills, workflow contracts, and sandboxes for
agentic workflows.

It is built around:
- a stable CLI contract for discovery and execution (`ts-agents workflow ...`, `ts-agents tool ...`)
- inspectable artifacts instead of chat-only outputs (plots, JSON, reports)
- reusable skills that encode time-series workflow guidance
- optional sandboxes for safer, reproducible execution (`local`, `subprocess`, `docker`, `daytona`, `modal`)
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
uv sync
uv run ts-agents workflow list
uv run ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
uv run ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3
uv run python data/make_synthetic_labeled_stream.py --scenario gait --seconds 40 --seed 1337 --out data/demo_labeled_stream.csv
uv run ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z
```

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

`ts-agents-hosted` is environment-variable driven rather than flag-driven.
Configure `HOST`, `PORT`, `GRADIO_SHARE`, `TS_AGENTS_ENABLE_AGENT`,
`TS_AGENTS_AGENT_TYPE`, `TS_AGENTS_PERSIST_SESSIONS`, and
`TS_AGENTS_UI_TITLE` before launch if you need non-default behavior.

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

- **CLI as the stable contract**: `ts-agents` is the primary interface for automation and reproducibility.
- **Framework adapters, not framework lock-in**: LangChain/deep-agent wrappers are convenience layers over the same tool registry.
- **Artifacts over chat**: tools produce inspectable files (plots, JSON, reports), and agents return summaries plus paths.
- **Swappable front-ends**: CLI agents, custom agents, and Gradio are interfaces around the same core tools.
- **Sandboxed execution**: backends can isolate dependencies and scale heavier workloads.

Canonical design doc:
- `docs/philosophy.qmd`

## Quickstart

```bash
uv sync
uv run ts-agents workflow list
uv run ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
uv run ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3
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

The workflow commands write artifacts under their `--output-dir`, for example
`outputs/inspect/summary.json` or `outputs/forecast/forecast.csv`.

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
deployments can still use the root `app.py` wrapper. It defaults to:
- manual analysis mode (`agent` disabled)
- no session persistence
- a public-safe configuration that does not require `OPENAI_API_KEY`

Launch it with:

```bash
ts-agents-hosted
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
ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3
```

### Run tools directly

```bash
ts-agents tool run stl_decompose_with_data --run Re200Rm200 --var bx001_real
ts-agents tool run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=30
```

### Save output and extract embedded images

```bash
ts-agents tool run forecast_theta_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --param horizon=30 \
  --save outputs/Re200Rm200/theta.txt \
  --extract-images outputs/Re200Rm200/assets
```

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
Use the `time-series-activity-recognition` skill. Generate a synthetic labeled stream with `uv run python data/make_synthetic_labeled_stream.py --scenario gait --seconds 40 --seed 1337 --out data/demo_labeled_stream.csv`, run `ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z --output-dir outputs/activity-recognition`, and produce `outputs/reports/activity-recognition.qmd` plus `outputs/reports/activity-recognition.pdf`.
```

Example prompt for Codex:

```text
Use the `forecasting` skill. Run `ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --output-dir outputs/forecasting`, summarize the artifacts, and generate `outputs/reports/forecasting-summary.qmd` plus `outputs/reports/forecasting-summary.pdf`.
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
```

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

- `main.py` - Gradio app entrypoint
- `ts_agents/cli/` - CLI parser, command handlers, input parsing, output helpers
- `ts_agents/contracts.py` - shared data contracts (ToolPayload, CLIEnvelope, ArtifactRef)
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
