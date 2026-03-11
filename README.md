# ts-agents

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://fnauman.github.io/ts-agents/)
[![Python](https://img.shields.io/badge/python-3.11--3.13-3776AB)](#installation)
[![License](https://img.shields.io/badge/license-MIT-2EA44F)](https://github.com/fnauman/ts-agents/blob/main/LICENSE)

`ts-agents` is a CLI-first toolkit for time-series analysis and agent-driven
automation. It combines:
- a stable CLI contract for reproducible runs (`ts-agents`)
- inspectable artifacts instead of chat-only outputs (plots, JSON, reports)
- optional sandboxes for safer execution (`local`, `subprocess`, `docker`, `daytona`, `modal`)
- both a Gradio UI (`ts-agents-ui`) and tool-driven agents (simple + deep)

It ships with two out-of-the-box demos:
- `window-classification` (synthetic labeled-stream window-size selection + evaluation)
- `forecasting` (baseline comparison and report artifacts)

Source-checkout-only datasets such as `data/wisdm_subset.csv` are documented
separately and are not part of the published wheel.

**Start here:** [Quickstart](#quickstart) | [Choose your path](#choose-your-path) | [Docs site](https://fnauman.github.io/ts-agents/) | [Distribution guide](https://fnauman.github.io/ts-agents/distribution.html) | [Demo walkthroughs](https://fnauman.github.io/ts-agents/walkthroughs.html)

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

### 1. Run a deterministic demo in under a minute

Use the scripted flows when you want a quick proof that the toolchain works,
without requiring an LLM key.

```bash
uv sync
uv run ts-agents demo window-classification --no-llm
uv run ts-agents demo forecasting --no-llm
```

### 2. Use the CLI on bundled or custom data

Use the CLI when you want reproducible commands, saved artifacts, and easy
automation.

```bash
ts-agents tool list --bundle demo
ts-agents run stl_decompose_with_data --run Re200Rm200 --var bx001_real
ts-agents demo window-classification --no-llm
```

### 3. Launch the UI or prepare a hosted demo

Use the Gradio app for interactive exploration, or the hosted entrypoint for a
public manual-mode deployment.

```bash
ts-agents-ui
ts-agents-hosted
```

For hosted deployment details, see the `ts-agents-hosted --help` output.

## Why ts-agents Instead of Using statsforecast/sktime/aeon Directly?

Those libraries are excellent algorithm/toolkit layers, and `ts-agents`
intentionally builds on that ecosystem rather than trying to replace it.

Use the underlying libraries directly when:
- you only need one modeling library inside a notebook or a custom pipeline
- you do not need artifacts, tool routing, or sandboxed execution

Use `ts-agents` when you want:
- a stable CLI contract that works the same across demos, agents, and automation
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

Canonical design docs:
- `docs/philosophy.qmd`
- `docs/architecture.qmd`

## Quickstart

```bash
uv sync
uv run ts-agents demo window-classification --no-llm
uv run ts-agents demo forecasting --no-llm
```

LLM-backed demo/report mode requires `OPENAI_API_KEY`. Either export it
directly or add it to `~/.env` (one `KEY=VALUE` per line; the app loads this
file automatically and will not overwrite variables already in your shell):

```bash
# Option A: export in your shell
export OPENAI_API_KEY=your-key

# Option B: store in ~/.env (loaded automatically)
echo 'OPENAI_API_KEY=your-key' >> ~/.env
```

```bash
uv run ts-agents demo window-classification
```

The demo writes plots to `outputs/demo/` (e.g. `window_scores.png`).

## Installation

Prerequisites:
- Python 3.11, 3.12, or 3.13
- [uv](https://github.com/astral-sh/uv)

Install from PyPI:

```bash
python -m pip install ts-agents
```

The default install is intentionally all-in-one and pulls a fairly heavy
forecasting/ML stack, including neural backends used by the shipped tool
surface. There is not yet a slim extras-based install profile, so plan for a
full scientific Python environment.
The dependency minimums also intentionally track the currently validated
`0.1.1` stack for this alpha release; widening lower-bound compatibility
is a follow-up task rather than part of the initial publish gate.

Run the packaged entrypoints:

```bash
ts-agents --help
ts-agents-ui --help
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

Run `ts-agents-hosted --help` for deployment options and optional agent-mode configuration.

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

### Run tools directly

```bash
ts-agents run stl_decompose_with_data --run Re200Rm200 --var bx001_real
ts-agents run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=30
```

### Save output and extract embedded images

```bash
ts-agents run forecast_theta_with_data \
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

### Demos

```bash
# Scripted (no API key required)
ts-agents demo window-classification --no-llm
ts-agents demo forecasting --no-llm

# LLM-backed report mode
ts-agents demo window-classification
ts-agents demo forecasting
```

Skill mapping for end-to-end demo runs:
- `demo window-classification` -> `activity-recognition` skill
- `demo forecasting` -> `forecasting` skill

Note: the WISDM example under `data/wisdm_subset.csv` is a source-checkout
workflow and is not bundled into the published wheel.

Example prompt for Claude Code:

```text
Use the `time-series-activity-recognition` skill. Run `ts-agents demo window-classification --no-llm`, save outputs under `outputs/demo/`, and produce `outputs/reports/REPORT.qmd` plus `outputs/reports/REPORT.pdf`.
```

Example prompt for Codex:

```text
Use the `forecasting` skill. Run `ts-agents demo forecasting --no-llm`, summarize the outputs, and generate `outputs/reports/REPORT.qmd` plus `outputs/reports/REPORT.pdf`.
```

For polished deliverables, generate a Quarto report and render to PDF:

```bash
quarto render outputs/reports/REPORT.qmd --to pdf
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

If the chosen backend is unavailable at runtime the executor falls back to
**local** with a warning.

For full details (env vars, resource limits, networking), see `SANDBOX.md`.

## Guides

- Quickstart: `docs/quickstart.qmd`
- Demo walkthroughs: `docs/walkthroughs.qmd`
- Demo scripts: `demo/README.md`
- Data generation and licensing notes: `data/README.md`
- Docs home: `docs/index.qmd`
- Project philosophy: `docs/philosophy.qmd`
- Architecture: `docs/architecture.qmd`
- Project roadmap and priorities: `ROADMAP.md`
- Design philosophy slides (Quarto source): `docs/talks/ts_agents_talk.qmd`

## Community

- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`

## Repository Layout

- `main.py` - Gradio app entrypoint
- `ts_agents/cli/` - CLI parser, command handlers, output helpers
- `ts_agents/core/` - pure time-series algorithms
- `ts_agents/tools/` - tool registry, wrappers, execution/sandbox routing
- `ts_agents/agents/` - simple and deep agent implementations
- `ts_agents/ui/` - Gradio tabs/components
- `ts_agents/persistence/` - cache/session/experiment logging
- `tests/` - unit and CLI tests
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
