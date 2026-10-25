# ts-agents

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://fnauman.github.io/ts-agents/)
[![Python](https://img.shields.io/badge/python-3.11--3.14-3776AB)](#installation)
[![License](https://img.shields.io/badge/license-MIT-2EA44F)](https://github.com/fnauman/ts-agents/blob/main/LICENSE)

`ts-agents` is a general-purpose, CLI-first time-series toolkit for external
agents (Claude Code, Codex, and similar runtimes) and automation. It gives
agents a stable, machine-readable surface to discover methods, run analyses on
their own data files, and produce inspectable artifacts across task families:

| Task family | Start with | Skill |
| --- | --- | --- |
| First look, data quality, periodicity | `workflow run inspect-series` | `diagnostics` |
| Anomalies, outliers, unusual events | `tool run find_discords --input ...` | `anomaly-detection` |
| Change-points and regimes | `tool run segment_changepoint --input ...` | `segmentation` |
| Spectra, frequencies, coherence | `tool run compute_psd --input ... --time-col t` | `spectral-analysis` |
| Trend and seasonality | `tool run stl_decompose --input ...` | `decomposition` |
| Activity recognition (labeled sensor streams) | `workflow run activity-recognition` | `activity-recognition` |
| Classification of labeled series | `tool run compare_classifiers` | `classification` |
| Forecasting (single series / panels) | `workflow run forecast-series` / `forecast-panel` | `forecasting` / `panel-forecasting` |

`ts-agents tool search <task words> --json` finds tools by task (for example
`anomaly`, `regime`, `spectrum`). Project-specific data preparation, custom
evaluation, and comprehensive reporting can still require code.

Foundation models are reached through [Darts](https://unit8co.github.io/darts/)
as an interoperability layer: a small catalog of pinned zero-shot checkpoints
(Chronos-2, TimesFM 2.5, PatchTST-FM) behind the optional `foundation` extra,
exercised through the same CLI/workflow contract as every other method.

It is built around:
- a stable CLI contract for bootstrap, discovery, and execution (`ts-agents capabilities`, `ts-agents workflow ...`, `ts-agents tool ...`)
- strict JSON envelopes with `schema_version`, typed exit codes, top-level `quality_status`/`degraded`/`requires_review`, and nested workflow `result.status`/`result.data.quality_flags`
- run lifecycle metadata: generated run IDs, `run_manifest.json`, non-clobbering default output directories, and `--resume` / `--overwrite` semantics
- inspectable artifacts instead of chat-only outputs (plots as `ArtifactRef` files, JSON payloads, Markdown reports)
- reusable skills that encode time-series workflow guidance as install-agnostic command templates
- optional sandboxes for safer, reproducible execution (`local`, `subprocess`, `docker`, `daytona`, `modal`) with readiness probes and explicit fallback flags
- optional tool wrappers for external agent frameworks; the package runs no built-in agents or UI

It ships with four first-class workflows:
- `inspect-series` (task-neutral diagnostics: sampling/gap checks, significance-tested periodicity, regime screen, next steps across task families)
- `forecast-series` (baseline comparison + forecast/report artifacts; optional zero-shot foundation models)
- `forecast-panel` (seasonal baseline, MLForecast GBMs, NeuralForecast NHITS and Darts zero-shot foundation models with rolling validation and saved models)
- `activity-recognition` (labeled-stream window-size selection + held-out evaluation, with `--group-col` for leave-subject-out scoring)

It also includes autoresearch loops for repeatable dataset/model/metric
experiments:
- `forecast-daytona` (M4 mini forecasting baselines under constrained resources)
- `classify-daytona` (windowed activity classification under constrained resources)
- `foundation-smoke` (optional Darts zero-shot foundation-model smoke run on M4 mini; `--models` selects checkpoints)
- `foundation-gpu-plan` (plan-only Darts Chronos-2 fine-tuning recipe with TimesFM/PatchTST-FM comparators and external MOMENT classification)

Source-checkout-only datasets such as `data/wisdm_subset.csv` are documented
separately and are not part of the published wheel.

**Start here:** [Quickstart](#quickstart) | [Choose your path](#choose-your-path) | [Docs site](https://fnauman.github.io/ts-agents/) | [Evaluation harness](https://fnauman.github.io/ts-agents/evaluation.html) | [Workflow walkthroughs](https://fnauman.github.io/ts-agents/walkthroughs.html)

## Table of Contents

- [Choose your path](#choose-your-path)
- [For autonomous agents](#for-autonomous-agents)
- [Why ts-agents instead of using statsforecast/sktime/aeon directly?](#why-ts-agents-instead-of-using-statsforecastsktimeaeon-directly)
- [Design principles](#design-principles)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [CLI usage](#cli-usage)
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
`ts-agents[recommended]` for the main workflow stack, including
ARIMA/ETS/Theta forecasting and `activity-recognition`. From a source
checkout, plain `uv sync` matches the base CLI-first install; use
`uv sync --extra recommended` for the same recommended workflow stack.

### 2. Run autoresearch loops

Use autoresearch loops when you want a bounded comparison plan with datasets,
models, metrics, budgets, and artifacts chosen up front. The Daytona-oriented
statistical/classical loops are benchmark-style searches: `--max-trials` counts
model/evaluation-spec rows. They default to vendored or generated datasets and
fit 4 vCPU / 8 GiB RAM / 10 GiB disk sandboxes.

```bash
ts-agents autoresearch list --json
ts-agents autoresearch show forecast-daytona --json
ts-agents autoresearch run forecast-daytona --profile smoke --models seasonal_naive --json
ts-agents autoresearch run classify-daytona --profile smoke --dataset synthetic --models knn --json
ts-agents autoresearch run foundation-smoke --dry-run --json
ts-agents autoresearch run foundation-smoke --models timesfm2p5 --json
ts-agents autoresearch run foundation-gpu-plan --json
```

Pass `--sandbox daytona` to run an autoresearch loop through the Daytona
backend after configuring `DAYTONA_API_KEY`; use `--dry-run` to materialize the
trial plan without fitting models. `foundation-smoke` is the executable
foundation-model loop: it scores zero-shot Darts checkpoints on the M4 mini
holdout, defaulting to `chronos2_small` (`--models` also accepts `chronos2`,
`timesfm2p5` and `patchtst_fm`). Only the selected checkpoints download into
`HF_HOME`. Real runs require `ts-agents[foundation]` and lazy-load
`darts`/`torch`/`huggingface_hub`, while dry-run mode needs no heavy TSFM
dependencies. The old `foundation-chronos-smoke` name is a deprecated alias
that resolves to `foundation-smoke` with a warning. `foundation-gpu-plan` is
intentionally plan-only and reports no trained-model metrics; it writes a
`darts_finetune_config.yaml` and labels MOMENT as external (plan-only, needs
`momentfm`).

### 3. Run individual analysis tools on your own files

Use the tool registry for direct access to individual analysis functions.
`--input` loads the `series` parameter from CSV, TSV, parquet, JSON, or JSONL;
`--time-col` orders rows, checks sampling (gaps, irregular steps), and — for a
regular numeric time column — sets `sample_rate` so periods and frequencies
come back in time units. `data/synthetic_signals.csv` (source checkout) is a
50 Hz table with known answers.

```bash
ts-agents tool search anomaly --json
ts-agents tool show find_discords --json
ts-agents tool run detect_periodicity --input data/synthetic_signals.csv --time-col t --value-col periodic --json
ts-agents tool run segment_changepoint --input data/synthetic_signals.csv --value-col regime --json
ts-agents tool run find_discords --input data/synthetic_signals.csv --value-col anomaly --param m=100 --json
ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}'
```

Every tool run reports `quality_status` (`ok`, or `review` when
`result.warnings` or `input.warnings` exist). Invalid, too-short, non-finite,
or multichannel input to a univariate tool fails with `ok: false` instead of
returning an empty answer. Arrays longer than 16 values are echoed in
`input.params` as length/hash summaries.

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
ts-agents tool show segment_changepoint --json
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

Autoresearch dry runs report `planned` (no fitting/scoring); the GPU recipe
reports `plan-only`. Opted-in execution fallback sets `requires_review`.

### 4. Run lifecycle and provenance

Workflow runs produce:

- a generated run ID and run-scoped output directory under `outputs/<workflow>/<run-id>/`
- a `run_manifest.json` capturing inputs, parameters, execution backend
  metadata, and emitted artifacts
- absolute paths on every `ArtifactRef` so an agent can materialize files from
  any working directory
- non-clobbering defaults, plus `--overwrite` / `--resume` semantics for retry
  loops and long sessions

`--resume` reruns the computation with the same workflow, normalized input
content, source interpretation, and options while keeping the run ID and
original creation time. It does not recover a computational checkpoint.
Changed analyses and legacy manifests without a resume fingerprint require a
new output directory. Active/interrupted runs are retained for inspection;
failed executions remain visible in the run catalog.

GC previews candidates by default and retains directories containing
unreadable, unclassified, or nonterminal run manifests. Stop concurrent work
and inspect the preview before applying cleanup.
Workflow and autoresearch CLI runs hold output leases throughout execution and
host manifest updates. These leases also protect unpublished nested runs from
concurrent parent cleanup or overwrite.

Past runs are a first-class surface. `runs` catalogs every manifest under the
outputs root, and `jobs` runs any CLI command in a detached background worker
with a durable record, log capture, and cancellation:

```bash
ts-agents runs list --json
ts-agents runs show <run-id> --json
ts-agents runs gc --older-than 30 --apply --json

ts-agents jobs start --json -- workflow run forecast-series \
  --input-json '{"series":[1,2,3,4,5,6,7,8,9,10,11,12]}' \
  --horizon 3 --methods seasonal_naive --skip-plots --json
ts-agents jobs status <job-id> --json
ts-agents jobs logs <job-id> --tail 50
ts-agents jobs cancel <job-id> --json
```

Background jobs require POSIX (Linux/macOS or WSL); native Windows supports
foreground commands. Cancellation is supervised: a timeout leaves the job active so
`jobs cancel <job-id> --force` can terminate a resistant local command process
group. A missing worker is reported as `stale`, with cancellation unconfirmed.
This does not establish termination of remote Docker/Daytona/Modal work or
processes that detach themselves into another session.

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
Daytona, and Modal all stage workflow and autoresearch artifacts back to the host output
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
- pinned zero-shot foundation models through Darts without taking on model-hub ownership
- optional sandbox backends for isolation, deployment, and heavier workloads
- stable CLI/tools and optional wrappers for external agent orchestration

## Design Principles

- **CLI as the stable contract**: `ts-agents` is the primary interface for automation and reproducibility. Autonomous agents plan against `capabilities`, `workflow show`, and `tool show` instead of hardcoded knowledge.
- **Strict machine envelopes**: `--json` output is versioned, strict, and typed — with status, quality flags, and exit codes — so agents can branch on failure mode rather than parsing prose.
- **External agent integration**: optional LangChain/deep-agent tool wrappers expose the same registry. External runtimes own orchestration and narrative generation.
- **Scoped TSFM interop, not a model hub**: Darts is the foundation-model interop layer and external projects such as TimeCopilot are comparator targets; `ts-agents` keeps a small catalog of pinned checkpoints (revision, licence and resolved context recorded in artifacts) rather than owning model breadth.
- **Artifacts over chat**: tools produce inspectable files (plots, JSON, reports), and agents return summaries plus paths.
- **Run lifecycle as first-class metadata**: every workflow run gets a run ID, a `run_manifest.json`, and non-clobbering defaults — so long, multi-turn sessions remain traceable and safe to rerun.
- **Broad analysis scope**: activity recognition, classification, anomaly/pattern detection, forecasting, decomposition, and spectral analysis share the CLI/tool contract.
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
uv sync --extra recommended
uv run ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive,arima,theta
uv run python data/make_synthetic_labeled_stream.py --scenario gait --seconds 40 --seed 1337 --out data/demo_labeled_stream.csv
uv run ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z
# Leave-subject-out activity evaluation on the vendored WISDM subset
uv run ts-agents workflow run activity-recognition --input data/wisdm_subset.csv --label-col label --group-col subject_id

# Non-forecasting analyses on a known-answer table (needs the patterns extra)
uv sync --extra recommended --extra patterns
uv run ts-agents tool run segment_changepoint --input data/synthetic_signals.csv --value-col regime --json
```

If you pass `--output-dir`, workflow artifacts are written there. If you omit
it, each workflow run creates a unique run directory such as
`outputs/<workflow>/<run-id>/` with `run_manifest.json`, JSON/CSV outputs, and
any generated plots or reports.

## Installation

Version 0.5.0 (unreleased) removes the `ts-agents run ...` and `ts-agents demo ...`
aliases and the 24 `*_with_data` tools that read the bundled demo dataset; use
`ts-agents tool run <tool> --input <file>` and `ts-agents workflow run ...`.
See the [changelog](CHANGELOG.md) for migration notes. Version 0.4.0 removed
the Gradio UI, hosted app, and built-in agents; `agents` installs only optional
LangChain tool wrappers.

For the GBM/NHITS forecasting and reporting profile:

```bash
python -m pip install "ts-agents[ml,neural,forecasting,viz]==0.4.0"
```

Prerequisites:
- Python 3.11-3.14 for the base CLI; use 3.11-3.13 for the qualified locked
  optional dependency stack. Heavy extras are not release-qualified on 3.14.
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
python -m pip install "ts-agents[classification,patterns]"
python -m pip install "ts-agents[recommended]"
python -m pip install "ts-agents[all]"
```

Feature extras:
- `agents`: optional LangChain tool wrappers for external runtimes; no built-in agents
- `decomposition`: STL, MSTL, Holt-Winters
- `forecasting`: statistical forecasting tools
- `patterns`: matrix profile (motifs, discords, FLUSS) and changepoint tooling
- `classification`: aeon/scikit-learn classification workflows
- `ml`: MLForecast LightGBM and histogram GBM panel models
- `neural`: NeuralForecast NHITS panel models
- `foundation`: Darts (`darts[torch]`) zero-shot foundation models; weights are not packaged and download per requested checkpoint into `HF_HOME` on first use
- `viz`: plotting helpers
- `recommended`: classical forecasting, decomposition, classification, and plotting
- `all`: the full optional stack

Install profiles:
- `ts-agents`: workflow discovery, `workflow show`, `inspect-series`, and a dependency-light `seasonal_naive` forecast baseline
- `ts-agents[forecasting]`: unlocks ARIMA, ETS, and Theta for `forecast-series`
- `ts-agents[ml]`: MLForecast panel models (`lightgbm`, `histgbm`; available since 0.3.0)
- `ts-agents[neural]`: NeuralForecast panel models (`nhits`; available since 0.3.0)
- `ts-agents[foundation]`: Darts zero-shot foundation models (`chronos2_small`, `chronos2`, `timesfm2p5`, `patchtst_fm`) for `forecast-series`, `forecast-panel` and `foundation-smoke`; Darts backend available since 0.3.0
- `ts-agents[classification]`: unlocks `activity-recognition`
- `ts-agents[recommended]`: the main workflow stack used in walkthroughs and demos
- `source checkout + uv sync`: same base CLI-first profile as `ts-agents`
- `source checkout + uv sync --extra recommended`: same recommended profile as `ts-agents[recommended]`

Run the packaged entrypoints:

```bash
ts-agents --help
ts-agents tool list
```

If you are running from a source checkout, prefix the CLI commands below with
`uv run` after syncing the extras you need.

Source checkout setup:

```bash
git clone https://github.com/fnauman/ts-agents.git
cd ts-agents
uv sync
```

Plain `uv sync` matches the base CLI-first install profile. Add extras as
needed:

```bash
uv sync --extra recommended
uv sync --all-extras
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

### Environment variables

All optional. Set them via `export` or in `~/.env`.

| Variable | Purpose | Default |
|----------|---------|---------|
| `TS_AGENTS_DATA_DIR` | Directory for `--run-id/--variable` bundled workflow inputs | bundled package data (or repo `./data`) |
| `TS_AGENTS_USE_TEST_DATA` | Use the bundled 5-point-per-run smoke fixture for `--run-id/--variable` | `true` |
| `TS_AGENTS_TEST_DATA_FILE` | Override the bundled fixture filename | `short_real.csv` |
| `TS_AGENTS_SANDBOX_MODE` | Default sandbox backend | `local` |
| `HF_HOME` | Hugging Face cache for foundation-model weights (`foundation` extra); set `HF_HUB_OFFLINE=1` to use a pre-populated cache without network | `~/.cache/huggingface` |

Sandbox-specific environment variables (Docker/Daytona/Modal auth, snapshots,
streaming, and log files) are documented in `SANDBOX.md`.

## Distribution

- Package metadata is configured for the `ts-agents` distribution name.
- GitHub Actions includes a PyPI publish workflow for tagged releases.
- GitHub Actions includes a GHCR workflow for publishing the sandbox image built
  from `Dockerfile.sandbox`.
- GitHub release/tag/docs flow is summarized in [Distribution guide](https://fnauman.github.io/ts-agents/distribution.html).

## CLI Usage

### Discover tools

```bash
ts-agents capabilities --json
ts-agents tool search changepoint --json
ts-agents tool list --category patterns
ts-agents tool list --bundle minimal
```

### Run workflows

```bash
ts-agents workflow list
ts-agents workflow show forecast-series --json
ts-agents workflow run inspect-series --input-json '{"series":[1,2,3,4]}'
ts-agents workflow run forecast-series --input-json '{"series":[1,2,3,4,5,6,7,8,9,10]}' --horizon 3 --methods seasonal_naive

# Panel forecasting on the bundled M4 mini-panel (train split only)
ts-agents workflow show forecast-panel --json
ts-agents data export-panel m4-monthly-mini --split train --out panel.csv --json
ts-agents workflow run forecast-panel --input panel.csv --freq MS --horizon 18 --season-length 12 --methods seasonal_naive --skip-plots
# With ts-agents[foundation]: add a zero-shot foundation model
ts-agents workflow run forecast-panel --input panel.csv --freq MS --horizon 18 --season-length 12 --methods seasonal_naive,chronos2_small --skip-plots
```

See the [panel and foundation-model forecasting guide](https://fnauman.github.io/ts-agents/panel-forecasting.html)
for method families, pinned checkpoints and validation semantics.

Use `workflow show` before automation to inspect required extras, supported
input modes, artifact outputs, and availability in the current environment.
If you omit `--output-dir`, the workflow creates a run-scoped directory under
`outputs/<workflow>/` and writes `run_manifest.json` plus the generated
artifacts there.

### Run tools directly

```bash
ts-agents tool run stl_decompose --input data.csv --value-col y --param period=12 --json
ts-agents tool run forecast_theta --input data.csv --value-col y --param horizon=30 --json
ts-agents tool run compute_psd --input signal.csv --time-col t --value-col u --json --save outputs/psd.json
```

Tools return data, not plots; use workflows (`--output-dir`) when you want
plots, CSVs, and reports written as artifacts. `--save` writes the exact
envelope that was printed.

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

## Sandbox Backends

Tools run inside a sandbox. Pick one with `--sandbox <mode>` or set
`TS_AGENTS_SANDBOX_MODE`.

| Mode | Isolation | Requirements |
|------|-----------|--------------|
| **local** _(default)_ | None (in-process) | — |
| **subprocess** | Separate Python process | — |
| **docker** | Container | Docker running; build image first: `./build_docker_sandbox.sh` (base install; pass extras as a second argument, e.g. `bash build_docker_sandbox.sh ts-agents-sandbox:ml ml`) |
| **daytona** | Cloud sandbox | `pip install daytona` + `DAYTONA_API_KEY` ([Daytona docs](https://www.daytona.io/docs)); default bootstrap clones this repo + runs `pip install -e` |
| **modal** | Serverless cloud | Source-checkout deployment path: `pip install modal`, run `modal token new` (opens browser auth) or set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, then from the repo root deploy with `modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox` |

If the chosen backend is unavailable at runtime, the executor fails with a
typed error unless you pass `--allow-fallback`. A substitution sets `requires_review`
and records the requested and actual backend. See `SANDBOX.md` for details.

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
- Historical design talk (includes pre-0.4 UI/agent architecture): `docs/talks/ts_agents_talk.qmd`

## Community

- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`

## Repository Layout

- `ts_agents/cli/` - CLI parser, command handlers, input parsing, output helpers
- `ts_agents/contracts.py` - shared data contracts (ArtifactRef, ToolPayload, CLIEnvelope, CLIError)
- `ts_agents/core/` - pure time-series algorithms
- `ts_agents/tools/` - tool registry, wrappers, execution/sandbox routing
- `ts_agents/workflows/` - first-class workflow implementations (inspect, forecast-series, forecast-panel, activity)
- `ts_agents/evals/` - deterministic evaluation harness
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
