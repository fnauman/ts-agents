# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-10-09

### Added

- Optional `ml` (MLForecast/LightGBM and histogram GBM) and `neural`
  (NeuralForecast/NHITS) profiles, also included in `all`. Base and recommended
  installs retain their existing dependency profiles.
- `forecast-panel` accepts regular long-format panels, refits at rolling
  validation origins, records train-only seasonal MASE plus MAE/RMSE, produces
  future forecasts, and persists native trained models with report/plot artifacts.
  It supports CLI discovery, input-content resume checks and sandbox serialization.
- Darts zero-shot foundation models `chronos2_small` (default), `chronos2`,
  `timesfm2p5` and `patchtst_fm` in `forecast-series` and `forecast-panel`,
  driven by one stdlib-only method catalog
  (`ts_agents.core.forecasting.catalog`) with pinned checkpoint revisions.
  Only the requested checkpoints download into `HF_HOME`; in both workflows
  `models/<fm>/` holds only `model_spec.json` (checkpoint, pinned revision,
  licence, resolved context; no weights) for each FM that produced forecasts,
  and reports carry zero-shot and pretraining-overlap caveats. Uncached
  weights without network fail as `backend_unavailable` (exit code 5) with the
  remediation in `hint`.
- `--context-length` on `forecast-series` and `forecast-panel` (default 512,
  validated against checkpoint-specific window limits) and
  `--accelerator {cpu,gpu}` on `forecast-series`.
  Horizon and context limits are validated before any work (exit code 2).
- Agent tools `forecast_foundation` and `forecast_foundation_with_data`
  (VERY_HIGH cost, `foundation` extra, checkpoint provenance in the payload;
  repeated calls reuse a bounded one-model cache),
  and `forecast_panel_from_csv`, which runs the `forecast-panel` workflow with
  `output_dir` defaulting under `TS_AGENTS_TOOL_ARTIFACT_DIR`. They are in the
  `full` and `forecasting` bundles and the forecasting subagent.
- `ts-agents data export-panel m4-monthly-mini --split {train,holdout,all}
  --out PATH [--train-end YYYY-MM-01]` writes a `unique_id, ds, y` panel at
  `MS` frequency. The dates are a synthetic alignment ending at `--train-end`.
- Daytona workflow runs set `TS_AGENTS_DAYTONA_INSTALL_EXTRAS` automatically:
  the workflow's required extras, the requested methods' extras, and `viz`
  unless `--skip-plots`. This applies to every workflow; a user-set value is
  never overridden.
- `Dockerfile.sandbox` accepts a `TS_AGENTS_EXTRAS` build argument, and
  `build_docker_sandbox.sh` takes extras as an optional second argument. The
  default image stays base-only and no weights are baked in.
- `workflow show forecast-panel` / `forecast-series` capabilities report
  `foundation_models`, `zero_shot_methods`, `method_extras` and
  `foundation_model_checkpoints`.

### Fixed

- Reject unknown forecasting option names, empty panel method selections and missing
  requested columns; keep existing JSON file paths ahead of inline parsing.
- Preserve native model directories and durable manifest/artifact paths across
  Docker, subprocess and remote panel-tool execution; normalize host CSV inputs
  before dispatch and install required Daytona extras for tools.
- Apply resource defaults without mutating caller contexts. Foundation inference
  preserves stdout/logger state, retains one model across repeated agent calls,
  and uses one checkpoint for series validation and future forecasts.
- Reports distinguish requested validation/future horizons from the shared model
  output chunk.
- Record the context actually used by autoresearch, validate checkpoint-specific
  window limits, align method schema defaults, and include alias notices in the
  initial manifest rather than an extra rewrite.

- Foundation-model inputs reject multidimensional histories and contexts outside
  the finite Float32 range before loading weights. `forecast-series` saves the
  actual validation configuration for nonwinning foundation models and both
  validation/future configurations for a winning foundation model.

- Large inline JSON inputs no longer fail filesystem filename-length probes.
- `forecast-panel` reruns and `--resume` replace each `models/<method>/`
  directory with a staged fit, so NHITS saves no longer fail on existing files and
  stale model files are never reported as artifacts.
- `skills validate` now exits non-zero with a `validation_error` envelope when
  any skill is invalid; the `panel-forecasting` skill gained its missing
  "When to use" section.

### Changed

- **Breaking:** the `foundation` extra is now `darts[torch]>=0.47,<0.48` plus
  `torch`, replacing `chronos-forecasting`; `all` follows. Base and
  recommended profiles are unchanged.
- **Breaking:** the `foundation-chronos-smoke` autoresearch loop is renamed
  `foundation-smoke` (scope `darts_foundation_zero_shot_smoke`). It runs
  `chronos2_small` by default, and `--models` accepts any of the four
  checkpoints. The weekly CI job is now `darts-fm-smoke`, with a dispatch
  `model` input.
- **Breaking:** `foundation-gpu-plan` writes `darts_finetune_config.yaml`
  (`Chronos2Model` with `enable_finetuning`, `amazon/chronos-2` pinned) with
  TimesFM 2.5 and PatchTST-FM zero-shot comparators. MOMENT stays in the plan
  labelled `external_plan_only` (requires `momentfm`, which no extra installs).
- `forecast-panel` availability is always `available` (the seasonal baseline
  works on every install). The ML, neural, foundation-model and plot families
  are reported under `optional_features`, and the install hint covers only
  missing families. `metrics.json` gains `foundation_models` and
  `options.resolved_context_length` when FMs are requested.
- `compare_forecasting_methods` reports unknown or panel-only methods as
  error entries instead of skipping them silently.
- Remote workflow artifact staging bundles `models/` last, so metrics, reports
  and manifests are staged first when size limits apply.
- `inspect-series` suggests `forecast-series --methods
  seasonal_naive,chronos2_small` for series shorter than 64 points.

### Deprecated

- The `foundation-chronos-smoke` loop name. It still resolves in
  `autoresearch show` and `run` (with a warning in the payload and manifest)
  but is no longer listed.

### Removed

- The `chronos-forecasting` dependency and its chronos-t5 models, the
  `chronos_finetune_config.yaml` gpu-plan artifact and the GluonTS `.arrow`
  adapter requirement.

## [0.2.2] - 2026-10-05

### Fixed

- Cancellation locks and rereads the job record after detecting lost worker ownership,
  preserving terminal status, exit code, and completion time when the supervisor
  finalized between the original read and ownership check. All job record
  writes share the same lock to prevent lost updates during finalization.
- Autoresearch CLI runs now hold hierarchy output leases before output mutation
  through host manifest synchronization, preventing concurrent GC, parent
  overwrite, and overlapping runs from removing or replacing active artifacts.

## [0.2.1] - 2026-10-05

### Fixed

- Source-archive verification now rejects untracked and duplicate members in
  maintained namespaces, path aliases, file/directory collisions, nonregular
  members, and unexpected files outside the known generated metadata. A late review identified
  the gate gap after 0.2.0 publication; independent inventory comparison
  confirmed that the published 0.2.0 archive contained no extra files.
  The runtime CLI behavior and dependency profiles are unchanged.

## [0.2.0] - 2026-10-05

First release since March, bundling the agent-facing surface work since 0.1.1
plus a run/jobs control plane.

### Added

- `ts-agents runs list/show/gc`: a catalog over every `run_manifest.json`
  under the outputs root, normalizing workflow and autoresearch manifests,
  with dry-run-by-default garbage collection.
- `ts-agents jobs start/list/status/logs/cancel`: background execution of any
  CLI command in a detached worker with a durable JSON job record, combined
  stdout/stderr log capture, exit-code finalization, and supervised local process-group
  cancellation on POSIX (Linux/macOS or WSL). Native Windows background jobs
  fail with an actionable hint; foreground commands remain available.
- `ts-agents capabilities`: machine-readable CLI discovery surface for
  autonomous agents (entrypoints, install profile, status contract, sandbox
  backends).
- `ts-agents autoresearch list/show/run`: constrained research loops
  (`forecast-daytona`, `classify-daytona`, `foundation-chronos-smoke`,
  `foundation-gpu-plan`) with budgets, trial artifacts, and rankings.
- `foundation-chronos-smoke`: an executable Chronos zero-shot forecasting
  path behind the new `foundation` extra, now exercised unmocked by a weekly
  scheduled CI workflow (real chronos-forecasting + torch on CPU).
- Workflow lifecycle contracts: run manifests, provenance, `--overwrite` /
  `--resume`, quality flags, and versioned strict-JSON envelopes with typed
  exit codes across the CLI.
- Deep-agent fallback observability: any deepagents-path failure records
  `fallback_used`, `fallback_reason`, and an install hint instead of failing
  or silently degrading.

### Changed

- `requires-python` is now `>=3.11` (previously capped `<3.14`). The base
  wheel installs and runs on Python 3.14; the locked optional dependency stack
  remains qualified on 3.11-3.13; 3.14 qualification covers the base CLI.
  CI and publish workflows smoke-test base wheels on 3.11-3.14.
- Shared artifact-staging module: the workflow and autoresearch sandbox
  executors now use one hardened implementation for path validation, symlink
  rejection, atomic writes, and staging limits.
- Autoresearch loop dispatch is metadata-driven: dependency rules live on the
  loop definition, model validation derives from the definition's model list,
  and the runner uses a single dispatch table.
- Generated reports and summaries no longer carry competitive-positioning
  prose or references to repo-only files that are not shipped in the wheel.
- The `agents` and `ui` extras are documented as experimental; the CLI is the
  supported contract surface.

### Fixed

- Cancellation now retains the supervisor through descendant termination;
  graceful timeout permits a later force request. Lost workers remain stale
  with termination unconfirmed. Log tails use bounded memory.
- Run GC retains unreadable/unclassified nested evidence and nonterminal runs,
  revalidates identities before deletion, and respects active output leases.
- Workflow resume requires the same workflow, input content/source, and options;
  compatible reruns retain creation time and ID. It is not checkpoint recovery.
  Legacy manifests without fingerprints require a new output directory.
- Active/failed workflow runs remain cataloged, concurrent writes to the same
  output directory are refused, and manifest-write errors are reported.
  All workflow manifest writes are atomic; POSIX leases validate private
  directory ownership and reject symlinks and multiply linked files.
- Installed-wheel checks execute the public CLI surface outside the checkout;
  source archives include tests, canonical skills, examples, and release tooling.
  Build gates verify source identity and wheel RECORD hashes.
- Runnable background-job example, release/support documentation, and scripted
  benchmark claim boundaries are corrected.

- Remote workflow artifact materialization created one temp directory per
  staged file when no output directory was requested, scattering a single
  bundle across many directories.
- The Docker workflow staging directory was static (`/io/artifacts`), so
  concurrent workflow runs could clobber each other's staged artifacts; it is
  now unique per run.

## [0.1.1] - 2026-03-10

Release-preparation and packaging hardening update for the first real PyPI
publish.

### Changed

- bumped the release version to `0.1.1` after the stale `v0.1.0` Git tag was
  found to point at an older pre-release commit
- aligned the documented PyPI user path with the installed wheel entrypoints
  and clarified which demo data is bundled versus source-checkout-only
- capped the advertised Python support range to the validated 3.11-3.13 matrix
  and declared missing direct runtime dependencies explicitly
- added artifact-level release gates in CI and publish workflows, including
  `twine check`, built-wheel smoke tests, TestPyPI validation, and tag/version
  matching for the real PyPI publish workflow
- tightened release metadata and tooling around the package surface, including
  `py.typed`, `__version__`, metadata tests, release-surface quality checks,
  deterministic pinned dev tools, and release helper scripts

### Notes

- This entry describes the published `0.1.1` release.

## [0.1.0] - 2026-03-05

Initial public release of `ts-agents`.

### Added

- CLI-first time-series toolkit with `ts-agents` entrypoint.
- Gradio app for manual analysis and agent-driven workflows.
- Tool registry covering decomposition, forecasting, patterns, spectral,
  classification, and statistics.
- Skill-based workflow system with export/validation commands.
- Sandbox backends: local, subprocess, docker, daytona, and modal.
- Deterministic demo workflows with `--no-llm`.
- Modal and Daytona sandbox documentation, including auth and deployment notes.
- Daytona/Modal log streaming support and optional log file output.

### Notes

- This entry describes the historical `0.1.0` tag.
