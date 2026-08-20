# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-08-20

First release since March, bundling five months of agent-facing surface work
plus a run/jobs control plane.

### Added

- `ts-agents runs list/show/gc`: a catalog over every `run_manifest.json`
  under the outputs root, normalizing workflow and autoresearch manifests,
  with dry-run-by-default garbage collection.
- `ts-agents jobs start/list/status/logs/cancel`: background execution of any
  CLI command in a detached worker with a durable JSON job record, combined
  stdout/stderr log capture, exit-code finalization, and process-group
  cancellation.
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
  wheel installs and runs on Python 3.14; heavy extras that depend on numba
  (`patterns`, `classification`) remain 3.11-3.13 until numba ships 3.14
  support. CI gains a 3.14 wheel-smoke job and the publish workflows smoke
  3.14.
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

- Current package version is `0.1.1` in `pyproject.toml`.

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

- Current package version is `0.1.0` in `pyproject.toml`.
