# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-10-05

### Fixed

- Source-archive verification now rejects untracked and duplicate members in
  maintained namespaces, path aliases, and nonregular members. A late review identified
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
