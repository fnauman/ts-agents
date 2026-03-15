# Changelog

All notable changes to this project will be documented in this file.

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
