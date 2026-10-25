# AGENTS.md

`ts-agents` is a CLI-first, general-purpose time-series toolkit (activity
recognition, classification, anomaly/pattern detection, change-points,
forecasting, decomposition, spectral analysis) driven by external agents
through the CLI, tools, and skills. Setup, install profiles, and check
commands live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Invariants

- No misleading success: errors, invalid or too-short input, and
  method/backend substitutions surface as failed or degraded results with
  warnings, never as `ok` with an empty or wrong answer. Report the method and
  backend that actually ran.
- Preserve units, timestamp meaning, metric direction, and train/test
  boundaries. Changing a numerical default needs a known-answer test and a
  CHANGELOG entry.
- Base CLI discovery works without optional extras; keep heavy imports lazy.
  Dependency versions live only in `pyproject.toml` and `uv.lock`.
- Routine tests do not download model weights, train substantial models,
  call paid services, or write into the checkout.
- `skills/` is canonical; `ts_agents/resources/skills/` stays byte-identical.

## Docs to update with behavior changes

| Subject | Authoritative location |
| --- | --- |
| Installation and distribution | `README.md`, `docs/distribution.qmd` |
| Contributor commands and check profiles | `CONTRIBUTING.md` |
| CLI contracts and run/job lifecycle | `docs/cli-examples.qmd` |
| Forecasting and panel limits | `docs/panel-forecasting.qmd` |
| Execution backends and resource limits | `SANDBOX.md` |
| Skills and exports | `skills/README.md`, `docs/skills-guide.qmd` |
| Evaluation evidence | `docs/evaluation.qmd`, `benchmarks/` |
| Shipped changes and future scope | `CHANGELOG.md`, `ROADMAP.md` |

## PRs

State behavior changes, docs updated, checks actually run (with results), and
risks. Never claim a check passed unless it ran.
