# Roadmap

This roadmap is intentionally lightweight and outcome-focused.
It complements `README.md` (what exists today) with direction (what is next).

## Working Principles

- Keep the `ts-agents` CLI contract stable.
- Treat artifacts (plots, reports, JSON, logs) as first-class outputs.
- Keep orchestration in external agent runtimes; maintain broad time-series analysis capabilities.
- Use sandbox backends to handle dependency and runtime isolation.

## Recently Shipped (0.4.0)

- Remove Gradio, hosted entrypoints, and built-in agents while retaining activity
  recognition, classification, anomaly/pattern detection, forecasting,
  decomposition, spectral analysis, skills, and external tool wrappers.
- Reject invalid ensemble selections and primary trial scores; validate
  single-series chronology; distinguish planned runs and backend substitutions.

## Earlier Shipped (0.3.0)

- Long-format panel forecasting with MLForecast LightGBM/histogram GBM,
  NeuralForecast NHITS, chronological backtests, train-only MASE scales, and
  persisted native models.
- Darts zero-shot foundation models (`chronos2_small`, `chronos2`,
  `timesfm2p5`, `patchtst_fm`) across series/panel workflows and agent tools.
  The `foundation` extra replaces `chronos-forecasting`; `foundation-smoke`
  replaces the old loop name (retained as a deprecated alias).
- Weekly unmocked Darts Chronos-small CPU smoke, installed-wheel checks, and
  source/artifact identity gates.

The first bounded electricity-demand dogfood completed on a local RTX 5090:
LightGBM, NHITS, and seasonal baselines ran on 30 hourly customer series with a
separate monthly holdout. This exercises one task, not general GPU or model
qualification. Preparation, frozen-model daily holdout evaluation, monitoring,
and comprehensive report regeneration required custom experiment code.
Validation covered only Sundays; scaled errors were sensitive to one nearly
constant customer, and NHITS produced a few negative demand predictions.

## Earlier Shipped (0.2.x)

- Run catalog: `ts-agents runs list/show/gc` over every `run_manifest.json`.
- Background jobs: `ts-agents jobs start/list/status/logs/cancel` with durable
  job records, log capture, and process-group cancellation.
- Python 3.14 base-install support; CI wheel smoke on 3.14.

## Current Focus

- Run further bounded dogfoods before choosing reusable preparation,
  frozen-model evaluation, and report-regeneration APIs. Record custom glue,
  domain constraints, and validation coverage instead of broadening capability
  claims from one successful task.
- External covariates, calibrated intervals/quantiles, foundation Modal
  extras, and FMs in `forecast-daytona` remain deferred.
- Keep published artifacts, install instructions, and source claims aligned
  through installed-wheel and source-identity release gates.
- Checkpoint recovery for interrupted autoresearch/workflow computations.
  Current workflow `--resume` is a compatible rerun, not checkpoint recovery.
- Job-aware progress reporting: stream trial/step progress into the job record
  so `jobs status` can show percent-complete for long runs.

## Decisions Needed

- **MCP:** evaluate an adapter when a named consumer needs it, including
  authentication, process lifecycle, and error semantics. Decide deliberately whether "CLI+skills only" is the bet, and
  record why.

## Next

- Better environment resolution and caching for heavy optional dependencies
  (the locked optional stack is not yet qualified on 3.14).
- Richer experiment history and diffing of run outputs/artifacts on top of the
  `runs` catalog.

## Later

- Human-in-the-loop gates for expensive/high-risk operations.
- Improved hybrid tool routing (heuristics + LLM + evaluation feedback).

## How This Roadmap Is Maintained

- Keep entries short and measurable.
- Prefer capability-level milestones over date promises.
- Update whenever a major direction changes or a milestone lands.
