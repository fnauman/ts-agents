# Roadmap

This roadmap is intentionally lightweight and outcome-focused.
It complements `README.md` (what exists today) with direction (what is next).

## Working Principles

- Keep the `ts-agents` CLI contract stable.
- Treat artifacts (plots, reports, JSON, logs) as first-class outputs.
- Keep orchestration in external agent runtimes; maintain broad time-series
  analysis capabilities across task families, not forecasting alone.
- Use sandbox backends to handle dependency and runtime isolation.

## Recently Shipped (0.4.1)

- Task-family parity: file input for every series tool, discovery by task
  words, and skills for anomaly detection, segmentation, and spectral
  analysis alongside diagnostics, decomposition, activity recognition,
  classification, and forecasting.
- Integrity of non-forecasting results: unit-invariant change-points,
  significance-tested periodicity, explicit failures instead of empty answers,
  and held-out (optionally leave-subject-out) activity evaluation.

## Earlier Shipped (0.4.0)

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

- **Pilots before context APIs.** Run bounded, agent-driven pilots outside
  forecasting with labeled ground truth: enterprise activity/classification
  (leave-subject-out), engineering anomaly detection on labeled events, and
  physics/DNS spectral or regime tasks with known answers. Score correctness,
  misleading-success incidents, custom glue code, and the context the agent
  had to be told.
- **Domain context.** Units, sampling conventions, channel meanings,
  operating regimes, physical constraints, and task objectives should inform
  preparation, evaluation, and interpretation. Start from the time-axis
  diagnostics now reported with every series input; add a recorded context
  file, then promote only the fields pilots show change a computation.
- **Evaluation per task family.** Event-level anomaly scoring, change-point
  annotation scoring, and known-answer spectral checks, plus agent-in-the-loop
  scenarios that grade answers rather than command success.
- Keep published artifacts, install instructions, and source claims aligned
  through installed-wheel and source-identity release gates.

## Decisions Needed

- **MCP:** CLI + skills is the current bet for Claude Code and Codex. Revisit
  when a named consumer needs an adapter, including authentication, process
  lifecycle, and error semantics.

## Next

- Task workflows chosen from pilot evidence (for example anomaly detection
  with event scoring, regime segmentation, or applying a trained activity model
  to unlabeled streams).
- Multichannel analysis beyond pairwise coherence (lag cross-correlation,
  multivariate change-points and discords).
- Missing-data and resampling policies recorded in provenance; optional
  loaders for scientific formats (HDF5/NetCDF) if pilots need them.
- Job-aware progress reporting for long runs.
- Better environment resolution and caching for heavy optional dependencies
  (the locked optional stack is not yet qualified on 3.14).

## Later

- External covariates, calibrated intervals/quantiles, foundation Modal
  extras, and FMs in `forecast-daytona`.
- Checkpoint recovery for interrupted autoresearch/workflow computations.
  Current workflow `--resume` is a compatible rerun, not checkpoint recovery.
- Richer experiment history and diffing of run outputs/artifacts.
- Human-in-the-loop gates for expensive/high-risk operations.

## How This Roadmap Is Maintained

- Keep entries short and measurable.
- Prefer capability-level milestones over date promises.
- Update whenever a major direction changes or a milestone lands.
