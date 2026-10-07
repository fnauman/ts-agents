# Roadmap

This roadmap is intentionally lightweight and outcome-focused.
It complements `README.md` (what exists today) with direction (what is next).

## Working Principles

- Keep the `ts-agents` CLI contract stable.
- Treat artifacts (plots, reports, JSON, logs) as first-class outputs.
- Keep agent frameworks and UI layers swappable.
- Use sandbox backends to handle dependency and runtime isolation.

## Recently Shipped (0.2.0)

- Run catalog: `ts-agents runs list/show/gc` over every `run_manifest.json`.
- Background jobs: `ts-agents jobs start/list/status/logs/cancel` with durable
  job records, log capture, and process-group cancellation.
- Python 3.14 base-install support; CI wheel smoke on 3.14.
- Weekly unmocked `foundation-chronos-smoke` CI run (real chronos + torch),
  so chronos API drift is caught in CI rather than by users. (Unreleased:
  replaced by the Darts-based `foundation-smoke` loop and `darts-fm-smoke` CI
  job; the old name is a deprecated alias.)

## Current Focus

- Qualify the new MLForecast GBM and NeuralForecast NHITS panel workflow on
  the bounded electricity-demand dogfood. External covariates and calibrated
  intervals remain future additions.
- Ship the Darts zero-shot foundation models (`chronos2_small`, `chronos2`,
  `timesfm2p5`, `patchtst_fm`) across `forecast-series`, `forecast-panel`,
  agent tools and `foundation-smoke` as a breaking `[foundation]` release
  (suggested 0.3.0). Quantile outputs, Modal extras, the UI forecasting tab
  and FMs in `forecast-daytona` are deferred.
- Keep published artifacts, install instructions, and source claims aligned
  through installed-wheel and source-identity release gates.
- Checkpoint recovery for interrupted autoresearch/workflow computations.
  Current workflow `--resume` is a compatible rerun, not checkpoint recovery.
- Job-aware progress reporting: stream trial/step progress into the job record
  so `jobs status` can show percent-complete for long runs.

## Decisions Needed

- **agents/ + ui/ (~20% of the package):** the deep-agent adapter and Gradio UI
  are experimental. Unit tests cover adapter contracts, but real LLM
  interactions and browser UI behavior remain outside release qualification. Decide within the
  0.3.x cycle: extract to a separate repo, delete, or commit with real tests.
  Until then, no polish-only PRs against these layers.
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
