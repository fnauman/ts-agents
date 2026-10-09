---
name: panel-forecasting
description: Compare a seasonal baseline, MLForecast LightGBM or histogram GBM, NeuralForecast NHITS and Darts zero-shot foundation models (Chronos-2, TimesFM 2.5, PatchTST-FM; `foundation` extra) on regular panels, with chronological backtests and saved artifacts.
metadata:
  domain: time-series
  tasks: [forecasting, panel, foundation-models]
  ts_agents:
    tool_category: forecasting
    preferred_workflow: forecast-panel
    artifact_checklist: [metrics.json, backtest_predictions.csv, forecast.csv, report.md, run_manifest.json]
---

# Panel forecasting

## When to use

Use this skill to compare global models across many related series in one
long-format panel, to persist trained GBM/NHITS models for reuse, or to score
zero-shot foundation models (FMs) against those baselines on rolling
validation. For a single series with statistical baselines, prefer the
forecasting skill and `forecast-series` (which also accepts the FM methods).

## Workflow

Use `ts-agents workflow show forecast-panel --json` to discover installed model
availability. The base install supports `seasonal_naive`; the workflow status
stays `available` and each optional family is reported under
`optional_features`. Install the `ml` extra for `lightgbm`/`histgbm`, `neural`
for `nhits`, and `foundation` for `chronos2_small`, `chronos2`, `timesfm2p5`
and `patchtst_fm`. Trainers are imported only when requested; an unavailable
requested method fails before any forecasting artifacts are written (only a
failed `run_manifest.json` is recorded).

Prepare a long-format file with exactly `unique_id`, `ds`, `y` (or supply
`--id-col`, `--time-col`, `--value-col`). Series must be regular at explicit
`--freq`, end at the same timestamp, and have unique keys and finite targets.
Audit timezones/missingness before running. For a bundled example panel, export
the M4 mini-panel train split (its month-start dates are a synthetic
alignment):

```bash
ts-agents data export-panel m4-monthly-mini --split train --out panel.csv --json
```

```bash
ts-agents workflow run forecast-panel --input panel.csv --freq h \
  --horizon 24 --season-length 168 --lags 1,24,168 \
  --methods seasonal_naive,lightgbm,nhits --n-windows 3 \
  --n-estimators 200 --max-steps 1000 --accelerator cpu --json
```

Covariates are unsupported. FMs run zero-shot: nothing is trained, and each
origin conditions only on its own cutoff history, truncated to
`--context-length` points (default 512, validated against each
model's window limit). Weights are fetched per requested model into `HF_HOME` on first use, so
request only the checkpoints you need; network-less sandboxes need a
pre-populated cache (`HF_HUB_OFFLINE=1`). `models/<fm>/` holds only
`model_spec.json` (checkpoint id, pinned revision, resolved context), never
weights. Run one FM at a time in memory-limited sandboxes.

```bash
ts-agents workflow run forecast-panel --input panel.csv --freq MS \
  --horizon 18 --season-length 12 --n-windows 2 \
  --methods seasonal_naive,chronos2_small --skip-plots --json
```

Set training caps and origin counts deliberately: every trained method is
refitted at each validation origin and once for future predictions.
`--max-steps` is a per-fit step cap, not a total wall-clock timeout. Keep
thread/device choices explicit (`--accelerator` applies to NHITS and FMs); use
existing job supervision for long runs.

Scores rank models on rolling validation, not an independent test. Keep final
test targets (for example the `--split holdout` export) outside the input,
choose settings on validation only, then score saved future predictions against
the untouched test. No random train/test split. MASE scales use each origin's
training prefix; zero scales are reported rather than assigned a fabricated
score. NHITS uses an internal training-tail split for early stopping. Public
benchmarks such as M4 may overlap FM pretraining data, so treat FM scores on
them as optimistic. Aggregate metrics pool all predictions; inspect per-series
and per-horizon results, and note overlapping origins if step size is below
horizon.

Use `metrics.json` (including `foundation_models` and
`options.resolved_context_length`), `backtest_predictions.csv`, `forecast.csv`,
`backtest.png`, and the run manifest to generate richer reports without
rerunning training. Native models are saved under `models/<method>/`;
MLForecast and NeuralForecast provide their own `load` APIs. Load only
artifacts from trusted training runs.
