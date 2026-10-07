---
name: panel-forecasting
description: Train MLForecast LightGBM or histogram GBM and NeuralForecast NHITS on regular panels, with chronological backtests and saved artifacts.
metadata:
  ts_agents:
    preferred_workflow: forecast-panel
    artifact_checklist: [metrics.json, backtest_predictions.csv, forecast.csv, report.md, run_manifest.json]
---

# Panel forecasting

Use `ts-agents workflow show forecast-panel --json` to discover installed model
availability. Install the `ml` extra for LightGBM/histogram GBM and `neural` for
NHITS. Trainers are imported only when requested; an unavailable requested
backend fails explicitly. The base install supports the seasonal baseline.

Prepare a long-format file with exactly `unique_id`, `ds`, `y` (or supply
`--id-col`, `--time-col`, `--value-col`). Series must be regular at explicit
`--freq`, end at the same timestamp, and have unique keys and finite targets.
Audit timezones/missingness before running. Covariates and foundation models are
outside this workflow's current contract.

```bash
ts-agents workflow run forecast-panel --input panel.csv --freq h \
  --horizon 24 --season-length 168 --lags 1,24,168 \
  --methods seasonal_naive,lightgbm,nhits --n-windows 3 \
  --n-estimators 200 --max-steps 1000 --accelerator cpu --json
```

Set training caps and origin counts deliberately: every method is refitted at
each validation origin and once for future predictions. `--max-steps` is a
per-fit step cap, not a total wall-clock timeout. Keep thread/device choices
explicit; use existing job supervision for long runs.

Scores rank models on rolling validation, not an independent test. Keep final
test targets outside the input, choose settings on validation only, then score
saved future predictions against the untouched test. No random train/test split.
MASE scales use each origin's training prefix; zero scales are reported rather
than assigned a fabricated score. NHITS uses an internal training-tail split for
early stopping. Aggregate metrics pool all predictions; inspect per-series and
per-horizon results, and note overlapping origins if step size is below horizon.

Use `metrics.json`, `backtest_predictions.csv`, `forecast.csv`, `backtest.png`,
and the run manifest to generate richer reports without rerunning training.
Native models are saved under `models/<method>/`; MLForecast and NeuralForecast
provide their own `load` APIs. Load only artifacts from trusted training runs.
