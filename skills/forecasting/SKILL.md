---
name: time-series-forecasting
description: >
  Forecast/predict future values of a time series, choose reasonable baselines, and
  compare forecasting methods on arbitrary series loaded from ts-agents data.
compatibility: "Best with the ts-agents repo + CLI (`ts-agents`)."
metadata:
  domain: time-series
  tasks: [forecasting, prediction, model-selection]
  ts_agents:
    tool_category: forecasting
    prefers_with_data_tools: true
    preferred_workflow: forecast-series
    preferred_tools: [detect_periodicity_with_data, forecast_seasonal_naive_with_data]
    artifact_checklist: [forecast_comparison.json, forecast.json, forecast.csv, report.md]
  claude_code:
    allowed-tools: [Bash, Read, Write, Edit]
    disable-model-invocation: false
---

# Time series forecasting

## What this skill does
Given a univariate time series, produce future predictions with a workflow that
is:

- baseline-first
- explicit about seasonality
- cost-aware
- easy to extend when more forecasting tools are added

If the user wants a reproducible benchmark/report workflow on the vendored M4
Monthly mini-panel, switch to [`SKILL-pro.md`](./SKILL-pro.md). This base skill
is for arbitrary `run_id` + `variable` series, not the fixed workflow contract.

## Minimal info to proceed
- A specific series (`run_id` + `variable`, or raw array in Python)
- Forecast horizon

Useful extras:
- season length / period if known
- whether the user wants quick projection vs holdout comparison
- whether the user needs prediction intervals

If the season length is unknown and seasonality matters, estimate it first.

## Model selection cheat sheet
- **Seasonal Naive**: minimum serious baseline for seasonal data. Cheap and easy to interpret.
- **Theta**: strong simple baseline; often competitive. Requires the optional `forecasting` extra.
- **ETS**: good default for level/trend/seasonality. Requires the optional `forecasting` extra.
- **ARIMA**: stronger but more fragile and higher cost. Requires the optional `forecasting` extra.
- **Ensemble**: use after you have compared individual methods; advanced combinations require the optional `forecasting` extra.

## Workflow
### 0) Estimate seasonality when needed
```bash
uv run ts-agents tool run detect_periodicity_with_data --run <RUN_ID> --var <VARIABLE> --param n_top=3
```

### 1) Check which methods are available in the current environment
```bash
uv run ts-agents workflow show forecast-series --json
```

Interpret `available_methods` / `unavailable_methods` before you choose models.
In a default/base install, expect `seasonal_naive` to be available and
`theta`, `ets`, `arima` to remain unavailable until the optional
`forecasting` extra is installed.

### 2) Run a base-safe baseline and artifact workflow
```bash
uv run ts-agents workflow run forecast-series \
  --run-id <RUN_ID> --variable <VARIABLE> \
  --horizon 50 \
  --methods seasonal_naive \
  --season-length <PERIOD>
```

If you only need a quick point forecast instead of the workflow artifact set:
```bash
uv run ts-agents tool run forecast_seasonal_naive_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 --param season_length=<PERIOD>
```

### 3) Expand the comparison only after confirming advanced backends are available

Notes:
- The comparison tool uses a simple historical split, not the professional
  rolling-origin + official holdout protocol.
- Use this when the user asks "which model looks best on this series?"
- If `workflow show forecast-series --json` lists `theta`, `ets`, or `arima`
  in `available_methods`, rerun the same workflow with those method names
  added to `--methods`.
- If those methods are listed under `unavailable_methods`, stay with
  `seasonal_naive` or install `ts-agents[forecasting]`.

### 4) Use an ensemble only after individual comparisons
Treat ensembles as an advanced path, not a base-profile default. Only use them
after individual methods are available and have been compared explicitly.

## Prediction intervals
The core forecasting functions support `level=[80, 95]`, while the
`*_with_data` wrappers are mainly geared toward point forecasts and plots.

If you need interval-capable statistical models such as ETS, Theta, or ARIMA,
install `ts-agents[forecasting]` and use Python directly:
```python
from ts_agents.data_access import get_series
from ts_agents.core.forecasting import forecast_ets

y = get_series("Re200Rm200", "bx001_real")
res = forecast_ets(y, horizon=50, level=[80, 95], season_length=12)
print(res.lower_bound[:5])
print(res.upper_bound[:5])
```

## Output expectations
Return:
- the chosen horizon
- which model(s) you ran and why
- the season length you assumed or estimated
- a short interpretation of the forecast
- if multiple models were compared: a recommendation tied to the reported
  metrics

## Reporting standard
For stakeholder-facing outputs, produce a short Markdown or Quarto report with:
- model comparison table
- forecast figure with labeled axes
- selected method rationale
- limitations and follow-up recommendation
