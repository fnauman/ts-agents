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
- **Theta**: strong simple baseline; often competitive.
- **ETS**: good default for level/trend/seasonality.
- **ARIMA**: stronger but more fragile and higher cost.
- **Ensemble**: use after you have compared individual methods.

## Workflow
### 0) Estimate seasonality when needed
```bash
uv run ts-agents run detect_periodicity_with_data --run <RUN_ID> --var <VARIABLE> --param n_top=3
```

### 1) Start with a baseline
If the series is seasonal and you know the period:
```bash
uv run ts-agents run forecast_seasonal_naive_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 --param season_length=<PERIOD>
```

If you do not know the period yet, or need a stronger fast baseline:
```bash
uv run ts-agents run forecast_theta_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 --param season_length=<PERIOD>
```

### 2) Add ETS or ARIMA when accuracy matters
```bash
uv run ts-agents run forecast_ets_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 --param season_length=<PERIOD>

uv run ts-agents run forecast_arima_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 --param season_length=<PERIOD>
```

### 3) Compare explicitly instead of relying on defaults
```bash
uv run ts-agents run compare_forecasts_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 \
  --param models=seasonal_naive,theta,ets,arima \
  --param season_length=<PERIOD>
```

Notes:
- The comparison tool uses a simple historical split, not the professional
  rolling-origin + official holdout protocol.
- Use this when the user asks "which model looks best on this series?"

### 4) Use an ensemble only after individual comparisons
```bash
uv run ts-agents run forecast_ensemble_with_data \
  --run <RUN_ID> --var <VARIABLE> \
  --param horizon=50 \
  --param models=seasonal_naive,theta,ets,arima \
  --param season_length=<PERIOD>
```

## Prediction intervals
The core forecasting functions support `level=[80, 95]`, while the
`*_with_data` wrappers are mainly geared toward point forecasts and plots.

If you need intervals, use Python directly:
```python
from src.data_access import get_series
from src.core.forecasting import forecast_ets

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
