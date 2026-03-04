---
name: time-series-forecasting
description: >
  Forecast/predict future values of a time series, choose reasonable baselines, and (optionally) compare forecasting methods.
  Use when the user asks to forecast, predict, extrapolate, project forward, or evaluate forecasting approaches on a series.
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
Given a univariate time series, produce **future predictions** (and, when possible, uncertainty) with a workflow that is:
- baseline-first (fast)
- cost-aware
- easy to extend as new forecasting tools are added

## Minimal info to proceed
- A specific series (either raw array, or `run_id` + `variable`)
- Forecast horizon (number of future steps).  
  If missing: default to the tool’s default (often `horizon=10`) and say so.

Optional but useful:
- sampling frequency / time step
- whether the user cares about **accuracy** (backtesting) vs “a plausible projection”
- whether the user needs **prediction intervals**

## Model selection cheat sheet (practical)
- **ETS**: good default for level/trend/seasonality; usually stable.
- **Theta**: strong simple baseline; often competitive.
- **ARIMA**: can be strong but higher cost; more fragile; use when you need AR structure or want a stronger statistical model.
- **Ensemble**: good when unsure; higher cost; use after a baseline.

## Cost-aware workflow
### 0) (Optional) Quick diagnostics for seasonality
If period is unknown and seasonality matters:
```bash
uv run ts-agents run detect_periodicity_with_data --run <RUN_ID> --var <VARIABLE> --param n_top=3
```

### 1) Run a fast baseline forecast
Pick one:
```bash
uv run ts-agents run forecast_theta_with_data --run <RUN_ID> --var <VARIABLE> --param horizon=50
uv run ts-agents run forecast_ets_with_data   --run <RUN_ID> --var <VARIABLE> --param horizon=50
```

### 2) If needed, run a stronger / more expensive model
ARIMA is higher cost:
```bash
uv run ts-agents run forecast_arima_with_data --run <RUN_ID> --var <VARIABLE> --param horizon=50
```

### 3) If unsure, compare multiple methods on holdout data
```bash
uv run ts-agents run compare_forecasts_with_data --run <RUN_ID> --var <VARIABLE> --param horizon=50
```

Notes:
- The comparison tool typically does a simple historical split; interpret metrics accordingly.
- Use this when the user asks “which model is best?” or you need evidence-based selection.

### 4) If you want a robust choice, use an ensemble
```bash
uv run ts-agents run forecast_ensemble_with_data --run <RUN_ID> --var <VARIABLE> --param horizon=50
```

## Prediction intervals (uncertainty)
Some core forecasting functions support `level=[80,95]` to produce intervals, but the current `*_with_data` wrappers may only plot point forecasts.

If you need intervals and wrappers don’t expose them:
1. Load the series with `get_series`
2. Run the **series-based** forecasting tool in Python, passing `level`

Example sketch:
```python
import numpy as np
from src.data_access import get_series
from src.core.forecasting import forecast_ets

y = get_series("Re200Rm200", "bx001_real")
res = forecast_ets(y, horizon=50, level=[80, 95])
print(res.intervals.keys())
```

## Tool discovery (future-proofing)
Always prefer discovery over hard-coding when new models appear:
```bash
uv run ts-agents tool list --category forecasting --json
uv run ts-agents tool list --category forecasting --max-cost medium --json
```

## Output expectations
Return:
- the chosen horizon
- which model(s) you ran and why
- a short interpretation (“trend continues”, “reverts”, “strong seasonality”)
- if multiple models: a small comparison and your recommendation
- plots when available (many `*_with_data` tools include an inline image payload)

## Report generation standard (Quarto PDF)
For demo and stakeholder-facing outputs, produce a Quarto report and render to PDF:

```bash
quarto render reports/REPORT.qmd --to pdf
```

The report should include:
- model comparison table (MAE/RMSE/MAPE)
- forecast visualization with labeled axes and caption
- selected method rationale and limitations
