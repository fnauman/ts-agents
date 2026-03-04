---
name: time-series-decomposition
description: >
  Decompose a time series into trend/seasonal/residual components (STL, MSTL, HP filter, Holt-Winters).
  Use when the user asks about trend, seasonality, detrending, or wants residuals for anomaly detection/forecasting.
compatibility: "Best with the ts-agents repo + CLI (`ts-agents`)."
metadata:
  domain: time-series
  tasks: [decomposition, trend, seasonality, detrending]
  ts_agents:
    tool_category: decomposition
    prefers_with_data_tools: true
  claude_code:
    allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
    disable-model-invocation: false
---

# Time series decomposition

## When to use
Use decomposition when you need to:
- separate **trend** and **seasonality**
- inspect the **residual** (noise) component
- detrend / deseasonalize before other analyses (anomalies, spectral slope, etc.)
- pick forecasting strategies based on observed components

## Pick a method (simple rubric)
1. **STL** (`stl_decompose_with_data`): best default for a single dominant seasonality.
2. **MSTL** (`mstl_decompose_with_data`): when there are multiple seasonalities (e.g., daily + weekly).
3. **HP filter** (`hp_filter_with_data`): when you mainly want a smooth trend and don’t trust a seasonal model.
4. **Holt-Winters decomposition** (`holt_winters_decompose_with_data`): forecasting-oriented decomposition; supports additive/multiplicative components.

## Step-by-step workflow
### 0) Decide (or estimate) the seasonal period
If the user didn’t specify a period:
```bash
uv run ts-agents run detect_periodicity_with_data --run <RUN_ID> --var <VARIABLE> --param n_top=3
```

Use the most plausible period as `period` for STL/Holt-Winters.

### 1) Run STL (default)
```bash
uv run ts-agents run stl_decompose_with_data --run <RUN_ID> --var <VARIABLE> --param period=<PERIOD> --param robust=true
```

Notes:
- `robust=true` is usually safer with outliers.
- If STL looks unstable, try a different period or use HP filter as a fallback.

### 2) Run MSTL (multiple seasonalities)
If you have multiple periods (e.g., `[24, 168]`):
```bash
uv run ts-agents run mstl_decompose_with_data --run <RUN_ID> --var <VARIABLE> --param periods=[24,168]
```

### 3) Run HP filter (trend extraction)
```bash
uv run ts-agents run hp_filter_with_data --run <RUN_ID> --var <VARIABLE> --param lamb=1600
```

Heuristic: larger `lamb` → smoother trend.

### 4) Run Holt-Winters decomposition (additive/multiplicative)
```bash
uv run ts-agents run holt_winters_decompose_with_data --run <RUN_ID> --var <VARIABLE> --param period=<PERIOD> --param trend=add --param seasonal=add
```

## Tool discovery (future-proofing)
List current decomposition tools and parameters:
```bash
uv run ts-agents tool list --category decomposition --json
```

## Output expectations
Summarize:
- period(s) used
- whether seasonality/trend is strong or weak
- residual behavior (variance, obvious outliers)
- recommended next step:
  - forecast on original vs residual
  - anomaly detection on residual
  - spectral analysis on residual (if trend dominates the spectrum)
