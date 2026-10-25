---
name: time-series-decomposition
description: >
  Decompose a time series into trend/seasonal/residual components (STL, MSTL, Holt-Winters).
  Use when the user asks about trend, seasonality, detrending, or wants residuals for anomaly
  detection, spectral analysis, or forecasting.
compatibility: "ts-agents CLI (`ts-agents`) with the `decomposition` extra."
metadata:
  domain: time-series
  tasks: [decomposition, trend, seasonality, detrending]
  ts_agents:
    tool_category: decomposition
    preferred_tools: [detect_periodicity, stl_decompose, mstl_decompose, holt_winters_decompose]
  claude_code:
    allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
    disable-model-invocation: false
---

# Time series decomposition

## When to use
Use decomposition when you need to:
- separate **trend** and **seasonality**
- inspect the **residual** component (e.g., for anomaly detection)
- detrend or deseasonalize before spectral analysis
- pick forecasting strategies from observed components

## Pick a method
1. **STL** (`stl_decompose`): default for one dominant seasonality.
2. **MSTL** (`mstl_decompose`): several seasonalities (e.g., daily + weekly).
3. **Holt-Winters** (`holt_winters_decompose`): forecasting-oriented; supports
   additive/multiplicative components. If the fit fails, the result is labeled
   `holt_winters_fallback` with a warning explaining why — report that.

## Workflow
### 0) Establish the period from evidence
```bash
ts-agents tool run detect_periodicity --input data.csv --value-col value --json
```

Use `dominant_period` (samples) only when `detected` is true. If it is false,
there is no significant seasonality: say so, and only decompose with a period
the user supplies from domain knowledge (e.g., 24 for hourly data with a daily
cycle). Without `period`, the decomposition tools auto-detect with the same test
and fail with an explanation rather than guessing.

### 1) Run STL (default)
```bash
ts-agents tool run stl_decompose --input data.csv --value-col value --param period=24 --param robust=true --json
```

### 2) Run MSTL (multiple seasonalities)
```bash
ts-agents tool run mstl_decompose --input data.csv --value-col value --param periods=[24,168] --json
```
Every period must be at most half the series length.

### 3) Run Holt-Winters
```bash
ts-agents tool run holt_winters_decompose --input data.csv --value-col value --param period=24 --param trend=add --param seasonal=add --json
```

Results include `period_source` (`user` or `auto`). Component arrays are as
long as the input; for long series, read the summary metrics
(`residual_variance`, `seasonal_strength`) first.

## Tool discovery
```bash
ts-agents tool list --category decomposition --json
```

## Output expectations
Summarize:
- period(s) used and where they came from (detected with p-value, or user-supplied)
- whether seasonality/trend is strong or weak (`seasonal_strength`)
- residual behavior (variance, obvious outliers)
- the recommended next step: forecast, anomaly detection on residuals, or
  spectral analysis on the residual
