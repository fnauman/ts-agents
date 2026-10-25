---
name: time-series-diagnostics
description: >
  First look at any time series: data quality (sampling, gaps, NaNs), descriptive stats,
  significance-tested periodicity, autocorrelation, and a quick regime screen, with next steps
  across anomaly detection, segmentation, decomposition, spectral analysis, and forecasting.
  Use when the user asks "what does this series look like?", "is there seasonality?", or before
  choosing parameters for any other analysis.
compatibility: "ts-agents CLI (`ts-agents`)."
metadata:
  domain: time-series
  tasks: [eda, diagnostics, data-quality, statistics, periodicity, autocorrelation]
  ts_agents:
    tool_categories: [statistics, spectral]
    preferred_workflow: inspect-series
    preferred_tools: [describe_series, detect_periodicity, compute_autocorrelation]
    artifact_checklist: [summary.json, report.md]
---

# Time series diagnostics

## Goal
Produce a fast, decision-useful snapshot of a series so later steps (anomaly
detection, segmentation, decomposition, spectral analysis, forecasting) are
configured from evidence rather than guesses.

## When to use
Start here for any new series, and before choosing a period, window length,
subsequence length, or forecasting horizon.

## Before running: pin down the data
- Which column holds the values (`--value-col`) and which holds time (`--time-col`)?
- What are the units of the values and of time? Record them in your answer; the
  CLI reports periods in samples or in time-column units, never in guessed units.
- Is it one series? Single-series inputs reject multiple IDs; use the
  `panel-forecasting` skill for panels and `--value-col` to pick one channel.

## Workflow
### 1) Run the inspect-series workflow (always start here)
```bash
ts-agents workflow run inspect-series --input data.csv --time-col t --value-col value --json
```

Read, in order:
- `quality_status` and `result.warnings` (irregular sampling, reordered rows,
  periodicity not assessed). `degraded` is a caveat to report, not a crash.
- `result.data.time_axis`: median step, `regular`, `gap_count`, `reordered`.
- `result.data.periodicity`: `detected`, `dominant_period` (samples),
  `dominant_period_time` (time units when the time step is regular and numeric),
  `p_value`, and `candidate_period` (strongest peak even when not significant).
- `result.data.regime_screen`: candidate change-points (needs the `patterns` extra).
- `result.data.recommended_next_steps`: suggestions across task families.

`detected: false` is a finding, not a failure: trends, random walks, and level
shifts are reported as aperiodic instead of returning the record length as a period.

### 2) Drill down with individual tools when needed
```bash
ts-agents tool run describe_series --input data.csv --value-col value --param extended=true --json
ts-agents tool run compute_autocorrelation --input data.csv --value-col value --param max_lag=200 --json
ts-agents tool run detect_periodicity --input data.csv --time-col t --value-col value --json
```

With a regular numeric `--time-col`, `sample_rate` is derived from the time
step (`input.series.sample_rate_source: time_column`) so periods come back in
time-column units. Datetime axes stay in samples; `input.series.time_axis`
reports the step in seconds.

Interpretation:
- Slow ACF decay → persistence or an unremoved trend.
- ACF peaks at a regular lag plus `detected: true` → periodicity.
- `nan_count > 0` → the input contained non-finite values (other tools reject them).

### 3) Choose the next skill
- Unusual subsequences or events → `anomaly-detection`
- Shifts in level/variance or regimes → `segmentation`
- Trend/seasonality separation → `decomposition`
- Frequencies, spectral slope, coherence → `spectral-analysis`
- Future values → `forecasting` (or `panel-forecasting`)

## Tool discovery
```bash
ts-agents tool search periodicity --json
ts-agents tool list --category statistics --json
```

## Output expectations
- What you ran and on which column/units.
- Data-quality caveats (gaps, irregular sampling, reordering, NaNs).
- 2–5 findings with numbers (e.g., "significant period 24 samples = 1 day, p<1e-10").
- The recommended next analysis and why.
