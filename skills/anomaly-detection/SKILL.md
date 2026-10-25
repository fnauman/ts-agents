---
name: time-series-anomaly-detection
description: >
  Find unusual subsequences, events, or outliers in a time series with matrix-profile discords,
  peaks, and decomposition residuals. Use when the user asks about anomalies, outliers, faults,
  novelties, or "what looks unusual?" in sensor, operational, or simulation data.
compatibility: "ts-agents CLI (`ts-agents`) with the `patterns` extra (stumpy)."
metadata:
  domain: time-series
  tasks: [anomaly-detection, outliers, discords, events]
  ts_agents:
    tool_category: patterns
    preferred_workflow: inspect-series
    preferred_tools: [find_discords, analyze_matrix_profile, detect_peaks, stl_decompose]
---

# Time series anomaly detection

## When to use
Use when the question is "what is unusual in this series?": shape anomalies,
faults, glitches, spikes, or rare events.

## What the tools can and cannot say
- `find_discords` returns the **top-k most unusual subsequences**. Top-k always
  exist, even in pure noise. Judge each discord by `robust_z`: its distance from
  the matrix-profile median in MAD units. Values far above the rest (for
  example 10+ while others are 2–3) indicate a real outlier in *this* series.
  `robust_z` is not a calibrated probability.
- Discords compare z-normalized **shapes**. A pure level shift or variance
  change is a regime change; use the `segmentation` skill for those.
- There is no built-in event-level scorer against labels yet. If the user has
  labeled events, compare discord indices to event windows yourself and report
  hits and misses with the tolerance you used.

## Workflow
### 1) Diagnose first
```bash
ts-agents workflow run inspect-series --input data.csv --time-col t --value-col value --json
```
Note the sampling caveats and any significant period.

### 2) Choose the subsequence length `m` (samples)
- With a significant period P: start with `m ≈ P` (one cycle).
- Otherwise: use the expected event duration from the domain (ask the user
  if unknown). Report `m` in time units when the sampling step is known.
- `m` must be at least 3, and the series at least `2m` samples long.

### 3) Find discords
```bash
ts-agents tool run find_discords --input data.csv --value-col value --param m=100 --param max_discords=5 --json
```
The default `exclusion_zone` is `m`, so one event does not fill several
top-k slots. Raise it if events are longer than `m`.

### 4) Cross-check
- Spikes and dropouts: `detect_peaks` with an explicit `prominence` (in data units).
- Seasonal series: decompose (see `decomposition`) and inspect residual
  extremes; seasonal peaks are not anomalies.
- Long series: `find_discords` is O(n^2); prefer a background job
  (`ts-agents jobs start -- tool run find_discords ...`) above ~100k samples.

## Worked example (known answer)
`data/synthetic_signals.csv` (50 Hz) has a flatlined oscillation at samples 1500–1529:
```bash
ts-agents tool run find_discords --input data/synthetic_signals.csv --value-col anomaly --param m=100 --json
```
The top discord starts within the flat segment and has a much larger
`robust_z` than the others.

## Output expectations
- `m`, exclusion zone, and how `m` was chosen.
- Each reported anomaly with index (and time), distance, and `robust_z`.
- Which discords you consider real and why (relative `robust_z`, domain context).
- Caveats: shape-based, top-k always exist, no labeled-event scoring.
