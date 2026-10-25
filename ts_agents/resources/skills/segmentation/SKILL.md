---
name: time-series-segmentation
description: >
  Detect change-points and regimes in a time series: shifts in mean or variance (ruptures) and
  changes in pattern/shape (FLUSS). Use when the user asks about regime changes, breaks, phases,
  operating modes, transitions, or wants to split a series before modeling.
compatibility: "ts-agents CLI (`ts-agents`) with the `patterns` extra (ruptures, stumpy)."
metadata:
  domain: time-series
  tasks: [change-point-detection, segmentation, regimes]
  ts_agents:
    tool_category: patterns
    preferred_workflow: inspect-series
    preferred_tools: [segment_changepoint, segment_fluss]
---

# Change-points and regime segmentation

## When to use
Use when the series may switch between regimes (operating modes, phases,
before/after an intervention), or before fitting one model to a whole series.

## Pick a method
- **`segment_changepoint`** (default): shifts in **mean and/or variance**. The
  default cost is `normal`. The input is standardized by a robust noise scale,
  so results do not depend on units (Pa vs kPa gives identical change-points).
  The default penalty is `3·log(n)` multiplied by `(1+ρ)/(1−ρ)`, where ρ is the
  lag-1 autocorrelation; persistent noise is therefore not cut into spurious
  regimes.
- **`segment_fluss`**: changes in **shape/pattern** via the matrix profile. It
  misses pure level shifts and ignores about `5·m` samples at each edge.

## Workflow
### 1) Diagnose first
```bash
ts-agents workflow run inspect-series --input data.csv --time-col t --value-col value --json
```
`result.data.regime_screen` already runs `segment_changepoint` with defaults.

### 2) Run change-point detection
```bash
ts-agents tool run segment_changepoint --input data.csv --value-col value --json
```
Read:
- `changepoints` (sample indices) and `segment_stats` (mean/std in original units)
- `method` (what ran) and `requested_method` (set when they differ, e.g. PELT
  cannot target a fixed number of segments, so binary segmentation ran instead)
- `settings`: `penalty`, `penalty_source`, `lag1_autocorrelation`, `noise_scale`, `jump`
- `warnings`: e.g. long series use `jump = ceil(n/500)`, so change-points are
  resolved to multiples of `jump`. Pass `--param jump=1` for full resolution
  (much slower).

Control the result explicitly when the user knows more:
- known number of regimes: `--param n_segments=3`
- more or fewer change-points: `--param penalty=<value>` (standardized units;
  higher means fewer)
- minimum regime duration: `--param min_size=<samples>`
- general distribution changes: `--param cost_model=rbf` (slow beyond a few thousand samples)

### 3) Shape-based regimes (optional)
```bash
ts-agents tool run segment_fluss --input data.csv --value-col value --param m=50 --param n_segments=3 --json
```
A warning reports when FLUSS returns fewer boundaries than requested.

## Worked example (known answer)
`data/synthetic_signals.csv` (`regime` column): the mean shifts at sample 1000
and the standard deviation triples at sample 2000.
```bash
ts-agents tool run segment_changepoint --input data/synthetic_signals.csv --value-col regime --json
```
Expect change-points near 1000 and 2000, within the reported `jump`.

## Output expectations
- Change-points in samples and, when the time column is known, in time.
- Per-segment mean/std in original units.
- The settings that produced them (penalty and its source, `jump`, cost model).
- Caveats: autocorrelated regimes, `jump` resolution, any algorithm substitution.
