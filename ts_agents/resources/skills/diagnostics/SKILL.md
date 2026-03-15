---
name: time-series-diagnostics
description: >
  Quick EDA and diagnostics for a time series: descriptive stats, autocorrelation, and periodicity.
  Use when the user asks "what does this series look like?", "is there seasonality?", "what's the period?",
  or before choosing decomposition/forecasting parameters.
compatibility: "Best with the ts-agents repo + CLI (`ts-agents`)."
metadata:
  domain: time-series
  tasks: [eda, diagnostics, statistics, periodicity, autocorrelation]
  ts_agents:
    tool_categories: [statistics, spectral]
    prefers_with_data_tools: true
---

# Time series diagnostics (EDA)

## Goal
Produce a **fast, decision-useful** snapshot of a time series so later steps (decomposition, forecasting, anomaly detection) can be configured correctly.

## When to use
Use when the user asks for a quick understanding of a series (seasonality, period, persistence) or before selecting decomposition/forecasting parameters.

## Default workflow (fast → slower)
### 1) Confirm what series we’re analyzing
If the user hasn’t pinned down `run_id` + `variable`, discover candidates with:
```bash
uv run ts-agents data list --runs
uv run ts-agents data vars
```
Then proceed with the selected `run_id` + `variable`.

### 2) Descriptive statistics (always start here)
Run:
```bash
uv run ts-agents run describe_series_with_data --run <RUN_ID> --var <VARIABLE>
```

Report at least: length, mean, std, min/max, skew/kurtosis (if available), and whether NaNs appear.

### 3) Autocorrelation (look for memory + seasonality)
Run:
```bash
uv run ts-agents run compute_autocorrelation_with_data --run <RUN_ID> --var <VARIABLE> --param max_lag=200
```

Interpretation heuristics:
- Slow ACF decay → persistent/long-memory-ish signal (or trend not removed)
- ACF spikes at regular lags → likely periodicity/seasonality
- Near-zero ACF beyond small lags → closer to white noise (after detrending)

### 4) Periodicity / dominant period (FFT-based quick check)
Run:
```bash
uv run ts-agents run detect_periodicity_with_data --run <RUN_ID> --var <VARIABLE> --param n_top=5
```

Use the top detected period(s) to:
- set `period` for STL/Holt-Winters
- pick subsequence lengths for matrix profile / motifs

### 5) Spectrum / PSD (when frequency content matters)
Run:
```bash
uv run ts-agents run compute_psd_with_data --run <RUN_ID> --var <VARIABLE> --param sampling_rate=1.0
```

Use PSD when:
- user asks about “frequencies”, “spectral slope”, “dominant oscillations”
- you want to confirm periodicity beyond the single “top period”

## Tool discovery (future-proofing)
If tools change or new diagnostics are added, list what’s available:
```bash
uv run ts-agents tool list --category statistics --json
uv run ts-agents tool list --category spectral --json
```

## Output expectations
Return a concise diagnostic summary:
- “What I ran” (stats, ACF, periodicity, PSD, etc.)
- 2–5 key findings (e.g., dominant period ≈ 48; strong persistence; heavy tails)
- Recommended next step(s) (decomposition vs forecasting vs anomaly detection)
