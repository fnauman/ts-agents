---
name: time-series-forecasting-pro
description: >
  Run the reproducible professional forecasting workflow on the vendored M4
  Monthly mini-panel, including rolling-origin validation, official holdout
  scoring, artifact generation, and report output.
compatibility: "Best with the ts-agents repo + Python/CLI."
metadata:
  domain: time-series
  tasks: [forecasting, benchmark, backtesting, reporting]
  ts_agents:
    tool_category: forecasting
    prefers_with_data_tools: false
---

# Professional forecasting workflow

## When to use
Use this skill when the user wants:
- a reproducible forecasting benchmark
- a professional artifact set instead of an ad hoc forecast
- model ranking on the vendored M4 Monthly mini-panel
- a baseline-first workflow with explicit train/backtest/holdout protocol

If the user wants to forecast an arbitrary repo series by `run_id` + `variable`,
use [`SKILL.md`](./SKILL.md) instead.

## Fixed workflow contract
This workflow uses the vendored reference dataset:
- `data/m4_monthly_mini.csv`
- monthly cadence
- season length `12`
- horizon `18`
- series IDs: `M4`, `M10`, `M100`, `M1000`, `M1002`

Required methods for the first contract-complete implementation:
- `seasonal_naive`
- `theta`
- `ets`
- `arima`

Validation layers:
1. expanding-window rolling-origin backtesting with `2` origins per series
2. final scoring on the official `18`-step holdout

## Primary command
Run the end-to-end example script:
```bash
uv run python examples/forecasting_m4_monthly_mini.py
```

Useful variants:
```bash
uv run python examples/forecasting_m4_monthly_mini.py \
  --methods seasonal_naive,theta,ets,arima \
  --output-dir outputs/reports/forecasting-workflow-m4-mini

uv run python examples/forecasting_m4_monthly_mini.py \
  --series M4,M100 \
  --methods seasonal_naive,theta \
  --output-dir outputs/reports/forecasting-workflow-smoke
```

## Artifact contract
The example writes:
- `metrics_by_series.csv`
- `summary.csv`
- `holdout_forecasts.csv`
- `run_summary.json`
- `plots/*.png`
- `REPORT.md`

These are the concrete artifacts to inspect, share, and later assert in Chunk
4C smoke tests.

## Interpretation rules
- Rank methods by holdout `sMAPE` first.
- Use `MAE` and `RMSE` as secondary checks.
- `seasonal_naive` must always be present as the minimum serious monthly
  baseline.
- Prefer the simplest method that wins or stays competitively close on holdout.

## Output expectations
Return:
- the methods evaluated
- the top holdout ranking
- the recommended method
- the location of the generated artifacts
- any failure modes or limitations observed in the run

## Known limitations
- This is a fixed reference workflow, not a general benchmark harness.
- It does not replace the existing built-in forecasting demo.
- Smoke-test artifact assertions are intentionally deferred to Chunk 4C.
