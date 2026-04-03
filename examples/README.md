# Examples

This directory holds end-to-end walkthroughs that are larger than a single tool
call but still meant to be run directly from the repo.

## `forecasting_m4_monthly_mini.py`

Reproducible professional forecasting workflow on the vendored M4 Monthly
mini-panel. It writes:

- per-series metrics
- aggregate summary
- holdout forecast outputs
- at least one plot
- a short Markdown report

Run it with:

```bash
uv run python examples/forecasting_m4_monthly_mini.py
```

For a reduced profile:

```bash
uv run python examples/forecasting_m4_monthly_mini.py \
  --series M4,M100 \
  --methods seasonal_naive \
  --output-dir outputs/reports/forecasting-workflow-smoke
```
