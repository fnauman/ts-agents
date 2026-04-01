# Benchmarks

This directory holds reproducible internal benchmark snapshots for the
refactor-era agent contract.

## Refactor benchmark

Run the deterministic harness from a source checkout:

```bash
uv run python -m ts_agents.evals.refactor_benchmark \
  --output-dir benchmarks/results/latest
```

The command writes:

- `benchmarks/results/latest/results.json` — full per-scenario results
- `benchmarks/results/latest/summary.md` — compact aggregate table
- `benchmarks/results/latest/artifacts/` — workflow artifacts produced by the benchmark
- `benchmarks/results/latest/workspace/` — deterministic input fixtures used during the run

The benchmark compares four assist levels:

1. `plain_model`
2. `plain_tools`
3. `structured_discovery`
4. `skills_workflows`

And three tasks:

- inspect a short univariate series
- compare forecasting baselines
- run labeled-stream activity recognition
