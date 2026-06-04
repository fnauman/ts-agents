# External Benchmark Context

Snapshot date: 2026-06-04

This directory records external benchmark and comparator context for `ts-agents`. It is intentionally separate from the internal refactor benchmark under `benchmarks/results/latest/`.

## GIFT-Eval

GIFT-Eval is the relevant external forecasting benchmark to reference before making foundation-model accuracy claims. The checked snapshot records its published scope: 23 datasets, 144k+ time series, 177M data points, seven domains, and 10 frequencies.

## TimeCopilot

TimeCopilot is tracked as a comparator and interoperability target because it focuses on foundation-model forecasting breadth. That is not the desired product direction for `ts-agents`; the repo should stay centered on CLI contracts, workflow manifests, artifacts, sandboxes, and reusable skills.

## Local TSFM Scope

`foundation-chronos-smoke` is the scoped executable TSFM path. It exists to validate one Chronos zero-shot route through the same autoresearch artifact contract, not to become a model hub.
