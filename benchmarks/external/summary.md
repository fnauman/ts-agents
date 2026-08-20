# External Benchmark Context

Snapshot date: 2026-06-04

This directory records external benchmark and comparator context for `ts-agents`. It is intentionally separate from the internal refactor benchmark under `benchmarks/results/latest/`.

## GIFT-Eval

GIFT-Eval ([paper](https://arxiv.org/abs/2410.10393), [leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval)) is the relevant external forecasting benchmark to reference before making foundation-model accuracy claims. As of the snapshot date its published scope was 23 datasets, 144k+ time series, 177M data points, seven domains, and 10 frequencies. Generate a GIFT-Eval-compatible export artifact before claiming external accuracy.

## TimeCopilot

TimeCopilot ([repository](https://github.com/TimeCopilot/timecopilot)) is tracked as a comparator and interoperability target because it focuses on foundation-model forecasting breadth. That is not the desired product direction for `ts-agents`; the repo should stay centered on CLI contracts, workflow manifests, artifacts, sandboxes, and reusable skills.

## Local TSFM Scope

`foundation-chronos-smoke` is the scoped executable TSFM path. It exists to validate one Chronos zero-shot route through the same autoresearch artifact contract, not to become a model hub.
