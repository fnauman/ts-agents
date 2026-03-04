#!/usr/bin/env bash
set -euo pipefail

# Forecasting comparison demo using the checked-in short_real.csv dataset.
# Run from repo root:
#   bash demo/run_demo_forecasting.sh

# Prefer uv if available (matches repo docs), but fall back gracefully.
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run ts-agents)
elif command -v ts-agents >/dev/null 2>&1; then
  RUNNER=(ts-agents)
else
  RUNNER=(python -m ts_agents)
fi

OUTDIR=outputs/demo_forecasting

mkdir -p "$OUTDIR"

echo "==> Forecasting comparison demo (ARIMA vs Theta)"
echo "    Dataset: data/short_real.csv  (run: Re200Rm200, var: bx001_real)"
echo

"${RUNNER[@]}" demo forecasting --no-llm \
  --methods arima,theta \
  --output-dir "$OUTDIR" \
  --report-path "${OUTDIR}/forecasting_report.md"

echo
echo "==> Done. Outputs:"
ls -lh "${OUTDIR}"
