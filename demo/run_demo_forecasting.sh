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

"${RUNNER[@]}" workflow run forecast-series \
  --run-id Re200Rm200 \
  --variable bx001_real \
  --horizon 1 \
  --methods arima,theta \
  --output-dir "$OUTDIR" \
  --overwrite \
  --json \
  --save "${OUTDIR}/workflow_result.json" >/dev/null

OUTDIR="$OUTDIR" python - <<'PY'
import json
import os

outdir = os.environ["OUTDIR"]
payload = json.load(open(f"{outdir}/workflow_result.json"))
data = payload["result"]["data"]
print("Demo complete.")
print(f"- Run ID: {data['source']['run_id']}")
print(f"- Variable: {data['source']['variable']}")
print(f"- Horizon: {data['horizon']}")
print(f"- Best method (RMSE): {data['best_method']}")
print(f"- Output dir: {data['output_dir']}")
PY

echo
echo "==> Done. Outputs:"
ls -lh "${OUTDIR}"
