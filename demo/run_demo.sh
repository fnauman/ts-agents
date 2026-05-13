#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash demo/run_demo.sh

# Prefer uv if available (matches repo docs), but fall back gracefully.
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run ts-agents)
elif command -v ts-agents >/dev/null 2>&1; then
  RUNNER=(ts-agents)
else
  RUNNER=(python -m ts_agents)
fi

mkdir -p data outputs/demo

echo "==> Generating synthetic labeled stream (stairs)..."
python data/make_synthetic_labeled_stream.py \
  --scenario stairs \
  --hz 20 \
  --minutes 4 \
  --seed 1337 \
  --out data/demo_labeled_stream.csv

echo
echo "==> Running activity-recognition workflow..."
"${RUNNER[@]}" workflow run activity-recognition \
  --input data/demo_labeled_stream.csv \
  --label-col label \
  --value-cols x,y,z \
  --metric balanced_accuracy \
  --classifier auto \
  --output-dir outputs/demo \
  --overwrite \
  --json \
  --save outputs/demo/workflow_result.json >/dev/null

echo
python - <<'PY'
import json
payload = json.load(open("outputs/demo/workflow_result.json"))
data = payload["result"]["data"]
print(f"Best window size: {data['best_window_size']}")
print(f"Metric: {data['metric']} | Score: {data['score']:.4f}")
print(f"Classifier: {data['classifier_used']}")
print(f"Output dir: {data['output_dir']}")
PY

echo
echo "==> Done. Outputs:"
ls -lh outputs/demo
