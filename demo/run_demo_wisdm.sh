#!/usr/bin/env bash
set -euo pipefail

# Activity-recognition demo using the checked-in WISDM subset.
# Run from repo root:
#   bash demo/run_demo_wisdm.sh

# Prefer uv if available (matches repo docs), but fall back gracefully.
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run ts-agents)
elif command -v ts-agents >/dev/null 2>&1; then
  RUNNER=(ts-agents)
else
  RUNNER=(python -m ts_agents)
fi

OUTDIR=outputs/demo_wisdm
CSV=data/wisdm_subset.csv

mkdir -p "$OUTDIR"

echo "==> WISDM activity-recognition demo"
echo "    Dataset: ${CSV}"
echo

echo "==> Selecting window size..."
"${RUNNER[@]}" run select_window_size_from_csv \
  --json \
  --save "${OUTDIR}/window_selection.json" \
  --param csv_path="${CSV}" \
  --param value_columns=x,y,z \
  --param label_column=label \
  --param window_sizes=32,64,96,128,160 \
  --param metric=balanced_accuracy \
  --param classifier=minirocket \
  --param balance=segment_cap \
  --param max_windows_per_segment=25 \
  --param test_size=0.25 \
  --param seed=1337

BEST_WINDOW=$(
  python - <<'PY'
import json
with open("outputs/demo_wisdm/window_selection.json", "r") as f:
    payload = json.load(f)
result = payload.get("result", payload)
print(result["best_window_size"])
PY
)

echo
echo "Best window size: ${BEST_WINDOW}"
echo

echo "==> Evaluating classifier with best window..."
"${RUNNER[@]}" run evaluate_windowed_classifier_from_csv \
  --json \
  --save "${OUTDIR}/eval.json" \
  --param csv_path="${CSV}" \
  --param value_columns=x,y,z \
  --param label_column=label \
  --param window_size="${BEST_WINDOW}" \
  --param metric=balanced_accuracy \
  --param classifier=minirocket \
  --param balance=segment_cap \
  --param max_windows_per_segment=25 \
  --param test_size=0.25 \
  --param seed=1337

echo
python - <<'PY'
import json
d = json.load(open("outputs/demo_wisdm/eval.json"))
result = d.get("result", d)
print(f"Metric: {result.get('metric')} | Score: {result.get('score'):.4f}")
c = result.get("classification") or {}
print(f"Accuracy: {c.get('accuracy')}")
print(f"F1 (macro): {c.get('f1_score')}")
print(f"n_windows: {result.get('n_windows')}")
print("Class window counts:", result.get("class_counts"))
PY
echo

echo "==> Writing plots..."
python demo/plot_window_selection.py "${OUTDIR}/window_selection.json" "${OUTDIR}/window_scores.png"
python demo/plot_confusion_matrix.py "${OUTDIR}/eval.json" "${OUTDIR}/confusion_matrix.png"

echo
echo "==> Done. Outputs:"
ls -lh "${OUTDIR}"
