---
name: time-series-classification
description: >
  Supervised time series classification: choose and run classifiers (KNN/DTW, ROCKET variants, HIVE-COTE), compare models, and report accuracy.
  Use when the user asks to classify/categorize time series, build a classifier, or compare time series classification algorithms.
compatibility: "Works in ts-agents via Python API or CLI. Classification tools typically expect arrays (X_train, y_train, X_test, y_test)."
metadata:
  domain: time-series
  tasks: [classification, supervised-learning, tsc]
  ts_agents:
    tool_category: classification
    preferred_tools: [rocket_classify, knn_classify, compare_classifiers]
---

# Time series classification

## What this skill is for
Time series classification (TSC) is supervised learning where:
- inputs: many labeled time series (`X_train`, `y_train`)
- output: predicted labels for new series (`X_test`)

In ts-agents, collection classifiers are **array-based**, so you usually:
1) build arrays in Python (or JSON for `--input-json`)
2) call `knn_classify`, `rocket_classify`, `hivecote_classify`, or `compare_classifiers`

Results report `accuracy`, `balanced_accuracy`, macro `f1_score`, and a
`confusion_matrix` with `labels`. Lead with balanced accuracy on imbalanced
data; `compare_classifiers` ranks by it by default (`rank_metric`). Pass
`random_state` to ROCKET-family classifiers for reproducible kernels; unknown
variants are rejected.

## Choose a classifier (practical rubric)
- **ROCKET / MiniRocket / MultiRocket**: strong default; fast; good starting point.
- **KNN (DTW distance)**: good for small datasets; interpretable; can be slower as data grows.
- **HIVE-COTE 2**: state-of-the-art accuracy but *very expensive*; use only when needed.

## Data shape conventions
The core wrappers accept flexible shapes and will coerce to aeon’s expected 3D:
- preferred: `(n_samples, n_channels, n_timepoints)`
- univariate common case: `(n_samples, n_timepoints)` is usually fine

Labels:
- `y_train`: shape `(n_samples,)` (strings or ints)

## Labeled-stream classification (windowing)
If you have **one long labeled stream** (label per timepoint or segment), first
pick a window size, then evaluate a classifier on windows.

Core tools:
- `select_window_size` / `select_window_size_from_csv`
- `evaluate_windowed_classifier` / `evaluate_windowed_classifier_from_csv`

CLI example:
```bash
ts-agents tool run select_window_size_from_csv \
  --param csv_path=data/labeled_stream.csv \
  --param value_columns=value \
  --param label_column=label \
  --param min_window=16 \
  --param max_window=256 \
  --param metric=balanced_accuracy \
  --param classifier=minirocket

ts-agents tool run evaluate_windowed_classifier_from_csv \
  --param csv_path=data/labeled_stream.csv \
  --param value_columns=value \
  --param label_column=label \
  --param window_size=64 \
  --param stride=32 \
  --param metric=balanced_accuracy
```

Practical tips:
- Streams that concatenate subjects or sessions need subject-wise evaluation:
  use the `activity-recognition` workflow with `--group-col`.
- If you see too few windows, reduce `window_size`, reduce `stride`, or use `labeling=majority`
  (training windows overlapping test windows are purged).
- For class imbalance, keep `balance=segment_cap` and tune `max_windows_per_segment`.

## Python workflow (recommended)
```python
import numpy as np
from ts_agents.core.classification import rocket_classify, compare_classifiers

# Example toy data
X_train = np.random.randn(50, 200)   # 50 series, length 200
y_train = np.array([0]*25 + [1]*25)

X_test  = np.random.randn(10, 200)
y_test  = np.random.randint(0, 2, size=10)

res = rocket_classify(X_train, y_train, X_test, y_test, variant="minirocket", random_state=0)
print(res.balanced_accuracy, res.f1_score)

cmp = compare_classifiers(X_train, y_train, X_test, y_test, classifiers=["rocket", "knn_dtw"])
print(cmp)
```

## CLI usage (only when data is already serialized)
You *can* pass arrays via `--param` as JSON, but it’s usually cumbersome. Prefer Python for non-trivial datasets.

## Tool discovery (future-proofing)
```bash
ts-agents tool list --category classification --json
```

## Output expectations
Always report:
- what classifier(s) you tried and why
- train/test sizes and series length
- accuracy (or another metric if requested)
- common failure modes (class imbalance, too few samples, inconsistent lengths)

If the user asks for “best model”:
- start with ROCKET + a simple baseline (DTW-KNN)
- only escalate to HIVE-COTE with explicit approval (it can be very slow)
