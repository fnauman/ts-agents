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
    prefers_with_data_tools: false
---

# Time series classification

## What this skill is for
Time series classification (TSC) is supervised learning where:
- inputs: many labeled time series (`X_train`, `y_train`)
- output: predicted labels for new series (`X_test`)

In ts-agents, current classification tools are **array-based** (not `*_with_data`), so you usually:
1) build arrays in Python
2) call `knn_classify`, `rocket_classify`, `hivecote_classify`, or `compare_classifiers`

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
uv run ts-agents run select_window_size_from_csv \
  --param csv_path=data/labeled_stream.csv \
  --param value_columns=value \
  --param label_column=label \
  --param min_window=16 \
  --param max_window=256 \
  --param metric=balanced_accuracy \
  --param classifier=minirocket

uv run ts-agents run evaluate_windowed_classifier_from_csv \
  --param csv_path=data/labeled_stream.csv \
  --param value_columns=value \
  --param label_column=label \
  --param window_size=64 \
  --param stride=32 \
  --param metric=balanced_accuracy
```

Practical tips:
- If you see too few windows, reduce `window_size`, reduce `stride`, or use `labeling=majority`.
- For class imbalance, keep `balance=segment_cap` and tune `max_windows_per_segment`.

## Python workflow (recommended)
```python
import numpy as np
from src.core.classification import rocket_classify, compare_classifiers

# Example toy data
X_train = np.random.randn(50, 200)   # 50 series, length 200
y_train = np.array([0]*25 + [1]*25)

X_test  = np.random.randn(10, 200)
y_test  = np.random.randint(0, 2, size=10)

res = rocket_classify(X_train, y_train, X_test, y_test, variant="minirocket")
print(res.accuracy)

cmp = compare_classifiers(X_train, y_train, X_test, y_test, classifiers=["rocket", "knn_dtw"])
print(cmp)
```

## CLI usage (only when data is already serialized)
You *can* pass arrays via `--param` as JSON, but it’s usually cumbersome. Prefer Python for non-trivial datasets.

## Tool discovery (future-proofing)
```bash
uv run ts-agents tool list --category classification --json
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
