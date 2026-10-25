### Report on Activity-Recognition Workflow

#### Inputs
- **Source**: `activity_fixture.csv`
- **Value Columns**: x, y, z
- **Label Column**: `label`
- **Preparation**: Used labels already present in the primary input table.

#### Controls
- **Classifier Requested**: `knn`
- **Classifier Resolved**: `knn`
- **Effective Backend**: `knn_dtw`
- **Metric**: `balanced_accuracy`
- **Labeling**: `strict`
- **Balance**: `segment_cap`
- **Stride**: auto (window_size // 2)
- **Selection Splits**: 3 (grouped, non-held-out data only)
- **Holdout**: whole label segments; 38 held-out windows, 76 training windows

#### Results
- **Best Window Size**: 8 samples
- **Held-out Score**: 1.0000
- **Chance (balanced accuracy)**: 0.5000
- **Accuracy**: 1.0000
- **Macro F1**: 1.0000
- **Retained Classes**: `idle`, `walk`
- **Dropped Classes**: none
- **Quality Flags**: fewer_valid_splits_than_requested, perfect_metrics
- **Warnings**: none

#### Window Sweep
| Window | Score | Windows | Retained | Dropped | Splits |
| --- | --- | --- | --- | --- | --- |
| 8 | 1.0000 | 76 | `idle`, `walk` | none | 1 |
| 16 | 1.0000 | 36 | `idle`, `walk` | none | 1 |

#### Note
Requested 3 grouped selection split(s), but only 1 were valid for the best window. All reported metrics are perfect; verify split balance and leakage assumptions.