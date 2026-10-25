"""Window size selection for sliding-window time series classification.

The repository currently focuses on algorithms that operate on *fixed-length*
collections of time series (typical for most classical TSC libraries). When you
have a *single long time series* with point-wise labels (or segment labels), you
need to:

1) pick a window size (and stride)
2) extract labeled windows
3) train/evaluate a classifier

This module implements a practical, segment-aware search strategy.

Key features
------------
* Grid search over candidate window sizes.
* Candidate generation that can be guided by the distribution of label-segment
  lengths (useful when class durations are uneven).
* Segment-grouped train/test splits to reduce leakage from overlapping windows.
* Optional per-segment cap (prevents long segments from dominating the score).
* Balanced metrics (balanced accuracy / macro-F1) suitable for class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import AnalysisResult
from ..classification.convolution import rocket_classify
from ..classification.distance_based import knn_classify


def _get_sklearn_windowing_tools():
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        from sklearn.model_selection import GroupShuffleSplit
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Windowed classification requires optional dependencies. Install with: pip install "ts-agents[classification]"'
        ) from exc
    return accuracy_score, f1_score, confusion_matrix, GroupShuffleSplit


@dataclass
class WindowedClassificationEvaluation(AnalysisResult):
    """Evaluation result for sliding-window classification on a labeled stream."""

    window_size: int
    stride: int
    classifier: str
    metric: str
    score: float
    n_windows: int
    class_counts: Dict[str, int]
    classification: Dict[str, Any]
    n_splits: int = 1
    split_scores: List[float] = field(default_factory=list)
    retained_classes: List[str] = field(default_factory=list)
    dropped_classes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]
ClassifierName = Literal["minirocket", "rocket", "knn"]
LabelingStrategy = Literal["strict", "majority"]
BalanceStrategy = Literal["none", "undersample", "segment_cap"]


def _as_2d_series(series: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]) -> np.ndarray:
    arr = np.asarray(series)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"series must be 1D or 2D, got shape {arr.shape}")


def _as_1d_labels(labels: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {arr.shape}")
    return arr


def _compute_segments(
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, Any]]:
    """Return contiguous segments as (start, end_exclusive, label).

    Segments break where the label changes and, when ``groups`` is given,
    where the group (subject, session, or contiguous block) changes.
    """
    if labels.size == 0:
        return []
    segments: List[Tuple[int, int, Any]] = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur or (groups is not None and groups[i] != groups[i - 1]):
            segments.append((start, i, cur))
            start = i
            cur = labels[i]
    segments.append((start, len(labels), cur))
    return segments


def _as_group_ids(groups: Optional[Union[np.ndarray, Sequence[Any]]], n: int) -> Optional[np.ndarray]:
    """Map arbitrary per-timepoint group labels to integer codes."""
    if groups is None:
        return None
    arr = np.asarray(groups)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"groups must be 1D with one entry per timepoint ({n}); got shape {arr.shape}")
    if bool(np.any(pd.isna(arr))):
        raise ValueError("groups must not contain missing values.")
    _, codes = np.unique(arr.astype(str), return_inverse=True)
    return codes.astype(int)


def _segment_ids_from_segments(
    segments: List[Tuple[int, int, Any]],
    n_timepoints: int,
) -> np.ndarray:
    """Create a per-timepoint segment id array from segments."""
    segment_ids = np.empty(n_timepoints, dtype=int)
    for seg_id, (start, end, _) in enumerate(segments):
        segment_ids[start:end] = seg_id
    return segment_ids


def _default_candidate_windows(
    segments: List[Tuple[int, int, Any]],
    *,
    min_window: int,
    max_window: int,
    max_candidates: int = 12,
) -> List[int]:
    """Generate a compact set of candidate window sizes.

    Heuristic: take quantiles of segment lengths across *all* labels, clamp to
    bounds, include bounds, and deduplicate.
    """
    lengths = np.array([end - start for start, end, _ in segments], dtype=int)
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return sorted({min_window, max_window})

    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    cand = set()
    for q in qs:
        w = int(np.round(np.quantile(lengths, q)))
        w = max(min_window, min(max_window, w))
        cand.add(w)
    cand.add(min_window)
    cand.add(max_window)

    # Keep smallest-to-largest and cap count.
    out = sorted(cand)
    if len(out) <= max_candidates:
        return out

    # If too many, keep evenly spaced subset.
    idx = np.linspace(0, len(out) - 1, max_candidates).round().astype(int)
    return [out[i] for i in sorted(set(idx))]


def _extract_windows_from_segments(
    series_2d: np.ndarray,
    segments: List[Tuple[int, int, Any]],
    *,
    window_size: int,
    stride: int,
    labeling: LabelingStrategy,
    min_purity: float = 0.8,
    labels: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract windows from within segments.

    Returns
    -------
    X : np.ndarray
        Shape (n_windows, window_size, n_channels)
    y : np.ndarray
        Shape (n_windows,)
    groups : np.ndarray
        Segment id for each window (the segment where it starts).
    spans : np.ndarray
        ``(start, end_exclusive)`` sample range of each window, used to purge
        training windows that overlap test windows (majority labeling).
    """
    X_list: List[np.ndarray] = []
    y_list: List[Any] = []
    g_list: List[int] = []
    span_list: List[Tuple[int, int]] = []

    if labeling == "majority":
        if labels is None:
            raise ValueError("labels are required when labeling='majority'")
        if len(labels) != series_2d.shape[0]:
            raise ValueError(
                f"labels length ({len(labels)}) must match series length ({series_2d.shape[0]})"
            )
        segment_ids = _segment_ids_from_segments(segments, len(labels))

        for s in range(0, len(labels) - window_size + 1, stride):
            e = s + window_size
            if group_ids is not None and group_ids[s] != group_ids[e - 1]:
                continue  # never mix subjects/sessions inside one window
            window_labels = labels[s:e]
            if window_labels.size == 0:
                continue
            values, counts = np.unique(window_labels, return_counts=True)
            best_idx = int(np.argmax(counts))
            purity = counts[best_idx] / float(window_size)
            if purity < min_purity:
                continue
            label = values[best_idx]

            window = series_2d[s:e, :]
            X_list.append(window)
            y_list.append(label)
            g_list.append(int(segment_ids[s]))
            span_list.append((s, e))
    else:
        # Strict labeling: keep windows fully inside a segment.
        for seg_id, (start, end, label) in enumerate(segments):
            seg_len = end - start
            if seg_len < window_size:
                continue

            for s in range(start, end - window_size + 1, stride):
                e = s + window_size
                window = series_2d[s:e, :]
                X_list.append(window)
                y_list.append(label)
                g_list.append(seg_id)
                span_list.append((s, e))

    if not X_list:
        return (
            np.empty((0, window_size, series_2d.shape[1])),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=int),
            np.empty((0, 2), dtype=int),
        )

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)
    groups = np.asarray(g_list, dtype=int)
    spans = np.asarray(span_list, dtype=int)
    return X, y, groups, spans


def _apply_balance_strategy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spans: np.ndarray,
    *,
    strategy: BalanceStrategy,
    max_windows_per_segment: int = 25,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample windows to mitigate duration/class imbalance."""
    if strategy == "none":
        return X, y, groups, spans

    rng = rng or np.random.default_rng(0)

    if strategy == "segment_cap":
        keep_idx: List[int] = []
        for seg_id in np.unique(groups):
            seg_idx = np.flatnonzero(groups == seg_id)
            if seg_idx.size <= max_windows_per_segment:
                keep_idx.extend(seg_idx.tolist())
            else:
                keep_idx.extend(rng.choice(seg_idx, size=max_windows_per_segment, replace=False).tolist())
        keep = np.array(sorted(keep_idx), dtype=int)
        return X[keep], y[keep], groups[keep], spans[keep]

    if strategy == "undersample":
        # Equalize number of windows per class.
        classes, counts = np.unique(y, return_counts=True)
        target = int(np.min(counts))
        keep_idx = []
        for c in classes:
            idx = np.flatnonzero(y == c)
            if idx.size <= target:
                keep_idx.extend(idx.tolist())
            else:
                keep_idx.extend(rng.choice(idx, size=target, replace=False).tolist())
        keep = np.array(sorted(keep_idx), dtype=int)
        return X[keep], y[keep], groups[keep], spans[keep]

    raise ValueError(f"Unknown balance strategy: {strategy}")


def _to_classifier_input(X: np.ndarray) -> np.ndarray:
    """Convert windows to classifier input shape (n_samples, n_channels, n_timepoints)."""
    X = np.asarray(X)
    if X.ndim == 3:
        # Window extraction yields (n_windows, window_size, n_channels).
        return X.transpose(0, 2, 1)
    if X.ndim == 2:
        # Treat as univariate windows (n_windows, window_size).
        return X[:, np.newaxis, :]
    if X.ndim == 1:
        return X.reshape(1, 1, -1)
    raise ValueError(f"windows must be 1D, 2D, or 3D, got shape {X.shape}")


def _balanced_accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy without emitting single-class warnings."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size == 0:
        return 0.0

    recalls: List[float] = []
    for label in labels:
        mask = y_true == label
        denom = int(np.sum(mask))
        if denom == 0:
            continue
        recalls.append(float(np.sum(y_pred[mask] == label) / denom))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _class_count_dict(labels: np.ndarray) -> Dict[str, int]:
    if labels.size == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def _class_retention_summary(all_labels: np.ndarray, retained_labels: np.ndarray) -> Dict[str, Any]:
    all_classes = [str(label) for label in np.unique(all_labels)]
    retained_classes = [str(label) for label in np.unique(retained_labels)] if retained_labels.size else []
    retained_lookup = set(retained_classes)
    dropped_classes = [label for label in all_classes if label not in retained_lookup]
    retained_fraction = (
        float(len(retained_classes) / len(all_classes))
        if all_classes
        else 0.0
    )
    return {
        "all_classes": all_classes,
        "retained_classes": retained_classes,
        "dropped_classes": dropped_classes,
        "retained_class_fraction": retained_fraction,
    }


def _run_window_classifier(
    *,
    classifier: ClassifierName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rocket_n_kernels: int,
    random_state: Optional[int] = None,
):
    X_train_3d = _to_classifier_input(X_train)
    X_test_3d = _to_classifier_input(X_test)

    if classifier in {"rocket", "minirocket"}:
        variant = "minirocket" if classifier == "minirocket" else "rocket"
        clf_res = rocket_classify(
            X_train_3d,
            y_train,
            X_test_3d,
            y_test,
            variant=variant,
            n_kernels=rocket_n_kernels,
            random_state=random_state,
        )
    elif classifier == "knn":
        clf_res = knn_classify(
            X_train_3d,
            y_train,
            X_test_3d,
            y_test,
            distance="dtw",
            n_neighbors=1,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return np.asarray(clf_res.predictions), clf_res


def _iter_valid_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
    min_classes_per_split: int = 2,
    spans: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return up to ``n_splits`` group splits with class coverage in train/test.

    When ``spans`` is given, training windows whose samples overlap any test
    window are purged, so train and test never share raw samples even when
    windows cross segment boundaries.
    """
    _, _, _, GroupShuffleSplit = _get_sklearn_windowing_tools()
    attempts = max(n_splits, n_splits * 4)
    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=test_size,
        random_state=random_state,
    )

    valid_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    fallback: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        if spans is not None and len(spans):
            train_idx = _purge_overlapping(train_idx, test_idx, spans)
        if fallback is None:
            fallback = (train_idx, test_idx)

        train_classes = np.unique(y[train_idx]).size
        test_classes = np.unique(y[test_idx]).size
        if train_classes >= min_classes_per_split and test_classes >= min_classes_per_split:
            valid_splits.append((train_idx, test_idx))
            if len(valid_splits) >= n_splits:
                break

    if valid_splits:
        return valid_splits
    if fallback is not None:
        return [fallback]
    return []


def _purge_overlapping(train_idx: np.ndarray, test_idx: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Drop training windows that share any sample with a test window."""
    n = int(spans[:, 1].max())
    covered = np.zeros(n + 1, dtype=int)
    for start, end in spans[test_idx]:
        covered[start] += 1
        covered[end] -= 1
    in_test = np.cumsum(covered)[:n] > 0
    prefix = np.concatenate([[0], np.cumsum(in_test)])
    starts, ends = spans[train_idx, 0], spans[train_idx, 1]
    overlaps = (prefix[ends] - prefix[starts]) > 0
    return train_idx[~overlaps]


def _score_metric(metric: MetricName, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy_score, f1_score, _, _ = _get_sklearn_windowing_tools()
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return _balanced_accuracy_safe(y_true, y_pred)
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")


@dataclass
class WindowSizeSelectionResult(AnalysisResult):
    """Result from window-size selection."""

    best_window_size: int
    metric: str
    scores_by_window: Dict[int, float]
    n_windows_by_window: Dict[int, int]
    details: Dict[str, Any]


def select_window_size(
    series: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    labels: Union[np.ndarray, Sequence[Any]],
    *,
    window_sizes: Optional[Sequence[int]] = None,
    min_window: int = 16,
    max_window: Optional[int] = None,
    stride: Optional[int] = None,
    metric: MetricName = "balanced_accuracy",
    classifier: ClassifierName = "minirocket",
    labeling: LabelingStrategy = "strict",
    balance: BalanceStrategy = "segment_cap",
    max_windows_per_segment: int = 25,
    n_splits: int = 3,
    test_size: float = 0.2,
    random_state: int = 0,
    seed: Optional[int] = None,
    rocket_n_kernels: int = 2000,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
) -> WindowSizeSelectionResult:
    """Select an appropriate window size for sliding-window classification.

    Parameters
    ----------
    series
        The time series, shape (n_timepoints,) or (n_timepoints, n_channels).
    labels
        Per-timepoint labels, shape (n_timepoints,). Labels should be constant
        within each segment.
    window_sizes
        Candidate window sizes. If None, they are derived from label-segment
        length quantiles.
    min_window
        Minimum window size if candidates are auto-generated.
    max_window
        Maximum window size. Defaults to the 95th percentile of segment lengths
        (clamped to series length).
    stride
        Stride for window extraction. Defaults to window_size // 2.
    metric
        Scoring metric: accuracy, balanced_accuracy, or f1_macro.
    classifier
        Classifier used for evaluation. "minirocket" is a strong, fast default.
    labeling
        "strict" keeps windows within a single segment label (recommended).
    balance
        Strategy to reduce duration/class imbalance.
    max_windows_per_segment
        Cap windows per segment when balance == "segment_cap".
    n_splits
        Number of random group splits.
    test_size
        Fraction of segments reserved for test per split.
    seed
        Optional alias for random_state (for CLI convenience). Also seeds
        ROCKET-family kernels so results are reproducible.
    rocket_n_kernels
        Number of kernels for ROCKET/MiniROCKET variants.
    groups
        Optional per-timepoint group labels (subject, session, or contiguous
        block). Windows never span two groups.

    The best window is the highest-scoring candidate that keeps every class;
    only when no candidate keeps every class does the overall best win, and
    ``details["selection_rule"]`` records which rule applied.
    """

    if seed is not None:
        random_state = seed

    series_2d = _as_2d_series(series)
    labels_1d = _as_1d_labels(labels)
    if len(series_2d) != len(labels_1d):
        raise ValueError(
            f"series and labels must have the same length; got {len(series_2d)} and {len(labels_1d)}"
        )

    group_ids = _as_group_ids(groups, len(labels_1d))
    segments = _compute_segments(labels_1d, group_ids)
    if not segments:
        raise ValueError("No label segments found; labels appear to be empty")

    seg_lengths = np.array([end - start for start, end, _ in segments], dtype=int)
    if max_window is None:
        max_window = int(np.clip(np.quantile(seg_lengths, 0.95), min_window, len(labels_1d)))
    max_window = int(min(max_window, len(labels_1d)))

    if window_sizes is None:
        window_sizes = _default_candidate_windows(
            segments,
            min_window=min_window,
            max_window=max_window,
        )
    window_sizes = [int(w) for w in window_sizes if w is not None]
    window_sizes = sorted({w for w in window_sizes if w >= 2 and w <= len(labels_1d)})
    if not window_sizes:
        raise ValueError("No valid candidate window sizes")

    rng = np.random.default_rng(random_state)
    original_classes = np.unique(labels_1d)

    scores_by_window: Dict[int, float] = {}
    n_windows_by_window: Dict[int, int] = {}
    split_scores: Dict[int, List[float]] = {}
    window_diagnostics: Dict[int, Dict[str, Any]] = {}

    for w in window_sizes:
        s = stride if stride is not None else max(1, w // 2)
        X, y, split_units, spans = _extract_windows_from_segments(
            series_2d,
            segments,
            window_size=w,
            stride=s,
            labeling=labeling,
            labels=labels_1d,
            group_ids=group_ids,
        )
        X, y, split_units, spans = _apply_balance_strategy(
            X,
            y,
            split_units,
            spans,
            strategy=balance,
            max_windows_per_segment=max_windows_per_segment,
            rng=rng,
        )

        n_windows_by_window[w] = int(X.shape[0])
        retention = _class_retention_summary(original_classes, y)
        window_diagnostics[w] = {
            "window_size": int(w),
            "stride": int(s),
            "n_windows": int(X.shape[0]),
            "class_counts": _class_count_dict(y),
            "retained_classes": retention["retained_classes"],
            "dropped_classes": retention["dropped_classes"],
            "retained_class_fraction": retention["retained_class_fraction"],
        }
        if X.shape[0] < 10:
            # Not enough data to evaluate.
            scores_by_window[w] = float("nan")
            split_scores[w] = []
            window_diagnostics[w]["valid_split_count"] = 0
            continue

        fold_scores: List[float] = []
        min_classes = min(2, np.unique(y).size)
        for train_idx, test_idx in _iter_valid_group_splits(
            X,
            y,
            split_units,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
            min_classes_per_split=min_classes,
            spans=spans if labeling == "majority" else None,
        ):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            y_pred, _ = _run_window_classifier(
                classifier=classifier,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                rocket_n_kernels=rocket_n_kernels,
                random_state=random_state,
            )
            fold_scores.append(_score_metric(metric, y_test, y_pred))

        split_scores[w] = fold_scores
        scores_by_window[w] = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        window_diagnostics[w]["valid_split_count"] = len(fold_scores)

    # Choose the best scoring window among those that keep every class; a
    # window that scores higher only by dropping hard classes is not "best".
    def _best(candidates: List[int]) -> Optional[int]:
        scored = [w for w in candidates if not np.isnan(scores_by_window[w])]
        return max(scored, key=lambda w: scores_by_window[w]) if scored else None

    full_coverage = [w for w in window_sizes if not window_diagnostics[w]["dropped_classes"]]
    best_w = _best(full_coverage)
    selection_rule = "best_score_with_all_classes"
    if best_w is None:
        best_w = _best(list(window_sizes))
        selection_rule = "best_score_some_classes_dropped"
    if best_w is None:
        # All NaN; pick median candidate.
        best_w = int(window_sizes[len(window_sizes) // 2])
        selection_rule = "no_valid_scores_median_candidate"

    return WindowSizeSelectionResult(
        method="window_size_selection",
        best_window_size=int(best_w),
        metric=str(metric),
        scores_by_window={int(k): float(v) for k, v in scores_by_window.items()},
        n_windows_by_window={int(k): int(v) for k, v in n_windows_by_window.items()},
        details={
            "candidates": window_sizes,
            "stride": stride,
            "labeling": labeling,
            "balance": balance,
            "max_windows_per_segment": max_windows_per_segment,
            "n_splits": n_splits,
            "test_size": test_size,
            "all_classes": [str(label) for label in original_classes],
            "selection_rule": selection_rule,
            "random_state": int(random_state),
            "grouped": group_ids is not None,
            "split_scores": {int(k): [float(x) for x in v] for k, v in split_scores.items()},
            "window_diagnostics": {int(k): v for k, v in window_diagnostics.items()},
        },
    )


def select_window_size_from_csv(
    csv_path: str,
    *,
    value_columns: Union[str, List[str]] = "value",
    label_column: str = "label",
    time_column: Optional[str] = None,
    **kwargs: Any,
) -> WindowSizeSelectionResult:
    """Convenience wrapper around :func:`select_window_size` for tabular inputs.

    The historical ``csv_path`` parameter accepts any tabular path supported by
    :func:`ts_agents.cli.input_parsing.load_labeled_stream_input`.

    Parameters
    ----------
    csv_path
        Path to a tabular file.
    value_columns
        Column name (or list of column names) containing the time series value(s).
    label_column
        Column name containing the per-timepoint label.
    time_column
        Optional time/index column for the tabular input.
    kwargs
        Forwarded to :func:`select_window_size`.
    """
    from ts_agents.cli.input_parsing import load_labeled_stream_input

    cols = (
        [c.strip() for c in value_columns.split(",") if c.strip()]
        if isinstance(value_columns, str)
        else list(value_columns)
    )
    stream_input = load_labeled_stream_input(
        input_path=csv_path,
        time_col=time_column,
        value_cols=cols,
        label_col=label_column,
    )
    return select_window_size(stream_input.values, stream_input.labels, **kwargs)


def evaluate_windowed_classifier(
    series: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    labels: Union[np.ndarray, Sequence[Any]],
    *,
    window_size: int,
    stride: Optional[int] = None,
    classifier: ClassifierName = "minirocket",
    metric: MetricName = "balanced_accuracy",
    labeling: LabelingStrategy = "strict",
    balance: BalanceStrategy = "segment_cap",
    max_windows_per_segment: Optional[int] = 25,
    n_splits: int = 1,
    test_size: float = 0.2,
    seed: int = 1337,
    rocket_n_kernels: int = 2000,
    min_label_purity: float = 0.8,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
) -> WindowedClassificationEvaluation:
    """Evaluate a classifier on sliding windows extracted from a labeled stream.

    This is a convenience wrapper that performs:
    - segment-contained window extraction (never across ``groups``)
    - imbalance handling (optional)
    - group-aware train/test split (by segment, or by connected segments
      for majority labeling, so train and test never share raw samples)
    - classification + metric computation
    """

    series2d = _as_2d_series(series)
    labels1d = _as_1d_labels(labels)
    group_ids = _as_group_ids(groups, len(labels1d))

    segments = _compute_segments(labels1d, group_ids)
    if not segments:
        raise ValueError("labels produced no segments")

    original_classes = np.unique(labels1d)
    stride_value = stride if stride is not None else max(1, window_size // 2)
    X, y, groups, spans = _extract_windows_from_segments(
        series2d,
        segments,
        window_size=window_size,
        stride=stride_value,
        labeling=labeling,
        min_purity=min_label_purity,
        labels=labels1d,
        group_ids=group_ids,
    )

    rng = np.random.default_rng(seed)
    cap = max_windows_per_segment if max_windows_per_segment is not None else 25
    X, y, groups, spans = _apply_balance_strategy(
        X,
        y,
        groups,
        spans,
        strategy=balance,
        max_windows_per_segment=cap,
        rng=rng,
    )

    if X.shape[0] < 10:
        raise ValueError(
            f"Too few windows ({X.shape[0]}) extracted. "
            "Try a smaller window size, smaller stride, or majority labeling."
        )

    min_classes = min(2, np.unique(y).size)
    splits = _iter_valid_group_splits(
        X,
        y,
        groups,
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
        min_classes_per_split=min_classes,
        spans=spans if labeling == "majority" else None,
    )
    if not splits:
        raise ValueError("Unable to construct a valid train/test split from extracted windows")

    split_scores: List[float] = []
    pooled_true: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    classification_method: Optional[str] = None
    classification_warnings: List[str] = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred, clf_res = _run_window_classifier(
            classifier=classifier,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            rocket_n_kernels=rocket_n_kernels,
            random_state=seed,
        )
        split_scores.append(_score_metric(metric, y_test, y_pred))
        pooled_true.append(np.asarray(y_test))
        pooled_pred.append(np.asarray(y_pred))
        if classification_method is None:
            classification_method = str(getattr(clf_res, "method", classifier))
        for warning in getattr(clf_res, "warnings", []) or []:
            if warning not in classification_warnings:
                classification_warnings.append(str(warning))

    y_true = np.concatenate(pooled_true, axis=0)
    y_pred = np.concatenate(pooled_pred, axis=0)
    score = float(np.nanmean(split_scores)) if split_scores else 0.0

    labels_order = np.unique(np.concatenate([y_true, y_pred]))
    accuracy_score, f1_score, confusion_matrix, _ = _get_sklearn_windowing_tools()
    retention = _class_retention_summary(original_classes, y)
    classification = {
        "method": classification_method or str(classifier),
        "predictions": y_pred.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_order).tolist(),
    }
    if classification_warnings:
        classification["warnings"] = classification_warnings
    class_counts = _class_count_dict(y)

    return WindowedClassificationEvaluation(
        method="windowed_classification",
        window_size=int(window_size),
        stride=int(stride_value),
        classifier=str(classifier),
        metric=str(metric),
        score=float(score),
        n_windows=int(X.shape[0]),
        class_counts=class_counts,
        classification=classification,
        n_splits=len(splits),
        split_scores=[float(value) for value in split_scores],
        retained_classes=retention["retained_classes"],
        dropped_classes=retention["dropped_classes"],
        details={
            "labeling": labeling,
            "balance": balance,
            "max_windows_per_segment": cap,
            "requested_n_splits": int(n_splits),
            "valid_split_count": int(len(splits)),
            "retained_class_fraction": retention["retained_class_fraction"],
            "all_classes": retention["all_classes"],
            "test_class_counts": _class_count_dict(y_true),
        },
    )


def split_holdout_units(
    labels: Union[np.ndarray, Sequence[Any]],
    *,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    test_size: float = 0.25,
    seed: int = 1337,
    max_attempts: int = 200,
) -> np.ndarray:
    """Choose a held-out set of whole groups (or label segments) for final scoring.

    Returns a boolean per-timepoint mask (True = held out). Units are the
    given ``groups`` (e.g. subjects) or, without groups, contiguous label
    segments. The holdout must contain at least two classes and every
    held-out class must also appear in the remaining data.
    """
    labels_1d = _as_1d_labels(labels)
    group_ids = _as_group_ids(groups, len(labels_1d))
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if group_ids is not None:
        unit_ids = group_ids
    else:
        unit_ids = _segment_ids_from_segments(_compute_segments(labels_1d), len(labels_1d))
    units = np.unique(unit_ids)
    if units.size < 2:
        raise ValueError(
            "A held-out evaluation needs at least two groups; "
            + ("the group column has only one value." if group_ids is not None else "labels form a single segment.")
        )
    target = test_size * len(labels_1d)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        chosen: List[int] = []
        held = 0
        for unit in rng.permutation(units):
            if held >= target or len(chosen) == units.size - 1:
                break
            chosen.append(int(unit))
            held += int(np.sum(unit_ids == unit))
        mask = np.isin(unit_ids, chosen)
        held_classes = set(np.unique(labels_1d[mask]).tolist())
        pool_classes = set(np.unique(labels_1d[~mask]).tolist())
        if len(held_classes) >= 2 and held_classes <= pool_classes:
            return mask
    raise ValueError(
        "Could not find a held-out split with at least two classes whose classes all appear "
        "in the remaining data; add more groups/segments or lower test_size."
    )


def contiguous_block_ids(
    holdout_mask: np.ndarray,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
) -> np.ndarray:
    """Integer ids for runs of constant (group, holdout membership)."""
    mask = np.asarray(holdout_mask, dtype=bool)
    group_ids = _as_group_ids(groups, len(mask))
    change = np.zeros(len(mask), dtype=bool)
    change[1:] = mask[1:] != mask[:-1]
    if group_ids is not None:
        change[1:] |= group_ids[1:] != group_ids[:-1]
    return np.cumsum(change)


def evaluate_windowed_classifier_holdout(
    series: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    labels: Union[np.ndarray, Sequence[Any]],
    *,
    holdout_mask: Union[np.ndarray, Sequence[bool]],
    window_size: int,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    stride: Optional[int] = None,
    classifier: ClassifierName = "minirocket",
    metric: MetricName = "balanced_accuracy",
    labeling: LabelingStrategy = "strict",
    seed: int = 1337,
    rocket_n_kernels: int = 2000,
    min_label_purity: float = 0.8,
) -> WindowedClassificationEvaluation:
    """Train on non-held-out windows and score once on held-out windows.

    Windows never span holdout boundaries or ``groups``, so the score is
    independent of any data used to choose the window size (when selection
    only saw the non-held-out rows).
    """
    series2d = _as_2d_series(series)
    labels1d = _as_1d_labels(labels)
    mask = np.asarray(holdout_mask, dtype=bool)
    if mask.shape != labels1d.shape:
        raise ValueError("holdout_mask must have one entry per timepoint.")
    block_ids = contiguous_block_ids(mask, groups)
    segments = _compute_segments(labels1d, block_ids)
    original_classes = np.unique(labels1d)
    stride_value = stride if stride is not None else max(1, window_size // 2)
    X, y, _, spans = _extract_windows_from_segments(
        series2d,
        segments,
        window_size=window_size,
        stride=stride_value,
        labeling=labeling,
        min_purity=min_label_purity,
        labels=labels1d,
        group_ids=block_ids,
    )
    in_holdout = mask[spans[:, 0]] if len(spans) else np.zeros(0, dtype=bool)
    train_idx, test_idx = np.flatnonzero(~in_holdout), np.flatnonzero(in_holdout)
    if train_idx.size < 2 or test_idx.size < 1:
        raise ValueError(
            f"Window size {window_size} leaves {train_idx.size} training and {test_idx.size} held-out windows; "
            "use a smaller window or a larger holdout."
        )
    if np.unique(y[train_idx]).size < 2:
        raise ValueError("Training windows contain fewer than two classes.")

    y_pred, clf_res = _run_window_classifier(
        classifier=classifier,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        rocket_n_kernels=rocket_n_kernels,
        random_state=seed,
    )
    y_true = y[test_idx]
    score = _score_metric(metric, y_true, y_pred)
    labels_order = np.unique(np.concatenate([y_true, y_pred]))
    accuracy_score, f1_score, confusion_matrix, _ = _get_sklearn_windowing_tools()
    retention = _class_retention_summary(original_classes, y)
    classification = {
        "method": str(getattr(clf_res, "method", classifier)),
        "predictions": np.asarray(y_pred).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_order).tolist(),
        "labels": [str(label) for label in labels_order],
    }
    if getattr(clf_res, "warnings", None):
        classification["warnings"] = [str(w) for w in clf_res.warnings]
    test_counts = _class_count_dict(y_true)
    majority_share = max(test_counts.values()) / float(len(y_true))
    return WindowedClassificationEvaluation(
        method="windowed_classification_holdout",
        window_size=int(window_size),
        stride=int(stride_value),
        classifier=str(classifier),
        metric=str(metric),
        score=float(score),
        n_windows=int(X.shape[0]),
        class_counts=_class_count_dict(y),
        classification=classification,
        n_splits=1,
        split_scores=[float(score)],
        retained_classes=retention["retained_classes"],
        dropped_classes=retention["dropped_classes"],
        details={
            "evaluation": "holdout",
            "labeling": labeling,
            "n_train_windows": int(train_idx.size),
            "n_holdout_windows": int(test_idx.size),
            "holdout_fraction_of_timepoints": float(mask.mean()),
            "test_class_counts": test_counts,
            # Chance references for interpreting the score.
            "chance_balanced_accuracy": float(1.0 / len(test_counts)),
            "majority_class_accuracy": float(majority_share),
            "random_state": int(seed),
            "all_classes": retention["all_classes"],
            "retained_class_fraction": retention["retained_class_fraction"],
        },
    )


def evaluate_windowed_classifier_from_csv(
    csv_path: str,
    *,
    window_size: int,
    value_columns: Union[str, List[str]] = "value",
    label_column: str = "label",
    time_column: Optional[str] = None,
    **kwargs: Any,
) -> WindowedClassificationEvaluation:
    """Tabular-file wrapper for :func:`evaluate_windowed_classifier`."""
    from ts_agents.cli.input_parsing import load_labeled_stream_input

    cols = (
        [c.strip() for c in value_columns.split(",") if c.strip()]
        if isinstance(value_columns, str)
        else list(value_columns)
    )
    stream_input = load_labeled_stream_input(
        input_path=csv_path,
        time_col=time_column,
        value_cols=cols,
        label_col=label_column,
    )
    return evaluate_windowed_classifier(
        stream_input.values,
        stream_input.labels,
        window_size=window_size,
        **kwargs,
    )
