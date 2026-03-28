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

from dataclasses import dataclass
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


def _compute_segments(labels: np.ndarray) -> List[Tuple[int, int, Any]]:
    """Return contiguous segments as (start, end_exclusive, label)."""
    if labels.size == 0:
        return []
    segments: List[Tuple[int, int, Any]] = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segments.append((start, i, cur))
            start = i
            cur = labels[i]
    segments.append((start, len(labels), cur))
    return segments


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract windows from within segments.

    Returns
    -------
    X : np.ndarray
        Shape (n_windows, window_size, n_channels)
    y : np.ndarray
        Shape (n_windows,)
    groups : np.ndarray
        Segment id for each window (used for group splitting).
    """
    X_list: List[np.ndarray] = []
    y_list: List[Any] = []
    g_list: List[int] = []

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

    if not X_list:
        return (
            np.empty((0, window_size, series_2d.shape[1])),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=int),
        )

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)
    groups = np.asarray(g_list, dtype=int)
    return X, y, groups


def _apply_balance_strategy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    strategy: BalanceStrategy,
    max_windows_per_segment: int = 25,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample windows to mitigate duration/class imbalance."""
    if strategy == "none":
        return X, y, groups

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
        return X[keep], y[keep], groups[keep]

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
        return X[keep], y[keep], groups[keep]

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


def _iter_valid_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
    min_classes_per_split: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return up to ``n_splits`` group splits with class coverage in train/test."""
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
        Optional alias for random_state (for CLI convenience).
    rocket_n_kernels
        Number of kernels for ROCKET/MiniROCKET variants.
    """

    if seed is not None:
        random_state = seed

    series_2d = _as_2d_series(series)
    labels_1d = _as_1d_labels(labels)
    if len(series_2d) != len(labels_1d):
        raise ValueError(
            f"series and labels must have the same length; got {len(series_2d)} and {len(labels_1d)}"
        )

    segments = _compute_segments(labels_1d)
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

    scores_by_window: Dict[int, float] = {}
    n_windows_by_window: Dict[int, int] = {}
    split_scores: Dict[int, List[float]] = {}

    for w in window_sizes:
        s = stride if stride is not None else max(1, w // 2)
        X, y, groups = _extract_windows_from_segments(
            series_2d,
            segments,
            window_size=w,
            stride=s,
            labeling=labeling,
            labels=labels_1d,
        )
        X, y, groups = _apply_balance_strategy(
            X,
            y,
            groups,
            strategy=balance,
            max_windows_per_segment=max_windows_per_segment,
            rng=rng,
        )

        n_windows_by_window[w] = int(X.shape[0])
        if X.shape[0] < 10:
            # Not enough data to evaluate.
            scores_by_window[w] = float("nan")
            continue

        fold_scores: List[float] = []
        min_classes = min(2, np.unique(y).size)
        for train_idx, test_idx in _iter_valid_group_splits(
            X,
            y,
            groups,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
            min_classes_per_split=min_classes,
        ):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

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
                )
                y_pred = np.asarray(clf_res.predictions)
            elif classifier == "knn":
                clf_res = knn_classify(
                    X_train_3d,
                    y_train,
                    X_test_3d,
                    y_test,
                    distance="dtw",
                    n_neighbors=1,
                )
                y_pred = np.asarray(clf_res.predictions)
            else:
                raise ValueError(f"Unknown classifier: {classifier}")

            fold_scores.append(_score_metric(metric, y_test, y_pred))

        split_scores[w] = fold_scores
        scores_by_window[w] = float(np.nanmean(fold_scores)) if fold_scores else float("nan")

    # Choose best (highest score), ignoring NaNs.
    best_w = None
    best_score = -float("inf")
    for w, score in scores_by_window.items():
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_w = w
    if best_w is None:
        # All NaN; pick median candidate.
        best_w = int(window_sizes[len(window_sizes) // 2])

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
            "split_scores": {int(k): [float(x) for x in v] for k, v in split_scores.items()},
        },
    )


def select_window_size_from_csv(
    csv_path: str,
    *,
    value_columns: Union[str, List[str]] = "value",
    label_column: str = "label",
    **kwargs: Any,
) -> WindowSizeSelectionResult:
    """Convenience wrapper around :func:`select_window_size` for CSV inputs.

    The CSV is expected to contain one row per timepoint.

    Parameters
    ----------
    csv_path
        Path to CSV.
    value_columns
        Column name (or list of column names) containing the time series value(s).
    label_column
        Column name containing the per-timepoint label.
    kwargs
        Forwarded to :func:`select_window_size`.
    """

    df = pd.read_csv(csv_path)
    if isinstance(value_columns, str):
        if "," in value_columns:
            cols = [c.strip() for c in value_columns.split(",") if c.strip()]
        else:
            cols = [value_columns]
    else:
        cols = list(value_columns)
    for c in cols + [label_column]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}. Available: {list(df.columns)}")
    series = df[cols].to_numpy()
    labels = df[label_column].to_numpy()
    return select_window_size(series, labels, **kwargs)


def evaluate_windowed_classifier(
    series: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    labels: Union[np.ndarray, Sequence[Any]],
    *,
    window_size: int,
    stride: Optional[int] = None,
    classifier: ClassifierName = "minirocket",
    metric: MetricName = "balanced_accuracy",
    balance: BalanceStrategy = "segment_cap",
    max_windows_per_segment: Optional[int] = 25,
    test_size: float = 0.2,
    seed: int = 1337,
    rocket_n_kernels: int = 2000,
) -> WindowedClassificationEvaluation:
    """Evaluate a classifier on sliding windows extracted from a labeled stream.

    This is a convenience wrapper that performs:
    - segment-contained window extraction
    - imbalance handling (optional)
    - group-aware train/test split (by segment)
    - classification + metric computation
    """

    series2d = _as_2d_series(series)
    labels1d = _as_1d_labels(labels)

    segments = _compute_segments(labels1d)
    if not segments:
        raise ValueError("labels produced no segments")

    stride_value = stride if stride is not None else max(1, window_size // 2)
    X, y, groups = _extract_windows_from_segments(
        series2d,
        segments,
        window_size=window_size,
        stride=stride_value,
        labeling="strict",
        min_purity=0.9,
        labels=labels1d,
    )

    rng = np.random.default_rng(seed)
    cap = max_windows_per_segment if max_windows_per_segment is not None else 25
    X, y, groups = _apply_balance_strategy(
        X,
        y,
        groups,
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
        n_splits=1,
        test_size=test_size,
        random_state=seed,
        min_classes_per_split=min_classes,
    )
    if not splits:
        raise ValueError("Unable to construct a valid train/test split from extracted windows")

    train_idx, test_idx = splits[0]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_3d = _to_classifier_input(X_train)
    X_test_3d = _to_classifier_input(X_test)

    if classifier in ("minirocket", "rocket"):
        variant = "minirocket" if classifier == "minirocket" else "rocket"
        clf_res = rocket_classify(
            X_train_3d,
            y_train,
            X_test_3d,
            y_test,
            variant=variant,
            n_kernels=rocket_n_kernels,
        )
    elif classifier == "knn":
        clf_res = knn_classify(
            X_train_3d,
            y_train,
            X_test_3d,
            y_test,
            distance="dtw",
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    y_pred = np.asarray(clf_res.predictions)
    score = _score_metric(metric, y_test, y_pred)

    labels_order = np.unique(np.concatenate([y_train, y_test, y_pred]))
    accuracy_score, f1_score, confusion_matrix, _ = _get_sklearn_windowing_tools()
    clf_res.accuracy = float(accuracy_score(y_test, y_pred))
    clf_res.f1_score = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    clf_res.confusion_matrix = confusion_matrix(y_test, y_pred, labels=labels_order)

    # Make class-counts JSON-friendly.
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {str(k): int(v) for k, v in zip(unique, counts)}

    return WindowedClassificationEvaluation(
        method="windowed_classification",
        window_size=int(window_size),
        stride=int(stride_value),
        classifier=str(classifier),
        metric=str(metric),
        score=float(score),
        n_windows=int(X.shape[0]),
        class_counts=class_counts,
        classification=clf_res.to_dict() if hasattr(clf_res, "to_dict") else dict(clf_res),
    )


def evaluate_windowed_classifier_from_csv(
    csv_path: str,
    *,
    window_size: int,
    value_columns: Union[str, List[str]] = "value",
    label_column: str = "label",
    **kwargs: Any,
) -> WindowedClassificationEvaluation:
    """CSV wrapper for :func:`evaluate_windowed_classifier`."""

    df = pd.read_csv(csv_path)
    if isinstance(value_columns, str):
        if "," in value_columns:
            cols = [c.strip() for c in value_columns.split(",") if c.strip()]
        else:
            cols = [value_columns]
    else:
        cols = list(value_columns)
    for c in cols + [label_column]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}. Available: {list(df.columns)}")
    series = df[cols].to_numpy()
    labels = df[label_column].to_numpy()
    return evaluate_windowed_classifier(series, labels, window_size=window_size, **kwargs)
