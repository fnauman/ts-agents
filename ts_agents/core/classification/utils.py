"""Shared utilities for classification modules."""

from typing import Any, Dict, Optional

import numpy as np


def ensure_3d(X: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (n_samples, n_channels, n_timepoints).

    Parameters
    ----------
    X : np.ndarray
        Input array. Can be 1D (single univariate sample),
        2D (n_samples, n_timepoints), or 3D (already correct).

    Returns
    -------
    np.ndarray
        3D array with shape (n_samples, n_channels, n_timepoints)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, 1, -1)
    elif X.ndim == 2:
        return X.reshape(X.shape[0], 1, X.shape[1])
    return X


def classification_metrics(y_true: Optional[Any], y_pred: Any) -> Dict[str, Any]:
    """Accuracy, balanced accuracy, macro-F1, and confusion matrix.

    Returns keyword arguments for :class:`ClassificationResult`; every metric
    is ``None`` when ``y_true`` is not provided. Balanced accuracy averages
    recall over classes present in ``y_true``; macro-F1 averages over the
    union of true and predicted labels (classes with no support score 0).
    """
    if y_true is None:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_score": None,
            "confusion_matrix": None,
            "labels": None,
        }
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_test has shape {y_true.shape} but predictions have shape {y_pred.shape}."
        )
    labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {label: i for i, label in enumerate(labels.tolist())}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true.tolist(), y_pred.tolist()):
        matrix[index[actual], index[predicted]] += 1

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    present = support > 0
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=present)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(len(labels)), where=predicted_count > 0
    )
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(labels)), where=denom > 0)

    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else None,
        "balanced_accuracy": float(np.mean(recall[present])) if np.any(present) else None,
        "f1_score": float(np.mean(f1)) if len(labels) else None,
        "confusion_matrix": matrix,
        "labels": [label.item() if hasattr(label, "item") else label for label in labels],
    }
