"""Distance-based time series classification.

This module provides functions for time series classification using
distance measures like DTW (Dynamic Time Warping) with K-Nearest Neighbors.
"""

from typing import Optional, Literal, Tuple
import numpy as np

from ..base import ClassificationResult
from .utils import ensure_3d


def knn_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    distance: Literal["dtw", "euclidean", "msm", "erp", "lcss"] = "dtw",
    n_neighbors: int = 1,
    weights: Literal["uniform", "distance"] = "uniform",
) -> ClassificationResult:
    """Time series classification using K-Nearest Neighbors with DTW or other distances.

    Parameters
    ----------
    X_train : np.ndarray
        Training data. Shape: (n_samples, n_channels, n_timepoints)
        or (n_samples, n_timepoints) for univariate.
    y_train : np.ndarray
        Training labels. Shape: (n_samples,)
    X_test : np.ndarray
        Test data. Same shape convention as X_train.
    y_test : np.ndarray, optional
        Test labels for accuracy computation.
    distance : str
        Distance metric:
        - 'dtw': Dynamic Time Warping
        - 'euclidean': Euclidean distance
        - 'msm': Move-Split-Merge
        - 'erp': Edit Distance with Real Penalty
        - 'lcss': Longest Common Subsequence
    n_neighbors : int
        Number of neighbors (default: 1 for classic DTW-NN).
    weights : str
        Weight function: 'uniform' or 'distance'.

    Returns
    -------
    ClassificationResult
        Predictions and accuracy metrics.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X_train = np.random.randn(50, 1, 100)  # 50 samples, 1 channel, 100 timepoints
    >>> y_train = np.array([0]*25 + [1]*25)
    >>> X_test = np.random.randn(10, 1, 100)
    >>> result = knn_classify(X_train, y_train, X_test, distance='dtw')
    >>> print(f"Predictions: {result.predictions}")
    """
    try:
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    except ImportError:
        return _fallback_knn_classify(X_train, y_train, X_test, y_test)

    # Ensure 3D format for aeon
    X_train = ensure_3d(X_train)
    X_test = ensure_3d(X_test)
    y_train = np.asarray(y_train)

    clf = KNeighborsTimeSeriesClassifier(
        distance=distance,
        n_neighbors=n_neighbors,
        weights=weights,
    )
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    # Get probabilities if available
    probabilities = None
    if hasattr(clf, 'predict_proba'):
        try:
            probabilities = clf.predict_proba(X_test)
        except Exception:
            pass

    # Compute accuracy if test labels provided
    accuracy = None
    if y_test is not None:
        y_test = np.asarray(y_test)
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method=f"knn_{distance}",
        predictions=predictions,
        probabilities=probabilities,
        accuracy=accuracy,
    )


def _fallback_knn_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
) -> ClassificationResult:
    """Fallback KNN implementation when aeon is not available."""
    from sklearn.neighbors import KNeighborsClassifier

    # Flatten to 2D for sklearn
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train_2d, y_train)

    predictions = clf.predict(X_test_2d)
    probabilities = clf.predict_proba(X_test_2d)

    accuracy = None
    if y_test is not None:
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method="knn_euclidean_fallback",
        predictions=predictions,
        probabilities=probabilities,
        accuracy=accuracy,
    )


def compute_dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Compute DTW distance between two time series.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    window : int, optional
        Sakoe-Chiba band width. If None, no constraint.

    Returns
    -------
    float
        DTW distance

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> y = np.sin(np.linspace(0.5, 2.5*np.pi, 100))
    >>> dist = compute_dtw_distance(x, y)
    >>> print(f"DTW distance: {dist:.3f}")
    """
    try:
        from aeon.distances import dtw_distance
        return float(dtw_distance(series1, series2, window=window))
    except ImportError:
        return _dtw_fallback(series1, series2, window)


def _dtw_fallback(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Simple DTW implementation as fallback."""
    n, m = len(series1), len(series2)

    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Apply window constraint if specified
    if window is None:
        window = max(n, m)

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = abs(series1[i - 1] - series2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1],     # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return float(dtw_matrix[n, m])
