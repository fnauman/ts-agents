"""Convolution-based time series classification (ROCKET family).

This module provides functions for time series classification using
random convolutional kernels (ROCKET and variants).
"""

from typing import Optional, Literal
import numpy as np

from ..base import ClassificationResult
from .utils import ensure_3d


def rocket_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    variant: Literal["rocket", "minirocket", "multirocket"] = "rocket",
    n_kernels: int = 10000,
) -> ClassificationResult:
    """Time series classification using ROCKET (Random Convolutional Kernel Transform).

    ROCKET generates random convolutional kernels and uses a linear classifier
    on the transformed features. Very fast and competitive accuracy.

    Parameters
    ----------
    X_train : np.ndarray
        Training data. Shape: (n_samples, n_channels, n_timepoints)
        or (n_samples, n_timepoints) for univariate.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray, optional
        Test labels for accuracy computation.
    variant : str
        ROCKET variant:
        - 'rocket': Original ROCKET
        - 'minirocket': Faster, deterministic variant
        - 'multirocket': Multivariate extension
    n_kernels : int
        Number of random kernels (default: 10000).

    Returns
    -------
    ClassificationResult
        Predictions and accuracy metrics.

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.random.randn(100, 1, 200)
    >>> y_train = np.array([0]*50 + [1]*50)
    >>> X_test = np.random.randn(20, 1, 200)
    >>> result = rocket_classify(X_train, y_train, X_test, variant='minirocket')
    >>> print(f"Predictions: {result.predictions}")
    """
    try:
        from aeon.classification.convolution_based import (
            RocketClassifier,
            MiniRocketClassifier,
            MultiRocketClassifier,
        )
    except ImportError:
        return _fallback_rocket_classify(X_train, y_train, X_test, y_test)

    # Ensure 3D format
    X_train = ensure_3d(X_train)
    X_test = ensure_3d(X_test)
    y_train = np.asarray(y_train)

    # Select classifier
    clf_map = {
        "rocket": RocketClassifier,
        "minirocket": MiniRocketClassifier,
        "multirocket": MultiRocketClassifier,
    }

    clf_class = clf_map.get(variant, RocketClassifier)

    # Different parameter names for different variants
    try:
        clf = clf_class(num_kernels=n_kernels)
    except TypeError:
        # Some variants use different parameter names
        clf = clf_class()

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Get probabilities
    probabilities = None
    if hasattr(clf, 'predict_proba'):
        try:
            probabilities = clf.predict_proba(X_test)
        except Exception:
            pass

    # Compute accuracy
    accuracy = None
    if y_test is not None:
        y_test = np.asarray(y_test)
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method=f"{variant}",
        predictions=predictions,
        probabilities=probabilities,
        accuracy=accuracy,
    )


def _fallback_rocket_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
) -> ClassificationResult:
    """Fallback implementation when aeon is not available."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler

    # Simple feature extraction: statistics per window
    def extract_features(X):
        X = ensure_3d(X)
        n_samples, n_channels, n_timepoints = X.shape

        # Simple features: mean, std, min, max per channel
        features = []
        for i in range(n_samples):
            sample_features = []
            for c in range(n_channels):
                series = X[i, c, :]
                sample_features.extend([
                    np.mean(series),
                    np.std(series),
                    np.min(series),
                    np.max(series),
                    np.median(series),
                ])
            features.append(sample_features)

        return np.array(features)

    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    clf = RidgeClassifierCV()
    clf.fit(X_train_scaled, y_train)

    predictions = clf.predict(X_test_scaled)

    accuracy = None
    if y_test is not None:
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method="rocket_fallback",
        predictions=predictions,
        probabilities=None,
        accuracy=accuracy,
    )


def transform_rocket(
    X: np.ndarray,
    n_kernels: int = 10000,
    fitted_transform=None,
) -> tuple:
    """Apply ROCKET transform to extract features.

    Parameters
    ----------
    X : np.ndarray
        Input data. Shape: (n_samples, n_channels, n_timepoints)
    n_kernels : int
        Number of random kernels.
    fitted_transform : object, optional
        Pre-fitted transform object for consistent transformation.

    Returns
    -------
    X_transformed : np.ndarray
        Transformed features. Shape: (n_samples, 2 * n_kernels)
    transform : object
        Fitted transform object (for reuse on test data)
    """
    try:
        from aeon.transformations.collection.convolution_based import Rocket
    except ImportError:
        raise ImportError("aeon is required for ROCKET transform")

    X = ensure_3d(X)

    if fitted_transform is None:
        transform = Rocket(num_kernels=n_kernels)
        X_transformed = transform.fit_transform(X)
    else:
        transform = fitted_transform
        X_transformed = transform.transform(X)

    return X_transformed, transform
