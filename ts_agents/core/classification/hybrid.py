"""Hybrid time series classification methods.

This module provides functions for ensemble and hybrid classification
methods like HIVE-COTE 2 (HC2).
"""

from typing import Optional, List, Dict, Any
import numpy as np

from ..base import ClassificationResult
from .utils import ensure_3d


def hivecote_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    time_limit_in_minutes: float = 1,
) -> ClassificationResult:
    """Time series classification using HIVE-COTE 2.

    HIVE-COTE 2 (Hierarchical Vote Collective of Transformation-based Ensembles)
    is a state-of-the-art ensemble classifier that combines multiple representations.

    WARNING: This is computationally expensive. Use time_limit_in_minutes to constrain.

    Parameters
    ----------
    X_train : np.ndarray
        Training data. Shape: (n_samples, n_channels, n_timepoints)
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray, optional
        Test labels for accuracy computation.
    time_limit_in_minutes : float
        Training budget in minutes (default: 1). Explicit 0 means no limit.

    Returns
    -------
    ClassificationResult
        Predictions and accuracy metrics.

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.random.randn(50, 1, 100)
    >>> y_train = np.array([0]*25 + [1]*25)
    >>> X_test = np.random.randn(10, 1, 100)
    >>> result = hivecote_classify(X_train, y_train, X_test, time_limit_in_minutes=1)
    """
    if not np.isfinite(time_limit_in_minutes) or time_limit_in_minutes < 0:
        raise ValueError("time_limit_in_minutes must be finite and nonnegative.")
    try:
        from aeon.classification.hybrid import HIVECOTEV2
    except ImportError as exc:
        # Fall back to simpler ensemble
        return _ensemble_fallback(
            X_train, y_train, X_test, y_test,
            warning_message=f"HIVE-COTE backend unavailable: {type(exc).__name__}: {exc}",
        )

    # Ensure 3D format
    X_train = ensure_3d(X_train)
    X_test = ensure_3d(X_test)
    y_train = np.asarray(y_train)

    clf = HIVECOTEV2(time_limit_in_minutes=time_limit_in_minutes)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    probabilities = None
    if hasattr(clf, 'predict_proba'):
        try:
            probabilities = clf.predict_proba(X_test)
        except Exception:
            pass

    accuracy = None
    if y_test is not None:
        y_test = np.asarray(y_test)
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method="hivecote2",
        predictions=predictions,
        probabilities=probabilities,
        accuracy=accuracy,
    )


def _ensemble_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    warning_message: Optional[str] = None,
) -> ClassificationResult:
    """Simple ensemble fallback when aeon is not available."""
    from .distance_based import knn_classify
    from .convolution import rocket_classify

    # Get predictions from multiple classifiers
    results = []
    warnings = [warning_message] if warning_message else []
    actual_methods = []

    # KNN with euclidean
    try:
        knn_result = knn_classify(X_train, y_train, X_test, distance='euclidean')
        results.append(knn_result.predictions)
        actual_methods.append(knn_result.method)
        warnings.extend(knn_result.warnings)
    except Exception as exc:
        warnings.append(f"KNN ensemble member failed: {type(exc).__name__}: {exc}")

    # ROCKET (fallback)
    try:
        rocket_result = rocket_classify(X_train, y_train, X_test, variant='rocket')
        results.append(rocket_result.predictions)
        actual_methods.append(rocket_result.method)
        warnings.extend(rocket_result.warnings)
    except Exception as exc:
        warnings.append(f"ROCKET ensemble member failed: {type(exc).__name__}: {exc}")

    if not results:
        raise RuntimeError("All HIVE-COTE fallback classifiers failed: " + "; ".join(warnings))
    warnings.append("HIVE-COTE replaced by ensemble of: " + ", ".join(actual_methods))
    # Majority voting
    results_array = np.array(results)
    predictions = []
    for i in range(results_array.shape[1]):
        votes = results_array[:, i]
        unique, counts = np.unique(votes, return_counts=True)
        predictions.append(unique[np.argmax(counts)])
    predictions = np.array(predictions)

    accuracy = None
    if y_test is not None:
        y_test = np.asarray(y_test)
        accuracy = float(np.mean(predictions == y_test))

    return ClassificationResult(
        method="ensemble_fallback",
        predictions=predictions,
        probabilities=None,
        accuracy=accuracy,
        warnings=warnings,
    )


def compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare multiple classifiers on the same data.

    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    classifiers : list of str, optional
        Classifiers to compare. Default: ['knn_dtw', 'rocket', 'minirocket']

    Returns
    -------
    dict
        Scores keyed by requested classifier, preserving actual method and fallback warnings.

    Examples
    --------
    >>> comparison = compare_classifiers(X_train, y_train, X_test, y_test)
    >>> for clf, metrics in comparison['results'].items():
    ...     print(f"{clf}: {metrics['accuracy']:.3f}")
    """
    from .distance_based import knn_classify
    from .convolution import rocket_classify

    if classifiers is None:
        classifiers = ['knn_euclidean', 'knn_dtw', 'rocket', 'minirocket']

    results = {}

    for clf_name in classifiers:
        try:
            if clf_name == 'knn_euclidean':
                result = knn_classify(X_train, y_train, X_test, y_test, distance='euclidean')
            elif clf_name == 'knn_dtw':
                result = knn_classify(X_train, y_train, X_test, y_test, distance='dtw')
            elif clf_name == 'rocket':
                result = rocket_classify(X_train, y_train, X_test, y_test, variant='rocket')
            elif clf_name == 'minirocket':
                result = rocket_classify(X_train, y_train, X_test, y_test, variant='minirocket')
            elif clf_name == 'multirocket':
                result = rocket_classify(X_train, y_train, X_test, y_test, variant='multirocket')
            elif clf_name == 'hivecote2':
                result = hivecote_classify(X_train, y_train, X_test, y_test, time_limit_in_minutes=1)
            else:
                raise ValueError(f"Unknown classifier: {clf_name}")

            if result.predictions.shape != np.asarray(y_test).shape:
                raise ValueError("Classifier must return one prediction per test label.")
            if result.accuracy is None or not np.isfinite(result.accuracy):
                raise ValueError("Classifier accuracy must be finite.")

            results[clf_name] = {
                'requested_method': clf_name,
                'actual_method': result.method,
                'warnings': result.warnings,
                'accuracy': result.accuracy,
                'predictions': result.predictions,
            }

        except Exception as e:
            results[clf_name] = {
                'requested_method': clf_name,
                'actual_method': None,
                'warnings': [],
                'accuracy': None,
                'error': str(e),
            }

    # Rank by accuracy
    valid_results = {k: v for k, v in results.items() if v.get('accuracy') is not None}
    if not valid_results:
        failures = "; ".join(f"{name}: {entry['error']}" for name, entry in results.items())
        raise RuntimeError("All classifiers failed: " + (failures or "no classifiers requested"))
    rankings = sorted(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'], reverse=True)
    warnings = []
    for requested, entry in results.items():
        warnings.extend(f"{requested}: {warning}" for warning in entry['warnings'])
        if entry.get('error'):
            warnings.append(f"{requested} failed: {entry['error']}")
        elif entry['actual_method'] != requested and not entry['warnings']:
            warnings.append(f"{requested} ran as {entry['actual_method']}")

    return {
        'results': results,
        'rankings': rankings,
        'best': rankings[0],
        'status': 'degraded' if warnings else 'ok',
        'warnings': warnings,
    }
