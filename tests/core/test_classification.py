"""Tests for the classification module."""

import sys
import types

import numpy as np
import pytest


def create_synthetic_data(n_samples=50, n_timepoints=100, n_classes=2):
    """Create synthetic time series classification data."""
    X = []
    y = []

    for i in range(n_samples):
        label = i % n_classes

        if label == 0:
            # Sine wave
            series = np.sin(np.linspace(0, 4 * np.pi, n_timepoints))
        else:
            # Cosine wave
            series = np.cos(np.linspace(0, 4 * np.pi, n_timepoints))

        # Add noise
        series = series + 0.1 * np.random.randn(n_timepoints)

        X.append(series)
        y.append(label)

    X = np.array(X).reshape(n_samples, 1, n_timepoints)
    y = np.array(y)

    return X, y


class TestKNNClassification:
    """Tests for KNN classification."""

    def test_knn_classify_euclidean(self):
        """Test KNN with Euclidean distance."""
        from ts_agents.core.classification import knn_classify

        X, y = create_synthetic_data(n_samples=40)
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        result = knn_classify(X_train, y_train, X_test, y_test, distance='euclidean')

        assert len(result.predictions) == 10
        assert result.accuracy is not None
        assert result.accuracy >= 0.5  # Should do better than random

    def test_knn_classify_dtw(self):
        """Test KNN with DTW distance."""
        from ts_agents.core.classification import knn_classify

        X, y = create_synthetic_data(n_samples=30)
        X_train, y_train = X[:20], y[:20]
        X_test, y_test = X[20:], y[20:]

        result = knn_classify(X_train, y_train, X_test, y_test, distance='dtw')

        assert len(result.predictions) == 10
        assert result.accuracy >= 0.5

    def test_compute_dtw_distance(self):
        """Test DTW distance computation."""
        from ts_agents.core.classification import compute_dtw_distance

        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        y = np.sin(np.linspace(0.5, 2.5 * np.pi, 100))

        dist = compute_dtw_distance(x, y)

        assert dist >= 0
        # Same series should have distance 0
        assert compute_dtw_distance(x, x) < 0.01

    def test_knn_classify_falls_back_when_aeon_runtime_breaks(self, monkeypatch):
        from ts_agents.core.classification import knn_classify

        broken_module = types.ModuleType("aeon.classification.distance_based")

        class BrokenKNNClassifier:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, X, y):
                raise RuntimeError("aeon runtime is broken")

        broken_module.KNeighborsTimeSeriesClassifier = BrokenKNNClassifier

        monkeypatch.setitem(sys.modules, "aeon", types.ModuleType("aeon"))
        monkeypatch.setitem(sys.modules, "aeon.classification", types.ModuleType("aeon.classification"))
        monkeypatch.setitem(sys.modules, "aeon.classification.distance_based", broken_module)

        X, y = create_synthetic_data(n_samples=20)
        result = knn_classify(X[:15], y[:15], X[15:], y[15:], distance="dtw")

        assert result.method == "knn_euclidean_fallback"
        assert len(result.predictions) == 5
        assert any("fallback activated" in warning for warning in result.warnings)


class TestROCKETClassification:
    """Tests for ROCKET classification."""

    def test_rocket_classify(self):
        """Test ROCKET classification."""
        from ts_agents.core.classification import rocket_classify

        X, y = create_synthetic_data(n_samples=40)
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        result = rocket_classify(X_train, y_train, X_test, y_test, variant='rocket')

        assert len(result.predictions) == 10
        assert result.accuracy is not None

    def test_rocket_classify_falls_back_when_aeon_runtime_breaks(self, monkeypatch):
        from ts_agents.core.classification import rocket_classify

        broken_module = types.ModuleType("aeon.classification.convolution_based")

        class BrokenRocketClassifier:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, X, y):
                raise RuntimeError("aeon runtime is broken")

        broken_module.RocketClassifier = BrokenRocketClassifier
        broken_module.MiniRocketClassifier = BrokenRocketClassifier
        broken_module.MultiRocketClassifier = BrokenRocketClassifier

        monkeypatch.setitem(sys.modules, "aeon", types.ModuleType("aeon"))
        monkeypatch.setitem(sys.modules, "aeon.classification", types.ModuleType("aeon.classification"))
        monkeypatch.setitem(sys.modules, "aeon.classification.convolution_based", broken_module)

        X, y = create_synthetic_data(n_samples=20)
        result = rocket_classify(X[:15], y[:15], X[15:], y[15:], variant="rocket")

        assert result.method == "rocket_fallback"
        assert len(result.predictions) == 5
        assert any("fallback activated" in warning for warning in result.warnings)


class TestEnsure3D:
    """Tests for the shared ensure_3d utility."""

    def test_ensure_3d_from_1d(self):
        """1D input should become (1, 1, n_timepoints)."""
        from ts_agents.core.classification.utils import ensure_3d

        x = np.array([1, 2, 3, 4, 5])
        result = ensure_3d(x)

        assert result.shape == (1, 1, 5)
        np.testing.assert_array_equal(result[0, 0, :], x)

    def test_ensure_3d_from_2d(self):
        """2D input should become (n_samples, 1, n_timepoints)."""
        from ts_agents.core.classification.utils import ensure_3d

        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = ensure_3d(x)

        assert result.shape == (2, 1, 3)
        np.testing.assert_array_equal(result[:, 0, :], x)

    def test_ensure_3d_passthrough(self):
        """3D input should pass through unchanged."""
        from ts_agents.core.classification.utils import ensure_3d

        x = np.random.randn(5, 2, 10)
        result = ensure_3d(x)

        assert result.shape == (5, 2, 10)
        np.testing.assert_array_equal(result, x)


class TestCompareClassifiers:
    """Tests for classifier comparison."""

    def test_compare_classifiers(self):
        """Test classifier comparison."""
        from ts_agents.core.classification import compare_classifiers

        X, y = create_synthetic_data(n_samples=40)
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        results = compare_classifiers(
            X_train, y_train, X_test, y_test,
            classifiers=['knn_euclidean', 'rocket']
        )

        assert 'results' in results
        assert 'rankings' in results
        assert len(results['results']) >= 1
