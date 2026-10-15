"""Tests for the classification module."""

import importlib.util
import sys
import types

import numpy as np
import pytest

requires_aeon = pytest.mark.skipif(
    importlib.util.find_spec("aeon") is None,
    reason="aeon not installed",
)


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

    @requires_aeon
    def test_knn_classify_euclidean(self):
        """Test KNN with Euclidean distance."""
        from ts_agents.core.classification import knn_classify

        X, y = create_synthetic_data(n_samples=40)
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        result = knn_classify(X_train, y_train, X_test, y_test, distance='euclidean')

        assert result.method == "knn_euclidean"
        assert len(result.predictions) == 10
        assert result.accuracy is not None
        assert result.accuracy >= 0.5  # Should do better than random

    @requires_aeon
    def test_knn_classify_dtw(self):
        """Test KNN with DTW distance."""
        from ts_agents.core.classification import knn_classify

        X, y = create_synthetic_data(n_samples=30)
        X_train, y_train = X[:20], y[:20]
        X_test, y_test = X[20:], y[20:]

        result = knn_classify(X_train, y_train, X_test, y_test, distance='dtw')

        assert result.method == "knn_dtw"
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
        pytest.importorskip("sklearn")

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

    @requires_aeon
    def test_rocket_classify(self):
        """Test ROCKET classification."""
        from ts_agents.core.classification import rocket_classify

        X, y = create_synthetic_data(n_samples=40)
        X_train, y_train = X[:30], y[:30]
        X_test, y_test = X[30:], y[30:]

        result = rocket_classify(X_train, y_train, X_test, y_test, variant='rocket')

        assert result.method == "rocket"
        assert len(result.predictions) == 10
        assert result.accuracy is not None

    def test_rocket_classify_falls_back_when_aeon_runtime_breaks(self, monkeypatch):
        from ts_agents.core.classification import rocket_classify
        pytest.importorskip("sklearn")

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

    def test_compare_classifiers(self, monkeypatch):
        """Comparison preserves fallback identity and cannot fabricate successful predictions."""
        from ts_agents.core.base import ClassificationResult
        from ts_agents.core.classification import compare_classifiers, hivecote_classify
        from ts_agents.core.classification import convolution, distance_based

        X = np.arange(8.0).reshape(4, 1, 2)
        y = np.array([0, 1, 0, 1])
        X_train, y_train, X_test, y_test = X[:2], y[:2], X[2:], y[2:]
        monkeypatch.setitem(sys.modules, "aeon.classification.hybrid", None)
        monkeypatch.setattr(
            distance_based, "knn_classify", lambda *args, **kwargs: ClassificationResult(
                method="knn_euclidean_fallback", predictions=y_test.copy(), accuracy=1.0,
                warnings=["KNN backend fallback activated"],
            ),
        )
        monkeypatch.setattr(
            convolution, "rocket_classify", lambda *args, **kwargs: ClassificationResult(
                method="rocket_fallback", predictions=y_test.copy(), accuracy=1.0,
                warnings=["ROCKET backend fallback activated"],
            ),
        )

        results = compare_classifiers(
            X_train, y_train, X_test, y_test,
            classifiers=['knn_euclidean', 'rocket', 'hivecote2', 'unknown']
        )

        assert results['rankings'] == ['knn_euclidean', 'rocket', 'hivecote2']
        assert results['status'] == 'degraded'
        assert results['warnings']
        assert "Unknown classifier" in results['results']['unknown']['error']
        for requested, actual in (("knn_euclidean", "knn_euclidean_fallback"), ("rocket", "rocket_fallback"), ("hivecote2", "ensemble_fallback")):
            entry = results['results'][requested]
            assert entry['requested_method'] == requested
            assert entry['actual_method'] == actual
            assert entry['warnings']
        assert any("knn_euclidean_fallback, rocket_fallback" in warning for warning in results['results']['hivecote2']['warnings'])

        def fail(*args, **kwargs):
            raise RuntimeError("backend failed")

        monkeypatch.setattr(distance_based, "knn_classify", fail)
        monkeypatch.setattr(convolution, "rocket_classify", fail)
        with pytest.raises(RuntimeError, match="All HIVE-COTE fallback classifiers failed"):
            hivecote_classify(X_train, y_train, X_test, np.zeros(2))
        with pytest.raises(RuntimeError, match="All classifiers failed.*backend failed"):
            compare_classifiers(X_train, y_train, X_test, np.zeros(2), classifiers=["hivecote2"])
