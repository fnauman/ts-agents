import numpy as np
import warnings

from ts_agents.core.base import ClassificationResult
from ts_agents.core.windowing import select_window_size, evaluate_windowed_classifier
import ts_agents.core.windowing.selection as window_selection


def _make_series():
    rng = np.random.default_rng(0)
    seg1 = rng.normal(0.0, 0.5, 50)
    seg2 = rng.normal(2.0, 0.5, 50)
    seg3 = rng.normal(-1.0, 0.5, 50)
    series = np.concatenate([seg1, seg2, seg3])
    labels = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 50)
    return series, labels


def test_select_window_size_knn_majority_smoke():
    series, labels = _make_series()
    result = select_window_size(
        series,
        labels,
        window_sizes=[8, 16],
        classifier="knn",
        metric="accuracy",
        labeling="majority",
        balance="none",
        n_splits=1,
        test_size=0.34,
    )
    assert result.method == "window_size_selection"
    assert result.best_window_size in {8, 16}
    assert set(result.scores_by_window.keys()) == {8, 16}


def test_evaluate_windowed_classifier_knn_smoke():
    series, labels = _make_series()
    result = evaluate_windowed_classifier(
        series,
        labels,
        window_size=8,
        classifier="knn",
        metric="accuracy",
        balance="none",
        test_size=0.34,
    )
    assert result.method == "windowed_classification"
    assert 0.0 <= result.score <= 1.0
    assert result.n_windows > 0


def test_evaluate_windowed_classifier_multivariate_smoke():
    series, labels = _make_series()
    series_multi = np.stack([series, series * 0.5], axis=1)
    result = evaluate_windowed_classifier(
        series_multi,
        labels,
        window_size=8,
        classifier="knn",
        metric="accuracy",
        balance="none",
        test_size=0.34,
    )
    assert result.n_windows > 0


def _stub_knn_factory(captured):
    def _stub_knn(X_train, y_train, X_test, y_test, **kwargs):
        captured.append((X_train.shape, X_test.shape))
        return ClassificationResult(
            method="knn_stub",
            predictions=np.asarray(y_test),
            accuracy=1.0,
        )

    return _stub_knn


def test_select_window_size_preserves_channels_multivariate(monkeypatch):
    series, labels = _make_series()
    series_multi = np.stack([series, series * 0.5], axis=1)

    captured = []
    monkeypatch.setattr(window_selection, "knn_classify", _stub_knn_factory(captured))

    window_selection.select_window_size(
        series_multi,
        labels,
        window_sizes=[8],
        classifier="knn",
        metric="accuracy",
        labeling="majority",
        balance="none",
        n_splits=1,
        test_size=0.34,
    )

    assert captured
    train_shape, test_shape = captured[0]
    assert train_shape[1:] == (series_multi.shape[1], 8)
    assert test_shape[1:] == (series_multi.shape[1], 8)


def test_evaluate_windowed_classifier_preserves_channels_multivariate(monkeypatch):
    series, labels = _make_series()
    series_multi = np.stack([series, series * 0.5], axis=1)

    captured = []
    monkeypatch.setattr(window_selection, "knn_classify", _stub_knn_factory(captured))

    window_selection.evaluate_windowed_classifier(
        series_multi,
        labels,
        window_size=8,
        classifier="knn",
        metric="accuracy",
        balance="none",
        test_size=0.34,
    )

    assert captured
    train_shape, test_shape = captured[0]
    assert train_shape[1:] == (series_multi.shape[1], 8)
    assert test_shape[1:] == (series_multi.shape[1], 8)


def test_balanced_accuracy_single_class_no_warning():
    y_true = np.array(["idle", "idle", "idle"])
    y_pred = np.array(["idle", "idle", "idle"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = window_selection._score_metric("balanced_accuracy", y_true, y_pred)

    assert score == 1.0
