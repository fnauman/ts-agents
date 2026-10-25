import pytest
import numpy as np
import warnings
import json

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


def test_evaluate_windowed_classifier_supports_multiple_group_splits(monkeypatch):
    series, labels = _make_series()

    observed = {}

    def fake_iter_valid_group_splits(X, y, groups, *, n_splits, **kwargs):
        observed["n_splits"] = n_splits
        return [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([2, 3, 4, 5]), np.array([0, 1])),
        ]

    def fake_knn(X_train, y_train, X_test, y_test, **kwargs):
        return ClassificationResult(
            method="knn_stub",
            predictions=np.asarray(y_test),
            accuracy=1.0,
        )

    monkeypatch.setattr(window_selection, "_iter_valid_group_splits", fake_iter_valid_group_splits)
    monkeypatch.setattr(window_selection, "knn_classify", fake_knn)

    result = window_selection.evaluate_windowed_classifier(
        series,
        labels,
        window_size=8,
        classifier="knn",
        metric="accuracy",
        balance="none",
        n_splits=2,
        test_size=0.34,
    )

    assert observed["n_splits"] == 2
    assert result.n_splits == 2
    assert result.split_scores == [1.0, 1.0]
    assert result.classification["method"] == "knn_stub"


def test_select_window_size_from_csv_accepts_json_records(tmp_path):
    series, labels = _make_series()
    records = [
        {"value": float(value), "label": label}
        for value, label in zip(series.tolist(), labels.tolist())
    ]
    json_path = tmp_path / "stream.json"
    json_path.write_text(json.dumps(records))

    result = window_selection.select_window_size_from_csv(
        str(json_path),
        value_columns="value",
        label_column="label",
        classifier="knn",
        metric="accuracy",
        balance="none",
        n_splits=1,
        test_size=0.34,
    )

    assert result.best_window_size in result.scores_by_window


def test_majority_splits_never_share_samples_between_train_and_test():
    """Purging removes training windows that overlap any test window."""
    series, labels = _make_series()
    segments = window_selection._compute_segments(labels)
    X, y, groups, spans = window_selection._extract_windows_from_segments(
        series.reshape(-1, 1), segments, window_size=20, stride=1, labeling="majority", labels=labels
    )
    splits = window_selection._iter_valid_group_splits(
        X, y, groups, n_splits=3, test_size=0.34, random_state=0, min_classes_per_split=1, spans=spans
    )

    assert splits
    for train_idx, test_idx in splits:
        test_samples = {t for start, end in spans[test_idx] for t in range(start, end)}
        train_samples = {t for start, end in spans[train_idx] for t in range(start, end)}
        assert not test_samples & train_samples


def test_windows_never_span_groups_and_holdout_keeps_whole_groups():
    """Subject groups bound windows, and the holdout takes whole subjects."""
    rng = np.random.default_rng(0)
    labels = np.array((["a"] * 40 + ["b"] * 40) * 3)
    groups = np.repeat(["s1", "s2", "s3"], 80)
    series = rng.normal(size=(240, 1)) + (labels == "b")[:, None] * 2.0

    group_ids = window_selection._as_group_ids(groups, len(labels))
    segments = window_selection._compute_segments(labels, group_ids)
    _, _, _, spans = window_selection._extract_windows_from_segments(
        series, segments, window_size=16, stride=4, labeling="majority", labels=labels, group_ids=group_ids
    )
    assert all(len(set(groups[start:end])) == 1 for start, end in spans)

    # Returning to subject A must not make an A-B-A window valid.
    repeated_groups = np.array(["a"] * 8 + ["b"] * 4 + ["a"] * 8)
    repeated_ids = window_selection._as_group_ids(repeated_groups, 20)
    repeated_labels = np.array(["x"] * 20)
    _, _, _, repeated_spans = window_selection._extract_windows_from_segments(
        np.arange(20)[:, None], window_selection._compute_segments(repeated_labels, repeated_ids),
        window_size=16, stride=1, labeling="majority", labels=repeated_labels, group_ids=repeated_ids,
    )
    assert len(repeated_spans) == 0

    mask = window_selection.split_holdout_units(labels, groups=groups, test_size=0.3, seed=1)
    held = set(groups[mask])
    assert len(held) == 1
    assert all(mask[groups == group].all() for group in held)

    # The requested fraction can be reached before the second required class.
    for fraction in (0.1, 0.25):
        stream_labels = np.repeat(["a", "b", "a", "b"], 40)
        mask = window_selection.split_holdout_units(stream_labels, test_size=fraction)
        assert set(stream_labels[mask]) == {"a", "b"}
        assert set(stream_labels[~mask]) == {"a", "b"}


def test_select_window_size_prefers_windows_keeping_every_class(monkeypatch):
    """A window that scores higher only by dropping a class is not chosen."""
    rng = np.random.default_rng(0)
    # Class "c" segments are 20 samples long, so 30-sample windows drop it.
    labels = np.array((["a"] * 60 + ["b"] * 60 + ["c"] * 20) * 3)
    series = rng.normal(size=len(labels))

    def fake_score(metric, y_true, y_pred):
        # The hard class "c" lowers the score whenever it is evaluated.
        return 0.6 if "c" in set(y_true) else 1.0

    monkeypatch.setattr(window_selection, "_score_metric", fake_score)
    monkeypatch.setattr(
        window_selection,
        "_run_window_classifier",
        lambda **kwargs: (kwargs["y_test"], ClassificationResult(method="fake", predictions=kwargs["y_test"])),
    )

    result = select_window_size(
        series, labels, window_sizes=[8, 30], classifier="knn", balance="none", n_splits=3, test_size=0.5,
    )

    assert result.details["window_diagnostics"][30]["dropped_classes"] == ["c"]
    assert result.scores_by_window[30] > result.scores_by_window[8]
    assert result.best_window_size == 8
    assert result.details["selection_rule"] == "best_score_with_all_classes"


@pytest.mark.parametrize("balance", ["segment_cap", "undersample"])
def test_final_holdout_balances_training_only(monkeypatch, balance):
    labels = np.repeat(["a", "b", "a", "b"], [80, 40, 80, 40])
    mask = np.arange(len(labels)) >= 120
    calls = []

    def classify(**kwargs):
        calls.append(kwargs)
        return kwargs["y_test"], ClassificationResult(method="fake", predictions=kwargs["y_test"])

    monkeypatch.setattr(window_selection, "_run_window_classifier", classify)
    result = window_selection.evaluate_windowed_classifier_holdout(
        np.arange(len(labels)), labels, holdout_mask=mask, window_size=8, stride=4,
        balance=balance, max_windows_per_segment=3,
    )
    train = calls[0]["y_train"]
    assert np.sum(train == "a") == np.sum(train == "b") == (3 if balance == "segment_cap" else 9)
    assert len(calls[0]["y_test"]) == 28
    assert result.details["n_train_windows"] == len(train)
    assert result.details["n_holdout_windows"] == 28
    assert result.n_windows == len(train) + 28
