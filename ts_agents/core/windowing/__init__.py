"""Windowing utilities for time series."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "WindowSizeSelectionResult": ("selection", "WindowSizeSelectionResult"),
    "WindowedClassificationEvaluation": ("selection", "WindowedClassificationEvaluation"),
    "evaluate_windowed_classifier": ("selection", "evaluate_windowed_classifier"),
    "evaluate_windowed_classifier_from_csv": ("selection", "evaluate_windowed_classifier_from_csv"),
    "select_window_size": ("selection", "select_window_size"),
    "select_window_size_from_csv": ("selection", "select_window_size_from_csv"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    "WindowSizeSelectionResult",
    "WindowedClassificationEvaluation",
    "select_window_size",
    "select_window_size_from_csv",
    "evaluate_windowed_classifier",
    "evaluate_windowed_classifier_from_csv",
]
