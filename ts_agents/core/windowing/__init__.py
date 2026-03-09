"""Windowing utilities for time series.

This package focuses on transforming long, labeled time series into
fixed-length windows suitable for classification, and selecting an
appropriate window size.

The key public entrypoints are:

- :func:`select_window_size` - segment-aware window-size search
- :func:`select_window_size_from_csv` - convenience wrapper for CSV inputs
"""

from .selection import (
    WindowSizeSelectionResult,
    WindowedClassificationEvaluation,
    evaluate_windowed_classifier,
    evaluate_windowed_classifier_from_csv,
    select_window_size,
    select_window_size_from_csv,
)

__all__ = [
    "WindowSizeSelectionResult",
    "WindowedClassificationEvaluation",
    "select_window_size",
    "select_window_size_from_csv",
    "evaluate_windowed_classifier",
    "evaluate_windowed_classifier_from_csv",
]
