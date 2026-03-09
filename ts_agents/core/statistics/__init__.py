"""Statistical analysis for time series.

This module provides functions for:
- Descriptive statistics (mean, std, RMS, etc.)
- Rolling statistics
- Correlation analysis
- Autocorrelation
"""

from .descriptive import (
    describe_series,
    compute_rolling_stats,
    compute_correlation,
    compute_autocorrelation,
    compare_series_stats,
)

__all__ = [
    "describe_series",
    "compute_rolling_stats",
    "compute_correlation",
    "compute_autocorrelation",
    "compare_series_stats",
]
