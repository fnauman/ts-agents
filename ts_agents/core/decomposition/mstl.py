"""MSTL (Multiple Seasonal-Trend decomposition using LOESS) for time series.

MSTL extends STL to handle multiple seasonal periods.
"""

from typing import Optional, List
import numpy as np
import pandas as pd

from ..base import DecompositionResult
from ..validation import as_univariate_series


def mstl_decompose(
    series: np.ndarray,
    periods: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> DecompositionResult:
    """Decompose time series using MSTL (Multi-Seasonal STL).

    MSTL can handle multiple seasonal periods simultaneously.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    periods : list of int, optional
        List of seasonal periods in samples. If None, the significant periods
        found by :func:`ts_agents.core.spectral.detect_periodicity` are used
        (``ValueError`` when none is significant).
    windows : list of int, optional
        List of seasonal smoothing windows (same length as periods).

    Returns
    -------
    DecompositionResult
        Decomposition with trend and combined seasonal component

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 10000)
    >>> # Daily and weekly seasonality
    >>> daily = np.sin(2 * np.pi * t)
    >>> weekly = 0.5 * np.sin(2 * np.pi * t / 7)
    >>> x = 0.1 * t + daily + weekly + 0.1 * np.random.randn(10000)
    >>> result = mstl_decompose(x, periods=[100, 700])
    """
    series = as_univariate_series(series, min_length=4)

    period_source = "user"
    if periods is None:
        periods = _detect_periods(series)
        period_source = "auto"
    periods = [int(p) for p in periods]
    if not periods or any(p < 2 or p > len(series) // 2 for p in periods):
        raise ValueError(
            f"periods must be between 2 and half the series length ({len(series) // 2}); got {periods}."
        )

    # Use statsmodels MSTL if available
    try:
        from statsmodels.tsa.seasonal import MSTL

        ts = pd.Series(series)
        mstl = MSTL(ts, periods=periods, windows=windows)
        result = mstl.fit()

        # Combine all seasonal components
        seasonal = result.seasonal.sum(axis=1).values if result.seasonal.ndim == 2 else result.seasonal.values

        return DecompositionResult(
            method="mstl",
            trend=result.trend.values,
            seasonal=seasonal,
            residual=result.resid.values,
            period=periods[0],  # Primary period
            period_source=period_source,
        )

    except ImportError as exc:
        raise ImportError(
            'MSTL decomposition requires optional dependencies. Install with: pip install "ts-agents[decomposition]"'
        ) from exc


def _detect_periods(series: np.ndarray, max_periods: int = 3) -> List[int]:
    """Significant integer periods (deduplicated) from the shared estimator."""
    from ..spectral.psd import detect_periodicity

    result = detect_periodicity(series, top_n=max_periods)
    periods: List[int] = []
    for value in result.top_periods:
        period = int(round(value))
        if 2 <= period <= len(series) // 2 and period not in periods:
            periods.append(period)
    if not periods:
        raise ValueError("No significant seasonal period detected; pass periods explicitly.")
    return sorted(periods)
