"""STL (Seasonal-Trend decomposition using LOESS) for time series.

STL is a robust method for decomposing a time series into three components:
trend, seasonal, and residual.
"""

from typing import Optional
import numpy as np
import pandas as pd

from ..base import DecompositionResult
from ..validation import as_univariate_series


def _get_stl() -> type:
    try:
        from statsmodels.tsa.seasonal import STL
    except ModuleNotFoundError as exc:
        raise ImportError(
            'STL decomposition requires optional dependencies. Install with: pip install "ts-agents[decomposition]"'
        ) from exc
    return STL


def stl_decompose(
    series: np.ndarray,
    period: Optional[int] = None,
    robust: bool = True,
    seasonal: int = 7,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
) -> DecompositionResult:
    """Decompose time series using STL (Seasonal-Trend LOESS).

    STL is robust to outliers when using robust=True.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    period : int, optional
        Seasonal period in samples. If None, it is auto-detected with
        :func:`ts_agents.core.spectral.detect_periodicity`; a ``ValueError``
        is raised when no significant period exists.
    robust : bool
        Use robust fitting (resistant to outliers). Default True.
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    trend : int, optional
        Length of the trend smoother. If None, auto-computed.
    low_pass : int, optional
        Length of the low-pass filter.

    Returns
    -------
    DecompositionResult
        Decomposition with trend, seasonal, residual components

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> trend = 0.5 * t
    >>> seasonal = np.sin(2 * np.pi * t)
    >>> noise = 0.1 * np.random.randn(1000)
    >>> x = trend + seasonal + noise
    >>> result = stl_decompose(x, period=100)
    >>> print(f"Residual variance: {result.residual_variance:.4f}")
    """
    series = as_univariate_series(series, min_length=4)
    period, period_source = _resolve_period(series, period)

    # Convert to pandas Series (STL requires this)
    ts = pd.Series(series)

    # Run STL
    stl = _get_stl()(
        ts,
        period=period,
        robust=robust,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
    )
    result = stl.fit()

    return DecompositionResult(
        method="stl",
        trend=result.trend.values,
        seasonal=result.seasonal.values,
        residual=result.resid.values,
        period=period,
        period_source=period_source,
    )


def _resolve_period(series: np.ndarray, period: Optional[int]) -> tuple[int, str]:
    """Validate a user period or auto-detect one with the shared estimator."""
    if period is None:
        from ..spectral.psd import detect_seasonal_period

        detected, _ = detect_seasonal_period(series)
        return detected, "auto"
    period = int(period)
    if period < 2 or period > len(series) // 2:
        raise ValueError(
            f"period must be between 2 and half the series length ({len(series) // 2}); got {period}."
        )
    return period, "user"
