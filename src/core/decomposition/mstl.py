"""MSTL (Multiple Seasonal-Trend decomposition using LOESS) for time series.

MSTL extends STL to handle multiple seasonal periods.
"""

from typing import Optional, List
import numpy as np
import pandas as pd

from ..base import DecompositionResult


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
        List of seasonal periods. If None, uses single auto-detected period.
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
    try:
        from statsforecast.models import MSTL as StatsMSTL
    except ImportError:
        # Fallback to regular STL
        from .stl import stl_decompose
        return stl_decompose(series, period=periods[0] if periods else None)

    series = np.asarray(series, dtype=np.float64).flatten()

    # Auto-detect periods if not provided
    if periods is None:
        periods = [_detect_periods(series)[0]]

    # Ensure periods are valid
    periods = [max(2, p) for p in periods]

    # Create DataFrame for statsforecast
    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    # Use statsmodels MSTL if available
    try:
        from statsmodels.tsa.seasonal import MSTL

        ts = pd.Series(series)
        mstl = MSTL(ts, periods=periods, windows=windows)
        result = mstl.fit()

        # Combine all seasonal components
        seasonal = result.seasonal.sum(axis=1).values if hasattr(result.seasonal, 'sum') else result.seasonal.values

        return DecompositionResult(
            method="mstl",
            trend=result.trend.values,
            seasonal=seasonal,
            residual=result.resid.values,
            period=periods[0],  # Primary period
        )

    except (ImportError, AttributeError):
        # Fallback to regular STL with first period
        from .stl import stl_decompose
        return stl_decompose(series, period=periods[0])


def _detect_periods(series: np.ndarray, max_periods: int = 3) -> List[int]:
    """Auto-detect multiple dominant periods using FFT."""
    # Remove mean
    sig = series - np.mean(series)

    # FFT
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig))
    power = np.abs(fft) ** 2

    # Ignore DC component
    power[0] = 0

    # Find top peaks
    top_indices = np.argsort(power)[-max_periods:][::-1]

    periods = []
    for idx in top_indices:
        freq = freqs[idx]
        if freq > 0:
            period = int(1 / freq)
            if 2 <= period <= len(series) // 2:
                periods.append(period)

    if not periods:
        periods = [max(2, len(series) // 10)]

    return periods
