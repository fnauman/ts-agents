"""STL (Seasonal-Trend decomposition using LOESS) for time series.

STL is a robust method for decomposing a time series into three components:
trend, seasonal, and residual.
"""

from typing import Optional
import numpy as np
import pandas as pd

from ..base import DecompositionResult


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
        Seasonal period. If None, auto-detected via FFT.
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
    series = np.asarray(series, dtype=np.float64).flatten()

    # Auto-detect period if not provided
    if period is None:
        period = _detect_period(series)

    # Ensure period is at least 2
    period = max(2, period)

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
    )


def _detect_period(series: np.ndarray) -> int:
    """Auto-detect dominant period using FFT."""
    # Remove mean
    sig = series - np.mean(series)

    # FFT
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig))
    power = np.abs(fft) ** 2

    # Ignore DC component
    power[0] = 0

    # Find dominant frequency
    peak_idx = np.argmax(power)
    dominant_freq = freqs[peak_idx]

    if dominant_freq > 0:
        period = int(1 / dominant_freq)
    else:
        period = len(series) // 10  # fallback

    # Clamp to reasonable range
    period = max(2, min(period, len(series) // 2))

    return period
