"""Holt-Winters Exponential Smoothing for time series decomposition.

Holt-Winters (also known as triple exponential smoothing) can model
time series with trend and seasonality.
"""

from typing import Optional, Literal
import numpy as np

from ..base import DecompositionResult


def holt_winters_decompose(
    series: np.ndarray,
    period: Optional[int] = None,
    trend: Literal["add", "mul", None] = "add",
    seasonal: Literal["add", "mul", None] = "add",
    damped_trend: bool = False,
) -> DecompositionResult:
    """Decompose time series using Holt-Winters exponential smoothing.

    Holt-Winters can handle trend and seasonality, with options for
    additive or multiplicative components.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    period : int, optional
        Seasonal period. If None, auto-detected.
    trend : str or None
        Type of trend component: 'add', 'mul', or None
    seasonal : str or None
        Type of seasonal component: 'add', 'mul', or None
    damped_trend : bool
        Whether to damp the trend component

    Returns
    -------
    DecompositionResult
        Decomposition with trend, seasonal, and residual components

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)
    >>> result = holt_winters_decompose(x, period=100)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = np.asarray(series, dtype=np.float64).flatten()

    # Auto-detect period if not provided
    if period is None:
        period = _detect_period(series)

    # Ensure period is at least 2
    period = max(2, period)

    # Fit the model
    try:
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=period,
            damped_trend=damped_trend,
        )
        fit = model.fit(optimized=True)

        # Extract components
        if seasonal:
            seasonal_component = fit.season
        else:
            seasonal_component = np.zeros_like(series)

        if trend:
            trend_component = fit.level + fit.trend if fit.trend is not None else fit.level
        else:
            trend_component = fit.level if hasattr(fit, 'level') else np.full_like(series, np.mean(series))

        # Compute residual
        fitted = fit.fittedvalues
        residual = series - fitted

        return DecompositionResult(
            method="holt_winters",
            trend=np.asarray(trend_component),
            seasonal=np.asarray(seasonal_component),
            residual=np.asarray(residual),
            period=period,
        )

    except Exception as e:
        # Fallback: simple decomposition
        return _simple_decompose(series, period)


def _detect_period(series: np.ndarray) -> int:
    """Auto-detect dominant period using FFT."""
    sig = series - np.mean(series)
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig))
    power = np.abs(fft) ** 2
    power[0] = 0

    peak_idx = np.argmax(power)
    dominant_freq = freqs[peak_idx]

    if dominant_freq > 0:
        period = int(1 / dominant_freq)
    else:
        period = len(series) // 10

    return max(2, min(period, len(series) // 2))


def _simple_decompose(series: np.ndarray, period: int) -> DecompositionResult:
    """Simple fallback decomposition using moving average."""
    n = len(series)

    # Trend: moving average
    if period > 1:
        kernel = np.ones(period) / period
        trend = np.convolve(series, kernel, mode='same')
    else:
        trend = np.full_like(series, np.mean(series))

    # Detrend
    detrended = series - trend

    # Seasonal: average by position in period
    seasonal = np.zeros_like(series)
    if period > 1:
        for i in range(period):
            seasonal[i::period] = np.mean(detrended[i::period])

    # Residual
    residual = series - trend - seasonal

    return DecompositionResult(
        method="holt_winters_fallback",
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        period=period,
    )
