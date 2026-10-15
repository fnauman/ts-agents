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
    additive or multiplicative models. Returned components are always additive
    contributions in the original data units: ``series = trend + seasonal + residual``.
    ``trend`` is the one-step nonseasonal prediction; a multiplicative seasonal
    factor is converted to its contribution in data units. ``residual`` remains
    the observed value minus the model's one-step fitted prediction.

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
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Holt-Winters decomposition requires optional dependencies. Install with: pip install "ts-agents[decomposition]"'
        ) from exc

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

        # fittedvalues uses seasonal states from one period earlier, not the
        # current states (which already incorporate the current observation).
        fitted = np.asarray(fit.fittedvalues)
        if seasonal:
            lagged_season = np.r_[fit.params["initial_seasons"], fit.season[:-period]]
            trend_component = fitted - lagged_season if seasonal == "add" else fitted / lagged_season
            seasonal_component = fitted - trend_component
        else:
            trend_component = fitted
            seasonal_component = np.zeros_like(series)

        # Compute residual
        residual = series - fitted

        return DecompositionResult(
            method="holt_winters",
            trend=np.asarray(trend_component),
            seasonal=np.asarray(seasonal_component),
            residual=np.asarray(residual),
            period=period,
        )

    except Exception:
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
