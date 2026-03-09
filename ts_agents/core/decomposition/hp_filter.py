"""Hodrick-Prescott (HP) Filter for trend extraction.

The HP filter separates a time series into trend and cyclical components.
"""

from typing import Optional
import numpy as np

from ..base import DecompositionResult


def hp_filter(
    series: np.ndarray,
    lamb: Optional[float] = None,
) -> DecompositionResult:
    """Apply Hodrick-Prescott filter to extract trend.

    The HP filter minimizes the sum of squared deviations from the trend
    subject to a smoothness penalty.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    lamb : float, optional
        Smoothing parameter (lambda). Higher values = smoother trend.
        Common values:
        - 1600 for quarterly data
        - 14400 for monthly data
        - 129600 for weekly data
        If None, auto-computed based on series length.

    Returns
    -------
    DecompositionResult
        Decomposition with trend and cyclical (seasonal/residual) components

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)
    >>> result = hp_filter(x, lamb=1600)
    >>> print(f"Trend smoothness: {result.trend_smoothness:.4f}")
    """
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
    except ImportError:
        # Manual implementation
        return _hp_filter_manual(series, lamb)

    series = np.asarray(series, dtype=np.float64).flatten()

    # Auto-compute lambda if not provided
    if lamb is None:
        # Rule of thumb based on series length
        n = len(series)
        if n < 100:
            lamb = 100
        elif n < 500:
            lamb = 1600
        else:
            lamb = 14400

    # Apply HP filter
    cycle, trend = hpfilter(series, lamb=lamb)

    return DecompositionResult(
        method="hp_filter",
        trend=trend,
        seasonal=cycle,  # HP filter calls this "cycle"
        residual=np.zeros_like(series),  # HP filter doesn't separate seasonal from residual
        period=0,  # Not applicable for HP filter
    )


def _hp_filter_manual(
    series: np.ndarray,
    lamb: Optional[float] = None,
) -> DecompositionResult:
    """Manual HP filter implementation (fallback if statsmodels not available)."""
    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    if lamb is None:
        lamb = 1600

    # Build the second-difference matrix
    # D is n x (n-2), D^T D is n x n
    # HP filter solves: min (y - tau)^T (y - tau) + lambda * (D^2 tau)^T (D^2 tau)
    # Solution: tau = (I + lambda * K)^{-1} y where K is the band matrix

    # Build K = D^T D where D is second difference operator
    K = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            K[i, 0] = 1
            K[i, 1] = -2
            K[i, 2] = 1
        elif i == 1:
            K[i, 0] = -2
            K[i, 1] = 5
            K[i, 2] = -4
            K[i, 3] = 1
        elif i == n - 2:
            K[i, n - 4] = 1
            K[i, n - 3] = -4
            K[i, n - 2] = 5
            K[i, n - 1] = -2
        elif i == n - 1:
            K[i, n - 3] = 1
            K[i, n - 2] = -2
            K[i, n - 1] = 1
        else:
            K[i, i - 2] = 1
            K[i, i - 1] = -4
            K[i, i] = 6
            K[i, i + 1] = -4
            K[i, i + 2] = 1

    # Solve (I + lambda * K) * trend = y
    A = np.eye(n) + lamb * K
    trend = np.linalg.solve(A, series)
    cycle = series - trend

    return DecompositionResult(
        method="hp_filter",
        trend=trend,
        seasonal=cycle,
        residual=np.zeros_like(series),
        period=0,
    )
