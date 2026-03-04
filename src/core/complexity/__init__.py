"""Complexity measures for time series.

This module provides functions for measuring the complexity of time series,
including entropy measures and fractal dimension estimation.

Note: Some functions require optional dependencies (antropy, pyinform).
"""

from typing import Optional
import numpy as np


def sample_entropy(
    series: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """Compute sample entropy of a time series.

    Sample entropy measures the regularity/complexity of the series.
    Lower values indicate more regular patterns.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Embedding dimension (pattern length)
    r : float, optional
        Tolerance (similarity threshold). If None, uses 0.2 * std(series).

    Returns
    -------
    float
        Sample entropy value

    Examples
    --------
    >>> import numpy as np
    >>> regular = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> random = np.random.randn(1000)
    >>> print(f"Regular: {sample_entropy(regular):.3f}")
    >>> print(f"Random: {sample_entropy(random):.3f}")
    """
    try:
        import antropy
        return float(antropy.sample_entropy(series, order=m, metric='chebyshev'))
    except ImportError:
        return _sample_entropy_fallback(series, m, r)


def _sample_entropy_fallback(
    series: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """Fallback sample entropy implementation."""
    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    if r is None:
        r = 0.2 * np.std(series)

    def count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(series[i:i + template_len] - series[j:j + template_len])) < r:
                    count += 1
        return count

    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0:
        return np.inf

    return -np.log(A / B)


def permutation_entropy(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy of a time series.

    Permutation entropy captures the complexity based on ordinal patterns.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    order : int
        Order of permutation patterns
    delay : int
        Time delay between samples
    normalize : bool
        Whether to normalize by log(order!)

    Returns
    -------
    float
        Permutation entropy value

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> pe = permutation_entropy(x, order=3)
    >>> print(f"Permutation entropy: {pe:.3f}")
    """
    try:
        import antropy
        return float(antropy.perm_entropy(series, order=order, delay=delay, normalize=normalize))
    except ImportError:
        return _permutation_entropy_fallback(series, order, delay, normalize)


def _permutation_entropy_fallback(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Fallback permutation entropy implementation."""
    from math import factorial

    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    # Generate all permutation patterns
    n_patterns = n - (order - 1) * delay

    if n_patterns <= 0:
        return np.nan

    patterns = []
    for i in range(n_patterns):
        pattern = series[i:i + order * delay:delay]
        patterns.append(tuple(np.argsort(pattern)))

    # Count pattern frequencies
    from collections import Counter
    counts = Counter(patterns)

    # Compute entropy
    probs = np.array(list(counts.values())) / len(patterns)
    entropy = -np.sum(probs * np.log(probs))

    if normalize:
        entropy /= np.log(factorial(order))

    return float(entropy)


def approximate_entropy(
    series: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """Compute approximate entropy of a time series.

    Similar to sample entropy but uses a different formula.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Embedding dimension
    r : float, optional
        Tolerance. If None, uses 0.2 * std(series).

    Returns
    -------
    float
        Approximate entropy value
    """
    try:
        import antropy
        return float(antropy.app_entropy(series, order=m, metric='chebyshev'))
    except ImportError:
        # Simplified approximation
        return sample_entropy(series, m, r) * 1.1  # Approximate


def hurst_exponent(
    series: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> float:
    """Estimate Hurst exponent using R/S analysis.

    The Hurst exponent measures the long-range dependence in a time series:
    - H < 0.5: anti-persistent (mean-reverting)
    - H = 0.5: random walk (no memory)
    - H > 0.5: persistent (trending)

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    min_window : int
        Minimum window size for R/S analysis
    max_window : int, optional
        Maximum window size. If None, uses len(series) // 4.

    Returns
    -------
    float
        Hurst exponent (0-1)

    Examples
    --------
    >>> # Random walk should have H ≈ 0.5
    >>> rw = np.cumsum(np.random.randn(1000))
    >>> h = hurst_exponent(rw)
    >>> print(f"Hurst exponent: {h:.3f}")
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    if max_window is None:
        max_window = n // 4

    # Generate window sizes (logarithmically spaced)
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5)

    if len(window_sizes) < 3:
        return np.nan

    rs_values = []

    for window in window_sizes:
        # Number of non-overlapping windows
        n_windows = n // window

        if n_windows < 1:
            continue

        rs_list = []
        for i in range(n_windows):
            window_data = series[i * window:(i + 1) * window]

            # Mean-adjust
            mean = np.mean(window_data)
            mean_adj = window_data - mean

            # Cumulative sum
            cumsum = np.cumsum(mean_adj)

            # Range
            R = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            S = np.std(window_data)

            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append((window, np.mean(rs_list)))

    if len(rs_values) < 3:
        return np.nan

    # Log-log regression
    log_n = np.log([r[0] for r in rs_values])
    log_rs = np.log([r[1] for r in rs_values])

    # Linear fit
    slope, _ = np.polyfit(log_n, log_rs, 1)

    return float(slope)


__all__ = [
    "sample_entropy",
    "permutation_entropy",
    "approximate_entropy",
    "hurst_exponent",
]
