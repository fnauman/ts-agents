"""Descriptive statistics for time series.

This module provides functions for computing summary statistics
of time series data.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from scipy import stats as scipy_stats

from ..base import DescriptiveStats


def describe_series(
    series: np.ndarray,
    extended: bool = False,
) -> DescriptiveStats:
    """Compute descriptive statistics for a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    extended : bool
        Whether to compute extended statistics (skewness, kurtosis)

    Returns
    -------
    DescriptiveStats
        Descriptive statistics of the series

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000) + 5
    >>> stats = describe_series(x, extended=True)
    >>> print(f"Mean: {stats.mean:.2f}, Std: {stats.std:.2f}")
    """
    series = np.asarray(series, dtype=np.float64).flatten()

    # Remove NaNs for statistics
    valid = series[~np.isnan(series)]

    length = len(valid)
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    min_val = float(np.min(valid))
    max_val = float(np.max(valid))
    rms = float(np.sqrt(np.mean(valid ** 2)))

    result = DescriptiveStats(
        method="descriptive",
        length=length,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        rms=rms,
    )

    if extended and length > 0:
        result.median = float(np.median(valid))
        result.skewness = float(scipy_stats.skew(valid))
        result.kurtosis = float(scipy_stats.kurtosis(valid))

    return result


def compute_rolling_stats(
    series: np.ndarray,
    window: int,
    stat: str = "mean",
) -> np.ndarray:
    """Compute rolling (moving window) statistics.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    window : int
        Window size
    stat : str
        Statistic to compute: 'mean', 'std', 'min', 'max', 'sum'

    Returns
    -------
    np.ndarray
        Rolling statistic values (same length as input, NaN-padded)

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> rolling_mean = compute_rolling_stats(x, window=10, stat='mean')
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = series[i - window + 1 : i + 1]

        if stat == "mean":
            result[i] = np.mean(window_data)
        elif stat == "std":
            result[i] = np.std(window_data)
        elif stat == "min":
            result[i] = np.min(window_data)
        elif stat == "max":
            result[i] = np.max(window_data)
        elif stat == "sum":
            result[i] = np.sum(window_data)

    return result


def compute_correlation(
    series1: np.ndarray,
    series2: np.ndarray,
    method: str = "pearson",
) -> Dict[str, float]:
    """Compute correlation between two time series.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'

    Returns
    -------
    dict
        Dictionary with correlation coefficient and p-value

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x + 0.5 * np.random.randn(100)
    >>> corr = compute_correlation(x, y)
    >>> print(f"Correlation: {corr['correlation']:.3f}")
    """
    series1 = np.asarray(series1, dtype=np.float64).flatten()
    series2 = np.asarray(series2, dtype=np.float64).flatten()

    # Align lengths
    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]

    if method == "pearson":
        corr, pvalue = scipy_stats.pearsonr(s1, s2)
    elif method == "spearman":
        corr, pvalue = scipy_stats.spearmanr(s1, s2)
    elif method == "kendall":
        corr, pvalue = scipy_stats.kendalltau(s1, s2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "correlation": float(corr),
        "p_value": float(pvalue),
        "method": method,
    }


def compute_autocorrelation(
    series: np.ndarray,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute autocorrelation function.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    max_lag : int, optional
        Maximum lag to compute. Default: len(series) // 4

    Returns
    -------
    np.ndarray
        Autocorrelation values for lags 0 to max_lag

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 500))
    >>> acf = compute_autocorrelation(x, max_lag=100)
    >>> print(f"ACF at lag 50: {acf[50]:.3f}")
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    series = series - np.mean(series)

    n = len(series)
    if max_lag is None:
        max_lag = n // 4

    max_lag = min(max_lag, n - 1)

    # Use FFT for efficient computation
    fft = np.fft.fft(series, n=2 * n)
    acf_full = np.fft.ifft(fft * np.conj(fft)).real[:n]
    acf_full /= acf_full[0]  # Normalize

    return acf_full[: max_lag + 1]


def compare_series_stats(
    series_dict: Dict[str, np.ndarray],
) -> Dict[str, DescriptiveStats]:
    """Compare statistics across multiple series.

    Parameters
    ----------
    series_dict : dict
        Dictionary mapping names to series data

    Returns
    -------
    dict
        Dictionary mapping names to DescriptiveStats

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = np.random.randn(100) + 2
    >>> comparison = compare_series_stats({'x': x, 'y': y})
    """
    results = {}
    for name, series in series_dict.items():
        results[name] = describe_series(series)

    return results
