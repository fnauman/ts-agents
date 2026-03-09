"""Coherence analysis between time series.

This module provides functions for computing the coherence (correlation
in frequency domain) between two time series.
"""

from typing import Optional
import numpy as np
from scipy import signal

from ..base import CoherenceResult


def _resolve_nperseg(length: int, nperseg: Optional[int]) -> int:
    """Resolve a safe nperseg for spectral methods."""
    if length < 2:
        raise ValueError("series must contain at least 2 samples for spectral analysis")
    if nperseg is None:
        return min(256, max(2, length // 4))
    nperseg = int(nperseg)
    if nperseg < 2:
        raise ValueError("nperseg must be at least 2")
    return min(nperseg, length)


def compute_coherence(
    series1: np.ndarray,
    series2: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: Optional[int] = None,
) -> CoherenceResult:
    """Compute coherence between two time series.

    Coherence measures the correlation between two signals as a function
    of frequency. Values range from 0 (no correlation) to 1 (perfect correlation).

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    sample_rate : float
        Sampling rate
    nperseg : int, optional
        Segment length. If None, auto-computed.

    Returns
    -------
    CoherenceResult
        Result with frequencies and coherence values

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> x = np.sin(2 * np.pi * t)
    >>> y = np.sin(2 * np.pi * t + 0.5) + 0.1 * np.random.randn(1000)
    >>> result = compute_coherence(x, y)
    >>> print(f"Mean coherence: {result.mean_coherence:.3f}")
    """
    series1 = np.asarray(series1, dtype=np.float64).flatten()
    series2 = np.asarray(series2, dtype=np.float64).flatten()

    # Ensure same length
    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]

    nperseg = _resolve_nperseg(min_len, nperseg)

    # Compute coherence
    freqs, coh = signal.coherence(s1, s2, fs=sample_rate, nperseg=nperseg)

    # Compute a weighted mean coherence to emphasize dominant frequencies
    try:
        _, pxx = signal.welch(s1, fs=sample_rate, nperseg=nperseg)
        _, pyy = signal.welch(s2, fs=sample_rate, nperseg=nperseg)
        weights = (pxx + pyy) / 2.0
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            mean_coherence = float(np.sum(coh * weights) / weight_sum)
        else:
            mean_coherence = float(np.mean(coh))
    except Exception:
        mean_coherence = float(np.mean(coh))
    max_idx = np.argmax(coh)
    max_coherence = float(coh[max_idx])
    dominant_freq = float(freqs[max_idx])

    return CoherenceResult(
        method="coherence",
        frequencies=freqs,
        coherence=coh,
        mean_coherence=mean_coherence,
        max_coherence=max_coherence,
        dominant_frequency=dominant_freq,
    )


def compute_cross_spectrum(
    series1: np.ndarray,
    series2: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: Optional[int] = None,
) -> tuple:
    """Compute cross power spectral density between two signals.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    sample_rate : float
        Sampling rate
    nperseg : int, optional
        Segment length

    Returns
    -------
    freqs : np.ndarray
        Frequency array
    csd : np.ndarray
        Cross spectral density (complex)
    """
    series1 = np.asarray(series1, dtype=np.float64).flatten()
    series2 = np.asarray(series2, dtype=np.float64).flatten()

    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]

    nperseg = _resolve_nperseg(min_len, nperseg)

    freqs, csd = signal.csd(s1, s2, fs=sample_rate, nperseg=nperseg)

    return freqs, csd


def compute_phase_coherence(
    series1: np.ndarray,
    series2: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: Optional[int] = None,
) -> tuple:
    """Compute phase difference between two coherent signals.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    sample_rate : float
        Sampling rate
    nperseg : int, optional
        Segment length

    Returns
    -------
    freqs : np.ndarray
        Frequency array
    phase : np.ndarray
        Phase difference in radians
    coherence : np.ndarray
        Coherence values
    """
    series1 = np.asarray(series1, dtype=np.float64).flatten()
    series2 = np.asarray(series2, dtype=np.float64).flatten()

    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]

    nperseg = _resolve_nperseg(min_len, nperseg)

    # Compute cross spectrum
    freqs, csd = signal.csd(s1, s2, fs=sample_rate, nperseg=nperseg)

    # Compute coherence
    _, coh = signal.coherence(s1, s2, fs=sample_rate, nperseg=nperseg)

    # Phase is the angle of the cross spectrum
    phase = np.angle(csd)

    return freqs, phase, coh
