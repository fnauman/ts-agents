"""Power Spectral Density (PSD) analysis for time series.

This module provides functions for computing and analyzing the power
spectrum of time series data, including spectral slope estimation
for turbulence analysis.
"""

from typing import Optional, Literal, Tuple
import numpy as np
from scipy import signal

from ..base import SpectralResult, PeriodicityResult


def _resolve_nperseg(length: int, nperseg: Optional[int]) -> int:
    """Resolve a safe nperseg for Welch-style methods."""
    if length < 2:
        raise ValueError("series must contain at least 2 samples for spectral analysis")
    if nperseg is None:
        return min(256, max(2, length // 4))
    nperseg = int(nperseg)
    if nperseg < 2:
        raise ValueError("nperseg must be at least 2")
    return min(nperseg, length)


def compute_psd(
    series: np.ndarray,
    sample_rate: float = 1.0,
    method: Literal["welch", "periodogram"] = "welch",
    nperseg: Optional[int] = None,
) -> SpectralResult:
    """Compute Power Spectral Density of a time series.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    sample_rate : float
        Sampling rate (samples per unit time)
    method : str
        Method for PSD estimation:
        - 'welch': Welch's method (smoother, recommended)
        - 'periodogram': Standard periodogram (more noisy)
    nperseg : int, optional
        Segment length for Welch's method. If None, auto-computed.

    Returns
    -------
    SpectralResult
        Result with frequencies, PSD, and spectral slope

    Examples
    --------
    >>> import numpy as np
    >>> # Colored noise with power law spectrum
    >>> x = np.cumsum(np.random.randn(1000))  # Random walk (1/f^2 spectrum)
    >>> result = compute_psd(x, sample_rate=1.0)
    >>> print(f"Spectral slope: {result.spectral_slope:.2f}")  # ~-2
    """
    series = np.asarray(series, dtype=np.float64).flatten()
    if series.size < 2:
        raise ValueError("series must contain at least 2 samples for spectral analysis")

    if method == "welch":
        nperseg = _resolve_nperseg(len(series), nperseg)
        freqs, psd = signal.welch(series, fs=sample_rate, nperseg=nperseg)
    else:
        freqs, psd = signal.periodogram(series, fs=sample_rate)

    # Compute spectral slope (for turbulence power law)
    slope, intercept = _fit_spectral_slope(freqs, psd)

    # Find dominant frequency
    mask = (freqs > 0) & (psd > 0)
    if np.any(mask):
        dom_idx = np.argmax(psd[mask])
        dominant_freq = freqs[mask][dom_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
    else:
        dominant_freq = 0.0
        dominant_period = 0.0

    return SpectralResult(
        method=method,
        frequencies=freqs,
        psd=psd,
        spectral_slope=slope,
        slope_intercept=intercept,
        dominant_frequency=dominant_freq,
        dominant_period=dominant_period,
    )


def _fit_spectral_slope(
    freqs: np.ndarray,
    psd: np.ndarray,
) -> Tuple[float, float]:
    """Fit power law to PSD in log-log space.

    Returns slope and intercept of log10(PSD) = slope * log10(freq) + intercept
    """
    # Filter valid points
    mask = (freqs > 0) & (psd > 0)
    valid_freqs = freqs[mask]
    valid_psd = psd[mask]

    if len(valid_freqs) < 10:
        return 0.0, 0.0

    log_f = np.log10(valid_freqs)
    log_p = np.log10(valid_psd)

    # Linear fit in log-log space
    slope, intercept = np.polyfit(log_f, log_p, 1)

    return float(slope), float(intercept)


def detect_periodicity(
    series: np.ndarray,
    sample_rate: float = 1.0,
    top_n: int = 3,
) -> PeriodicityResult:
    """Detect dominant periodicities in a time series using FFT.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    sample_rate : float
        Sampling rate
    top_n : int
        Number of top periods to return

    Returns
    -------
    PeriodicityResult
        Result with dominant period and confidence

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1000)
    >>> x = np.sin(2 * np.pi * t / 10) + 0.5 * np.sin(2 * np.pi * t / 25)
    >>> result = detect_periodicity(x)
    >>> print(f"Dominant period: {result.dominant_period:.1f}")
    """
    series = np.asarray(series, dtype=np.float64).flatten()

    # Remove mean
    sig = series - np.mean(series)

    # Compute FFT
    fft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1 / sample_rate)
    power = np.abs(fft) ** 2

    # Ignore DC component
    power[0] = 0

    # Find dominant peak
    peak_idx = np.argmax(power)
    dominant_freq = freqs[peak_idx]
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

    # Confidence: peak power / total power
    total_power = np.sum(power)
    confidence = power[peak_idx] / total_power if total_power > 0 else 0.0

    # Top N periods
    top_indices = np.argsort(power)[-top_n:][::-1]
    top_periods = []
    for idx in top_indices:
        f = freqs[idx]
        if f > 0:
            top_periods.append(float(1.0 / f))

    return PeriodicityResult(
        method="fft",
        dominant_period=float(dominant_period),
        dominant_frequency=float(dominant_freq),
        confidence=float(confidence),
        top_periods=top_periods,
    )


def compute_spectral_centroid(
    series: np.ndarray,
    sample_rate: float = 1.0,
) -> float:
    """Compute the spectral centroid (center of mass of spectrum).

    The spectral centroid indicates where the "center of mass" of the
    spectrum is located.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    sample_rate : float
        Sampling rate

    Returns
    -------
    float
        Spectral centroid (in frequency units)
    """
    result = compute_psd(series, sample_rate=sample_rate)

    # Weighted average of frequencies
    total_power = np.sum(result.psd)
    if total_power > 0:
        centroid = np.sum(result.frequencies * result.psd) / total_power
    else:
        centroid = 0.0

    return float(centroid)
