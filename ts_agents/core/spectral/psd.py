"""Power Spectral Density (PSD) analysis for time series.

This module provides functions for computing and analyzing the power
spectrum of time series data, including spectral slope estimation
for turbulence analysis.
"""

from typing import Optional, Literal, Tuple
import numpy as np
from scipy import signal

from ..base import SpectralResult, PeriodicityResult
from ..validation import as_univariate_series


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
    series = as_univariate_series(series, min_length=2)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive.")

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


MIN_PERIODICITY_SAMPLES = 16


def detect_periodicity(
    series: np.ndarray,
    sample_rate: float = 1.0,
    top_n: int = 3,
    alpha: float = 0.01,
    min_cycles: float = 3.0,
    detrend: Literal["linear", "constant"] = "linear",
) -> PeriodicityResult:
    """Detect dominant periodicities with a red-noise significance test.

    The series is detrended (linear by default), Hann-tapered, and its
    periodogram is compared against an AR(1) red-noise background fitted to
    the lag-1 autocorrelation. Adjacent bins belonging to one spectral peak
    are merged and peaks with fewer than ``min_cycles`` cycles in the record
    are ignored. A peak counts as a detected period only if its
    Bonferroni-corrected p-value is below ``alpha`` *and* the
    autocorrelation has a prominent positive peak at the same lag. Each
    confirmed sinusoid is removed before testing the next peak. Trends,
    level shifts, and random walks therefore do not report the record length
    (or its harmonics) as a "dominant period". The defaults were checked on
    synthetic white noise, AR(1) noise, random walks, trends, and level
    shifts (no detections in 100 seeds each) and on known sinusoids.

    Parameters
    ----------
    series : np.ndarray
        1D time series data with at least 16 finite samples.
    sample_rate : float
        Samples per unit time. Periods are returned in units of
        ``1 / sample_rate`` (samples when 1.0).
    top_n : int
        Maximum number of significant periods to return.
    alpha : float
        Family-wise significance level for declaring a period.
    min_cycles : float
        Minimum number of complete cycles a period must show in the record
        (default 3; with 2, a single level shift resembles half a cycle).
    detrend : str
        ``"linear"`` (default) or ``"constant"`` (mean removal only).

    Returns
    -------
    PeriodicityResult
        ``detected`` is False and ``dominant_period`` is None when no peak is
        significant; ``candidate_period`` then reports the strongest peak.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1000)
    >>> x = np.sin(2 * np.pi * t / 25) + 0.3 * np.random.randn(1000)
    >>> result = detect_periodicity(x)
    >>> print(f"Dominant period: {result.dominant_period:.1f}")
    """
    series = as_univariate_series(series, min_length=MIN_PERIODICITY_SAMPLES)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if min_cycles < 1:
        raise ValueError("min_cycles must be at least 1.")
    if detrend not in ("linear", "constant"):
        raise ValueError("detrend must be 'linear' or 'constant'.")

    n = len(series)
    base = PeriodicityResult(
        method="fft_red_noise",
        dominant_period=None,
        dominant_frequency=None,
        confidence=0.0,
        alpha=float(alpha),
        sample_rate=float(sample_rate),
        detrend=detrend,
        min_cycles=float(min_cycles),
    )

    sig = signal.detrend(series, type=detrend)
    if np.allclose(sig, 0.0, atol=1e-12 * max(1.0, float(np.max(np.abs(series))))):
        base.warnings.append("Series is constant after detrending; no periodicity can be assessed.")
        return base

    taper = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(sig * taper)) ** 2
    total_power = float(np.sum(power[1:]))

    # Candidate bins: at least min_cycles cycles in the record, below Nyquist.
    duration = n / sample_rate
    candidate_idx = np.flatnonzero((freqs >= min_cycles / duration) & (freqs < sample_rate / 2))
    if candidate_idx.size < 3:
        base.warnings.append(
            f"Only {candidate_idx.size} frequency bins satisfy min_cycles={min_cycles}; "
            "the series is too short to assess periodicity."
        )
        return base

    residual = sig.copy()
    accepted: list[float] = []
    accepted_bins: set[int] = set()
    first = True
    for _ in range(max(top_n, 1) + _MAX_REJECTED_CANDIDATES):
        residual_power = np.abs(np.fft.rfft(residual * taper)) ** 2
        ratio = residual_power / _red_noise_background(residual, freqs, sample_rate)
        # Background-scaled ordinates are ~chi2(2)/2 under the null, whose
        # median is ln 2; calibrate the scale robustly with the median.
        scale = float(np.median(ratio[candidate_idx])) / np.log(2.0)
        if not np.isfinite(scale) or scale <= 0:
            break
        ratio = ratio / scale
        peaks = [idx for idx in _local_peaks(ratio, candidate_idx) if idx not in accepted_bins]
        if not peaks:
            break
        best = max(peaks, key=lambda idx: ratio[idx])
        p_value = _bonferroni_p(float(ratio[best]), candidate_idx.size)
        frequency = _refine_peak_frequency(residual_power, freqs, best)
        if first:
            base.candidate_period = float(1.0 / frequency)
            base.p_value = p_value
            base.confidence = float(power[best] / total_power) if total_power > 0 else 0.0
            first = False
        if p_value >= alpha:
            break
        # Every tested peak is excluded from later rounds whether or not it
        # is confirmed, so the loop always makes progress.
        accepted_bins.update({best - 1, best, best + 1})
        period_samples = sample_rate / frequency
        if _acf_confirms_period(_autocorrelation(residual), period_samples, n):
            accepted.append(float(1.0 / frequency))
            if len(accepted) == 1:
                base.detected = True
                base.dominant_frequency = float(frequency)
                base.dominant_period = float(1.0 / frequency)
                base.p_value = p_value
                base.confidence = float(power[best] / total_power) if total_power > 0 else 0.0
            if len(accepted) >= top_n:
                break
            # Remove only confirmed components so later rounds test the remainder.
            residual = _remove_sinusoid(residual, frequency / sample_rate)

    base.top_periods = accepted
    if not base.detected and base.candidate_period is not None and base.p_value is not None and base.p_value < alpha:
        base.warnings.append(
            "Spectral peaks exceeded the red-noise background but none was confirmed by an "
            "autocorrelation peak at the same lag; treating the series as aperiodic."
        )
    return base


_MAX_REJECTED_CANDIDATES = 8


def _bonferroni_p(ratio: float, n_tests: int) -> float:
    p_single = min(float(np.exp(-ratio)), 1.0 - 1e-16)
    return float(-np.expm1(n_tests * np.log1p(-p_single)))


def _autocorrelation(sig: np.ndarray) -> np.ndarray:
    n = len(sig)
    spectrum = np.fft.rfft(sig, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n)[:n]
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def _acf_confirms_period(acf: np.ndarray, period: float, n: int) -> bool:
    """Require a prominent autocorrelation peak near ``period``.

    The peak must be a positive local maximum that rises above the trough
    before it by more than four white-noise standard errors.

    Red noise, trends, and level shifts concentrate spectral power at low
    frequencies but produce monotonically decaying autocorrelation, whereas
    a genuine oscillation repeats in the ACF at its period.
    """
    tolerance = max(1, int(round(0.1 * period)))
    lo = max(1, int(np.floor(period)) - tolerance)
    hi = min(len(acf) - 2, int(np.ceil(period)) + tolerance)
    if hi <= lo:
        return False
    window = np.arange(lo, hi + 1)
    local_max = [
        lag for lag in window
        if acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]
    ]
    if not local_max:
        return False
    lag = max(local_max, key=lambda value: acf[value])
    # White-noise standard error of a sample autocorrelation; the peak must
    # be clearly positive and rise well above the trough before it. Red noise,
    # trends, and level shifts give monotone or flat ACFs whose wiggles do
    # not clear this prominence threshold.
    noise_se = 1.0 / np.sqrt(n)
    trough = float(np.min(acf[1:lag])) if lag > 1 else float(acf[0])
    return bool(acf[lag] > 2.0 * noise_se and acf[lag] - trough > 4.0 * noise_se)


def _remove_sinusoid(sig: np.ndarray, cycles_per_sample: float) -> np.ndarray:
    t = np.arange(len(sig))
    design = np.column_stack([
        np.cos(2.0 * np.pi * cycles_per_sample * t),
        np.sin(2.0 * np.pi * cycles_per_sample * t),
    ])
    coef, *_ = np.linalg.lstsq(design, sig, rcond=None)
    return sig - design @ coef


def _red_noise_background(sig: np.ndarray, freqs: np.ndarray, sample_rate: float) -> np.ndarray:
    """Unscaled AR(1) spectrum shape from the lag-1 autocorrelation of ``sig``."""
    denom = float(np.dot(sig, sig))
    r = float(np.dot(sig[1:], sig[:-1]) / denom) if denom > 0 else 0.0
    r = float(np.clip(r, -0.99, 0.99))
    omega = 2.0 * np.pi * freqs / sample_rate
    return (1.0 - r**2) / (1.0 - 2.0 * r * np.cos(omega) + r**2)


def _local_peaks(values: np.ndarray, candidate_idx: np.ndarray) -> list[int]:
    """Indices of local maxima within the candidate band (adjacent bins merged)."""
    lo, hi = int(candidate_idx[0]), int(candidate_idx[-1])
    peaks = []
    for idx in candidate_idx:
        left = values[idx - 1] if idx - 1 >= lo else -np.inf
        right = values[idx + 1] if idx + 1 <= hi else -np.inf
        if values[idx] >= left and values[idx] > right:
            peaks.append(int(idx))
    return peaks or [int(candidate_idx[np.argmax(values[candidate_idx])])]


def _refine_peak_frequency(power: np.ndarray, freqs: np.ndarray, idx: int) -> float:
    """Quadratic interpolation of the log-power peak between neighbouring bins."""
    if idx <= 0 or idx >= len(power) - 1:
        return float(freqs[idx])
    a, b, c = np.log(power[idx - 1 : idx + 2] + 1e-300)
    denom = a - 2.0 * b + c
    offset = 0.5 * (a - c) / denom if denom < 0 else 0.0
    offset = float(np.clip(offset, -0.5, 0.5))
    step = float(freqs[1] - freqs[0])
    return float(freqs[idx] + offset * step)


def detect_seasonal_period(series: np.ndarray) -> tuple[int, PeriodicityResult]:
    """Return an integer seasonal period for decomposition auto-detection.

    Raises ``ValueError`` when no significant period exists, so decomposition
    never silently uses a trend or the record length as its "season".
    """
    result = detect_periodicity(series)
    if not result.detected or result.dominant_period is None:
        hint = (
            f" Strongest non-significant candidate: {result.candidate_period:.2f} samples (p={result.p_value:.3g})."
            if result.candidate_period is not None and result.p_value is not None
            else ""
        )
        raise ValueError(
            "No significant seasonal period detected; pass the period explicitly." + hint
        )
    period = int(round(result.dominant_period))
    if period < 2 or period > len(np.asarray(series)) // 2:
        raise ValueError(
            f"Detected period {result.dominant_period:.2f} is outside the decomposable range "
            f"[2, {len(np.asarray(series)) // 2}]; pass the period explicitly."
        )
    return period, result


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
