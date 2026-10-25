"""Time series segmentation for regime detection.

This module provides functions to segment time series into distinct regimes
using various algorithms including Matrix Profile (FLUSS) and changepoint
detection (ruptures).
"""

from typing import List, Optional, Dict, Any, Literal
import warnings as warnings_module

import numpy as np

from ..base import SegmentResult
from ..validation import as_univariate_series


def _sanitize_changepoints(
    changepoints: List[int],
    *,
    n: int,
    min_size: int = 1,
) -> List[int]:
    """Normalize changepoints to strictly increasing in-range boundaries."""
    if n <= 0:
        return []

    safe_min = max(1, int(min_size))
    cleaned: List[int] = []
    for cp in sorted({int(x) for x in changepoints}):
        if cp <= 0 or cp >= n:
            continue
        if cp < safe_min or (n - cp) < safe_min:
            continue
        if cleaned and (cp - cleaned[-1]) < safe_min:
            continue
        cleaned.append(cp)
    return cleaned


def segment_fluss(
    series: np.ndarray,
    m: int = 50,
    n_segments: int = 3,
    n_regimes: Optional[int] = None,
) -> SegmentResult:
    """Segment time series using FLUSS (Fast Low-cost Unipotent Semantic Segmentation).

    FLUSS uses the Matrix Profile to find semantic segments (regime changes)
    in time series data. It compares z-normalized subsequence *shapes*, so it
    detects changes in pattern, not pure level or variance shifts (use
    :func:`segment_changepoint` for those), and it cannot place boundaries
    within about ``5 * m`` samples of either end.

    Parameters
    ----------
    series : np.ndarray
        1D time series data with at least ``4 * m`` samples
    m : int
        Subsequence length for Matrix Profile
    n_segments : int
        Number of regimes to find

    Returns
    -------
    SegmentResult
        Result with changepoint locations and segment statistics. A warning
        is attached when fewer boundaries than requested were found.

    Examples
    --------
    >>> # Create a series with regime changes
    >>> x = np.concatenate([np.random.randn(200), np.random.randn(200) + 3, np.random.randn(200)])
    >>> result = segment_fluss(x, m=50, n_segments=3)
    >>> print(f"Changepoints: {result.changepoints}")
    """
    try:
        import stumpy
    except ModuleNotFoundError as exc:
        raise ImportError(
            'FLUSS segmentation requires optional dependencies. Install with: pip install "ts-agents[patterns]"'
        ) from exc

    if n_regimes is not None:
        n_segments = int(n_regimes)
    n_segments = int(n_segments)
    m = int(m)
    if n_segments < 2:
        raise ValueError("n_segments must be at least 2 for FLUSS.")
    if m < 3:
        raise ValueError("Subsequence length m must be at least 3.")

    series = as_univariate_series(series)
    n = len(series)
    if n < 4 * m:
        raise ValueError(
            f"FLUSS needs at least 4*m = {4 * m} samples for m={m}; got {n}. "
            "Use a smaller m or segment_changepoint."
        )

    mp = stumpy.stump(series, m)
    _, regime_locations = stumpy.fluss(mp[:, 1], L=m, n_regimes=n_segments)

    # STUMPY may return placeholders/boundaries (e.g., 0). Drop invalid/degenerate points.
    raw_changepoints = [int(x) for x in np.asarray(regime_locations).tolist()]
    changepoints = _sanitize_changepoints(raw_changepoints, n=n, min_size=1)

    warnings: List[str] = []
    requested = n_segments - 1
    if len(changepoints) < requested:
        warnings.append(
            f"FLUSS returned {len(changepoints)} of {requested} requested boundaries "
            f"(edges within ~{5 * m} samples are excluded and placeholders were dropped)."
        )

    return SegmentResult(
        method="fluss",
        changepoints=changepoints,
        n_segments=len(changepoints) + 1,
        segment_stats=_compute_segment_stats(series, changepoints),
        settings={"m": m, "n_regimes_requested": n_segments, "edge_exclusion": 5 * m},
        warnings=warnings,
    )


_ALGORITHMS = ("pelt", "binseg", "bottomup", "window")
_COST_MODELS = ("rbf", "l1", "l2", "normal", "ar", "linear")
# Default penalty = factor * log(n) on robustly standardized input. Likelihood
# costs use a BIC-style factor (3 = segment parameters + location, with a
# small margin for l1/l2); rbf (kernel cost, scale-free in ruptures) was
# calibrated on synthetic white noise with single mean/variance shifts.
_PENALTY_LOG_FACTOR = {"rbf": 2.0, "l1": 3.0, "l2": 3.0, "normal": 3.0, "ar": 3.0, "linear": 3.0}
_FULL_RESOLUTION_MAX_N = 500


def _robust_noise_scale(series: np.ndarray) -> float:
    """Noise standard deviation from first differences (insensitive to shifts)."""
    diffs = np.diff(series)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    scale = mad / 0.6744897501960817 / np.sqrt(2.0)
    if scale > 0:
        return scale
    return float(np.std(series))


def segment_changepoint(
    series: np.ndarray,
    n_segments: Optional[int] = None,
    n_bkps: Optional[int] = None,
    algorithm: Literal["pelt", "binseg", "bottomup", "window"] = "pelt",
    cost_model: Literal["rbf", "l1", "l2", "normal", "ar", "linear"] = "normal",
    penalty: Optional[float] = None,
    min_size: int = 5,
    jump: Optional[int] = None,
) -> SegmentResult:
    """Segment time series using changepoint detection (ruptures library).

    The series is standardized by a robust noise scale (MAD of first
    differences) before detection, so results do not depend on measurement
    units: multiplying the data by any positive constant gives the same
    changepoints. ``penalty`` is therefore in standardized units.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    n_segments : int, optional
        Number of segments to find. If None, uses penalty-based detection.
        PELT cannot target a segment count; binary segmentation runs instead
        and the result reports that substitution.
    algorithm : str
        Algorithm to use:
        - 'pelt': Pruned Exact Linear Time (optimal, fast)
        - 'binseg': Binary Segmentation (greedy, fast)
        - 'bottomup': Bottom-up (starts with many segments, merges)
        - 'window': Sliding window approach
    cost_model : str
        Cost function:
        - 'normal': Normal distribution (mean and variance shifts; default,
          linear-time per evaluation)
        - 'rbf': Radial basis function (general distribution changes;
          quadratic memory/time, slow beyond a few thousand samples)
        - 'l1', 'l2': L1/L2 norm (mean shifts only)
        - 'ar': Autoregressive model
        - 'linear': Linear regression
    penalty : float, optional
        Penalty on standardized data. Higher = fewer changepoints. Defaults
        to ``3 * log(n)`` for likelihood costs and ``2 * log(n)`` for rbf,
        multiplied by ``(1 + rho) / (1 - rho)`` where ``rho`` is the series'
        lag-1 autocorrelation (clipped to [0, 0.99]).
    min_size : int
        Minimum segment size in samples
    jump : int, optional
        Candidate changepoints are restricted to multiples of ``jump``.
        Defaults to 1 up to 500 samples and ``ceil(n / 500)`` beyond, to
        bound runtime; the effective value is reported in ``settings``.

    Returns
    -------
    SegmentResult
        Changepoints, segment statistics in original units, and the effective
        ``settings`` (penalty, scale, jump, cost model).

    Examples
    --------
    >>> x = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
    >>> result = segment_changepoint(x, n_segments=2)
    >>> print(f"Changepoint at: {result.changepoints[0]}")
    """
    try:
        import ruptures as rpt
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Changepoint detection requires optional dependencies. Install with: pip install "ts-agents[patterns]"'
        ) from exc

    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {', '.join(_ALGORITHMS)}.")
    if cost_model not in _COST_MODELS:
        raise ValueError(f"Unknown cost_model '{cost_model}'. Supported: {', '.join(_COST_MODELS)}.")

    # Backward-compatible alias used by older UI code.
    if n_segments is None and n_bkps is not None:
        n_segments = int(n_bkps) + 1
    if n_segments is not None and int(n_segments) < 1:
        raise ValueError("n_segments must be at least 1.")
    min_size = int(min_size)
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")

    series = as_univariate_series(series)
    n = len(series)
    if n < 2 * min_size:
        raise ValueError(
            f"Series length {n} is too short for changepoint detection with min_size={min_size}; "
            f"need at least {2 * min_size} samples."
        )
    if n_segments is not None and int(n_segments) * min_size > n:
        raise ValueError(
            f"Cannot place {n_segments} segments of at least {min_size} samples in {n} samples."
        )

    if jump is None:
        jump = 1 if n <= _FULL_RESOLUTION_MAX_N else int(np.ceil(n / _FULL_RESOLUTION_MAX_N))
    jump = int(jump)
    if jump < 1:
        raise ValueError("jump must be at least 1.")

    warnings: List[str] = []
    actual_algorithm = algorithm
    if algorithm == "pelt" and n_segments is not None:
        actual_algorithm = "binseg"
        warnings.append(
            "PELT cannot target a fixed number of segments; binary segmentation ran instead."
        )
    if jump > 1:
        warnings.append(
            f"Changepoints are resolved to multiples of jump={jump} samples to bound runtime "
            "on long series; pass jump=1 for full resolution."
        )

    scale = _robust_noise_scale(series)
    settings: Dict[str, Any] = {
        "cost_model": cost_model,
        "min_size": min_size,
        "jump": jump,
        "standardization": "median / robust noise scale (MAD of first differences)",
        "noise_scale": float(scale),
    }
    if scale <= 0:
        warnings.append("Series is constant; no changepoints can exist.")
        return SegmentResult(
            method=f"changepoint_{actual_algorithm}",
            requested_method=f"changepoint_{algorithm}" if actual_algorithm != algorithm else None,
            changepoints=[],
            n_segments=1,
            segment_stats=_compute_segment_stats(series, []),
            settings=settings,
            warnings=warnings,
        )
    standardized = (series - np.median(series)) / scale

    if n_segments is None:
        penalty_source = "user" if penalty is not None else "default"
        if penalty is None:
            # BIC-style penalty, inflated by the AR(1) variance factor
            # (1 + rho) / (1 - rho): autocorrelated samples carry less
            # independent evidence, so smooth or persistent signals are not
            # cut into spurious "regimes". Real shifts raise the series'
            # lag-1 autocorrelation only modestly relative to their gain.
            centered = standardized - np.mean(standardized)
            denom = float(np.dot(centered, centered))
            rho = float(np.dot(centered[1:], centered[:-1]) / denom) if denom > 0 else 0.0
            rho = float(np.clip(rho, 0.0, 0.99))
            inflation = (1.0 + rho) / (1.0 - rho)
            penalty = _PENALTY_LOG_FACTOR[cost_model] * float(np.log(n)) * inflation
            settings.update({"lag1_autocorrelation": rho, "penalty_autocorrelation_inflation": inflation})
        penalty = float(penalty)
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative.")
        settings.update({"penalty": penalty, "penalty_source": penalty_source})
    else:
        settings["n_segments_requested"] = int(n_segments)

    algorithm_classes = {
        "pelt": rpt.Pelt,
        "binseg": rpt.Binseg,
        "bottomup": rpt.BottomUp,
    }
    try:
        with warnings_module.catch_warnings():
            # ruptures >= 1.1.5 announces its covariance bias whenever a
            # 'normal' cost is constructed; it is not about the user's data.
            warnings_module.filterwarnings("ignore", message="New behaviour in v1.1.5", category=UserWarning)
            if actual_algorithm == "window":
                width = max(2 * min_size, n // 20)
                settings["window_width"] = width
                model = rpt.Window(width=width, model=cost_model, min_size=min_size, jump=jump)
            else:
                model = algorithm_classes[actual_algorithm](model=cost_model, min_size=min_size, jump=jump)
            model.fit(standardized)
            if n_segments is not None:
                raw = model.predict(n_bkps=int(n_segments) - 1)
            else:
                raw = model.predict(pen=penalty)
    except rpt.exceptions.BadSegmentationParameters as exc:
        raise ValueError(f"Changepoint detection cannot satisfy the requested settings: {exc}") from exc

    changepoints = _sanitize_changepoints(
        [int(cp) for cp in raw if int(cp) < n],
        n=n,
        min_size=min_size,
    )
    if n_segments is not None and len(changepoints) != int(n_segments) - 1:
        warnings.append(
            f"Requested {int(n_segments) - 1} changepoints but {len(changepoints)} were found."
        )

    return SegmentResult(
        method=f"changepoint_{actual_algorithm}",
        requested_method=f"changepoint_{algorithm}" if actual_algorithm != algorithm else None,
        changepoints=changepoints,
        n_segments=len(changepoints) + 1,
        segment_stats=_compute_segment_stats(series, changepoints),
        settings=settings,
        warnings=warnings,
    )


def _compute_segment_stats(
    series: np.ndarray,
    changepoints: List[int],
) -> List[Dict[str, Any]]:
    """Compute statistics for each segment."""
    n = len(series)
    boundaries = [0] + list(changepoints) + [n]

    segment_stats = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        segment = series[start:end]
        if segment.size == 0:
            continue

        segment_stats.append({
            "start": int(start),
            "end": int(end),
            "length": int(end - start),
            "mean": float(np.mean(segment)),
            "std": float(np.std(segment)),
            "min": float(np.min(segment)),
            "max": float(np.max(segment)),
        })

    if not segment_stats:
        return [{
            "start": 0,
            "end": int(n),
            "length": int(n),
            "mean": float(np.mean(series)) if n > 0 else 0.0,
            "std": float(np.std(series)) if n > 0 else 0.0,
            "min": float(np.min(series)) if n > 0 else 0.0,
            "max": float(np.max(series)) if n > 0 else 0.0,
        }]

    return segment_stats
