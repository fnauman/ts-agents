"""Time series segmentation for regime detection.

This module provides functions to segment time series into distinct regimes
using various algorithms including Matrix Profile (FLUSS) and changepoint
detection (ruptures).
"""

from typing import List, Optional, Dict, Any, Literal
import numpy as np

from ..base import SegmentResult


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
    in time series data.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length for Matrix Profile
    n_segments : int
        Number of regimes to find

    Returns
    -------
    SegmentResult
        Result with changepoint locations and segment statistics

    Examples
    --------
    >>> # Create a series with regime changes
    >>> x = np.concatenate([np.random.randn(200), np.random.randn(200) + 3, np.random.randn(200)])
    >>> result = segment_fluss(x, m=50, n_segments=3)
    >>> print(f"Changepoints: {result.changepoints}")
    """
    import stumpy

    if n_regimes is not None:
        n_segments = int(n_regimes)

    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    if n < 4 * m:
        return SegmentResult(
            method="fluss",
            changepoints=[],
            n_segments=1,
            segment_stats=[{
                "start": 0,
                "end": n,
                "length": n,
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
            }],
        )

    try:
        mp = stumpy.stump(series, m)
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=m, n_regimes=n_segments)

        # STUMPY may return placeholders/boundaries (e.g., 0). Drop invalid/degenerate points.
        raw_changepoints = [int(x) for x in np.asarray(regime_locations).tolist()]
        changepoints = _sanitize_changepoints(raw_changepoints, n=n, min_size=1)

        # Compute segment statistics
        segment_stats = _compute_segment_stats(series, changepoints)

        return SegmentResult(
            method="fluss",
            changepoints=changepoints,
            n_segments=len(changepoints) + 1,
            segment_stats=segment_stats,
        )

    except Exception as e:
        return SegmentResult(
            method="fluss",
            changepoints=[],
            n_segments=1,
            segment_stats=[{
                "start": 0,
                "end": n,
                "length": n,
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "error": str(e),
            }],
        )


def segment_changepoint(
    series: np.ndarray,
    n_segments: Optional[int] = None,
    n_bkps: Optional[int] = None,
    algorithm: Literal["pelt", "binseg", "bottomup", "window"] = "pelt",
    cost_model: Literal["rbf", "l1", "l2", "normal", "ar", "linear"] = "rbf",
    penalty: Optional[float] = None,
    min_size: int = 5,
) -> SegmentResult:
    """Segment time series using changepoint detection (ruptures library).

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    n_segments : int, optional
        Number of segments to find. If None, uses penalty-based detection.
    algorithm : str
        Algorithm to use:
        - 'pelt': Pruned Exact Linear Time (optimal, fast)
        - 'binseg': Binary Segmentation (greedy, fast)
        - 'bottomup': Bottom-up (starts with many segments, merges)
        - 'window': Sliding window approach
    cost_model : str
        Cost function:
        - 'rbf': Radial basis function (good for general changes)
        - 'l1', 'l2': L1/L2 norm (mean shifts)
        - 'normal': Normal distribution (mean and variance)
        - 'ar': Autoregressive model
        - 'linear': Linear regression
    penalty : float, optional
        Penalty value for PELT. Higher = fewer changepoints.
        Auto-estimated if None.
    min_size : int
        Minimum segment size

    Returns
    -------
    SegmentResult
        Result with changepoint locations and segment statistics

    Examples
    --------
    >>> x = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
    >>> result = segment_changepoint(x, n_segments=2)
    >>> print(f"Changepoint at: {result.changepoints[0]}")
    """
    import ruptures as rpt

    # Backward-compatible alias used by older UI code.
    if n_segments is None and n_bkps is not None:
        n_segments = int(n_bkps) + 1

    series = np.asarray(series, dtype=np.float64).flatten()
    n = len(series)

    if n < 10:
        return SegmentResult(
            method=f"changepoint_{algorithm}",
            changepoints=[],
            n_segments=1,
            segment_stats=[{
                "start": 0,
                "end": n,
                "length": n,
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
            }],
        )

    # Auto-estimate penalty if not provided
    if penalty is None:
        penalty = np.log(n) * np.var(series) * 0.5

    try:
        # Select algorithm
        if algorithm == "pelt":
            if n_segments is not None:
                # PELT doesn't support n_bkps, fall back to binseg
                model = rpt.Binseg(model=cost_model, min_size=min_size, jump=1)
                model.fit(series)
                changepoints = model.predict(n_bkps=n_segments - 1)
            else:
                model = rpt.Pelt(model=cost_model, min_size=min_size, jump=1)
                model.fit(series)
                changepoints = model.predict(pen=penalty)

        elif algorithm == "binseg":
            model = rpt.Binseg(model=cost_model, min_size=min_size, jump=1)
            model.fit(series)
            if n_segments is not None:
                changepoints = model.predict(n_bkps=n_segments - 1)
            else:
                changepoints = model.predict(pen=penalty)

        elif algorithm == "bottomup":
            model = rpt.BottomUp(model=cost_model, min_size=min_size, jump=1)
            model.fit(series)
            if n_segments is not None:
                changepoints = model.predict(n_bkps=n_segments - 1)
            else:
                changepoints = model.predict(pen=penalty)

        elif algorithm == "window":
            width = max(10, n // 20)
            model = rpt.Window(width=width, model=cost_model, min_size=min_size, jump=1)
            model.fit(series)
            if n_segments is not None:
                changepoints = model.predict(n_bkps=n_segments - 1)
            else:
                changepoints = model.predict(pen=penalty)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # ruptures includes n as final breakpoint, remove it
        if changepoints and changepoints[-1] == n:
            changepoints = changepoints[:-1]

        changepoints = _sanitize_changepoints(
            [int(cp) for cp in changepoints],
            n=n,
            min_size=min_size,
        )
        segment_stats = _compute_segment_stats(series, changepoints)

        return SegmentResult(
            method=f"changepoint_{algorithm}",
            changepoints=changepoints,
            n_segments=len(changepoints) + 1,
            segment_stats=segment_stats,
        )

    except Exception as e:
        return SegmentResult(
            method=f"changepoint_{algorithm}",
            changepoints=[],
            n_segments=1,
            segment_stats=[{
                "start": 0,
                "end": n,
                "length": n,
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "error": str(e),
            }],
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
