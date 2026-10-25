"""Recurrence plot analysis for time series.

This module provides functions to generate recurrence plots and compute
Recurrence Quantification Analysis (RQA) metrics.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.ndimage as ndimage

from ..base import RecurrenceResult
from ..validation import as_univariate_series


def _downsample(series: np.ndarray, max_points: int) -> Tuple[np.ndarray, int]:
    """Decimate to at most ``max_points`` samples; returns (series, step)."""
    if max_points < 2:
        raise ValueError("max_points must be at least 2.")
    step = int(np.ceil(len(series) / max_points)) if len(series) > max_points else 1
    return series[::step], step


def compute_recurrence_matrix(
    series: np.ndarray,
    threshold: Optional[float] = None,
    max_points: int = 1000,
) -> Tuple[np.ndarray, float]:
    """Compute the recurrence matrix for a time series.

    The recurrence matrix R(i,j) = 1 if |x_i - x_j| < threshold, else 0.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    threshold : float, optional
        Distance threshold for recurrence. If None, auto-computed as 10% of range.
    max_points : int
        Maximum number of points to use. Longer series are decimated with
        step ``ceil(n / max_points)`` (no anti-alias filtering).

    Returns
    -------
    recurrence_matrix : np.ndarray
        Binary recurrence matrix
    threshold : float
        The threshold used

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 4*np.pi, 200))
    >>> R, thresh = compute_recurrence_matrix(x)
    >>> print(f"Recurrence matrix shape: {R.shape}")
    """
    series = as_univariate_series(series, min_length=2)
    series, _ = _downsample(series, max_points)

    # Compute distance matrix
    s = series[:, None]
    e = series[None, :]
    dist_mat = np.abs(s - e)

    # Auto-compute threshold if not provided
    if threshold is None:
        threshold = (np.max(series) - np.min(series)) * 0.1
    if not threshold > 0:
        raise ValueError("Recurrence threshold must be positive (the series may be constant).")

    # Create binary recurrence matrix
    recurrence_matrix = (dist_mat < threshold).astype(np.int8)

    return recurrence_matrix, float(threshold)


def compute_rqa_metrics(
    recurrence_matrix: np.ndarray,
    min_line_length: int = 2,
) -> dict:
    """Compute Recurrence Quantification Analysis (RQA) metrics.

    Parameters
    ----------
    recurrence_matrix : np.ndarray
        Binary recurrence matrix
    min_line_length : int
        Minimum length to consider a diagonal/vertical line

    Returns
    -------
    dict
        Dictionary with RQA metrics:
        - RR: Recurrence Rate (density of recurrence points)
        - DET: Determinism (fraction in diagonal lines)
        - LAM: Laminarity (fraction in vertical lines)
    """
    if recurrence_matrix.size == 0:
        return {"RR": 0.0, "DET": 0.0, "LAM": 0.0}

    # Recurrence Rate
    RR = float(np.mean(recurrence_matrix))

    total_recurrence_points = np.sum(recurrence_matrix)
    if total_recurrence_points == 0:
        return {"RR": 0.0, "DET": 0.0, "LAM": 0.0}

    # Determinism (diagonal lines)
    # Create a structure for diagonal connectivity
    diag_structure = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    labeled, n_features = ndimage.label(recurrence_matrix, structure=diag_structure)

    if n_features > 0:
        # Calculate lengths of each labeled region
        features = ndimage.sum(recurrence_matrix, labeled, index=range(1, n_features + 1))
        diagonal_points = np.sum(features[features >= min_line_length])
        DET = float(diagonal_points / total_recurrence_points)
    else:
        DET = 0.0

    # Laminarity (vertical lines)
    v_structure = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    v_labeled, v_n_features = ndimage.label(recurrence_matrix, structure=v_structure)

    if v_n_features > 0:
        v_features = ndimage.sum(recurrence_matrix, v_labeled, index=range(1, v_n_features + 1))
        vertical_points = np.sum(v_features[v_features >= min_line_length])
        LAM = float(vertical_points / total_recurrence_points)
    else:
        LAM = 0.0

    return {"RR": RR, "DET": DET, "LAM": LAM}


def analyze_recurrence(
    series: np.ndarray,
    threshold: Optional[float] = None,
    max_points: int = 1000,
    min_line_length: int = 2,
    include_matrix: bool = False,
) -> RecurrenceResult:
    """Perform recurrence analysis on a scalar time series (no embedding).

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    threshold : float, optional
        Distance threshold for recurrence (default: 10% of the value range)
    max_points : int
        Maximum number of points for analysis; longer inputs are decimated
        and the step is reported in ``downsample_step`` with a warning.
    min_line_length : int
        Minimum line length for RQA metrics
    include_matrix : bool
        Return the binary recurrence matrix (up to ``max_points``^2 values).

    Returns
    -------
    RecurrenceResult
        RQA metrics, plus the matrix when requested

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 4*np.pi, 500)) + 0.1 * np.random.randn(500)
    >>> result = analyze_recurrence(x, threshold=0.2)
    >>> print(f"Recurrence Rate: {result.recurrence_rate:.3f}")
    >>> print(f"Determinism: {result.determinism:.3f}")
    """
    series = as_univariate_series(series, min_length=2)
    _, step = _downsample(series, max_points)
    recurrence_matrix, threshold = compute_recurrence_matrix(
        series, threshold=threshold, max_points=max_points
    )

    metrics = compute_rqa_metrics(recurrence_matrix, min_line_length=min_line_length)
    warnings = []
    if step > 1:
        warnings.append(
            f"Input of {len(series)} samples was decimated with step {step} "
            f"({recurrence_matrix.shape[0]} points used, no anti-alias filter); "
            "RQA metrics describe the decimated series."
        )

    return RecurrenceResult(
        method="recurrence_plot",
        recurrence_matrix=recurrence_matrix if include_matrix else None,
        threshold=threshold,
        recurrence_rate=metrics["RR"],
        determinism=metrics["DET"],
        laminarity=metrics["LAM"],
        n_input_points=int(len(series)),
        n_points_used=int(recurrence_matrix.shape[0]),
        downsample_step=int(step),
        warnings=warnings,
    )
