"""Matrix Profile analysis for time series.

This module provides functions for motif discovery (recurring patterns) and
discord detection (anomalies) using the STUMPY library.
"""

from typing import List, Optional
import numpy as np

from ..base import (
    DiscordResult,
    DiscordSearchResult,
    MatrixProfileResult,
    MotifResult,
    MotifSearchResult,
)
from ..validation import as_univariate_series


def _get_stumpy():
    try:
        import stumpy
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Matrix profile analysis requires optional dependencies. Install with: pip install "ts-agents[patterns]"'
        ) from exc
    return stumpy


def _validate_subsequence_length(series: np.ndarray, m: int) -> int:
    m = int(m)
    if m < 3:
        raise ValueError("Subsequence length m must be at least 3.")
    if len(series) < 2 * m:
        raise ValueError(
            f"Series length {len(series)} is too short for subsequence length m={m}; "
            f"need at least {2 * m} samples."
        )
    return m


def _resolve_exclusion_zone(m: int, exclusion_zone: Optional[int]) -> int:
    zone = m if exclusion_zone is None else int(exclusion_zone)
    if zone < 0:
        raise ValueError("exclusion_zone must be nonnegative.")
    return zone


def compute_matrix_profile(
    series: np.ndarray,
    m: int = 50,
) -> np.ndarray:
    """Compute the Matrix Profile for a time series.

    The Matrix Profile is a vector where each element is the z-normalized
    Euclidean distance between a subsequence and its nearest neighbor.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length (window size)

    Returns
    -------
    np.ndarray
        Matrix profile array with shape (n - m + 1, 4):
        - Column 0: Matrix profile values (distances)
        - Column 1: Matrix profile indices (nearest neighbor indices)
        - Column 2: Left matrix profile indices
        - Column 3: Right matrix profile indices

    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> mp = compute_matrix_profile(x, m=50)
    >>> print(f"Matrix profile shape: {mp.shape}")
    """
    series = as_univariate_series(series)
    m = _validate_subsequence_length(series, m)
    return _get_stumpy().stump(series, m)


def _top_motifs(
    series: np.ndarray,
    mp: np.ndarray,
    m: int,
    max_motifs: int,
    exclusion_zone: int,
    include_subsequences: bool,
) -> List[MotifResult]:
    mp_values = mp[:, 0].astype(np.float64).copy()
    mp_indices = mp[:, 1].astype(int)
    motifs: List[MotifResult] = []
    for _ in range(max_motifs):
        idx = int(np.argmin(mp_values))
        val = float(mp_values[idx])
        if not np.isfinite(val):
            break
        neighbor_idx = int(mp_indices[idx])
        motif = MotifResult(index=idx, neighbor_index=neighbor_idx, distance=val)
        if include_subsequences:
            motif.subsequence = series[idx : idx + m].copy()
        motifs.append(motif)
        # Exclude around both members so the same pair is not reported twice.
        for center in (idx, neighbor_idx):
            start = max(0, center - exclusion_zone)
            end = min(len(mp_values), center + exclusion_zone + 1)
            mp_values[start:end] = np.inf
    return motifs


def _top_discords(
    series: np.ndarray,
    mp: np.ndarray,
    m: int,
    max_discords: int,
    exclusion_zone: int,
    include_subsequences: bool,
) -> List[DiscordResult]:
    profile = mp[:, 0].astype(np.float64)
    finite = profile[np.isfinite(profile)]
    median = float(np.median(finite)) if finite.size else 0.0
    mad = float(np.median(np.abs(finite - median))) * 1.4826 if finite.size else 0.0
    mp_values = profile.copy()
    discords: List[DiscordResult] = []
    for _ in range(max_discords):
        idx = int(np.argmax(mp_values))
        val = float(mp_values[idx])
        if not np.isfinite(val):
            break
        discord = DiscordResult(
            index=idx,
            distance=val,
            robust_z=float((val - median) / mad) if mad > 0 else None,
        )
        if include_subsequences:
            discord.subsequence = series[idx : idx + m].copy()
        discords.append(discord)
        start = max(0, idx - exclusion_zone)
        end = min(len(mp_values), idx + exclusion_zone + 1)
        mp_values[start:end] = -np.inf
    return discords


def find_motifs(
    series: np.ndarray,
    m: int = 50,
    max_motifs: int = 3,
    exclusion_zone: Optional[int] = None,
    include_subsequences: bool = False,
) -> MotifSearchResult:
    """Find the top motifs (recurring patterns) in a time series.

    Motifs are subsequences with the smallest matrix profile values,
    meaning they have very similar patterns elsewhere in the series.
    Distances are z-normalized, so motifs match shape, not level or scale.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length in samples
    max_motifs : int
        Maximum number of motifs to return
    exclusion_zone : int, optional
        Samples excluded around each reported motif and its neighbor.
        Default is ``m`` (non-overlapping subsequences).
    include_subsequences : bool
        Whether to include the actual subsequence arrays

    Returns
    -------
    MotifSearchResult
        Motifs sorted by distance (best first) plus the settings used.

    Examples
    --------
    >>> pattern = np.sin(np.linspace(0, 2*np.pi, 50))
    >>> x = np.random.randn(1000)
    >>> x[100:150] = pattern
    >>> x[500:550] = pattern * 1.05  # Similar pattern
    >>> result = find_motifs(x, m=50, max_motifs=3)
    >>> print(f"Found {len(result.motifs)} motifs")
    """
    series = as_univariate_series(series)
    m = _validate_subsequence_length(series, m)
    zone = _resolve_exclusion_zone(m, exclusion_zone)
    mp = _get_stumpy().stump(series, m)
    motifs = _top_motifs(series, mp, m, max_motifs, zone, include_subsequences)
    warnings = []
    if len(motifs) < max_motifs:
        warnings.append(
            f"Only {len(motifs)} of {max_motifs} requested motifs fit outside the exclusion zones."
        )
    return MotifSearchResult(
        method="stumpy_motifs",
        motifs=motifs,
        subsequence_length=m,
        exclusion_zone=zone,
        warnings=warnings,
    )


def find_discords(
    series: np.ndarray,
    m: int = 50,
    max_discords: int = 3,
    exclusion_zone: Optional[int] = None,
    include_subsequences: bool = False,
) -> DiscordSearchResult:
    """Find the top discords (most unusual subsequences) in a time series.

    Discords are subsequences with the largest matrix profile values,
    meaning they are most different from all other subsequences. The top-k
    discords always exist, even in pure noise; use ``robust_z`` to judge how
    unusual each one is relative to the rest of the matrix profile. Distances
    are z-normalized, so pure level shifts can be missed.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length in samples
    max_discords : int
        Maximum number of discords to return
    exclusion_zone : int, optional
        Samples excluded around each reported discord. Default is ``m`` so
        one event cannot occupy several of the top-k slots.
    include_subsequences : bool
        Whether to include the actual subsequence arrays

    Returns
    -------
    DiscordSearchResult
        Discords sorted by distance (most unusual first) plus the settings used.

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 20*np.pi, 1000))
    >>> x[500:550] += 5  # Inject an anomaly
    >>> result = find_discords(x, m=50, max_discords=3)
    >>> print(f"Most anomalous index: {result.discords[0].index}")
    """
    series = as_univariate_series(series)
    m = _validate_subsequence_length(series, m)
    zone = _resolve_exclusion_zone(m, exclusion_zone)
    mp = _get_stumpy().stump(series, m)
    discords = _top_discords(series, mp, m, max_discords, zone, include_subsequences)
    warnings = []
    if len(discords) < max_discords:
        warnings.append(
            f"Only {len(discords)} of {max_discords} requested discords fit outside the exclusion zones."
        )
    return DiscordSearchResult(
        method="stumpy_discords",
        discords=discords,
        subsequence_length=m,
        exclusion_zone=zone,
        warnings=warnings,
    )


def analyze_matrix_profile(
    series: np.ndarray,
    m: int = 50,
    max_motifs: int = 3,
    max_discords: int = 3,
    include_subsequences: bool = False,
    subsequence_length: Optional[int] = None,
    n_motifs: Optional[int] = None,
    n_discords: Optional[int] = None,
    exclusion_zone: Optional[int] = None,
    include_profile: bool = False,
) -> MatrixProfileResult:
    """Perform comprehensive Matrix Profile analysis.

    Finds both motifs (recurring patterns) and discords (unusual
    subsequences) from one matrix-profile computation.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length in samples
    max_motifs : int
        Maximum number of motifs to find
    max_discords : int
        Maximum number of discords to find
    include_subsequences : bool
        Whether to include subsequence arrays
    exclusion_zone : int, optional
        Samples excluded around reported motifs/discords (default ``m``).
    include_profile : bool
        Return the full profile arrays (``mp_values``/``mp_indices``).
        Off by default to keep machine-readable output bounded.

    Returns
    -------
    MatrixProfileResult
        Comprehensive result with matrix profile summary and patterns

    Examples
    --------
    >>> x = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> result = analyze_matrix_profile(x, m=50)
    >>> print(f"MP range: [{result.mp_min:.3f}, {result.mp_max:.3f}]")
    >>> print(f"Top motif distance: {result.motifs[0].distance:.3f}")
    """
    # Backward-compatible aliases used by earlier UI code.
    if subsequence_length is not None:
        m = int(subsequence_length)
    if n_motifs is not None:
        max_motifs = int(n_motifs)
    if n_discords is not None:
        max_discords = int(n_discords)

    series = as_univariate_series(series)
    m = _validate_subsequence_length(series, m)
    zone = _resolve_exclusion_zone(m, exclusion_zone)

    mp = _get_stumpy().stump(series, m)
    mp_values = mp[:, 0].astype(np.float64)
    mp_indices = mp[:, 1].astype(int)

    return MatrixProfileResult(
        method="stumpy_stump",
        mp_values=mp_values if include_profile else None,
        mp_indices=mp_indices if include_profile else None,
        subsequence_length=m,
        motifs=_top_motifs(series, mp, m, max_motifs, zone, include_subsequences),
        discords=_top_discords(series, mp, m, max_discords, zone, include_subsequences),
        mp_min=float(np.min(mp_values)),
        mp_max=float(np.max(mp_values)),
    )


def compute_distance_profile(
    query: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    """Compute the distance profile of a query against a series.

    Uses MASS (Mueen's Algorithm for Similarity Search) for fast computation.

    Parameters
    ----------
    query : np.ndarray
        Query subsequence
    series : np.ndarray
        Target time series

    Returns
    -------
    np.ndarray
        Distance profile (distance to each position in series)

    Examples
    --------
    >>> pattern = np.sin(np.linspace(0, 2*np.pi, 50))
    >>> x = np.random.randn(1000)
    >>> x[300:350] = pattern
    >>> dp = compute_distance_profile(pattern, x)
    >>> best_match = np.argmin(dp)
    >>> print(f"Best match at index: {best_match}")
    """
    query = as_univariate_series(query, name="query", min_length=3)
    series = as_univariate_series(series, min_length=len(query))

    return _get_stumpy().mass(query, series)
