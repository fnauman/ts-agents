"""Matrix Profile analysis for time series.

This module provides functions for motif discovery (recurring patterns) and
discord detection (anomalies) using the STUMPY library.
"""

from typing import List, Optional
import numpy as np
import stumpy

from ..base import MatrixProfileResult, MotifResult, DiscordResult


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
    series = np.asarray(series, dtype=np.float64).flatten()
    return stumpy.stump(series, m)


def find_motifs(
    series: np.ndarray,
    m: int = 50,
    max_motifs: int = 3,
    exclusion_zone: Optional[int] = None,
    include_subsequences: bool = False,
) -> List[MotifResult]:
    """Find the top motifs (recurring patterns) in a time series.

    Motifs are subsequences with the smallest matrix profile values,
    meaning they have very similar patterns elsewhere in the series.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length
    max_motifs : int
        Maximum number of motifs to return
    exclusion_zone : int, optional
        Minimum distance between motifs. Default is m // 2.
    include_subsequences : bool
        Whether to include the actual subsequence arrays

    Returns
    -------
    List[MotifResult]
        List of motif results, sorted by distance (best first)

    Examples
    --------
    >>> # Create a series with a repeating pattern
    >>> pattern = np.sin(np.linspace(0, 2*np.pi, 50))
    >>> x = np.random.randn(1000)
    >>> x[100:150] = pattern
    >>> x[500:550] = pattern * 1.05  # Similar pattern
    >>> motifs = find_motifs(x, m=50, max_motifs=3)
    >>> print(f"Found {len(motifs)} motifs")
    """
    series = np.asarray(series, dtype=np.float64).flatten()

    if exclusion_zone is None:
        exclusion_zone = m // 2

    mp = stumpy.stump(series, m)
    mp_values = mp[:, 0].copy()
    mp_indices = mp[:, 1].astype(int)

    motifs = []
    for _ in range(max_motifs):
        # Find minimum
        idx = int(np.argmin(mp_values))
        val = float(mp_values[idx])

        if val == np.inf:
            break

        neighbor_idx = int(mp_indices[idx])

        motif = MotifResult(
            index=idx,
            neighbor_index=neighbor_idx,
            distance=val,
        )

        if include_subsequences:
            motif.subsequence = series[idx : idx + m].copy()

        motifs.append(motif)

        # Apply exclusion zone
        start = max(0, idx - exclusion_zone)
        end = min(len(mp_values), idx + exclusion_zone)
        mp_values[start:end] = np.inf

    return motifs


def find_discords(
    series: np.ndarray,
    m: int = 50,
    max_discords: int = 3,
    exclusion_zone: Optional[int] = None,
    include_subsequences: bool = False,
) -> List[DiscordResult]:
    """Find the top discords (anomalies) in a time series.

    Discords are subsequences with the largest matrix profile values,
    meaning they are most different from all other subsequences.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length
    max_discords : int
        Maximum number of discords to return
    exclusion_zone : int, optional
        Minimum distance between discords. Default is m // 2.
    include_subsequences : bool
        Whether to include the actual subsequence arrays

    Returns
    -------
    List[DiscordResult]
        List of discord results, sorted by distance (worst first)

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 20*np.pi, 1000))
    >>> x[500:550] += 5  # Inject an anomaly
    >>> discords = find_discords(x, m=50, max_discords=3)
    >>> print(f"Most anomalous index: {discords[0].index}")
    """
    series = np.asarray(series, dtype=np.float64).flatten()

    if exclusion_zone is None:
        exclusion_zone = m // 2

    mp = stumpy.stump(series, m)
    mp_values = mp[:, 0].copy()

    discords = []
    for _ in range(max_discords):
        # Find maximum
        idx = int(np.argmax(mp_values))
        val = float(mp_values[idx])

        if val == -np.inf:
            break

        discord = DiscordResult(
            index=idx,
            distance=val,
        )

        if include_subsequences:
            discord.subsequence = series[idx : idx + m].copy()

        discords.append(discord)

        # Apply exclusion zone
        start = max(0, idx - exclusion_zone)
        end = min(len(mp_values), idx + exclusion_zone)
        mp_values[start:end] = -np.inf

    return discords


def analyze_matrix_profile(
    series: np.ndarray,
    m: int = 50,
    max_motifs: int = 3,
    max_discords: int = 3,
    include_subsequences: bool = False,
    subsequence_length: Optional[int] = None,
    n_motifs: Optional[int] = None,
    n_discords: Optional[int] = None,
) -> MatrixProfileResult:
    """Perform comprehensive Matrix Profile analysis.

    Finds both motifs (recurring patterns) and discords (anomalies).

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    m : int
        Subsequence length
    max_motifs : int
        Maximum number of motifs to find
    max_discords : int
        Maximum number of discords to find
    include_subsequences : bool
        Whether to include subsequence arrays

    Returns
    -------
    MatrixProfileResult
        Comprehensive result with matrix profile and pattern analysis

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

    series = np.asarray(series, dtype=np.float64).flatten()

    mp = stumpy.stump(series, m)
    mp_values = mp[:, 0]
    mp_indices = mp[:, 1].astype(int)

    motifs = find_motifs(
        series, m=m, max_motifs=max_motifs, include_subsequences=include_subsequences
    )
    discords = find_discords(
        series, m=m, max_discords=max_discords, include_subsequences=include_subsequences
    )

    return MatrixProfileResult(
        method="stumpy_stump",
        mp_values=mp_values,
        mp_indices=mp_indices,
        subsequence_length=m,
        motifs=motifs,
        discords=discords,
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
    query = np.asarray(query, dtype=np.float64).flatten()
    series = np.asarray(series, dtype=np.float64).flatten()

    return stumpy.mass(query, series)
