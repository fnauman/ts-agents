"""Peak detection and analysis for time series.

This module provides functions to detect, count, and analyze peaks (local maxima)
in time series data using scipy's signal processing capabilities.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.signal import find_peaks

from ..base import PeakResult


def detect_peaks(
    series: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[float] = None,
    threshold: Optional[float] = None,
) -> PeakResult:
    """Detect peaks in a time series with detailed statistics.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    height : float, optional
        Minimum height of peaks. Can be a number or array.
    distance : int, optional
        Minimum horizontal distance (in samples) between peaks.
    prominence : float, optional
        Minimum prominence of peaks.
    width : float, optional
        Minimum width of peaks.
    threshold : float, optional
        Minimum threshold for peak detection.

    Returns
    -------
    PeakResult
        Result containing peak indices, values, and statistics.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000) * 0.1
    >>> result = detect_peaks(x, distance=50, prominence=0.5)
    >>> print(f"Found {result.count} peaks")
    """
    series = np.asarray(series).flatten()

    # Find peaks with all specified parameters
    peaks, properties = find_peaks(
        series,
        height=height,
        distance=distance,
        prominence=prominence,
        width=width,
        threshold=threshold,
    )

    # Extract peak values
    peak_values = series[peaks]

    # Compute statistics
    count = len(peaks)

    if count == 0:
        return PeakResult(
            method="scipy_find_peaks",
            peak_indices=np.array([], dtype=int),
            peak_values=np.array([]),
            count=0,
        )

    # Spacing statistics
    if count > 1:
        spacings = np.diff(peaks)
        mean_spacing = float(np.mean(spacings))
        std_spacing = float(np.std(spacings))
        spacing_cv = std_spacing / mean_spacing if mean_spacing > 0 else 0.0
    else:
        mean_spacing = 0.0
        std_spacing = 0.0
        spacing_cv = 0.0

    # Regularity classification
    if spacing_cv < 0.2:
        regularity = "High (Quasi-periodic)"
    elif spacing_cv < 0.5:
        regularity = "Moderate"
    else:
        regularity = "Low (Irregular)"

    # Height statistics
    if "peak_heights" in properties:
        mean_height = float(np.mean(properties["peak_heights"]))
    else:
        mean_height = float(np.mean(peak_values))

    # Prominence
    mean_prominence = None
    if "prominences" in properties:
        mean_prominence = float(np.mean(properties["prominences"]))

    return PeakResult(
        method="scipy_find_peaks",
        peak_indices=peaks,
        peak_values=peak_values,
        count=count,
        mean_spacing=mean_spacing,
        std_spacing=std_spacing,
        spacing_cv=spacing_cv,
        regularity=regularity,
        mean_height=mean_height,
        mean_prominence=mean_prominence,
    )


def count_peaks(
    series: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
) -> int:
    """Count the number of peaks in a time series.

    This is a convenience function that returns just the count.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    height : float, optional
        Minimum height of peaks
    distance : int, optional
        Minimum horizontal distance between peaks
    prominence : float, optional
        Minimum prominence of peaks

    Returns
    -------
    int
        Number of peaks found

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> n = count_peaks(x, distance=50)
    """
    result = detect_peaks(series, height=height, distance=distance, prominence=prominence)
    return result.count


def find_peak_properties(
    series: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Find peaks and return raw scipy properties.

    Lower-level function that returns scipy's native format.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    height : float, optional
        Minimum height of peaks
    distance : int, optional
        Minimum horizontal distance between peaks

    Returns
    -------
    peaks : np.ndarray
        Indices of peaks
    properties : dict
        Dictionary of peak properties from scipy
    """
    series = np.asarray(series).flatten()
    return find_peaks(series, height=height, distance=distance)
