"""Pattern detection and analysis for time series.

This module provides functions for:
- Peak detection and analysis
- Recurrence plot analysis and RQA metrics
- Matrix Profile analysis (motifs and discords)
- Time series segmentation (regime detection)
"""

from .peaks import (
    detect_peaks,
    count_peaks,
    find_peak_properties,
)

from .recurrence import (
    compute_recurrence_matrix,
    compute_rqa_metrics,
    analyze_recurrence,
)

from .matrix_profile import (
    compute_matrix_profile,
    find_motifs,
    find_discords,
    analyze_matrix_profile,
    compute_distance_profile,
)

from .segmentation import (
    segment_fluss,
    segment_changepoint,
)

__all__ = [
    # Peaks
    "detect_peaks",
    "count_peaks",
    "find_peak_properties",
    # Recurrence
    "compute_recurrence_matrix",
    "compute_rqa_metrics",
    "analyze_recurrence",
    # Matrix Profile
    "compute_matrix_profile",
    "find_motifs",
    "find_discords",
    "analyze_matrix_profile",
    "compute_distance_profile",
    # Segmentation
    "segment_fluss",
    "segment_changepoint",
]
