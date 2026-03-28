"""Pattern detection and analysis for time series."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "detect_peaks": ("peaks", "detect_peaks"),
    "count_peaks": ("peaks", "count_peaks"),
    "find_peak_properties": ("peaks", "find_peak_properties"),
    "compute_recurrence_matrix": ("recurrence", "compute_recurrence_matrix"),
    "compute_rqa_metrics": ("recurrence", "compute_rqa_metrics"),
    "analyze_recurrence": ("recurrence", "analyze_recurrence"),
    "compute_matrix_profile": ("matrix_profile", "compute_matrix_profile"),
    "find_motifs": ("matrix_profile", "find_motifs"),
    "find_discords": ("matrix_profile", "find_discords"),
    "analyze_matrix_profile": ("matrix_profile", "analyze_matrix_profile"),
    "compute_distance_profile": ("matrix_profile", "compute_distance_profile"),
    "segment_fluss": ("segmentation", "segment_fluss"),
    "segment_changepoint": ("segmentation", "segment_changepoint"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

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
