"""Spectral analysis for time series.

This module provides functions for:
- Power Spectral Density (PSD) estimation
- Periodicity detection
- Coherence analysis between signals
- Spectral slope estimation (for turbulence)
"""

from .psd import (
    compute_psd,
    detect_periodicity,
    compute_spectral_centroid,
)

from .coherence import (
    compute_coherence,
    compute_cross_spectrum,
    compute_phase_coherence,
)

__all__ = [
    # PSD
    "compute_psd",
    "detect_periodicity",
    "compute_spectral_centroid",
    # Coherence
    "compute_coherence",
    "compute_cross_spectrum",
    "compute_phase_coherence",
]
