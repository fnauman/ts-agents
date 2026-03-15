"""Core time series analysis library.

Submodules are imported lazily to keep lightweight CLI paths fast.
"""

from __future__ import annotations

from importlib import import_module

from .base import (
    AnalysisResult,
    DecompositionResult,
    ForecastResult,
    MultiForecastResult,
    PeakResult,
    MatrixProfileResult,
    RecurrenceResult,
    SegmentResult,
    ClassificationResult,
    SpectralResult,
    CoherenceResult,
    PeriodicityResult,
    DescriptiveStats,
)

_LAZY_SUBMODULES = {
    "decomposition",
    "forecasting",
    "patterns",
    "classification",
    "spectral",
    "statistics",
    "comparison",
}

_LAZY_COMPARISON_EXPORTS = {
    "ComparisonResult",
    "compare_methods",
    "compare_decomposition_methods",
    "compare_forecasting_methods",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in _LAZY_COMPARISON_EXPORTS:
        comparison = import_module(f"{__name__}.comparison")
        value = getattr(comparison, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Submodules
    "decomposition",
    "forecasting",
    "patterns",
    "classification",
    "spectral",
    "statistics",
    "comparison",
    # Result types
    "AnalysisResult",
    "DecompositionResult",
    "ForecastResult",
    "MultiForecastResult",
    "PeakResult",
    "MatrixProfileResult",
    "RecurrenceResult",
    "SegmentResult",
    "ClassificationResult",
    "SpectralResult",
    "CoherenceResult",
    "PeriodicityResult",
    "DescriptiveStats",
    # Comparison
    "ComparisonResult",
    "compare_methods",
    "compare_decomposition_methods",
    "compare_forecasting_methods",
]
