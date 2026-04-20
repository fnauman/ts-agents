"""Base classes and result types for the core analysis library.

This module provides the foundational types used across all analysis modules.
All result classes are dataclasses that can be easily serialized and compared.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import numpy as np


class AnalysisCategory(Enum):
    """Categories of time series analysis."""
    DECOMPOSITION = "decomposition"
    FORECASTING = "forecasting"
    PATTERNS = "patterns"
    CLASSIFICATION = "classification"
    SPECTRAL = "spectral"
    STATISTICS = "statistics"


@dataclass
class AnalysisResult:
    """Base class for all analysis results.

    All analysis functions should return a subclass of this that includes:
    - The computed results
    - Metrics for comparison with other methods
    - Metadata about the computation
    """
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


# =============================================================================
# Decomposition Results
# =============================================================================

@dataclass
class DecompositionResult(AnalysisResult):
    """Result of time series decomposition."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    period: int

    # Metrics for comparison
    residual_variance: float = 0.0
    trend_smoothness: float = 0.0
    seasonal_strength: float = 0.0

    def __post_init__(self):
        """Compute metrics if not provided."""
        if self.residual_variance == 0.0 and len(self.residual) > 0:
            self.residual_variance = float(np.var(self.residual[~np.isnan(self.residual)]))
        if self.trend_smoothness == 0.0 and len(self.trend) > 1:
            self.trend_smoothness = compute_smoothness(self.trend)
        if self.seasonal_strength == 0.0:
            self.seasonal_strength = compute_seasonal_strength(
                self.seasonal, self.residual
            )


# =============================================================================
# Forecasting Results
# =============================================================================

@dataclass
class ForecastResult(AnalysisResult):
    """Result of time series forecasting."""
    forecast: np.ndarray
    horizon: int

    # Optional prediction intervals
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence_level: float = 0.95

    # Metrics (if validation data available)
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


@dataclass
class MultiForecastResult(AnalysisResult):
    """Result containing forecasts from multiple models."""
    forecasts: Dict[str, np.ndarray]
    horizon: int

    def get_ensemble(self, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Get weighted ensemble of forecasts."""
        if weights is None:
            weights = {k: 1.0 / len(self.forecasts) for k in self.forecasts}

        result = np.zeros(self.horizon)
        for model, forecast in self.forecasts.items():
            result += weights.get(model, 0) * forecast
        return result


# =============================================================================
# Pattern Detection Results
# =============================================================================

@dataclass
class PeakResult(AnalysisResult):
    """Result of peak detection."""
    peak_indices: np.ndarray
    peak_values: np.ndarray
    count: int

    # Statistics
    mean_spacing: float = 0.0
    std_spacing: float = 0.0
    spacing_cv: float = 0.0  # Coefficient of variation
    regularity: str = "N/A"
    mean_height: float = 0.0
    mean_prominence: Optional[float] = None


@dataclass
class MotifResult:
    """A single motif (recurring pattern) found in the time series."""
    index: int
    neighbor_index: int
    distance: float
    subsequence: Optional[np.ndarray] = None


@dataclass
class DiscordResult:
    """A single discord (anomaly) found in the time series."""
    index: int
    distance: float
    subsequence: Optional[np.ndarray] = None


@dataclass
class MatrixProfileResult(AnalysisResult):
    """Result of matrix profile analysis."""
    mp_values: np.ndarray
    mp_indices: np.ndarray
    subsequence_length: int

    motifs: List[MotifResult] = field(default_factory=list)
    discords: List[DiscordResult] = field(default_factory=list)

    mp_min: float = 0.0
    mp_max: float = 0.0


@dataclass
class RecurrenceResult(AnalysisResult):
    """Result of recurrence analysis."""
    recurrence_matrix: np.ndarray
    threshold: float

    # RQA metrics
    recurrence_rate: float = 0.0
    determinism: float = 0.0
    laminarity: float = 0.0


@dataclass
class SegmentResult(AnalysisResult):
    """Result of time series segmentation."""
    changepoints: List[int]
    n_segments: int

    # Per-segment statistics
    segment_stats: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Classification Results
# =============================================================================

@dataclass
class ClassificationResult(AnalysisResult):
    """Result of time series classification."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None

    # Metrics (if test labels provided)
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Spectral Results
# =============================================================================

@dataclass
class SpectralResult(AnalysisResult):
    """Result of spectral analysis (PSD)."""
    frequencies: np.ndarray
    psd: np.ndarray

    # Turbulence metrics
    spectral_slope: float = 0.0
    slope_intercept: float = 0.0

    # Dominant frequency
    dominant_frequency: float = 0.0
    dominant_period: float = 0.0


@dataclass
class CoherenceResult(AnalysisResult):
    """Result of coherence analysis between two signals."""
    frequencies: np.ndarray
    coherence: np.ndarray

    mean_coherence: float = 0.0
    max_coherence: float = 0.0
    dominant_frequency: float = 0.0


@dataclass
class PeriodicityResult(AnalysisResult):
    """Result of periodicity detection."""
    dominant_period: float
    dominant_frequency: float
    confidence: float

    top_periods: List[float] = field(default_factory=list)


# =============================================================================
# Statistics Results
# =============================================================================

@dataclass
class DescriptiveStats(AnalysisResult):
    """Descriptive statistics of a time series."""
    length: int
    mean: float
    std: float
    min: float
    max: float
    rms: float

    # Optional additional stats
    median: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


# =============================================================================
# Helper Functions
# =============================================================================

def compute_smoothness(series: np.ndarray) -> float:
    """Compute smoothness of a series (lower = smoother).

    Uses second derivative as a measure of curvature.
    """
    if len(series) < 3:
        return 0.0

    # Remove NaNs
    valid = series[~np.isnan(series)]
    if len(valid) < 3:
        return 0.0

    # Second difference (discrete approximation of second derivative)
    second_diff = np.diff(valid, n=2)
    return float(np.std(second_diff))


def compute_seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    """Compute strength of seasonality (0-1 scale).

    Based on: 1 - Var(residual) / Var(seasonal + residual)
    """
    # Remove NaNs
    s = seasonal[~np.isnan(seasonal)]
    r = residual[~np.isnan(residual)]

    if len(s) == 0 or len(r) == 0:
        return 0.0

    # Align lengths
    min_len = min(len(s), len(r))
    s = s[:min_len]
    r = r[:min_len]

    var_remainder = np.var(r)
    var_detrended = np.var(s + r)

    if var_detrended == 0:
        return 0.0

    strength = max(0, 1 - var_remainder / var_detrended)
    return float(strength)
