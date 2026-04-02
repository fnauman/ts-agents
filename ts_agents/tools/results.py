"""Structured Result Types and Formatting Layer.

This module provides:
- ToolResult: Base class for structured tool results
- Specialized result types for different tool categories
- ResultFormatter: Convert structured results to LLM-friendly strings

The structured format enables:
1. IPC serialization for sandboxed execution
2. Consistent result handling across agents
3. Separation of computation from presentation

Example usage:
    >>> from ts_agents.tools.results import ToolResult, ResultFormatter
    >>>
    >>> # Tools return structured results
    >>> result = DecompositionResult(
    ...     trend=trend_array,
    ...     seasonal=seasonal_array,
    ...     residual=residual_array,
    ...     period=12,
    ...     visualization={"format": "png", "data": base64_data}
    ... )
    >>>
    >>> # Format for LLM consumption
    >>> formatter = ResultFormatter()
    >>> formatted = formatter.format(result)
"""

from __future__ import annotations

import base64
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from ts_agents.cli.output import dump_json
from ts_agents.contracts import ArtifactRef, ToolPayload


def _serialize_value(value: Any) -> Any:
    """Convert numpy arrays and other types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return value


def serialize_result(result: Any) -> Any:
    """Serialize a result for JSON/IPC transport."""
    if result is None:
        return None

    if isinstance(result, ToolResult):
        return result.to_dict()

    if hasattr(result, "to_dict") and callable(result.to_dict):
        return serialize_result(result.to_dict())

    if is_dataclass(result):
        return serialize_result(asdict(result))

    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}

    if isinstance(result, (list, tuple)):
        return [serialize_result(v) for v in result]

    return _serialize_value(result)


def _is_tool_payload_dict(result: Any) -> bool:
    return isinstance(result, dict) and "summary" in result and "data" in result and "artifacts" in result


def _format_artifact_ref(artifact: Any) -> str:
    if isinstance(artifact, ArtifactRef):
        artifact = asdict(artifact)
    description = artifact.get("description") or artifact.get("kind") or "artifact"
    mime_type = artifact.get("mime_type")
    suffix = f" ({mime_type})" if mime_type else ""
    return f"- {description}: {artifact.get('path')}{suffix}"


def _format_tool_payload_sections(
    *,
    summary: str,
    data: Any,
    artifacts: Any,
    warnings: Any,
    format_value,
) -> str:
    parts = [summary or "(no summary)"]

    warnings = warnings or []
    if warnings:
        parts.append("Warnings:")
        parts.extend(f"- {warning}" for warning in warnings)

    artifacts = artifacts or []
    if artifacts:
        parts.append("Artifacts:")
        parts.extend(_format_artifact_ref(artifact) for artifact in artifacts)

    if data not in ({}, [], None):
        rendered_data = format_value(data)
        if rendered_data and rendered_data != "(no result)":
            parts.append("Data:")
            parts.append(rendered_data)

    return "\n".join(parts)


@dataclass
class ToolResult(ABC):
    """Base class for structured tool results.

    All tool results should inherit from this class to ensure
    consistent serialization and formatting.
    """

    @abstractmethod
    def get_summary(self) -> str:
        """Get a brief summary for LLM consumption."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in asdict(self).items():
            result[key] = _serialize_value(value)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return dump_json(self.to_dict(), indent=None)


@dataclass
class Visualization:
    """Visualization data for tool results.

    Attributes
    ----------
    format : str
        Image format (png, svg, etc.)
    data : str
        Base64-encoded image data
    width : int, optional
        Image width in pixels
    height : int, optional
        Image height in pixels
    title : str, optional
        Plot title
    """
    format: str = "png"
    data: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format,
            "data": self.data,
            "width": self.width,
            "height": self.height,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Visualization":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_matplotlib(cls, fig, title: Optional[str] = None) -> "Visualization":
        """Create visualization from matplotlib figure."""
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return cls(
            format="png",
            data=img_base64,
            title=title,
        )


@dataclass
class DecompositionResult(ToolResult):
    """Result from decomposition tools (STL, MSTL, Holt-Winters).

    Attributes
    ----------
    trend : List[float]
        Trend component
    seasonal : List[float]
        Seasonal component
    residual : List[float]
        Residual component
    period : int, optional
        Detected or used period
    method : str
        Decomposition method used
    residual_variance : float
        Variance of residual component
    visualization : Visualization, optional
        Plot of decomposition
    metadata : Dict[str, Any]
        Additional metadata
    """
    trend: List[float] = field(default_factory=list)
    seasonal: List[float] = field(default_factory=list)
    residual: List[float] = field(default_factory=list)
    period: Optional[int] = None
    method: str = "stl"
    residual_variance: float = 0.0
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [
            f"{self.method.upper()} Decomposition:",
            f"- Period: {self.period}" if self.period else "- Period: auto-detected",
            f"- Residual variance: {self.residual_variance:.4f}",
            f"- Series length: {len(self.trend)}",
        ]
        return "\n".join(lines)


@dataclass
class ForecastResult(ToolResult):
    """Result from forecasting tools.

    Attributes
    ----------
    forecast : List[float]
        Forecasted values
    lower_bound : List[float], optional
        Lower confidence bound
    upper_bound : List[float], optional
        Upper confidence bound
    horizon : int
        Forecast horizon
    method : str
        Forecasting method used
    confidence_level : float
        Confidence level for bounds
    visualization : Visualization, optional
        Forecast plot
    metadata : Dict[str, Any]
        Additional metadata (model params, etc.)
    """
    forecast: List[float] = field(default_factory=list)
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    horizon: int = 0
    method: str = "arima"
    confidence_level: float = 0.95
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [
            f"{self.method.upper()} Forecast:",
            f"- Horizon: {self.horizon}",
            f"- First 5 values: {self.forecast[:5]}",
        ]
        if self.lower_bound:
            lines.append(f"- Confidence level: {self.confidence_level * 100:.0f}%")
        return "\n".join(lines)


@dataclass
class PeakResult(ToolResult):
    """Result from peak detection tools.

    Attributes
    ----------
    peak_indices : List[int]
        Indices of detected peaks
    peak_values : List[float]
        Values at peak positions
    count : int
        Number of peaks detected
    mean_spacing : float
        Mean spacing between peaks
    regularity : str
        Regularity classification (regular, irregular, etc.)
    visualization : Visualization, optional
        Peak detection plot
    metadata : Dict[str, Any]
        Additional metadata
    """
    peak_indices: List[int] = field(default_factory=list)
    peak_values: List[float] = field(default_factory=list)
    count: int = 0
    mean_spacing: float = 0.0
    regularity: str = "unknown"
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [
            f"Peak Detection:",
            f"- Peaks detected: {self.count}",
            f"- Mean spacing: {self.mean_spacing:.2f}",
            f"- Regularity: {self.regularity}",
        ]
        return "\n".join(lines)


@dataclass
class SpectralResult(ToolResult):
    """Result from spectral analysis tools.

    Attributes
    ----------
    frequencies : List[float]
        Frequency values
    psd : List[float]
        Power spectral density
    dominant_frequency : float
        Most prominent frequency
    dominant_period : float
        Period corresponding to dominant frequency
    spectral_slope : float
        Slope of log-log PSD
    visualization : Visualization, optional
        PSD plot
    metadata : Dict[str, Any]
        Additional metadata
    """
    frequencies: List[float] = field(default_factory=list)
    psd: List[float] = field(default_factory=list)
    dominant_frequency: float = 0.0
    dominant_period: float = 0.0
    spectral_slope: float = 0.0
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [
            f"Spectral Analysis:",
            f"- Dominant frequency: {self.dominant_frequency:.4f}",
            f"- Dominant period: {self.dominant_period:.2f}",
            f"- Spectral slope: {self.spectral_slope:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class MotifResult(ToolResult):
    """Result for a single motif (repeating pattern).

    Attributes
    ----------
    index : int
        Starting index of the motif
    neighbor_index : int
        Index of the most similar subsequence
    distance : float
        Distance to the nearest neighbor
    subsequence : List[float], optional
        The actual subsequence values
    """
    index: int = 0
    neighbor_index: int = 0
    distance: float = 0.0
    subsequence: Optional[List[float]] = None

    def get_summary(self) -> str:
        """Get a brief summary."""
        return f"Motif at {self.index} matches {self.neighbor_index} (dist={self.distance:.4f})"


@dataclass
class DiscordResult(ToolResult):
    """Result for a single discord (anomaly).

    Attributes
    ----------
    index : int
        Starting index of the discord
    distance : float
        Distance to nearest neighbor (higher = more anomalous)
    subsequence : List[float], optional
        The actual subsequence values
    """
    index: int = 0
    distance: float = 0.0
    subsequence: Optional[List[float]] = None

    def get_summary(self) -> str:
        """Get a brief summary."""
        return f"Discord at {self.index} (dist={self.distance:.4f})"


@dataclass
class MatrixProfileResult(ToolResult):
    """Result from matrix profile analysis.

    Attributes
    ----------
    mp_values : List[float]
        Matrix profile values
    mp_indices : List[int]
        Matrix profile indices
    motifs : List[MotifResult]
        Top motifs found
    discords : List[DiscordResult]
        Top discords found
    window_size : int
        Window size used
    visualization : Visualization, optional
        Matrix profile plot
    metadata : Dict[str, Any]
        Additional metadata
    """
    mp_values: List[float] = field(default_factory=list)
    mp_indices: List[int] = field(default_factory=list)
    motifs: List[MotifResult] = field(default_factory=list)
    discords: List[DiscordResult] = field(default_factory=list)
    window_size: int = 50
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [f"Matrix Profile (m={self.window_size}):"]
        if self.motifs:
            m = self.motifs[0]
            lines.append(f"- Top motif: {m.index} matches {m.neighbor_index} (dist={m.distance:.4f})")
        if self.discords:
            d = self.discords[0]
            lines.append(f"- Top discord: {d.index} (dist={d.distance:.4f})")
        return "\n".join(lines)


@dataclass
class ChangePointResult(ToolResult):
    """Result from changepoint detection.

    Attributes
    ----------
    changepoints : List[int]
        Indices of detected changepoints
    n_segments : int
        Number of segments
    method : str
        Method used
    visualization : Visualization, optional
        Changepoint plot
    metadata : Dict[str, Any]
        Additional metadata
    """
    changepoints: List[int] = field(default_factory=list)
    n_segments: int = 0
    method: str = "pelt"
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        return f"Changepoint Detection: {len(self.changepoints)} changepoints at {self.changepoints}"


@dataclass
class StatisticsResult(ToolResult):
    """Result from statistical analysis.

    Attributes
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    rms : float
        Root mean square
    skewness : float
        Skewness
    kurtosis : float
        Kurtosis
    n_points : int
        Number of data points
    visualization : Visualization, optional
        Statistics plot
    metadata : Dict[str, Any]
        Extended statistics
    """
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    rms: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    n_points: int = 0
    visualization: Optional[Visualization] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        lines = [
            f"Descriptive Statistics (n={self.n_points}):",
            f"- Mean: {self.mean:.4f}",
            f"- Std: {self.std:.4f}",
            f"- Range: [{self.min_val:.4f}, {self.max_val:.4f}]",
            f"- RMS: {self.rms:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class ScalarResult(ToolResult):
    """Result for scalar values (score, count, estimated period, etc.).

    Attributes
    ----------
    value : float
        The scalar value
    name : str
        Name of the metric
    interpretation : str
        Human-readable interpretation
    metadata : Dict[str, Any]
        Additional context
    """
    value: float = 0.0
    name: str = ""
    interpretation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a brief summary."""
        if self.interpretation:
            return f"{self.name}: {self.value:.4f} ({self.interpretation})"
        return f"{self.name}: {self.value:.4f}"


@dataclass
class GenericResult(ToolResult):
    """Generic result for tools that don't fit other categories.

    Attributes
    ----------
    data : Dict[str, Any]
        Result data
    summary_text : str
        Pre-formatted summary text
    visualization : Visualization, optional
        Optional visualization
    """
    data: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    visualization: Optional[Visualization] = None

    def get_summary(self) -> str:
        """Get a brief summary."""
        return self.summary_text if self.summary_text else str(self.data)


class ResultFormatter:
    """Format structured results for LLM consumption.

    This formatter converts structured ToolResult objects into
    strings suitable for agent consumption, including embedded
    image data when visualizations are present.

    Parameters
    ----------
    include_images : bool
        Whether to include base64 image data
    max_array_length : int
        Maximum number of array elements to include

    Examples
    --------
    >>> formatter = ResultFormatter()
    >>> result = DecompositionResult(...)
    >>> formatted = formatter.format(result)
    """

    def __init__(
        self,
        include_images: bool = True,
        max_array_length: int = 10,
    ):
        self.include_images = include_images
        self.max_array_length = max_array_length

    def format(self, result: Union[ToolResult, Dict[str, Any], str, Any]) -> str:
        """Format a result for LLM consumption.

        Parameters
        ----------
        result : Union[ToolResult, Dict, str, Any]
            The result to format

        Returns
        -------
        str
            Formatted output string
        """
        if isinstance(result, ToolPayload):
            return self._format_tool_payload(result)

        if isinstance(result, ArtifactRef):
            return _format_artifact_ref(result)

        # Handle string results (legacy format)
        if isinstance(result, str):
            return result

        # Handle structured results
        if isinstance(result, ToolResult):
            return self._format_tool_result(result)

        # Handle objects with to_dict (e.g., AnalysisResult)
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return self._format_dict_result(serialize_result(result.to_dict()))

        # Handle dataclasses
        if is_dataclass(result):
            if _is_tool_payload_dict(serialize_result(result)):
                payload = serialize_result(result)
                return _format_tool_payload_sections(
                    summary=payload.get("summary", ""),
                    data=payload.get("data", {}),
                    artifacts=payload.get("artifacts", []),
                    warnings=payload.get("warnings", []),
                    format_value=self._format_payload_data,
                )
            return self._format_dict_result(serialize_result(result))

        # Handle dict results (transitional format)
        if isinstance(result, dict):
            return self._format_dict_result(result)

        # Fallback
        return str(result)

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format a ToolResult."""
        parts = [result.get_summary()]

        # Add visualization if present
        if self.include_images:
            viz = getattr(result, 'visualization', None)
            if viz and viz.data:
                parts.append(f"\n[IMAGE_DATA:{viz.data}]")

        return "\n".join(parts)

    def _format_dict_result(self, result: Dict[str, Any]) -> str:
        """Format a dictionary result."""
        if _is_tool_payload_dict(result):
            return _format_tool_payload_sections(
                summary=result.get("summary", ""),
                data=result.get("data", {}),
                artifacts=result.get("artifacts", []),
                warnings=result.get("warnings", []),
                format_value=self._format_payload_data,
            )

        parts = []

        # Handle summary or result field
        if "summary" in result:
            parts.append(str(result["summary"]))
        elif "result" in result:
            parts.append(str(result["result"]))
        else:
            # Format key-value pairs
            for key, value in result.items():
                if key not in ("visualization", "image_data", "metadata"):
                    if isinstance(value, (list, np.ndarray)):
                        value = self._truncate_array(value)
                    parts.append(f"{key}: {value}")

        # Handle visualization
        if self.include_images and "visualization" in result:
            viz = result["visualization"]
            if isinstance(viz, dict) and viz.get("data"):
                parts.append(f"\n[IMAGE_DATA:{viz['data']}]")
            elif isinstance(viz, Visualization) and viz.data:
                parts.append(f"\n[IMAGE_DATA:{viz.data}]")

        return "\n".join(parts) if parts else str(result)

    def _format_tool_payload(self, result: ToolPayload) -> str:
        return _format_tool_payload_sections(
            summary=result.summary,
            data=result.data,
            artifacts=result.artifacts,
            warnings=result.warnings,
            format_value=self._format_payload_data,
        )

    def _format_payload_data(self, value: Any) -> str:
        if value is None:
            return "(no result)"
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if _is_tool_payload_dict(value):
                return _format_tool_payload_sections(
                    summary=value.get("summary", ""),
                    data=value.get("data", {}),
                    artifacts=value.get("artifacts", []),
                    warnings=value.get("warnings", []),
                    format_value=self._format_payload_data,
                )
            return self._format_dict_result(value)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]"
            if len(value) > self.max_array_length:
                return self._truncate_array(value)
            return str(serialize_result(value))
        if isinstance(value, np.ndarray):
            return self._truncate_array(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return self._format_payload_data(serialize_result(value.to_dict()))
        if is_dataclass(value):
            return self._format_payload_data(serialize_result(value))
        return str(serialize_result(value))

    def _truncate_array(self, arr: Union[List, np.ndarray]) -> str:
        """Truncate array for display."""
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()

        if len(arr) <= self.max_array_length:
            return str(arr)

        head = arr[:self.max_array_length // 2]
        tail = arr[-(self.max_array_length // 2):]
        return f"{head} ... {tail} (length={len(arr)})"


# Default formatter instance
_default_formatter: Optional[ResultFormatter] = None


def get_formatter() -> ResultFormatter:
    """Get the default formatter instance."""
    global _default_formatter
    if _default_formatter is None:
        _default_formatter = ResultFormatter()
    return _default_formatter


def format_result(result: Union[ToolResult, Dict[str, Any], str, Any]) -> str:
    """Convenience function to format a result using the default formatter."""
    return get_formatter().format(result)
