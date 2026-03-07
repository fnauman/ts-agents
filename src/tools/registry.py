"""Tool Registry - Central management of all analysis tools with metadata.

This module provides:
- ToolMetadata: Dataclass for tool information
- ToolCategory: Enum for categorizing tools
- ComputationalCost: Enum for estimated runtime
- ToolRegistry: Central registry for all tools
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Sequence
from enum import Enum
import inspect
import threading
import numpy as np


class ToolCategory(Enum):
    """Categories of analysis tools."""
    DECOMPOSITION = "decomposition"
    FORECASTING = "forecasting"
    PATTERNS = "patterns"
    CLASSIFICATION = "classification"
    SPECTRAL = "spectral"
    COMPLEXITY = "complexity"
    STATISTICS = "statistics"
    COMPARISON = "comparison"
    DATA = "data"


class ComputationalCost(Enum):
    """Estimated computational cost/runtime of tools.

    LOW: < 1 second
    MEDIUM: 1-30 seconds
    HIGH: 30s - 5 minutes
    VERY_HIGH: > 5 minutes, may require user approval
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Cost ordering for comparison
_COST_ORDER = [
    ComputationalCost.LOW,
    ComputationalCost.MEDIUM,
    ComputationalCost.HIGH,
    ComputationalCost.VERY_HIGH,
]


@dataclass
class ToolParameter:
    """Description of a single tool parameter."""
    name: str
    type: str
    description: str
    optional: bool = False
    default: Any = None


@dataclass
class ToolMetadata:
    """Metadata for a registered tool.

    Attributes
    ----------
    name : str
        Unique identifier for the tool
    description : str
        Human-readable description for LLM prompts
    category : ToolCategory
        Analysis category
    cost : ComputationalCost
        Estimated runtime
    core_function : Callable
        The underlying core function to call
    parameters : List[ToolParameter]
        Parameter descriptions
    dependencies : List[str]
        Optional package dependencies
    examples : List[str]
        Example usage prompts for agents
    returns : str
        Description of return type
    timeout_seconds : int
        Maximum execution time in seconds (for sandbox)
    memory_mb : int
        Memory limit in MB (for sandbox)
    disk_mb : int
        Disk limit in MB (for sandbox)
    input_validation_fn : Callable, optional
        Custom validation function for parameters
    """
    name: str
    description: str
    category: ToolCategory
    cost: ComputationalCost
    core_function: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    returns: str = ""
    timeout_seconds: int = 300
    memory_mb: int = 512
    disk_mb: int = 100
    input_validation_fn: Optional[Callable] = None

    def get_signature(self) -> str:
        """Get function signature as string."""
        sig = inspect.signature(self.core_function)
        return f"{self.name}{sig}"

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for tool definitions."""
        properties = {}
        required = []

        for param in self.parameters:
            param_schema = {
                "type": _python_type_to_json_type(param.type),
                "description": param.description,
            }
            if param.default is not None:
                param_schema["default"] = param.default
            properties[param.name] = param_schema

            if not param.optional:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def _python_type_to_json_type(type_str: str) -> str:
    """Convert Python type string to JSON schema type."""
    mapping = {
        "int": "integer",
        "float": "number",
        "str": "string",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "np.ndarray": "array",
        "numpy.ndarray": "array",
        "ndarray": "array",
    }
    return mapping.get(type_str.lower(), "string")


class ToolRegistry:
    """Central registry of all analysis tools.

    This is a singleton-style registry that maintains metadata for all
    registered tools. Tools can be queried by name, category, or cost.

    Examples
    --------
    >>> from ts_agents.tools.registry import ToolRegistry, ToolMetadata
    >>>
    >>> # Get a tool
    >>> tool = ToolRegistry.get("stl_decompose")
    >>> print(tool.description)
    >>>
    >>> # List tools by category
    >>> decomp_tools = ToolRegistry.list_by_category(ToolCategory.DECOMPOSITION)
    >>>
    >>> # Filter by computational cost
    >>> fast_tools = ToolRegistry.list_by_max_cost(ComputationalCost.LOW)
    """

    _tools: Dict[str, ToolMetadata] = {}
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def register(cls, metadata: ToolMetadata) -> None:
        """Register a tool with the registry (thread-safe).

        Parameters
        ----------
        metadata : ToolMetadata
            Tool metadata to register
        """
        with cls._lock:
            cls._tools[metadata.name] = metadata

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a tool from the registry (thread-safe).

        Parameters
        ----------
        name : str
            Tool name to unregister
        """
        with cls._lock:
            if name in cls._tools:
                del cls._tools[name]

    @classmethod
    def get(cls, name: str) -> ToolMetadata:
        """Get a tool by name.

        Parameters
        ----------
        name : str
            Tool name

        Returns
        -------
        ToolMetadata
            Tool metadata

        Raises
        ------
        KeyError
            If tool not found
        """
        cls._ensure_initialized()
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not found in registry. "
                          f"Available: {list(cls._tools.keys())}")
        return cls._tools[name]

    @classmethod
    def get_optional(cls, name: str) -> Optional[ToolMetadata]:
        """Get a tool by name, returning None if not found."""
        cls._ensure_initialized()
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[ToolMetadata]:
        """List all registered tools.

        Returns
        -------
        List[ToolMetadata]
            All registered tools
        """
        cls._ensure_initialized()
        return list(cls._tools.values())

    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered tool names.

        Returns
        -------
        List[str]
            All tool names
        """
        cls._ensure_initialized()
        return list(cls._tools.keys())

    @classmethod
    def list_by_category(cls, category: ToolCategory) -> List[ToolMetadata]:
        """List tools in a specific category.

        Parameters
        ----------
        category : ToolCategory
            Category to filter by

        Returns
        -------
        List[ToolMetadata]
            Tools in the category
        """
        cls._ensure_initialized()
        return [t for t in cls._tools.values() if t.category == category]

    @classmethod
    def list_by_max_cost(cls, max_cost: ComputationalCost) -> List[ToolMetadata]:
        """List tools with cost at or below the specified level.

        Parameters
        ----------
        max_cost : ComputationalCost
            Maximum cost level

        Returns
        -------
        List[ToolMetadata]
            Tools at or below the cost level
        """
        cls._ensure_initialized()
        max_idx = _COST_ORDER.index(max_cost)
        return [
            t for t in cls._tools.values()
            if _COST_ORDER.index(t.cost) <= max_idx
        ]

    @classmethod
    def list_by_cost(cls, cost: ComputationalCost) -> List[ToolMetadata]:
        """List tools with exact cost level.

        Parameters
        ----------
        cost : ComputationalCost
            Exact cost level

        Returns
        -------
        List[ToolMetadata]
            Tools with the specified cost
        """
        cls._ensure_initialized()
        return [t for t in cls._tools.values() if t.cost == cost]

    @classmethod
    def search(
        cls,
        query: str,
        category: Optional[ToolCategory] = None,
        max_cost: Optional[ComputationalCost] = None,
    ) -> List[ToolMetadata]:
        """Search tools by name or description.

        Parameters
        ----------
        query : str
            Search query (matched against name and description)
        category : ToolCategory, optional
            Filter by category
        max_cost : ComputationalCost, optional
            Filter by maximum cost

        Returns
        -------
        List[ToolMetadata]
            Matching tools
        """
        cls._ensure_initialized()
        query = query.lower()
        results = []

        for tool in cls._tools.values():
            # Category filter
            if category is not None and tool.category != category:
                continue

            # Cost filter
            if max_cost is not None:
                if _COST_ORDER.index(tool.cost) > _COST_ORDER.index(max_cost):
                    continue

            # Text search
            if (query in tool.name.lower() or
                query in tool.description.lower()):
                results.append(tool)

        return results

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools. Mainly for testing (thread-safe)."""
        with cls._lock:
            cls._tools.clear()
            cls._initialized = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure default tools are registered (thread-safe)."""
        with cls._lock:
            if not cls._initialized:
                _register_default_tools()
                cls._initialized = True

    @classmethod
    def get_tools_for_category_summary(cls) -> Dict[str, List[str]]:
        """Get a summary of tools organized by category.

        Returns
        -------
        Dict[str, List[str]]
            Mapping of category name to tool names
        """
        cls._ensure_initialized()
        result: Dict[str, List[str]] = {}
        for tool in cls._tools.values():
            cat_name = tool.category.value
            if cat_name not in result:
                result[cat_name] = []
            result[cat_name].append(tool.name)
        return result


# =============================================================================
# Default Tool Registrations
# =============================================================================

def _register_default_tools() -> None:
    """Register all default tools from the core library."""

    # Core implementations (series-based)
    from ..core.decomposition import (
        stl_decompose,
        mstl_decompose,
        hp_filter,
        holt_winters_decompose,
    )
    from ..core.forecasting import (
        forecast_arima,
        forecast_ets,
        forecast_theta,
        forecast_ensemble,
        compare_forecasts,
    )
    from ..core.patterns import (
        detect_peaks,
        count_peaks,
        analyze_recurrence,
        analyze_matrix_profile,
        find_motifs,
        find_discords,
        segment_changepoint,
        segment_fluss,
    )
    from ..core.spectral import (
        compute_psd,
        detect_periodicity,
        compute_coherence,
    )
    from ..core.complexity import (
        sample_entropy,
        permutation_entropy,
        hurst_exponent,
    )
    from ..core.statistics import (
        describe_series,
        compute_autocorrelation,
        compare_series_stats,
    )
    from ..core.classification import (
        knn_classify,
        rocket_classify,
        hivecote_classify,
        compare_classifiers,
    )

    from ..core.windowing import (
        select_window_size,
        select_window_size_from_csv,
        evaluate_windowed_classifier,
        evaluate_windowed_classifier_from_csv,
    )

    # With-data wrappers (variable_name/run_id)
    from .agent_tools import (
        stl_decompose_with_data,
        mstl_decompose_with_data,
        hp_filter_with_data,
        holt_winters_decompose_with_data,
        forecast_arima_with_data,
        forecast_ets_with_data,
        forecast_theta_with_data,
        forecast_ensemble_with_data,
        compare_forecasts_with_data,
        detect_peaks_with_data,
        count_peaks_with_data,
        analyze_recurrence_with_data,
        analyze_matrix_profile_with_data,
        find_motifs_with_data,
        find_discords_with_data,
        segment_changepoint_with_data,
        segment_fluss_with_data,
        compute_psd_with_data,
        detect_periodicity_with_data,
        compute_coherence_with_data,
        sample_entropy_with_data,
        permutation_entropy_with_data,
        hurst_exponent_with_data,
        describe_series_with_data,
        compute_autocorrelation_with_data,
        compare_series_stats_with_data,
    )

    def _register_tool(
        name: str,
        description: str,
        category: ToolCategory,
        cost: ComputationalCost,
        core_function,
        dependencies,
        parameters,
        examples,
        returns: str,
    ) -> None:
        ToolRegistry.register(ToolMetadata(
            name=name,
            description=description,
            category=category,
            cost=cost,
            core_function=core_function,
            dependencies=dependencies or [],
            parameters=parameters or [],
            examples=examples or [],
            returns=returns,
        ))

    def _register_with_data(
        base_name: str,
        description: str,
        category: ToolCategory,
        cost: ComputationalCost,
        core_function,
        dependencies,
        parameters,
        examples,
        returns: str,
    ) -> None:
        _register_tool(
            name=f"{base_name}_with_data",
            description=f"{description} (loads data by run/variable)",
            category=category,
            cost=cost,
            core_function=core_function,
            dependencies=dependencies,
            parameters=parameters,
            examples=examples,
            returns=returns,
        )

    # ---------------------------------------------------------------------
    # Data Loading Tool
    # ---------------------------------------------------------------------
    from ..data_access import get_series as _get_series_raw

    def get_series(run_id: str, variable: str) -> np.ndarray:
        """Load a time series from the dataset."""
        return _get_series_raw(run_id, variable)

    _register_tool(
        name="get_series",
        description="Load a time series from the dataset. Returns the raw data array.",
        category=ToolCategory.DATA,
        cost=ComputationalCost.LOW,
        core_function=get_series,
        dependencies=[],
        parameters=[
            ToolParameter("run_id", "str", "The simulation run ID (e.g., 'Re200Rm200')"),
            ToolParameter("variable", "str", "Variable name (e.g., 'bx001_real')"),
        ],
        examples=[
            "Load bx001_real for Re200Rm200",
            "Get by001_real for Re175Rm175",
        ],
        returns="numpy array containing the time series data",
    )

    # ---------------------------------------------------------------------
    # Decomposition Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="stl_decompose",
        description="Decompose time series using STL (Seasonal-Trend LOESS).",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=stl_decompose,
        dependencies=["statsmodels"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("period", "int", "Seasonal period", optional=True),
            ToolParameter("robust", "bool", "Use robust fitting", optional=True, default=True),
        ],
        examples=["Decompose a series to extract trend and seasonality"],
        returns="DecompositionResult with trend, seasonal, residual components",
    )
    _register_with_data(
        base_name="stl_decompose",
        description="Decompose time series using STL (Seasonal-Trend LOESS).",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=stl_decompose_with_data,
        dependencies=["statsmodels", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("period", "int", "Seasonal period", optional=True),
            ToolParameter("robust", "bool", "Use robust fitting", optional=True, default=True),
        ],
        examples=["Decompose bx001_real for Re105Rm105"],
        returns="DecompositionResult with plot",
    )

    _register_tool(
        name="mstl_decompose",
        description="Multi-seasonal STL decomposition for series with multiple periodicities.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.MEDIUM,
        core_function=mstl_decompose,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("periods", "list", "List of seasonal periods", optional=True),
            ToolParameter("windows", "list", "Seasonal smoothing windows", optional=True),
        ],
        examples=["Decompose series with daily and weekly seasonality"],
        returns="DecompositionResult with trend, seasonal, residual components",
    )
    _register_with_data(
        base_name="mstl_decompose",
        description="Multi-seasonal STL decomposition for series with multiple periodicities.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.MEDIUM,
        core_function=mstl_decompose_with_data,
        dependencies=["statsforecast", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("periods", "list", "List of seasonal periods", optional=True),
        ],
        examples=["Decompose series with multiple seasonalities"],
        returns="DecompositionResult with plot",
    )

    _register_tool(
        name="hp_filter",
        description="Hodrick-Prescott filter for smooth trend extraction.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=hp_filter,
        dependencies=["statsmodels"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("lamb", "float", "Smoothing parameter", optional=True, default=1600),
        ],
        examples=["Extract smooth trend from noisy data"],
        returns="DecompositionResult with trend and residual components",
    )
    _register_with_data(
        base_name="hp_filter",
        description="Hodrick-Prescott filter for smooth trend extraction.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=hp_filter_with_data,
        dependencies=["statsmodels", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("lamb", "float", "Smoothing parameter", optional=True, default=1600),
        ],
        examples=["Extract smooth trend from noisy data"],
        returns="DecompositionResult with plot",
    )

    _register_tool(
        name="holt_winters_decompose",
        description="Holt-Winters exponential smoothing decomposition.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=holt_winters_decompose,
        dependencies=["statsmodels"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("period", "int", "Seasonal period", optional=True),
            ToolParameter("trend", "str", "Trend type: add or mul", optional=True, default="add"),
            ToolParameter("seasonal", "str", "Seasonal type: add or mul", optional=True, default="add"),
        ],
        examples=["Decompose series for forecasting"],
        returns="DecompositionResult with trend, seasonal, residual components",
    )
    _register_with_data(
        base_name="holt_winters_decompose",
        description="Holt-Winters exponential smoothing decomposition.",
        category=ToolCategory.DECOMPOSITION,
        cost=ComputationalCost.LOW,
        core_function=holt_winters_decompose_with_data,
        dependencies=["statsmodels", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("period", "int", "Seasonal period", optional=True),
            ToolParameter("trend", "str", "Trend type: add or mul", optional=True, default="add"),
            ToolParameter("seasonal", "str", "Seasonal type: add or mul", optional=True, default="add"),
        ],
        examples=["Decompose series for forecasting"],
        returns="DecompositionResult with plot",
    )

    # ---------------------------------------------------------------------
    # Forecasting Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="forecast_arima",
        description="Forecast using AutoARIMA.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=forecast_arima,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("level", "list", "Confidence levels", optional=True),
        ],
        examples=["Forecast a series for the next 20 steps"],
        returns="ForecastResult with predictions",
    )
    _register_with_data(
        base_name="forecast_arima",
        description="Forecast using AutoARIMA.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=forecast_arima_with_data,
        dependencies=["statsforecast", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to forecast"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
        ],
        examples=["Forecast bx001_real for next 20 steps"],
        returns="ForecastResult with plot",
    )

    _register_tool(
        name="forecast_ets",
        description="Forecast using AutoETS (Exponential Smoothing).",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.MEDIUM,
        core_function=forecast_ets,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("level", "list", "Confidence levels", optional=True),
        ],
        examples=["Forecast with Exponential Smoothing"],
        returns="ForecastResult with predictions",
    )
    _register_with_data(
        base_name="forecast_ets",
        description="Forecast using AutoETS (Exponential Smoothing).",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.MEDIUM,
        core_function=forecast_ets_with_data,
        dependencies=["statsforecast", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to forecast"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
        ],
        examples=["Forecast with Exponential Smoothing"],
        returns="ForecastResult with plot",
    )

    _register_tool(
        name="forecast_theta",
        description="Forecast using AutoTheta method.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.LOW,
        core_function=forecast_theta,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("level", "list", "Confidence levels", optional=True),
        ],
        examples=["Forecast using Theta method"],
        returns="ForecastResult with predictions",
    )
    _register_with_data(
        base_name="forecast_theta",
        description="Forecast using AutoTheta method.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.LOW,
        core_function=forecast_theta_with_data,
        dependencies=["statsforecast", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to forecast"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
        ],
        examples=["Forecast using Theta method"],
        returns="ForecastResult with plot",
    )

    _register_tool(
        name="forecast_ensemble",
        description="Ensemble forecasting combining multiple models.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=forecast_ensemble,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("models", "list", "Models to include", optional=True),
        ],
        examples=["Ensemble forecast for improved accuracy"],
        returns="MultiForecastResult with individual and combined forecasts",
    )
    _register_with_data(
        base_name="forecast_ensemble",
        description="Ensemble forecasting combining multiple models.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=forecast_ensemble_with_data,
        dependencies=["statsforecast", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to forecast"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("models", "list", "Models to include", optional=True),
        ],
        examples=["Ensemble forecast for improved accuracy"],
        returns="MultiForecastResult with plot",
    )

    _register_tool(
        name="compare_forecasts",
        description="Compare multiple forecasting methods on historical data.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=compare_forecasts,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("test_size", "int", "Test size", optional=True),
            ToolParameter("models", "list", "Models to compare", optional=True),
        ],
        examples=["Compare ARIMA vs ETS accuracy"],
        returns="Comparison results with metrics",
    )
    _register_with_data(
        base_name="compare_forecasts",
        description="Compare multiple forecasting methods on historical data.",
        category=ToolCategory.FORECASTING,
        cost=ComputationalCost.HIGH,
        core_function=compare_forecasts_with_data,
        dependencies=["statsforecast"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to forecast"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("horizon", "int", "Forecast horizon", optional=True, default=10),
            ToolParameter("models", "list", "Models to compare", optional=True),
            ToolParameter("methods", "list", "Alias for models (backward-compatible)", optional=True),
        ],
        examples=["Compare ARIMA vs ETS accuracy"],
        returns="Comparison results with metrics",
    )

    # ---------------------------------------------------------------------
    # Pattern Detection Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="detect_peaks",
        description="Detect peaks (local maxima) in the time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.LOW,
        core_function=detect_peaks,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("height", "float", "Minimum peak height", optional=True),
            ToolParameter("distance", "int", "Min distance between peaks", optional=True),
            ToolParameter("prominence", "float", "Min peak prominence", optional=True),
            ToolParameter("width", "float", "Min peak width", optional=True),
            ToolParameter("threshold", "float", "Min peak threshold", optional=True),
        ],
        examples=["Find peaks in the signal"],
        returns="PeakResult with indices and statistics",
    )
    _register_with_data(
        base_name="detect_peaks",
        description="Detect peaks (local maxima) in the time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.LOW,
        core_function=detect_peaks_with_data,
        dependencies=["scipy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("distance", "int", "Min distance between peaks", optional=True),
            ToolParameter("prominence", "float", "Min peak prominence", optional=True),
        ],
        examples=["Find peaks in the signal"],
        returns="PeakResult with plot",
    )

    _register_tool(
        name="count_peaks",
        description="Count the number of peaks in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.LOW,
        core_function=count_peaks,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("height", "float", "Minimum peak height", optional=True),
            ToolParameter("distance", "int", "Minimum distance between peaks", optional=True),
            ToolParameter("prominence", "float", "Minimum peak prominence", optional=True),
        ],
        examples=["Count oscillations in the signal"],
        returns="Integer count of peaks",
    )
    _register_with_data(
        base_name="count_peaks",
        description="Count the number of peaks in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.LOW,
        core_function=count_peaks_with_data,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("distance", "int", "Minimum distance between peaks", optional=True),
            ToolParameter("prominence", "float", "Minimum peak prominence", optional=True),
        ],
        examples=["Count oscillations in the signal"],
        returns="Integer count of peaks",
    )

    _register_tool(
        name="analyze_recurrence",
        description="Generate recurrence plot and compute RQA metrics.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=analyze_recurrence,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("threshold", "float", "Distance threshold", optional=True),
            ToolParameter("max_points", "int", "Max points", optional=True, default=1000),
            ToolParameter("min_line_length", "int", "Min line length", optional=True, default=2),
        ],
        examples=["Analyze recurrence patterns"],
        returns="RecurrenceResult with metrics",
    )
    _register_with_data(
        base_name="analyze_recurrence",
        description="Generate recurrence plot and compute RQA metrics.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=analyze_recurrence_with_data,
        dependencies=["scipy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("threshold", "float", "Distance threshold", optional=True),
        ],
        examples=["Analyze recurrence patterns"],
        returns="RecurrenceResult with plot",
    )

    _register_tool(
        name="analyze_matrix_profile",
        description="Compute Matrix Profile to find motifs and discords.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.HIGH,
        core_function=analyze_matrix_profile,
        dependencies=["stumpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("m", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("max_motifs", "int", "Max motifs", optional=True, default=3),
            ToolParameter("max_discords", "int", "Max discords", optional=True, default=3),
            ToolParameter("include_subsequences", "bool", "Include subsequences", optional=True, default=False),
        ],
        examples=["Find motifs and anomalies using Matrix Profile"],
        returns="MatrixProfileResult with motifs and discords",
    )
    _register_with_data(
        base_name="analyze_matrix_profile",
        description="Compute Matrix Profile to find motifs and discords.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.HIGH,
        core_function=analyze_matrix_profile_with_data,
        dependencies=["stumpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("window_size", "int", "Subsequence window size", optional=True, default=50),
        ],
        examples=["Find motifs and anomalies using Matrix Profile"],
        returns="MatrixProfileResult with plot",
    )

    _register_tool(
        name="find_motifs",
        description="Find motifs (recurring patterns) in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=find_motifs,
        dependencies=["stumpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("m", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("max_motifs", "int", "Number of motifs", optional=True, default=3),
            ToolParameter("exclusion_zone", "int", "Exclusion zone", optional=True),
            ToolParameter("include_subsequences", "bool", "Include subsequences", optional=True, default=False),
        ],
        examples=["Find the most common repeating pattern"],
        returns="List of MotifResult objects",
    )
    _register_with_data(
        base_name="find_motifs",
        description="Find motifs (recurring patterns) in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=find_motifs_with_data,
        dependencies=["stumpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("window_size", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("n_motifs", "int", "Number of motifs", optional=True, default=3),
        ],
        examples=["Find the most common repeating pattern"],
        returns="List of MotifResult objects and plot",
    )

    _register_tool(
        name="find_discords",
        description="Find discords (anomalies) in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=find_discords,
        dependencies=["stumpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("m", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("max_discords", "int", "Number of discords", optional=True, default=3),
            ToolParameter("exclusion_zone", "int", "Exclusion zone", optional=True),
            ToolParameter("include_subsequences", "bool", "Include subsequences", optional=True, default=False),
        ],
        examples=["Find unusual patterns in the data"],
        returns="List of DiscordResult objects",
    )
    _register_with_data(
        base_name="find_discords",
        description="Find discords (anomalies) in a time series.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=find_discords_with_data,
        dependencies=["stumpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("window_size", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("n_discords", "int", "Number of discords", optional=True, default=3),
        ],
        examples=["Find unusual patterns in the data"],
        returns="List of DiscordResult objects and plot",
    )

    _register_tool(
        name="segment_changepoint",
        description="Segment time series by detecting changepoints.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=segment_changepoint,
        dependencies=["ruptures"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("n_segments", "int", "Number of segments", optional=True),
            ToolParameter("algorithm", "str", "Algorithm: pelt/binseg/bottomup/window", optional=True, default="pelt"),
            ToolParameter("cost_model", "str", "Cost model", optional=True, default="rbf"),
            ToolParameter("penalty", "float", "Penalty value", optional=True),
            ToolParameter("min_size", "int", "Minimum segment size", optional=True, default=5),
        ],
        examples=["Detect regime changes in the signal"],
        returns="SegmentResult with changepoint locations",
    )
    _register_with_data(
        base_name="segment_changepoint",
        description="Segment time series by detecting changepoints.",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=segment_changepoint_with_data,
        dependencies=["ruptures", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("n_segments", "int", "Number of segments to find", optional=True),
            ToolParameter(
                "n_changepoints",
                "int",
                "Compatibility alias for changepoints count (maps to n_segments = n_changepoints + 1)",
                optional=True,
            ),
            ToolParameter("algorithm", "str", "Algorithm: pelt/binseg/bottomup/window", optional=True, default="pelt"),
            ToolParameter("cost_model", "str", "Cost model", optional=True, default="rbf"),
            ToolParameter("penalty", "float", "Penalty value", optional=True),
            ToolParameter("min_size", "int", "Minimum segment size", optional=True, default=5),
        ],
        examples=["Detect regime changes in the signal"],
        returns="SegmentResult with plot",
    )

    _register_tool(
        name="segment_fluss",
        description="Segment time series using FLUSS (matrix profile).",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=segment_fluss,
        dependencies=["stumpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("m", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("n_segments", "int", "Number of segments", optional=True, default=3),
        ],
        examples=["Find natural segment boundaries"],
        returns="SegmentResult with segment boundaries",
    )
    _register_with_data(
        base_name="segment_fluss",
        description="Segment time series using FLUSS (matrix profile).",
        category=ToolCategory.PATTERNS,
        cost=ComputationalCost.MEDIUM,
        core_function=segment_fluss_with_data,
        dependencies=["stumpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("window_size", "int", "Subsequence length", optional=True, default=50),
            ToolParameter("n_segments", "int", "Number of segments", optional=True, default=3),
        ],
        examples=["Find natural segment boundaries"],
        returns="SegmentResult with plot",
    )

    # ---------------------------------------------------------------------
    # Classification Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="knn_classify",
        description="K-Nearest Neighbors classification with DTW distance.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.MEDIUM,
        core_function=knn_classify,
        dependencies=["aeon"],
        parameters=[
            ToolParameter("X_train", "np.ndarray", "Training data (n_samples, n_channels, n_timepoints)"),
            ToolParameter("y_train", "np.ndarray", "Training labels"),
            ToolParameter("X_test", "np.ndarray", "Test data"),
            ToolParameter("y_test", "np.ndarray", "Test labels", optional=True),
            ToolParameter("distance", "str", "Distance metric", optional=True, default="dtw"),
            ToolParameter("n_neighbors", "int", "Number of neighbors", optional=True, default=1),
        ],
        examples=["Classify time series using DTW-KNN"],
        returns="ClassificationResult with predictions and accuracy",
    )

    _register_tool(
        name="rocket_classify",
        description="ROCKET classification using random convolutional kernels.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.LOW,
        core_function=rocket_classify,
        dependencies=["aeon"],
        parameters=[
            ToolParameter("X_train", "np.ndarray", "Training data"),
            ToolParameter("y_train", "np.ndarray", "Training labels"),
            ToolParameter("X_test", "np.ndarray", "Test data"),
            ToolParameter("y_test", "np.ndarray", "Test labels", optional=True),
            ToolParameter("variant", "str", "Variant: rocket/minirocket/multirocket", optional=True, default="rocket"),
            ToolParameter("n_kernels", "int", "Number of random kernels", optional=True, default=10000),
        ],
        examples=["Quick classification with ROCKET"],
        returns="ClassificationResult with predictions and accuracy",
    )

    _register_tool(
        name="hivecote_classify",
        description="HIVE-COTE 2 classification - state-of-the-art ensemble.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.VERY_HIGH,
        core_function=hivecote_classify,
        dependencies=["aeon"],
        parameters=[
            ToolParameter("X_train", "np.ndarray", "Training data"),
            ToolParameter("y_train", "np.ndarray", "Training labels"),
            ToolParameter("X_test", "np.ndarray", "Test data"),
            ToolParameter("y_test", "np.ndarray", "Test labels", optional=True),
        ],
        examples=["State-of-the-art classification"],
        returns="ClassificationResult with predictions and accuracy",
    )

    _register_tool(
        name="compare_classifiers",
        description="Compare multiple classification algorithms on the same data.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.HIGH,
        core_function=compare_classifiers,
        dependencies=["aeon"],
        parameters=[
            ToolParameter("X_train", "np.ndarray", "Training data"),
            ToolParameter("y_train", "np.ndarray", "Training labels"),
            ToolParameter("X_test", "np.ndarray", "Test data"),
            ToolParameter("y_test", "np.ndarray", "Test labels"),
            ToolParameter("classifiers", "list", "List of classifier names", optional=True),
        ],
        examples=["Compare DTW-KNN vs ROCKET"],
        returns="Comparison results with accuracy for each classifier",
    )

    # ---------------------------------------------------------------------
    # Window size selection (for sliding-window classification)
    # ---------------------------------------------------------------------
    _register_tool(
        name="select_window_size",
        description=(
            "Select an appropriate window size for sliding-window classification on a long, labeled time series. "
            "Uses segment-aware candidate generation, group splits, and balanced metrics."
        ),
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.HIGH,
        core_function=select_window_size,
        dependencies=["numpy", "pandas", "scikit-learn"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data (n_timepoints,) or (n_timepoints, n_channels)"),
            ToolParameter("labels", "np.ndarray", "Per-timepoint labels (n_timepoints,)"),
            ToolParameter("window_sizes", "list[int]", "Candidate window sizes", optional=True),
            ToolParameter("min_window", "int", "Minimum window size (auto candidates)", optional=True, default=16),
            ToolParameter("max_window", "int", "Maximum window size (auto candidates)", optional=True),
            ToolParameter("stride", "int", "Window stride (defaults to window_size//2)", optional=True),
            ToolParameter(
                "metric",
                "str",
                "Scoring metric: accuracy | balanced_accuracy | f1_macro",
                optional=True,
                default="balanced_accuracy",
            ),
            ToolParameter(
                "classifier",
                "str",
                "Evaluation classifier: minirocket | rocket | knn",
                optional=True,
                default="minirocket",
            ),
            ToolParameter(
                "labeling",
                "str",
                "Labeling: strict (recommended) | majority",
                optional=True,
                default="strict",
            ),
            ToolParameter(
                "balance",
                "str",
                "Balancing: none | undersample | segment_cap",
                optional=True,
                default="segment_cap",
            ),
            ToolParameter(
                "max_windows_per_segment",
                "int",
                "Cap windows per segment when balance=segment_cap",
                optional=True,
                default=25,
            ),
            ToolParameter("n_splits", "int", "Number of group splits", optional=True, default=3),
            ToolParameter("test_size", "float", "Test fraction per split", optional=True, default=0.2),
            ToolParameter("seed", "int", "Random seed", optional=True, default=1337),
            ToolParameter(
                "rocket_n_kernels",
                "int",
                "ROCKET/MiniROCKET kernels (speed/accuracy tradeoff)",
                optional=True,
                default=2000,
            ),
        ],
        examples=["Find best window size for labeled sensor stream"],
        returns="WindowSizeSelectionResult with best window size and scores",
    )

    _register_tool(
        name="select_window_size_from_csv",
        description="Window-size selection from a CSV with columns for values and labels.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.HIGH,
        core_function=select_window_size_from_csv,
        dependencies=["pandas", "numpy", "scikit-learn"],
        parameters=[
            ToolParameter("csv_path", "str", "Path to CSV (one row per timepoint)"),
            ToolParameter(
                "value_columns",
                "str | list[str]",
                "Value column name(s)",
                optional=True,
                default="value",
            ),
            ToolParameter(
                "label_column",
                "str",
                "Label column name",
                optional=True,
                default="label",
            ),
            # Forwarded kwargs (must be whitelisted to pass tool validation)
            ToolParameter("window_sizes", "list[int]", "Candidate window sizes", optional=True),
            ToolParameter("min_window", "int", "Minimum window size", optional=True, default=16),
            ToolParameter("max_window", "int", "Maximum window size", optional=True),
            ToolParameter("stride", "int", "Window stride", optional=True),
            ToolParameter("metric", "str", "Scoring metric", optional=True, default="balanced_accuracy"),
            ToolParameter("classifier", "str", "Evaluation classifier", optional=True, default="minirocket"),
            ToolParameter("labeling", "str", "Labeling", optional=True, default="strict"),
            ToolParameter("balance", "str", "Balancing strategy", optional=True, default="segment_cap"),
            ToolParameter(
                "max_windows_per_segment",
                "int",
                "Cap windows per segment",
                optional=True,
                default=25,
            ),
            ToolParameter("n_splits", "int", "Number of splits", optional=True, default=3),
            ToolParameter("test_size", "float", "Test fraction", optional=True, default=0.2),
            ToolParameter("seed", "int", "Random seed", optional=True, default=1337),
            ToolParameter("rocket_n_kernels", "int", "Kernels", optional=True, default=2000),
        ],
        examples=["Select window size directly from labeled CSV"],
        returns="WindowSizeSelectionResult with best window size and scores",
    )

    _register_tool(
        name="evaluate_windowed_classifier",
        description=(
            "Evaluate a classifier on sliding windows extracted from a labeled stream (array input). "
            "Uses segment-grouped train/test split to reduce leakage."
        ),
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.HIGH,
        core_function=evaluate_windowed_classifier,
        dependencies=["numpy", "scikit-learn", "aeon"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series (T,) or (T,C)"),
            ToolParameter("labels", "np.ndarray", "Per-timepoint labels (T,)"),
            ToolParameter("window_size", "int", "Window size"),
            ToolParameter("stride", "int", "Stride", optional=True),
            ToolParameter("classifier", "str", "minirocket|rocket|knn", optional=True, default="minirocket"),
            ToolParameter(
                "metric",
                "str",
                "accuracy|balanced_accuracy|f1_macro",
                optional=True,
                default="balanced_accuracy",
            ),
            ToolParameter("balance", "str", "none|undersample|segment_cap", optional=True, default="segment_cap"),
            ToolParameter(
                "max_windows_per_segment",
                "int",
                "Cap windows per segment (segment_cap)",
                optional=True,
                default=25,
            ),
            ToolParameter("test_size", "float", "Test fraction", optional=True, default=0.2),
            ToolParameter("seed", "int", "Random seed", optional=True, default=1337),
            ToolParameter("rocket_n_kernels", "int", "Kernels", optional=True, default=2000),
        ],
        examples=["Evaluate minirocket on windows of size 64"],
        returns="WindowedClassificationEvaluation with metric + embedded ClassificationResult",
    )

    _register_tool(
        name="evaluate_windowed_classifier_from_csv",
        description="Evaluate a classifier on sliding windows extracted from a labeled CSV.",
        category=ToolCategory.CLASSIFICATION,
        cost=ComputationalCost.HIGH,
        core_function=evaluate_windowed_classifier_from_csv,
        dependencies=["pandas", "numpy", "scikit-learn", "aeon"],
        parameters=[
            ToolParameter("csv_path", "str", "Path to CSV (one row per timepoint)"),
            ToolParameter("window_size", "int", "Window size"),
            ToolParameter(
                "value_columns",
                "str | list[str]",
                "Value column name(s)",
                optional=True,
                default="value",
            ),
            ToolParameter("label_column", "str", "Label column name", optional=True, default="label"),
            ToolParameter("stride", "int", "Stride", optional=True),
            ToolParameter("classifier", "str", "minirocket|rocket|knn", optional=True, default="minirocket"),
            ToolParameter(
                "metric",
                "str",
                "accuracy|balanced_accuracy|f1_macro",
                optional=True,
                default="balanced_accuracy",
            ),
            ToolParameter("balance", "str", "none|undersample|segment_cap", optional=True, default="segment_cap"),
            ToolParameter(
                "max_windows_per_segment",
                "int",
                "Cap windows per segment (segment_cap)",
                optional=True,
                default=25,
            ),
            ToolParameter("test_size", "float", "Test fraction", optional=True, default=0.2),
            ToolParameter("seed", "int", "Random seed", optional=True, default=1337),
            ToolParameter("rocket_n_kernels", "int", "Kernels", optional=True, default=2000),
        ],
        examples=["Evaluate classifier directly from labeled CSV"],
        returns="WindowedClassificationEvaluation",
    )

    # ---------------------------------------------------------------------
    # Spectral Analysis Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="compute_psd",
        description="Compute Power Spectral Density using Welch's method.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=compute_psd,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("sample_rate", "float", "Sampling rate", optional=True, default=1.0),
            ToolParameter("method", "str", "Method: welch or periodogram", optional=True, default="welch"),
            ToolParameter("nperseg", "int", "Segment length", optional=True),
        ],
        examples=["Compute power spectrum of the signal"],
        returns="SpectralResult with frequencies and PSD",
    )
    _register_with_data(
        base_name="compute_psd",
        description="Compute Power Spectral Density using Welch's method.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=compute_psd_with_data,
        dependencies=["scipy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("sampling_rate", "float", "Sampling rate", optional=True, default=1.0),
        ],
        examples=["Compute PSD for bx001_real"],
        returns="SpectralResult with plot",
    )

    _register_tool(
        name="detect_periodicity",
        description="Detect dominant periodicity using FFT.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=detect_periodicity,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("sample_rate", "float", "Sampling rate", optional=True, default=1.0),
            ToolParameter("top_n", "int", "Number of top periods", optional=True, default=3),
        ],
        examples=["Detect oscillation period"],
        returns="PeriodicityResult with dominant period",
    )
    _register_with_data(
        base_name="detect_periodicity",
        description="Detect dominant periodicity using FFT.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=detect_periodicity_with_data,
        dependencies=["numpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("n_top", "int", "Number of top periods", optional=True, default=5),
        ],
        examples=["Detect oscillation period"],
        returns="PeriodicityResult with plot",
    )

    _register_tool(
        name="compute_coherence",
        description="Compute coherence between two signals.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=compute_coherence,
        dependencies=["scipy"],
        parameters=[
            ToolParameter("series1", "np.ndarray", "First time series"),
            ToolParameter("series2", "np.ndarray", "Second time series"),
            ToolParameter("sample_rate", "float", "Sampling rate", optional=True, default=1.0),
            ToolParameter("nperseg", "int", "Segment length", optional=True),
        ],
        examples=["Compute coherence between two signals"],
        returns="CoherenceResult with frequency-dependent coherence",
    )
    _register_with_data(
        base_name="compute_coherence",
        description="Compute coherence between two signals.",
        category=ToolCategory.SPECTRAL,
        cost=ComputationalCost.LOW,
        core_function=compute_coherence_with_data,
        dependencies=["scipy", "matplotlib"],
        parameters=[
            ToolParameter("variable1", "str", "First variable name"),
            ToolParameter("unique_id1", "str", "Run ID for first variable"),
            ToolParameter("variable2", "str", "Second variable name"),
            ToolParameter("unique_id2", "str", "Run ID for second variable"),
            ToolParameter("sample_rate", "float", "Sampling rate", optional=True),
            ToolParameter("sampling_rate", "float", "Compatibility alias for sample_rate", optional=True),
            ToolParameter("fs", "float", "Compatibility alias for sample_rate", optional=True),
        ],
        examples=["Analyze coupling between two signals"],
        returns="CoherenceResult with plot",
    )

    # ---------------------------------------------------------------------
    # Complexity Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="sample_entropy",
        description="Compute sample entropy - measures regularity/complexity.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.MEDIUM,
        core_function=sample_entropy,
        dependencies=["antropy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("m", "int", "Embedding dimension", optional=True, default=2),
            ToolParameter("r", "float", "Tolerance", optional=True),
        ],
        examples=["How regular is this time series?"],
        returns="Float sample entropy value",
    )
    _register_with_data(
        base_name="sample_entropy",
        description="Compute sample entropy - measures regularity/complexity.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.MEDIUM,
        core_function=sample_entropy_with_data,
        dependencies=["antropy"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("m", "int", "Embedding dimension", optional=True, default=2),
            ToolParameter("r", "float", "Tolerance", optional=True),
        ],
        examples=["How regular is this time series?"],
        returns="Float sample entropy value",
    )

    _register_tool(
        name="permutation_entropy",
        description="Compute permutation entropy - ordinal pattern complexity.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.LOW,
        core_function=permutation_entropy,
        dependencies=["antropy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("order", "int", "Order of permutation patterns", optional=True, default=3),
            ToolParameter("delay", "int", "Time delay", optional=True, default=1),
            ToolParameter("normalize", "bool", "Normalize by max entropy", optional=True, default=True),
        ],
        examples=["Measure complexity using permutation entropy"],
        returns="Float permutation entropy value",
    )
    _register_with_data(
        base_name="permutation_entropy",
        description="Compute permutation entropy - ordinal pattern complexity.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.LOW,
        core_function=permutation_entropy_with_data,
        dependencies=["antropy"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("order", "int", "Order of permutation patterns", optional=True, default=3),
            ToolParameter("delay", "int", "Time delay", optional=True, default=1),
            ToolParameter("normalize", "bool", "Normalize by max entropy", optional=True, default=True),
        ],
        examples=["Measure complexity using permutation entropy"],
        returns="Float permutation entropy value",
    )

    _register_tool(
        name="hurst_exponent",
        description="Estimate Hurst exponent using R/S analysis.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.LOW,
        core_function=hurst_exponent,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("min_window", "int", "Minimum window", optional=True, default=10),
            ToolParameter("max_window", "int", "Maximum window", optional=True),
        ],
        examples=["Compute Hurst exponent"],
        returns="Float Hurst exponent (0-1)",
    )
    _register_with_data(
        base_name="hurst_exponent",
        description="Estimate Hurst exponent using R/S analysis.",
        category=ToolCategory.COMPLEXITY,
        cost=ComputationalCost.LOW,
        core_function=hurst_exponent_with_data,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("min_window", "int", "Minimum window", optional=True, default=10),
            ToolParameter("max_window", "int", "Maximum window", optional=True),
        ],
        examples=["Compute Hurst exponent"],
        returns="Float Hurst exponent (0-1)",
    )

    # ---------------------------------------------------------------------
    # Statistics Tools (series-based)
    # ---------------------------------------------------------------------
    _register_tool(
        name="describe_series",
        description="Compute descriptive statistics: mean, std, min, max, RMS, etc.",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=describe_series,
        dependencies=["numpy", "scipy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("extended", "bool", "Compute extended stats", optional=True, default=False),
        ],
        examples=["Describe basic statistics of a series"],
        returns="DescriptiveStats with mean, std, min, max, RMS, etc.",
    )
    _register_with_data(
        base_name="describe_series",
        description="Compute descriptive statistics: mean, std, min, max, RMS, etc.",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=describe_series_with_data,
        dependencies=["numpy", "scipy"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
        ],
        examples=["Describe bx001_real"],
        returns="DescriptiveStats with summary and plot",
    )

    _register_tool(
        name="compute_autocorrelation",
        description="Compute autocorrelation function (ACF).",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=compute_autocorrelation,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("series", "np.ndarray", "Time series data"),
            ToolParameter("max_lag", "int", "Maximum lag", optional=True),
        ],
        examples=["Compute autocorrelation of the signal"],
        returns="Array of autocorrelation values",
    )
    _register_with_data(
        base_name="compute_autocorrelation",
        description="Compute autocorrelation function (ACF).",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=compute_autocorrelation_with_data,
        dependencies=["numpy", "matplotlib"],
        parameters=[
            ToolParameter("variable_name", "str", "Variable name to analyze"),
            ToolParameter("unique_id", "str", "Run ID"),
            ToolParameter("max_lag", "int", "Maximum lag", optional=True),
        ],
        examples=["Compute autocorrelation of the signal"],
        returns="Array of autocorrelation values and plot",
    )

    _register_tool(
        name="compare_series_stats",
        description="Compare statistics between multiple time series.",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=compare_series_stats,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("series_dict", "dict", "Mapping of name to series"),
        ],
        examples=["Compare statistics across multiple series"],
        returns="Dict of DescriptiveStats by name",
    )
    _register_with_data(
        base_name="compare_series_stats",
        description="Compare statistics between multiple time series.",
        category=ToolCategory.STATISTICS,
        cost=ComputationalCost.LOW,
        core_function=compare_series_stats_with_data,
        dependencies=["numpy"],
        parameters=[
            ToolParameter("variables", "list", "List of variable names"),
            ToolParameter("run_ids", "list", "List of run IDs"),
        ],
        examples=["Compare statistics across all runs"],
        returns="DataFrame comparing statistics across series",
    )
