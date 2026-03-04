"""Tool Bundles - Curated sets of tools for different use cases.

This module provides predefined bundles of tools:
- demo: Meta-bundle combining the windowing and forecasting demo workflows
- demo_windowing: Focused tools for the windowing demo workflow
- demo_forecasting: Focused tools for the forecasting demo workflow
- minimal: 5 core tools for baseline testing
- standard: 15 tools for typical analysis
- full: 25+ tools for comprehensive analysis
- all: Every registered tool

Plus specialized bundles:
- decomposition: All decomposition tools
- forecasting: All forecasting tools
- patterns: All pattern detection tools
- classification: All classification tools
- spectral: All spectral analysis tools
- orchestrator: Tools for deep agent orchestrators
"""

from typing import List, Dict, Any, Optional, Union

from .registry import (
    ToolRegistry,
    ToolMetadata,
    ToolCategory,
    ComputationalCost,
)


# =============================================================================
# Bundle Definitions
# =============================================================================

# Demo bundle: Focused toolset for the window-size selection demo
DEMO_WINDOWING_BUNDLE = [
    # Windowing (hero demo)
    "select_window_size_from_csv",
    "evaluate_windowed_classifier_from_csv",
]

# Demo bundle: Focused toolset for forecasting demo workflows
DEMO_FORECASTING_BUNDLE = [
    "describe_series_with_data",
    "forecast_theta_with_data",
    "compare_forecasts_with_data",
]

# Meta-bundle: combines both demo tracks.
DEMO_BUNDLE = list(dict.fromkeys(DEMO_WINDOWING_BUNDLE + DEMO_FORECASTING_BUNDLE))

# Minimal bundle: Just enough tools for basic analysis
MINIMAL_BUNDLE = [
    "describe_series_with_data",      # Basic statistics
    "detect_peaks_with_data",         # Peak detection
    "stl_decompose_with_data",        # Decomposition
    "forecast_arima_with_data",       # Forecasting
    "detect_periodicity_with_data",   # Spectral
]

# Standard bundle: Well-rounded set for typical analysis
STANDARD_BUNDLE = [
    # Statistics
    "describe_series_with_data",
    "compare_series_stats_with_data",

    # Decomposition
    "stl_decompose_with_data",
    "mstl_decompose_with_data",

    # Patterns
    "detect_peaks_with_data",
    "count_peaks_with_data",
    "analyze_matrix_profile_with_data",
    "find_motifs_with_data",

    # Forecasting
    "forecast_arima_with_data",
    "forecast_ets_with_data",
    "forecast_theta_with_data",

    # Spectral
    "compute_psd_with_data",
    "detect_periodicity_with_data",
    "compute_coherence_with_data",

    # Complexity
    "hurst_exponent_with_data",
]

# Full bundle: Comprehensive set for in-depth analysis
FULL_BUNDLE = [
    # Statistics
    "describe_series_with_data",
    "compute_autocorrelation_with_data",
    "compare_series_stats_with_data",

    # Decomposition - all methods
    "stl_decompose_with_data",
    "mstl_decompose_with_data",
    "hp_filter_with_data",
    "holt_winters_decompose_with_data",

    # Patterns - all methods
    "detect_peaks_with_data",
    "count_peaks_with_data",
    "analyze_recurrence_with_data",
    "analyze_matrix_profile_with_data",
    "find_motifs_with_data",
    "find_discords_with_data",
    "segment_changepoint_with_data",
    "segment_fluss_with_data",

    # Forecasting - all methods
    "forecast_arima_with_data",
    "forecast_ets_with_data",
    "forecast_theta_with_data",
    "forecast_ensemble_with_data",
    "compare_forecasts_with_data",

    # Classification
    "knn_classify",
    "rocket_classify",
    "compare_classifiers",

    # Spectral - all methods
    "compute_psd_with_data",
    "detect_periodicity_with_data",
    "compute_coherence_with_data",

    # Complexity - all methods
    "sample_entropy_with_data",
    "permutation_entropy_with_data",
    "hurst_exponent_with_data",
]

# Orchestrator bundle: Tools for the orchestrator agent (high-level)
ORCHESTRATOR_BUNDLE = [
    # Overview tools
    "describe_series_with_data",
    "detect_periodicity_with_data",

    # Comparison tools
    "compare_series_stats_with_data",
    "compare_forecasts_with_data",
    "compare_classifiers",

    # High-level analysis
    "stl_decompose_with_data",
    "forecast_ensemble_with_data",
    "analyze_matrix_profile_with_data",
]


# Category-specific bundles (dynamically populated)
CATEGORY_BUNDLES: Dict[str, List[str]] = {
    "data": [
        "get_series",
    ],
    "decomposition": [
        "stl_decompose_with_data",
        "mstl_decompose_with_data",
        "hp_filter_with_data",
        "holt_winters_decompose_with_data",
    ],
    "forecasting": [
        "forecast_arima_with_data",
        "forecast_ets_with_data",
        "forecast_theta_with_data",
        "forecast_ensemble_with_data",
        "compare_forecasts_with_data",
    ],
    "patterns": [
        "detect_peaks_with_data",
        "count_peaks_with_data",
        "analyze_recurrence_with_data",
        "analyze_matrix_profile_with_data",
        "find_motifs_with_data",
        "find_discords_with_data",
        "segment_changepoint_with_data",
        "segment_fluss_with_data",
    ],
    "classification": [
        "knn_classify",
        "rocket_classify",
        "hivecote_classify",
        "compare_classifiers",
    ],
    "spectral": [
        "compute_psd_with_data",
        "detect_periodicity_with_data",
        "compute_coherence_with_data",
    ],
    "complexity": [
        "sample_entropy_with_data",
        "permutation_entropy_with_data",
        "hurst_exponent_with_data",
    ],
    "statistics": [
        "describe_series_with_data",
        "compute_autocorrelation_with_data",
        "compare_series_stats_with_data",
    ],
}


# =============================================================================
# Bundle Functions
# =============================================================================

def get_bundle(name: str) -> List[ToolMetadata]:
    """Get a predefined bundle of tools.

    Parameters
    ----------
    name : str
        Bundle name: "demo", "demo_windowing", "demo_forecasting",
        "minimal", "standard", "full", "all", "orchestrator", or a category name

    Returns
    -------
    List[ToolMetadata]
        List of tool metadata objects

    Examples
    --------
    >>> tools = get_bundle("standard")
    >>> print(f"Standard bundle has {len(tools)} tools")

    >>> decomp_tools = get_bundle("decomposition")
    """
    name = name.lower()

    if name == "demo":
        tool_names = DEMO_BUNDLE
    elif name == "demo_windowing":
        tool_names = DEMO_WINDOWING_BUNDLE
    elif name == "demo_forecasting":
        tool_names = DEMO_FORECASTING_BUNDLE
    elif name == "minimal":
        tool_names = MINIMAL_BUNDLE
    elif name == "standard":
        tool_names = STANDARD_BUNDLE
    elif name == "full":
        tool_names = FULL_BUNDLE
    elif name == "all":
        return ToolRegistry.list_all()
    elif name == "orchestrator":
        tool_names = ORCHESTRATOR_BUNDLE
    elif name in CATEGORY_BUNDLES:
        tool_names = CATEGORY_BUNDLES[name]
    else:
        raise ValueError(
            f"Unknown bundle: {name}. Available: demo, demo_windowing, demo_forecasting, minimal, standard, full, all, "
            f"orchestrator, {', '.join(CATEGORY_BUNDLES.keys())}"
        )

    return [ToolRegistry.get(n) for n in tool_names]


def get_bundle_names(name: str) -> List[str]:
    """Get tool names for a predefined bundle.

    Parameters
    ----------
    name : str
        Bundle name

    Returns
    -------
    List[str]
        List of tool names

    Examples
    --------
    >>> names = get_bundle_names("minimal")
    >>> print(names)
    ['describe_series', 'detect_peaks', 'stl_decompose', ...]
    """
    name = name.lower()

    if name == "demo":
        return list(DEMO_BUNDLE)
    elif name == "demo_windowing":
        return list(DEMO_WINDOWING_BUNDLE)
    elif name == "demo_forecasting":
        return list(DEMO_FORECASTING_BUNDLE)
    elif name == "minimal":
        return list(MINIMAL_BUNDLE)
    elif name == "standard":
        return list(STANDARD_BUNDLE)
    elif name == "full":
        return list(FULL_BUNDLE)
    elif name == "all":
        return ToolRegistry.list_names()
    elif name == "orchestrator":
        return list(ORCHESTRATOR_BUNDLE)
    elif name in CATEGORY_BUNDLES:
        return list(CATEGORY_BUNDLES[name])
    else:
        raise ValueError(f"Unknown bundle: {name}")


def list_available_bundles() -> List[str]:
    """List all available bundle names.

    Returns
    -------
    List[str]
        Available bundle names
    """
    return [
        "demo",
        "demo_windowing",
        "demo_forecasting",
        "minimal",
        "standard",
        "full",
        "all",
        "orchestrator",
    ] + list(CATEGORY_BUNDLES.keys())


def get_bundle_summary() -> Dict[str, Dict[str, Any]]:
    """Get summary information about all bundles.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Bundle summaries with tool counts and descriptions

    Examples
    --------
    >>> summary = get_bundle_summary()
    >>> for name, info in summary.items():
    ...     print(f"{name}: {info['count']} tools")
    """
    return {
        "demo": {
            "count": len(DEMO_BUNDLE),
            "description": f"Meta-bundle covering both demo tracks ({len(DEMO_BUNDLE)} tools)",
            "tools": DEMO_BUNDLE,
        },
        "demo_windowing": {
            "count": len(DEMO_WINDOWING_BUNDLE),
            "description": f"Focused windowing essentials for LLM-first demo ({len(DEMO_WINDOWING_BUNDLE)} tools)",
            "tools": DEMO_WINDOWING_BUNDLE,
        },
        "demo_forecasting": {
            "count": len(DEMO_FORECASTING_BUNDLE),
            "description": f"Focused forecasting demo tools ({len(DEMO_FORECASTING_BUNDLE)} tools)",
            "tools": DEMO_FORECASTING_BUNDLE,
        },
        "minimal": {
            "count": len(MINIMAL_BUNDLE),
            "description": "Core tools for baseline testing (5 tools)",
            "tools": MINIMAL_BUNDLE,
        },
        "standard": {
            "count": len(STANDARD_BUNDLE),
            "description": "Well-rounded set for typical analysis (15 tools)",
            "tools": STANDARD_BUNDLE,
        },
        "full": {
            "count": len(FULL_BUNDLE),
            "description": "Comprehensive set for in-depth analysis (25+ tools)",
            "tools": FULL_BUNDLE,
        },
        "all": {
            "count": len(ToolRegistry.list_all()),
            "description": "Every registered tool",
            "tools": ToolRegistry.list_names(),
        },
        "orchestrator": {
            "count": len(ORCHESTRATOR_BUNDLE),
            "description": "High-level tools for agent orchestration",
            "tools": ORCHESTRATOR_BUNDLE,
        },
        **{
            cat: {
                "count": len(tools),
                "description": f"All {cat} tools",
                "tools": tools,
            }
            for cat, tools in CATEGORY_BUNDLES.items()
        }
    }


# =============================================================================
# Custom Bundle Creation
# =============================================================================

def create_custom_bundle(
    tools: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    max_cost: Optional[ComputationalCost] = None,
    include_comparison: bool = True,
) -> List[ToolMetadata]:
    """Create a custom bundle with specified criteria.

    Parameters
    ----------
    tools : List[str], optional
        Specific tool names to include
    categories : List[str], optional
        Categories to include (e.g., ["decomposition", "forecasting"])
    max_cost : ComputationalCost, optional
        Maximum computational cost
    include_comparison : bool
        Whether to include comparison tools

    Returns
    -------
    List[ToolMetadata]
        Custom bundle of tools

    Examples
    --------
    >>> # Fast tools for decomposition and forecasting
    >>> bundle = create_custom_bundle(
    ...     categories=["decomposition", "forecasting"],
    ...     max_cost=ComputationalCost.LOW,
    ... )
    """
    result = []
    seen_names = set()

    # Add specific tools
    if tools:
        for name in tools:
            if name not in seen_names:
                result.append(ToolRegistry.get(name))
                seen_names.add(name)

    # Add category tools
    if categories:
        for cat_name in categories:
            cat = ToolCategory(cat_name)
            for tool in ToolRegistry.list_by_category(cat):
                if tool.name not in seen_names:
                    # Check cost
                    if max_cost is not None:
                        from .registry import _COST_ORDER
                        if _COST_ORDER.index(tool.cost) > _COST_ORDER.index(max_cost):
                            continue
                    result.append(tool)
                    seen_names.add(tool.name)

    # Add comparison tools if requested
    if include_comparison:
        comparison_tools = ["compare_forecasts", "compare_classifiers", "compare_series_stats"]
        for name in comparison_tools:
            if name not in seen_names:
                try:
                    tool = ToolRegistry.get(name)
                    if max_cost is None or \
                       _COST_ORDER.index(tool.cost) <= _COST_ORDER.index(max_cost):
                        result.append(tool)
                        seen_names.add(name)
                except KeyError:
                    pass

    return result


# =============================================================================
# Subagent Bundle Definitions
# =============================================================================

def get_subagent_bundle(subagent_name: str) -> List[ToolMetadata]:
    """Get tools for a specific subagent type.

    Parameters
    ----------
    subagent_name : str
        Subagent name: "decomposition", "forecasting", "patterns",
        "classification", "turbulence"

    Returns
    -------
    List[ToolMetadata]
        Tools appropriate for that subagent

    Examples
    --------
    >>> tools = get_subagent_bundle("decomposition")
    """
    subagent_bundles = {
        "decomposition": [
            "stl_decompose_with_data",
            "mstl_decompose_with_data",
            "hp_filter_with_data",
            "holt_winters_decompose_with_data",
            "detect_periodicity_with_data",
            "describe_series_with_data",
        ],
        "forecasting": [
            "forecast_arima_with_data",
            "forecast_ets_with_data",
            "forecast_theta_with_data",
            "forecast_ensemble_with_data",
            "compare_forecasts_with_data",
            "detect_periodicity_with_data",
        ],
        "patterns": [
            "detect_peaks_with_data",
            "count_peaks_with_data",
            "analyze_recurrence_with_data",
            "analyze_matrix_profile_with_data",
            "find_motifs_with_data",
            "find_discords_with_data",
            "segment_changepoint_with_data",
            "segment_fluss_with_data",
        ],
        "classification": [
            "knn_classify",
            "rocket_classify",
            "hivecote_classify",
            "compare_classifiers",
        ],
        "turbulence": [
            "compute_psd_with_data",
            "compute_coherence_with_data",
            "detect_periodicity_with_data",
            "describe_series_with_data",
            "hurst_exponent_with_data",
            "sample_entropy_with_data",
        ],
    }

    if subagent_name not in subagent_bundles:
        raise ValueError(
            f"Unknown subagent: {subagent_name}. "
            f"Available: {list(subagent_bundles.keys())}"
        )

    return [ToolRegistry.get(n) for n in subagent_bundles[subagent_name]]


# =============================================================================
# Convenience Functions for Agent Creation
# =============================================================================

def get_langchain_bundle(
    bundle_name: str = "standard",
    return_direct: bool = False,
) -> List[Any]:
    """Get tools wrapped for LangChain.

    Parameters
    ----------
    bundle_name : str
        Bundle name to get
    return_direct : bool
        Whether tools should return directly to user

    Returns
    -------
    List[BaseTool]
        LangChain-compatible tools

    Examples
    --------
    >>> tools = get_langchain_bundle("standard")
    >>> agent = create_openai_functions_agent(llm, tools, prompt)
    """
    from .wrappers import wrap_tools_for_langchain

    bundle = get_bundle(bundle_name)
    return wrap_tools_for_langchain(bundle, return_direct=return_direct)


def get_deepagent_bundle(bundle_name: str = "standard") -> List[Dict[str, Any]]:
    """Get tools wrapped for deepagents.

    Parameters
    ----------
    bundle_name : str
        Bundle name to get

    Returns
    -------
    List[dict]
        deepagent-compatible tools

    Examples
    --------
    >>> tools = get_deepagent_bundle("full")
    >>> agent = create_deep_agent(tools=tools)
    """
    from .wrappers import wrap_tools_for_deepagent

    bundle = get_bundle(bundle_name)
    return wrap_tools_for_deepagent(bundle)
