"""Tool Bundles - Curated sets of tools for different use cases.

This module provides predefined bundles of tools:
- minimal: one entry point per task family (5 tools)
- standard: well-rounded set across task families
- full: comprehensive set
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

# Minimal bundle: one entry point per task family
MINIMAL_BUNDLE = [
    "describe_series",          # Statistics
    "detect_periodicity",       # Spectral
    "segment_changepoint",      # Regime changes
    "find_discords",            # Anomalies
    "forecast_theta",           # Forecasting
]

# Standard bundle: Well-rounded set for typical analysis
STANDARD_BUNDLE = [
    # Statistics
    "describe_series",
    "compute_autocorrelation",

    # Decomposition
    "stl_decompose",
    "mstl_decompose",

    # Patterns, anomalies, regimes
    "detect_peaks",
    "find_motifs",
    "find_discords",
    "segment_changepoint",

    # Forecasting
    "forecast_seasonal_naive",
    "forecast_theta",
    "compare_forecasts",

    # Classification
    "compare_classifiers",

    # Spectral
    "compute_psd",
    "detect_periodicity",
    "compute_coherence",
]

# Full bundle: Comprehensive set for in-depth analysis
FULL_BUNDLE = [
    # Statistics
    "describe_series",
    "compute_autocorrelation",
    "compare_series_stats",

    # Decomposition - all methods
    "stl_decompose",
    "mstl_decompose",
    "holt_winters_decompose",

    # Patterns - all methods
    "detect_peaks",
    "count_peaks",
    "analyze_recurrence",
    "analyze_matrix_profile",
    "find_motifs",
    "find_discords",
    "segment_changepoint",
    "segment_fluss",

    # Forecasting - all methods
    "forecast_arima",
    "forecast_ets",
    "forecast_seasonal_naive",
    "forecast_theta",
    "forecast_ensemble",
    "compare_forecasts",
    # Optional extras: Darts zero-shot FMs [foundation]; panel GBM/NHITS/FM families
    "forecast_foundation",
    "forecast_panel_from_csv",

    # Classification and labeled streams
    "knn_classify",
    "rocket_classify",
    "compare_classifiers",
    "select_window_size_from_csv",
    "evaluate_windowed_classifier_from_csv",

    # Spectral - all methods
    "compute_psd",
    "detect_periodicity",
    "compute_coherence",
]

# Orchestrator bundle: overview and comparison tools across task families
ORCHESTRATOR_BUNDLE = [
    # Overview tools
    "describe_series",
    "detect_periodicity",
    "segment_changepoint",
    "find_discords",

    # Comparison tools
    "compare_series_stats",
    "compare_forecasts",
    "compare_classifiers",

    # High-level analysis
    "stl_decompose",
    "forecast_ensemble",
    "analyze_matrix_profile",
]


# Category-specific bundles
CATEGORY_BUNDLES: Dict[str, List[str]] = {
    "data": [
        "get_series",
    ],
    "decomposition": [
        "stl_decompose",
        "mstl_decompose",
        "holt_winters_decompose",
    ],
    "forecasting": [
        "forecast_arima",
        "forecast_ets",
        "forecast_seasonal_naive",
        "forecast_theta",
        "forecast_ensemble",
        "compare_forecasts",
        "forecast_foundation",
        "forecast_panel_from_csv",
    ],
    "patterns": [
        "detect_peaks",
        "count_peaks",
        "analyze_recurrence",
        "analyze_matrix_profile",
        "find_motifs",
        "find_discords",
        "segment_changepoint",
        "segment_fluss",
    ],
    "classification": [
        "knn_classify",
        "rocket_classify",
        "hivecote_classify",
        "compare_classifiers",
        "select_window_size",
        "select_window_size_from_csv",
        "evaluate_windowed_classifier",
        "evaluate_windowed_classifier_from_csv",
    ],
    "spectral": [
        "compute_psd",
        "detect_periodicity",
        "compute_coherence",
    ],
    "statistics": [
        "describe_series",
        "compute_autocorrelation",
        "compare_series_stats",
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
        Bundle name: "minimal", "standard", "full", "all", "orchestrator",
        or a category name

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

    if name == "minimal":
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
            f"Unknown bundle: {name}. Available: minimal, standard, full, all, "
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

    if name == "minimal":
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
        "minimal": {
            "count": len(MINIMAL_BUNDLE),
            "description": f"One entry point per task family ({len(MINIMAL_BUNDLE)} tools)",
            "tools": MINIMAL_BUNDLE,
        },
        "standard": {
            "count": len(STANDARD_BUNDLE),
            "description": f"Well-rounded set across task families ({len(STANDARD_BUNDLE)} tools)",
            "tools": STANDARD_BUNDLE,
        },
        "full": {
            "count": len(FULL_BUNDLE),
            "description": f"Comprehensive set for in-depth analysis ({len(FULL_BUNDLE)} tools)",
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
            "stl_decompose",
            "mstl_decompose",
            "holt_winters_decompose",
            "detect_periodicity",
            "describe_series",
        ],
        "forecasting": [
            "forecast_arima",
            "forecast_ets",
            "forecast_seasonal_naive",
            "forecast_theta",
            "forecast_ensemble",
            "compare_forecasts",
            "forecast_foundation",
            "forecast_panel_from_csv",
            "detect_periodicity",
        ],
        "patterns": list(CATEGORY_BUNDLES["patterns"]),
        "classification": list(CATEGORY_BUNDLES["classification"]),
        "turbulence": [
            "compute_psd",
            "compute_coherence",
            "detect_periodicity",
            "describe_series",
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
