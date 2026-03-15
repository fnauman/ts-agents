"""Method Comparison - Compare multiple analysis methods and generate recommendations.

This module provides utilities for:
- Running multiple methods on the same data
- Computing comparison metrics
- Generating rankings and recommendations
- Creating comparison visualizations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np

from .base import AnalysisResult, AnalysisCategory


@dataclass
class ComparisonResult:
    """Result of comparing multiple analysis methods.

    Attributes
    ----------
    category : str
        Analysis category (e.g., "decomposition", "forecasting")
    methods : List[str]
        Names of methods compared
    results : Dict[str, AnalysisResult]
        Individual results from each method
    metrics : Dict[str, Dict[str, float]]
        Metrics for each method: {method: {metric: value}}
    rankings : Dict[str, List[str]]
        Rankings for each metric: {metric: [best, ..., worst]}
    recommendation : str
        Text recommendation based on comparison
    computation_times : Dict[str, float]
        Time taken by each method (seconds)
    """
    category: str
    methods: List[str]
    results: Dict[str, AnalysisResult]
    metrics: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    recommendation: str = ""
    computation_times: Dict[str, float] = field(default_factory=dict)

    def get_best_method(self, metric: str) -> str:
        """Get the best method for a specific metric.

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        str
            Name of the best method
        """
        if metric not in self.rankings:
            raise KeyError(f"Metric '{metric}' not found. Available: {list(self.rankings.keys())}")
        return self.rankings[metric][0]

    def get_overall_best(self) -> str:
        """Get the overall best method (most first-place rankings).

        Returns
        -------
        str
            Name of the best overall method
        """
        wins = {m: 0 for m in self.methods}
        for ranking in self.rankings.values():
            if ranking:
                wins[ranking[0]] += 1
        return max(wins.items(), key=lambda x: x[1])[0]

    def to_table(self) -> str:
        """Format comparison as a markdown table.

        Returns
        -------
        str
            Markdown table comparing methods
        """
        if not self.metrics:
            return "No metrics available for comparison."

        # Get all metric names
        all_metrics = set()
        for method_metrics in self.metrics.values():
            all_metrics.update(method_metrics.keys())
        all_metrics = sorted(all_metrics)

        # Build header
        lines = ["| Method | " + " | ".join(all_metrics) + " |"]
        lines.append("|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|")

        # Build rows
        for method in self.methods:
            values = []
            for metric in all_metrics:
                val = self.metrics.get(method, {}).get(metric, float('nan'))
                if np.isnan(val):
                    values.append("N/A")
                else:
                    values.append(f"{val:.4f}")
            lines.append(f"| {method} | " + " | ".join(values) + " |")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category,
            "methods": self.methods,
            "metrics": self.metrics,
            "rankings": self.rankings,
            "recommendation": self.recommendation,
            "computation_times": self.computation_times,
        }


# =============================================================================
# Decomposition Comparison
# =============================================================================

def compare_decomposition_methods(
    series: np.ndarray,
    methods: Optional[List[str]] = None,
    period: Optional[int] = None,
    **kwargs
) -> ComparisonResult:
    """Compare multiple decomposition methods on the same data.

    Parameters
    ----------
    series : np.ndarray
        Time series to decompose
    methods : List[str], optional
        Methods to compare. Default: ["stl", "mstl", "holt_winters"]
    period : int, optional
        Seasonal period for methods that need it

    Returns
    -------
    ComparisonResult
        Comparison with metrics, rankings, and recommendation

    Examples
    --------
    >>> result = compare_decomposition_methods(series, period=150)
    >>> print(result.recommendation)
    >>> print(result.to_table())
    """
    import time
    from .decomposition import (
        stl_decompose,
        mstl_decompose,
        holt_winters_decompose,
    )

    if methods is None:
        methods = ["stl", "mstl", "holt_winters"]

    method_funcs = {
        "stl": stl_decompose,
        "mstl": mstl_decompose,
        "holt_winters": holt_winters_decompose,
    }

    results = {}
    metrics = {}
    computation_times = {}

    for method in methods:
        if method not in method_funcs:
            continue

        try:
            start_time = time.time()

            # Call the method
            if method in {"stl", "holt_winters"}:
                result = method_funcs[method](series, period=period, **kwargs)
            else:
                periods = [period] if period else None
                result = method_funcs[method](series, periods=periods, **kwargs)

            computation_times[method] = time.time() - start_time
            results[method] = result

            # Extract metrics
            metrics[method] = {
                "residual_variance": result.residual_variance,
                "trend_smoothness": result.trend_smoothness,
                "seasonal_strength": result.seasonal_strength,
            }

        except Exception as e:
            # Log error but continue with other methods
            metrics[method] = {"error": str(e)}

    # Compute rankings
    rankings = _compute_rankings(metrics)

    # Generate recommendation
    recommendation = _generate_decomposition_recommendation(results, metrics, rankings)

    return ComparisonResult(
        category="decomposition",
        methods=[m for m in methods if m in results],
        results=results,
        metrics=metrics,
        rankings=rankings,
        recommendation=recommendation,
        computation_times=computation_times,
    )


# =============================================================================
# Forecasting Comparison
# =============================================================================

def compare_forecasting_methods(
    series: np.ndarray,
    horizon: int,
    methods: Optional[List[str]] = None,
    validation_size: Optional[int] = None,
    **kwargs
) -> ComparisonResult:
    """Compare multiple forecasting methods using holdout validation.

    Parameters
    ----------
    series : np.ndarray
        Historical time series
    horizon : int
        Forecast horizon
    methods : List[str], optional
        Methods to compare. Default: ["arima", "ets", "theta"]
    validation_size : int, optional
        Size of validation set (default: horizon)

    Returns
    -------
    ComparisonResult
        Comparison with metrics, rankings, and recommendation
    """
    import time
    from .forecasting import (
        forecast_arima,
        forecast_ets,
        forecast_theta,
    )

    if methods is None:
        methods = ["arima", "ets", "theta"]

    if validation_size is None:
        validation_size = horizon

    method_funcs = {
        "arima": forecast_arima,
        "ets": forecast_ets,
        "theta": forecast_theta,
    }

    # Split data for validation
    train = series[:-validation_size]
    actual = series[-validation_size:]

    results = {}
    metrics = {}
    computation_times = {}

    for method in methods:
        if method not in method_funcs:
            continue

        try:
            start_time = time.time()

            # Forecast on training data
            result = method_funcs[method](train, horizon=validation_size, **kwargs)
            computation_times[method] = time.time() - start_time
            results[method] = result

            # Compute error metrics
            forecast = result.forecast[:len(actual)]
            mae = np.mean(np.abs(forecast - actual))
            rmse = np.sqrt(np.mean((forecast - actual) ** 2))
            mape = np.mean(np.abs((forecast - actual) / actual)) * 100 if np.all(actual != 0) else np.nan

            metrics[method] = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }

        except Exception as e:
            metrics[method] = {"error": str(e)}

    # Compute rankings (lower is better for error metrics)
    rankings = _compute_rankings(metrics, lower_is_better=["mae", "rmse", "mape"])

    # Generate recommendation
    recommendation = _generate_forecasting_recommendation(results, metrics, rankings)

    return ComparisonResult(
        category="forecasting",
        methods=[m for m in methods if m in results],
        results=results,
        metrics=metrics,
        rankings=rankings,
        recommendation=recommendation,
        computation_times=computation_times,
    )


# =============================================================================
# Generic Comparison
# =============================================================================

def compare_methods(
    series: np.ndarray,
    category: str,
    methods: Optional[List[str]] = None,
    **kwargs
) -> ComparisonResult:
    """Compare methods within a category.

    This is a convenience function that dispatches to category-specific
    comparison functions.

    Parameters
    ----------
    series : np.ndarray
        Time series data
    category : str
        Category: "decomposition", "forecasting", etc.
    methods : List[str], optional
        Methods to compare
    **kwargs
        Additional arguments for the specific comparison function

    Returns
    -------
    ComparisonResult
        Comparison results

    Examples
    --------
    >>> result = compare_methods(series, "decomposition", period=150)
    >>> print(result.recommendation)
    """
    category = category.lower()

    if category == "decomposition":
        return compare_decomposition_methods(series, methods=methods, **kwargs)
    elif category == "forecasting":
        return compare_forecasting_methods(series, methods=methods, **kwargs)
    else:
        raise ValueError(f"Comparison not implemented for category: {category}")


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_rankings(
    metrics: Dict[str, Dict[str, float]],
    lower_is_better: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Compute rankings for each metric.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Metrics by method
    lower_is_better : List[str], optional
        Metrics where lower values are better

    Returns
    -------
    Dict[str, List[str]]
        Rankings: {metric: [best_method, ..., worst_method]}
    """
    if lower_is_better is None:
        lower_is_better = []

    # Get all metric names
    all_metrics = set()
    for method_metrics in metrics.values():
        for key in method_metrics.keys():
            if key != "error":
                all_metrics.add(key)

    rankings = {}
    for metric in all_metrics:
        # Collect (method, value) pairs
        values = []
        for method, method_metrics in metrics.items():
            if metric in method_metrics and not np.isnan(method_metrics[metric]):
                values.append((method, method_metrics[metric]))

        if not values:
            continue

        # Sort
        reverse = metric not in lower_is_better
        values.sort(key=lambda x: x[1], reverse=reverse)

        rankings[metric] = [v[0] for v in values]

    return rankings


def _generate_decomposition_recommendation(
    results: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    rankings: Dict[str, List[str]],
) -> str:
    """Generate text recommendation for decomposition methods."""
    if not rankings:
        return "Unable to generate recommendation - no valid results."

    parts = ["## Decomposition Method Comparison\n"]

    # Find overall best
    wins = {m: 0 for m in results.keys()}
    for ranking in rankings.values():
        if ranking:
            wins[ranking[0]] += 1

    if wins:
        best_method = max(wins.items(), key=lambda x: x[1])[0]
        parts.append(f"**Recommended: {best_method.upper()}**\n")

    # Explain rankings
    if "residual_variance" in rankings:
        best_resid = rankings["residual_variance"][0]
        parts.append(f"- Lowest residual variance: **{best_resid}**")

    if "trend_smoothness" in rankings:
        smoothest = rankings["trend_smoothness"][0]
        parts.append(f"- Smoothest trend: **{smoothest}**")

    if "seasonal_strength" in rankings:
        strongest = rankings["seasonal_strength"][0]
        parts.append(f"- Strongest seasonality: **{strongest}**")

    # Add specific recommendations
    parts.append("\n### Method Notes:")
    parts.append("- **STL**: Good general choice, robust to outliers")
    parts.append("- **MSTL**: Use when multiple seasonal periods exist")
    parts.append("- **Holt-Winters**: Best when forecasting is the end goal")

    return "\n".join(parts)


def _generate_forecasting_recommendation(
    results: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    rankings: Dict[str, List[str]],
) -> str:
    """Generate text recommendation for forecasting methods."""
    if not rankings:
        return "Unable to generate recommendation - no valid results."

    parts = ["## Forecasting Method Comparison\n"]

    # Find best by RMSE (most commonly used)
    if "rmse" in rankings and rankings["rmse"]:
        best_method = rankings["rmse"][0]
        best_rmse = metrics[best_method]["rmse"]
        parts.append(f"**Recommended: {best_method.upper()}** (RMSE: {best_rmse:.4f})\n")

    # Show all metrics
    parts.append("### Accuracy Metrics:")
    for method in results.keys():
        if method in metrics:
            m = metrics[method]
            if "error" not in m:
                parts.append(
                    f"- **{method}**: MAE={m.get('mae', float('nan')):.4f}, "
                    f"RMSE={m.get('rmse', float('nan')):.4f}, "
                    f"MAPE={m.get('mape', float('nan')):.1f}%"
                )

    # Add method notes
    parts.append("\n### Method Notes:")
    method_notes = {
        "arima": "- **ARIMA**: Best for non-seasonal or differenced series",
        "ets": "- **ETS**: Best for seasonal data with clear patterns",
        "theta": "- **Theta**: Simple but effective, won M3 competition",
    }
    for method in results.keys():
        note = method_notes.get(method.lower())
        if note:
            parts.append(note)

    return "\n".join(parts)


# =============================================================================
# Visualization Support
# =============================================================================

def plot_decomposition_comparison(
    comparison: ComparisonResult,
    series: np.ndarray,
    figsize: Tuple[int, int] = (14, 10),
) -> Any:
    """Create comparison plot for decomposition methods.

    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result from compare_decomposition_methods
    series : np.ndarray
        Original series
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Comparison figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    n_methods = len(comparison.results)
    fig, axes = plt.subplots(n_methods + 1, 3, figsize=figsize)

    # Plot original series
    axes[0, 1].plot(series, 'k-', alpha=0.7)
    axes[0, 1].set_title("Original Series")
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    # Plot each method's decomposition
    for i, (method, result) in enumerate(comparison.results.items(), 1):
        # Trend
        axes[i, 0].plot(result.trend)
        axes[i, 0].set_ylabel(method.upper())
        if i == 1:
            axes[i, 0].set_title("Trend")

        # Seasonal
        axes[i, 1].plot(result.seasonal)
        if i == 1:
            axes[i, 1].set_title("Seasonal")

        # Residual
        axes[i, 2].plot(result.residual)
        if i == 1:
            axes[i, 2].set_title("Residual")

    plt.tight_layout()
    return fig


def plot_forecast_comparison(
    comparison: ComparisonResult,
    series: np.ndarray,
    figsize: Tuple[int, int] = (12, 6),
) -> Any:
    """Create comparison plot for forecasting methods.

    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result from compare_forecasting_methods
    series : np.ndarray
        Original series (including validation portion)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Comparison figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(series, 'k-', label='Actual', linewidth=2)

    # Check if there are any results to plot
    if not comparison.results:
        ax.set_title("Forecast Comparison (no results)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.tight_layout()
        return fig

    # Plot each method's forecast
    colors = plt.cm.tab10.colors
    forecast_len = None
    for i, (method, result) in enumerate(comparison.results.items()):
        forecast_len = len(result.forecast)
        n_train = len(series) - forecast_len
        forecast_idx = np.arange(n_train, n_train + forecast_len)
        ax.plot(forecast_idx, result.forecast, '--',
               color=colors[i % len(colors)],
               label=f'{method.upper()} (RMSE: {comparison.metrics[method].get("rmse", float("nan")):.4f})')

    ax.axvline(x=len(series) - forecast_len, color='gray',
               linestyle=':', label='Forecast start')
    ax.legend()
    ax.set_title("Forecast Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    plt.tight_layout()
    return fig
