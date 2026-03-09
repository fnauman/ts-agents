"""Decomposition analysis tab component.

This component provides manual decomposition analysis with:
- Method selection (STL, MSTL, HP Filter, Holt-Winters)
- Parameter configuration
- Visualization of trend, seasonal, and residual components
- Method comparison
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple

from ...config import AVAILABLE_RUNS, REAL_VARIABLES
from ..state import UIState, load_series_data


def create_decomposition_tab(state: gr.State):
    """Create the decomposition analysis tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    """
    gr.Markdown("""
    ## Decomposition Analysis

    Decompose time series into trend, seasonal, and residual components.
    Select a method or compare all methods to find the best fit.
    """)

    with gr.Row():
        # Left column: Parameters
        with gr.Column(scale=1):
            gr.Markdown("### Data Selection")
            run_id = gr.Dropdown(
                choices=AVAILABLE_RUNS,
                label="Run ID",
                value=AVAILABLE_RUNS[0] if AVAILABLE_RUNS else None
            )
            variable = gr.Dropdown(
                choices=REAL_VARIABLES,
                label="Variable",
                value=REAL_VARIABLES[0] if REAL_VARIABLES else None
            )

            gr.Markdown("### Decomposition Methods")
            methods = gr.CheckboxGroup(
                choices=["STL", "MSTL", "HP Filter", "Holt-Winters"],
                value=["STL"],
                label="Select methods to run"
            )

            gr.Markdown("### Parameters")
            period = gr.Slider(
                minimum=0,
                maximum=500,
                value=0,
                step=10,
                label="Seasonal Period (0 = auto-detect)"
            )
            robust = gr.Checkbox(value=True, label="Robust fitting (STL)")

            with gr.Accordion("Advanced Options", open=False):
                hp_lambda = gr.Number(
                    value=1600,
                    label="HP Filter Lambda",
                    info="Larger values = smoother trend"
                )
                mstl_periods = gr.Textbox(
                    value="",
                    label="MSTL Periods (comma-separated)",
                    placeholder="e.g., 12, 52 for monthly and weekly"
                )

            with gr.Row():
                run_btn = gr.Button("Run Selected", variant="primary")
                compare_btn = gr.Button("Compare All")

        # Right column: Results
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Decomposition Results")

            with gr.Tabs():
                with gr.Tab("Metrics"):
                    metrics_output = gr.Dataframe(
                        label="Method Metrics",
                        headers=["Method", "Residual Var", "Trend Smoothness", "Seasonal Strength"],
                        row_count=4
                    )
                with gr.Tab("Recommendation"):
                    recommendation = gr.Markdown(label="Recommendation")

            status_text = gr.Textbox(label="Status", interactive=False)

    # Note: load_series_data is now imported from ..state

    def run_decomposition(
        run_id: str,
        variable: str,
        methods: list,
        period: int,
        robust: bool,
        hp_lambda: float,
        mstl_periods: str,
        ui_state: UIState
    ) -> Tuple[Any, pd.DataFrame, str, str, UIState]:
        """Run selected decomposition methods."""
        import matplotlib.pyplot as plt

        if not methods:
            return None, pd.DataFrame(), "", "Please select at least one method", ui_state

        # Load data
        series = load_series_data(run_id, variable)
        if series is None:
            return None, pd.DataFrame(), "", f"Failed to load data for {run_id}/{variable}", ui_state

        period_val = period if period > 0 else None

        # Import decomposition functions
        from ...core.decomposition import (
            stl_decompose,
            mstl_decompose,
            hp_filter,
            holt_winters_decompose,
        )

        method_map = {
            "STL": ("stl", lambda: stl_decompose(series, period=period_val, robust=robust)),
            "MSTL": ("mstl", lambda: mstl_decompose(
                series,
                periods=[int(p.strip()) for p in mstl_periods.split(",") if p.strip()] if mstl_periods.strip() else None
            )),
            "HP Filter": ("hp_filter", lambda: hp_filter(series, lamb=hp_lambda)),
            "Holt-Winters": ("holt_winters", lambda: holt_winters_decompose(series, period=period_val)),
        }

        results = {}
        metrics_data = []

        for method in methods:
            if method not in method_map:
                continue

            method_key, method_func = method_map[method]
            try:
                result = method_func()
                results[method] = result
                metrics_data.append([
                    method,
                    f"{result.residual_variance:.4f}",
                    f"{result.trend_smoothness:.6f}",
                    f"{result.seasonal_strength:.4f}"
                ])
            except Exception as e:
                metrics_data.append([method, "Error", str(e)[:50], ""])

        if not results:
            return None, pd.DataFrame(), "", "All methods failed", ui_state

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods, 3, figsize=(14, 3 * n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)

        for i, (method, result) in enumerate(results.items()):
            # Trend
            axes[i, 0].plot(result.trend, color='blue')
            axes[i, 0].set_ylabel(method)
            if i == 0:
                axes[i, 0].set_title("Trend")

            # Seasonal
            axes[i, 1].plot(result.seasonal, color='green')
            if i == 0:
                axes[i, 1].set_title("Seasonal")

            # Residual
            axes[i, 2].plot(result.residual, color='red', alpha=0.7)
            if i == 0:
                axes[i, 2].set_title("Residual")

        plt.suptitle(f"Decomposition: {variable} ({run_id})", fontsize=12)
        plt.tight_layout()

        # Create metrics dataframe
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=["Method", "Residual Var", "Trend Smoothness", "Seasonal Strength"]
        )

        # Update UI state
        if ui_state is not None:
            ui_state.add_analysis(
                "decomposition",
                {"methods": methods, "period": period_val, "run_id": run_id, "variable": variable},
                f"Ran {len(methods)} method(s)"
            )
            ui_state.current_decomposition = {m: r.to_dict() for m, r in results.items()}

        status = f"Completed {len(results)} decomposition(s)"
        return fig, metrics_df, "", status, ui_state

    def run_comparison(
        run_id: str,
        variable: str,
        period: int,
        ui_state: UIState
    ) -> Tuple[Any, pd.DataFrame, str, str, UIState]:
        """Run all decomposition methods and compare."""
        import matplotlib.pyplot as plt

        # Load data
        series = load_series_data(run_id, variable)
        if series is None:
            return None, pd.DataFrame(), "", f"Failed to load data for {run_id}/{variable}", ui_state

        period_val = period if period > 0 else None

        # Use comparison module
        from ...core.comparison import (
            compare_decomposition_methods,
            plot_decomposition_comparison,
        )

        try:
            comparison = compare_decomposition_methods(series, period=period_val)

            # Create comparison plot
            fig = plot_decomposition_comparison(comparison, series)

            # Create metrics dataframe
            metrics_data = []
            for method in comparison.methods:
                m = comparison.metrics.get(method, {})
                if "error" not in m:
                    metrics_data.append([
                        method,
                        f"{m.get('residual_variance', 0):.4f}",
                        f"{m.get('trend_smoothness', 0):.6f}",
                        f"{m.get('seasonal_strength', 0):.4f}"
                    ])
                else:
                    metrics_data.append([method, "Error", m.get("error", "")[:50], ""])

            metrics_df = pd.DataFrame(
                metrics_data,
                columns=["Method", "Residual Var", "Trend Smoothness", "Seasonal Strength"]
            )

            # Update UI state
            if ui_state is not None:
                ui_state.add_analysis(
                    "decomposition_comparison",
                    {"period": period_val, "run_id": run_id, "variable": variable},
                    f"Best method: {comparison.get_overall_best()}"
                )

            status = f"Comparison complete. Best: {comparison.get_overall_best()}"
            return fig, metrics_df, comparison.recommendation, status, ui_state

        except Exception as e:
            return None, pd.DataFrame(), "", f"Comparison failed: {e}", ui_state

    # Connect events
    run_btn.click(
        run_decomposition,
        inputs=[run_id, variable, methods, period, robust, hp_lambda, mstl_periods, state],
        outputs=[plot_output, metrics_output, recommendation, status_text, state]
    )

    compare_btn.click(
        run_comparison,
        inputs=[run_id, variable, period, state],
        outputs=[plot_output, metrics_output, recommendation, status_text, state]
    )

    # Update state when data selection changes
    def update_data_selection(run_id: str, variable: str, ui_state: UIState):
        if ui_state is not None:
            ui_state.run_id = run_id
            ui_state.variable = variable
        return ui_state

    run_id.change(
        update_data_selection,
        inputs=[run_id, variable, state],
        outputs=[state]
    )

    variable.change(
        update_data_selection,
        inputs=[run_id, variable, state],
        outputs=[state]
    )
