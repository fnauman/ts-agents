"""Forecasting tab component.

This component provides time series forecasting with:
- Multiple methods (ARIMA, ETS, Theta, Ensemble)
- Horizon configuration
- Confidence intervals visualization
- Method comparison
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple

from ...config import AVAILABLE_RUNS, REAL_VARIABLES
from ..state import UIState, load_series_data


def create_forecasting_tab(state: gr.State):
    """Create the forecasting analysis tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    """
    gr.Markdown("""
    ## Forecasting

    Generate forecasts using statistical methods with confidence intervals.
    Compare methods using holdout validation.
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

            gr.Markdown("### Forecasting Methods")
            methods = gr.CheckboxGroup(
                choices=["ARIMA", "ETS", "Theta", "Ensemble"],
                value=["ARIMA"],
                label="Select methods to run"
            )

            gr.Markdown("### Parameters")
            horizon = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="Forecast Horizon"
            )
            confidence = gr.Slider(
                minimum=0.50,
                maximum=0.99,
                value=0.95,
                step=0.05,
                label="Confidence Level"
            )

            with gr.Accordion("Validation Options", open=False):
                validation_split = gr.Slider(
                    minimum=0.0,
                    maximum=0.3,
                    value=0.1,
                    step=0.05,
                    label="Validation Split",
                    info="Fraction of data for validation (0 = no validation)"
                )

            with gr.Row():
                run_btn = gr.Button("Run Forecast", variant="primary")
                compare_btn = gr.Button("Compare Methods")

        # Right column: Results
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Forecast Results")

            with gr.Tabs():
                with gr.Tab("Metrics"):
                    metrics_output = gr.Dataframe(
                        label="Forecast Metrics",
                        headers=["Method", "MAE", "RMSE", "MAPE (%)"],
                        row_count=4
                    )
                with gr.Tab("Forecast Values"):
                    forecast_table = gr.Dataframe(
                        label="Forecast Values",
                        headers=["Step", "Forecast", "Lower", "Upper"],
                        row_count=10
                    )
                with gr.Tab("Recommendation"):
                    recommendation = gr.Markdown(label="Recommendation")

            status_text = gr.Textbox(label="Status", interactive=False)

    # Note: load_series_data is now imported from ..state

    def run_forecast(
        run_id: str,
        variable: str,
        methods: list,
        horizon: int,
        confidence: float,
        validation_split: float,
        ui_state: UIState
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, str, str, UIState]:
        """Run selected forecasting methods."""
        import matplotlib.pyplot as plt

        if not methods:
            return None, pd.DataFrame(), pd.DataFrame(), "", "Please select at least one method", ui_state

        # Load data
        series = load_series_data(run_id, variable)
        if series is None:
            return None, pd.DataFrame(), pd.DataFrame(), "", f"Failed to load data", ui_state

        # Split data if validation is enabled
        if validation_split > 0:
            n_val = int(len(series) * validation_split)
            train = series[:-n_val]
            actual = series[-n_val:]
        else:
            train = series
            actual = None

        # Import forecasting functions
        from ...core.forecasting import (
            forecast_arima,
            forecast_ets,
            forecast_theta,
            forecast_ensemble,
        )

        method_funcs = {
            "ARIMA": forecast_arima,
            "ETS": forecast_ets,
            "Theta": forecast_theta,
            "Ensemble": forecast_ensemble,
        }

        results = {}
        metrics_data = []

        forecast_horizon = n_val if validation_split > 0 else horizon
        level = [int(confidence * 100)] if confidence is not None else None

        for method in methods:
            if method not in method_funcs:
                continue

            try:
                if method == "Ensemble":
                    raw_result = method_funcs[method](train, horizon=forecast_horizon)
                    from ...core.base import ForecastResult
                    ensemble_forecast = raw_result.get_ensemble()
                    result = ForecastResult(
                        method="ensemble",
                        forecast=ensemble_forecast,
                        horizon=forecast_horizon,
                        lower_bound=None,
                        upper_bound=None,
                        confidence_level=confidence,
                    )
                else:
                    result = method_funcs[method](
                        train,
                        horizon=forecast_horizon,
                        level=level,
                    )
                results[method] = result

                # Compute validation metrics if available
                if actual is not None and result.forecast is not None:
                    forecast = result.forecast[:len(actual)]
                    mae = np.mean(np.abs(forecast - actual))
                    rmse = np.sqrt(np.mean((forecast - actual) ** 2))
                    mape = np.mean(np.abs((forecast - actual) / (actual + 1e-8))) * 100
                    metrics_data.append([method, f"{mae:.4f}", f"{rmse:.4f}", f"{mape:.2f}"])
                else:
                    metrics_data.append([method, "N/A", "N/A", "N/A"])

            except Exception as e:
                metrics_data.append([method, "Error", str(e)[:40], ""])

        if not results:
            return None, pd.DataFrame(), pd.DataFrame(), "", "All methods failed", ui_state

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(range(len(train)), train, 'k-', label='Historical', linewidth=1)

        if actual is not None:
            ax.plot(range(len(train), len(series)), actual, 'k--', label='Actual', linewidth=1)

        # Plot forecasts
        colors = plt.cm.tab10.colors
        forecast_start = len(train)

        for i, (method, result) in enumerate(results.items()):
            forecast_idx = range(forecast_start, forecast_start + len(result.forecast))
            ax.plot(forecast_idx, result.forecast, '--',
                   color=colors[i % len(colors)],
                   label=f'{method}', linewidth=1.5)

            # Plot confidence intervals if available
            if result.lower_bound is not None and result.upper_bound is not None:
                ax.fill_between(
                    forecast_idx,
                    result.lower_bound,
                    result.upper_bound,
                    color=colors[i % len(colors)],
                    alpha=0.2
                )

        ax.axvline(x=len(train), color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        ax.set_title(f"Forecast: {variable} ({run_id})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.tight_layout()

        # Create metrics dataframe
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=["Method", "MAE", "RMSE", "MAPE (%)"]
        )

        # Create forecast values table (for first method)
        first_result = list(results.values())[0]
        forecast_data = []
        for i in range(min(20, len(first_result.forecast))):
            row = [
                i + 1,
                f"{first_result.forecast[i]:.4f}",
                f"{first_result.lower_bound[i]:.4f}" if first_result.lower_bound is not None else "N/A",
                f"{first_result.upper_bound[i]:.4f}" if first_result.upper_bound is not None else "N/A",
            ]
            forecast_data.append(row)

        forecast_df = pd.DataFrame(
            forecast_data,
            columns=["Step", "Forecast", "Lower", "Upper"]
        )

        # Update UI state
        if ui_state is not None:
            ui_state.add_analysis(
                "forecasting",
                {"methods": methods, "horizon": horizon, "run_id": run_id, "variable": variable},
                f"Ran {len(methods)} method(s)"
            )
            ui_state.current_forecast = {m: r.to_dict() for m, r in results.items()}

        status = f"Completed {len(results)} forecast(s)"
        return fig, metrics_df, forecast_df, "", status, ui_state

    def run_comparison(
        run_id: str,
        variable: str,
        horizon: int,
        ui_state: UIState
    ) -> Tuple[Any, pd.DataFrame, pd.DataFrame, str, str, UIState]:
        """Compare all forecasting methods."""
        import matplotlib.pyplot as plt

        # Load data
        series = load_series_data(run_id, variable)
        if series is None:
            return None, pd.DataFrame(), pd.DataFrame(), "", f"Failed to load data", ui_state

        # Use comparison module
        from ...core.comparison import (
            compare_forecasting_methods,
            plot_forecast_comparison,
        )

        try:
            comparison = compare_forecasting_methods(series, horizon=horizon)

            # Create comparison plot
            fig = plot_forecast_comparison(comparison, series)

            # Create metrics dataframe
            metrics_data = []
            for method in comparison.methods:
                m = comparison.metrics.get(method, {})
                if "error" not in m:
                    metrics_data.append([
                        method,
                        f"{m.get('mae', 0):.4f}",
                        f"{m.get('rmse', 0):.4f}",
                        f"{m.get('mape', 0):.2f}"
                    ])
                else:
                    metrics_data.append([method, "Error", m.get("error", "")[:40], ""])

            metrics_df = pd.DataFrame(
                metrics_data,
                columns=["Method", "MAE", "RMSE", "MAPE (%)"]
            )

            # Get best method forecast for table
            best_method = comparison.get_overall_best() if comparison.results else None
            forecast_data = []
            if best_method and best_method in comparison.results:
                result = comparison.results[best_method]
                for i in range(min(20, len(result.forecast))):
                    row = [
                        i + 1,
                        f"{result.forecast[i]:.4f}",
                        f"{result.lower_bound[i]:.4f}" if result.lower_bound is not None else "N/A",
                        f"{result.upper_bound[i]:.4f}" if result.upper_bound is not None else "N/A",
                    ]
                    forecast_data.append(row)

            forecast_df = pd.DataFrame(
                forecast_data,
                columns=["Step", "Forecast", "Lower", "Upper"]
            )

            # Update UI state
            if ui_state is not None:
                ui_state.add_analysis(
                    "forecasting_comparison",
                    {"horizon": horizon, "run_id": run_id, "variable": variable},
                    f"Best method: {comparison.get_overall_best() if comparison.results else 'N/A'}"
                )

            best = comparison.get_overall_best() if comparison.results else "N/A"
            status = f"Comparison complete. Best: {best}"
            return fig, metrics_df, forecast_df, comparison.recommendation, status, ui_state

        except Exception as e:
            return None, pd.DataFrame(), pd.DataFrame(), "", f"Comparison failed: {e}", ui_state

    # Connect events
    run_btn.click(
        run_forecast,
        inputs=[run_id, variable, methods, horizon, confidence, validation_split, state],
        outputs=[plot_output, metrics_output, forecast_table, recommendation, status_text, state]
    )

    compare_btn.click(
        run_comparison,
        inputs=[run_id, variable, horizon, state],
        outputs=[plot_output, metrics_output, forecast_table, recommendation, status_text, state]
    )
