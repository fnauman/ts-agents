"""Comparison tab component.

This component provides cross-run and cross-method comparisons:
- Compare statistics across runs
- Compare decomposition/forecasting methods
- Spectral analysis comparison
- Export comparison results
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, List

from ...config import AVAILABLE_RUNS, REAL_VARIABLES
from ..state import UIState


def create_comparison_tab(state: gr.State):
    """Create the comparison tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    """
    gr.Markdown("""
    ## Cross-Run Comparison

    Compare time series characteristics and analysis results across different
    simulation runs to identify patterns and differences.
    """)

    with gr.Row():
        # Left column: Parameters
        with gr.Column(scale=1):
            gr.Markdown("### Data Selection")

            runs_to_compare = gr.CheckboxGroup(
                choices=AVAILABLE_RUNS,
                label="Runs to Compare",
                value=AVAILABLE_RUNS[:3] if len(AVAILABLE_RUNS) >= 3 else AVAILABLE_RUNS,
                info="Select 2 or more runs"
            )

            variables_to_compare = gr.CheckboxGroup(
                choices=REAL_VARIABLES,
                label="Variables to Compare",
                value=[REAL_VARIABLES[0]] if REAL_VARIABLES else [],
                info="Select one or more variables"
            )

            gr.Markdown("### Comparison Type")
            comparison_type = gr.Radio(
                choices=[
                    "Descriptive Statistics",
                    "Peak Analysis",
                    "Spectral Analysis",
                    "Correlation Analysis"
                ],
                value="Descriptive Statistics",
                label="Select Comparison"
            )

            with gr.Accordion("Spectral Options", open=False) as spectral_opts:
                freq_range = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.5,
                    step=0.01,
                    label="Max Frequency (Nyquist fraction)"
                )

            with gr.Row():
                run_btn = gr.Button("Run Comparison", variant="primary")
                export_btn = gr.Button("Export Results")

        # Right column: Results
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Comparison Results")

            with gr.Tabs():
                with gr.Tab("Summary Table"):
                    table_output = gr.Dataframe(
                        label="Comparison Results",
                        row_count=10
                    )

                with gr.Tab("Rankings"):
                    rankings_output = gr.Markdown(label="Rankings")

                with gr.Tab("Details"):
                    details_output = gr.Markdown(label="Detailed Analysis")

            status_text = gr.Textbox(label="Status", interactive=False)

    def load_series_data(run_id: str, variable: str) -> Optional[np.ndarray]:
        """Load time series data."""
        try:
            from ...data_access import get_series
            return get_series(run_id, variable)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def compare_descriptive_stats(
        runs: List[str],
        variables: List[str]
    ) -> Tuple[Any, pd.DataFrame, str, str]:
        """Compare descriptive statistics across runs."""
        import matplotlib.pyplot as plt
        from ...core.statistics import describe_series

        results = []

        for run_id in runs:
            for variable in variables:
                series = load_series_data(run_id, variable)
                if series is not None:
                    stats = describe_series(series)
                    results.append({
                        "Run": run_id,
                        "Variable": variable,
                        "Mean": stats.mean,
                        "Std": stats.std,
                        "Min": stats.min,
                        "Max": stats.max,
                        "RMS": stats.rms,
                    })

        if not results:
            return None, pd.DataFrame(), "", "No data loaded"

        df = pd.DataFrame(results)

        # Create visualization
        n_vars = len(variables)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean comparison
        ax = axes[0, 0]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Mean"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Mean")
        ax.legend()

        # Std comparison
        ax = axes[0, 1]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Std"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Standard Deviation")
        ax.legend()

        # RMS comparison
        ax = axes[1, 0]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["RMS"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("RMS")
        ax.legend()

        # Range comparison
        ax = axes[1, 1]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ranges = var_data["Max"].values - var_data["Min"].values
            ax.bar(np.arange(len(runs)) + i * 0.2, ranges,
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Range (Max - Min)")
        ax.legend()

        plt.tight_layout()

        # Rankings
        rankings = "### Rankings by Metric\n\n"
        for metric in ["Mean", "Std", "RMS"]:
            sorted_df = df.sort_values(metric, ascending=False)
            rankings += f"**{metric}** (highest first):\n"
            for i, row in sorted_df.head(5).iterrows():
                rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {row[metric]:.4f}\n"
            rankings += "\n"

        # Details
        details = "### Detailed Statistics\n\n"
        for run_id in runs:
            details += f"#### {run_id}\n"
            run_data = df[df["Run"] == run_id]
            for _, row in run_data.iterrows():
                details += f"- **{row['Variable']}**: mean={row['Mean']:.4f}, std={row['Std']:.4f}, rms={row['RMS']:.4f}\n"
            details += "\n"

        return fig, df, rankings, details

    def compare_peak_analysis(
        runs: List[str],
        variables: List[str]
    ) -> Tuple[Any, pd.DataFrame, str, str]:
        """Compare peak characteristics across runs."""
        import matplotlib.pyplot as plt
        from ...core.patterns import detect_peaks

        results = []

        for run_id in runs:
            for variable in variables:
                series = load_series_data(run_id, variable)
                if series is not None:
                    try:
                        peak_result = detect_peaks(series, distance=10)
                        results.append({
                            "Run": run_id,
                            "Variable": variable,
                            "Peak Count": peak_result.count,
                            "Mean Height": peak_result.mean_height,
                            "Mean Spacing": peak_result.mean_spacing,
                            "Regularity": peak_result.regularity,
                        })
                    except Exception as e:
                        print(f"Error analyzing {run_id}/{variable}: {e}")

        if not results:
            return None, pd.DataFrame(), "", "No data loaded"

        df = pd.DataFrame(results)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        n_vars = len(variables)

        # Peak count comparison
        ax = axes[0]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Peak Count"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Peak Count")
        ax.legend()

        # Mean height comparison
        ax = axes[1]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Mean Height"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Mean Peak Height")
        ax.legend()

        # Mean spacing comparison
        ax = axes[2]
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Mean Spacing"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Mean Peak Spacing")
        ax.legend()

        plt.tight_layout()

        # Rankings
        rankings = "### Peak Analysis Rankings\n\n"
        rankings += "**Most Peaks**:\n"
        sorted_df = df.sort_values("Peak Count", ascending=False)
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {int(row['Peak Count'])} peaks\n"

        rankings += "\n**Most Regular**:\n"
        regular_df = df[df["Regularity"] == "regular"]
        for _, row in regular_df.iterrows():
            rankings += f"  - {row['Run']}/{row['Variable']}\n"

        # Details
        details = "### Peak Analysis Details\n\n"
        for _, row in df.iterrows():
            details += f"- **{row['Run']}/{row['Variable']}**: {int(row['Peak Count'])} peaks, "
            details += f"height={row['Mean Height']:.4f}, spacing={row['Mean Spacing']:.1f}, "
            details += f"regularity={row['Regularity']}\n"

        return fig, df, rankings, details

    def compare_spectral_analysis(
        runs: List[str],
        variables: List[str],
        max_freq: float
    ) -> Tuple[Any, pd.DataFrame, str, str]:
        """Compare spectral characteristics across runs."""
        import matplotlib.pyplot as plt
        from ...core.spectral import compute_psd, detect_periodicity

        results = []
        psd_data = {}

        for run_id in runs:
            for variable in variables:
                series = load_series_data(run_id, variable)
                if series is not None:
                    try:
                        psd_result = compute_psd(series)
                        period_result = detect_periodicity(series)

                        results.append({
                            "Run": run_id,
                            "Variable": variable,
                            "Dominant Period": period_result.dominant_period,
                            "Dominant Freq": period_result.dominant_frequency,
                            "Confidence": period_result.confidence,
                            "Spectral Slope": psd_result.spectral_slope,
                        })

                        key = f"{run_id}/{variable}"
                        psd_data[key] = (psd_result.frequencies, psd_result.psd)

                    except Exception as e:
                        print(f"Error analyzing {run_id}/{variable}: {e}")

        if not results:
            return None, pd.DataFrame(), "", "No data loaded"

        df = pd.DataFrame(results)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PSD comparison
        ax = axes[0]
        for key, (freqs, psd) in psd_data.items():
            mask = freqs <= max_freq
            ax.semilogy(freqs[mask], psd[mask], label=key, alpha=0.7)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_title("Power Spectral Density")
        ax.legend(fontsize=8)

        # Dominant period comparison
        ax = axes[1]
        n_vars = len(variables)
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Dominant Period"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Dominant Period")
        ax.legend()

        plt.tight_layout()

        # Rankings
        rankings = "### Spectral Analysis Rankings\n\n"
        rankings += "**Longest Dominant Period**:\n"
        sorted_df = df.sort_values("Dominant Period", ascending=False)
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {row['Dominant Period']:.2f}\n"

        rankings += "\n**Steepest Spectral Slope** (more turbulent):\n"
        sorted_df = df.sort_values("Spectral Slope", ascending=True)
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {row['Spectral Slope']:.4f}\n"

        # Details
        details = "### Spectral Analysis Details\n\n"
        for _, row in df.iterrows():
            details += f"- **{row['Run']}/{row['Variable']}**: "
            details += f"period={row['Dominant Period']:.2f}, "
            details += f"freq={row['Dominant Freq']:.4f}, "
            details += f"slope={row['Spectral Slope']:.4f}\n"

        return fig, df, rankings, details

    def compare_correlation_analysis(
        runs: List[str],
        variables: List[str]
    ) -> Tuple[Any, pd.DataFrame, str, str]:
        """Compare correlation structures across runs."""
        import matplotlib.pyplot as plt
        from ...core.statistics import compute_autocorrelation

        results = []
        acf_data = {}

        for run_id in runs:
            for variable in variables:
                series = load_series_data(run_id, variable)
                if series is not None:
                    try:
                        # Compute autocorrelation
                        max_lag = min(100, len(series) // 4)
                        lags = np.arange(max_lag)
                        acf = np.correlate(series - np.mean(series),
                                          series - np.mean(series), mode='full')
                        acf = acf[len(acf)//2:]
                        acf = acf[:max_lag] / acf[0]

                        # Find decorrelation time (first zero crossing)
                        zero_crossings = np.where(np.diff(np.sign(acf)))[0]
                        decorr_time = zero_crossings[0] if len(zero_crossings) > 0 else max_lag

                        results.append({
                            "Run": run_id,
                            "Variable": variable,
                            "Lag-1 ACF": acf[1] if len(acf) > 1 else 0,
                            "Lag-10 ACF": acf[10] if len(acf) > 10 else 0,
                            "Decorr Time": decorr_time,
                        })

                        key = f"{run_id}/{variable}"
                        acf_data[key] = (lags, acf)

                    except Exception as e:
                        print(f"Error analyzing {run_id}/{variable}: {e}")

        if not results:
            return None, pd.DataFrame(), "", "No data loaded"

        df = pd.DataFrame(results)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ACF comparison
        ax = axes[0]
        for key, (lags, acf) in acf_data.items():
            ax.plot(lags, acf, label=key, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation Function")
        ax.legend(fontsize=8)

        # Decorrelation time comparison
        ax = axes[1]
        n_vars = len(variables)
        for i, var in enumerate(variables):
            var_data = df[df["Variable"] == var]
            ax.bar(np.arange(len(runs)) + i * 0.2, var_data["Decorr Time"],
                   width=0.2, label=var)
        ax.set_xticks(np.arange(len(runs)) + 0.1 * n_vars)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_title("Decorrelation Time")
        ax.legend()

        plt.tight_layout()

        # Rankings
        rankings = "### Correlation Analysis Rankings\n\n"
        rankings += "**Highest Persistence (Lag-1 ACF)**:\n"
        sorted_df = df.sort_values("Lag-1 ACF", ascending=False)
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {row['Lag-1 ACF']:.4f}\n"

        rankings += "\n**Longest Decorrelation Time**:\n"
        sorted_df = df.sort_values("Decorr Time", ascending=False)
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rankings += f"  {i+1}. {row['Run']}/{row['Variable']}: {int(row['Decorr Time'])} lags\n"

        # Details
        details = "### Correlation Analysis Details\n\n"
        for _, row in df.iterrows():
            details += f"- **{row['Run']}/{row['Variable']}**: "
            details += f"lag-1={row['Lag-1 ACF']:.4f}, "
            details += f"lag-10={row['Lag-10 ACF']:.4f}, "
            details += f"decorr={int(row['Decorr Time'])} lags\n"

        return fig, df, rankings, details

    def run_comparison(
        runs: List[str],
        variables: List[str],
        comparison_type: str,
        freq_range: float,
        ui_state: UIState
    ) -> Tuple[Any, pd.DataFrame, str, str, str, UIState]:
        """Run the selected comparison."""
        if len(runs) < 2:
            return (None, pd.DataFrame(), "", "",
                   "Please select at least 2 runs to compare", ui_state)

        if not variables:
            return (None, pd.DataFrame(), "", "",
                   "Please select at least one variable", ui_state)

        try:
            if comparison_type == "Descriptive Statistics":
                fig, df, rankings, details = compare_descriptive_stats(runs, variables)
            elif comparison_type == "Peak Analysis":
                fig, df, rankings, details = compare_peak_analysis(runs, variables)
            elif comparison_type == "Spectral Analysis":
                fig, df, rankings, details = compare_spectral_analysis(runs, variables, freq_range)
            elif comparison_type == "Correlation Analysis":
                fig, df, rankings, details = compare_correlation_analysis(runs, variables)
            else:
                return (None, pd.DataFrame(), "", "",
                       f"Unknown comparison type: {comparison_type}", ui_state)

            # Update UI state
            if ui_state is not None:
                ui_state.add_analysis(
                    f"comparison_{comparison_type.lower().replace(' ', '_')}",
                    {"runs": runs, "variables": variables},
                    f"Compared {len(runs)} runs"
                )
                ui_state.current_comparison = {"type": comparison_type, "results": df.to_dict()}

            status = f"Comparison complete: {len(runs)} runs, {len(variables)} variables"
            return fig, df, rankings, details, status, ui_state

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, pd.DataFrame(), "", "", f"Comparison failed: {e}", ui_state

    def export_results(table_data: pd.DataFrame, comparison_type: str):
        """Export comparison results."""
        from datetime import datetime

        if table_data.empty:
            return gr.Info("No results to export")

        try:
            filename = f"comparison_{comparison_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            table_data.to_csv(filename, index=False)
            return gr.Info(f"Results exported to {filename}")
        except Exception as e:
            return gr.Warning(f"Export failed: {e}")

    # Connect events
    run_btn.click(
        run_comparison,
        inputs=[runs_to_compare, variables_to_compare, comparison_type, freq_range, state],
        outputs=[plot_output, table_output, rankings_output, details_output, status_text, state]
    )

    export_btn.click(
        export_results,
        inputs=[table_output, comparison_type],
        outputs=[]
    )
