"""Patterns analysis tab component.

This component provides pattern detection and analysis:
- Peak detection
- Recurrence plot analysis (RQA)
- Matrix profile (motifs and discords)
- Changepoint detection
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple

from ...config import AVAILABLE_RUNS, REAL_VARIABLES
from ..state import UIState, load_series_data


def create_patterns_tab(state: gr.State):
    """Create the patterns analysis tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    """
    gr.Markdown("""
    ## Pattern Analysis

    Detect and analyze patterns in time series including peaks, motifs, anomalies,
    recurrence structures, and regime changes.
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

            gr.Markdown("### Analysis Type")
            analysis_type = gr.Radio(
                choices=["Peak Detection", "Matrix Profile", "Recurrence Plot", "Changepoint Detection"],
                value="Peak Detection",
                label="Select Analysis"
            )

            # Peak detection parameters
            with gr.Group(visible=True) as peak_params:
                gr.Markdown("#### Peak Parameters")
                peak_height = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0,
                    step=0.1,
                    label="Min Height (0 = auto)"
                )
                peak_distance = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Min Distance"
                )
                peak_prominence = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=0.0,
                    step=0.1,
                    label="Min Prominence (0 = none)"
                )

            # Matrix profile parameters
            with gr.Group(visible=False) as mp_params:
                gr.Markdown("#### Matrix Profile Parameters")
                subsequence_length = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=5,
                    label="Subsequence Length"
                )
                n_motifs = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Motifs"
                )
                n_discords = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Discords"
                )

            # Recurrence plot parameters
            with gr.Group(visible=False) as rqa_params:
                gr.Markdown("#### Recurrence Plot Parameters")
                embedding_dim = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Embedding Dimension"
                )
                time_delay = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="Time Delay"
                )
                threshold = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                    label="Recurrence Threshold"
                )

            # Changepoint parameters
            with gr.Group(visible=False) as cp_params:
                gr.Markdown("#### Changepoint Parameters")
                n_changepoints = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max Changepoints"
                )
                cp_method = gr.Dropdown(
                    choices=["fluss", "ruptures"],
                    value="fluss",
                    label="Detection Method"
                )

            run_btn = gr.Button("Run Analysis", variant="primary")

        # Right column: Results
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Analysis Results")

            with gr.Tabs():
                with gr.Tab("Summary"):
                    summary_output = gr.Markdown(label="Summary")
                with gr.Tab("Details"):
                    details_output = gr.Dataframe(
                        label="Detailed Results",
                        row_count=10
                    )

            status_text = gr.Textbox(label="Status", interactive=False)

    # Note: load_series_data is now imported from ..state

    def run_peak_detection(
        series: np.ndarray,
        height: float,
        distance: int,
        prominence: float
    ) -> Tuple[Any, str, pd.DataFrame]:
        """Run peak detection analysis."""
        import matplotlib.pyplot as plt
        from ...core.patterns import detect_peaks

        # Prepare parameters
        height_val = height if height > 0 else None
        prominence_val = prominence if prominence > 0 else None

        result = detect_peaks(
            series,
            height=height_val,
            distance=distance,
            prominence=prominence_val
        )

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot series with peaks
        axes[0].plot(series, 'b-', alpha=0.7, label='Series')
        axes[0].plot(result.peak_indices, result.peak_values, 'ro',
                    markersize=8, label=f'Peaks ({result.count})')
        axes[0].set_title("Peak Detection")
        axes[0].legend()
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Value")

        # Plot peak spacing histogram
        if result.count > 1:
            spacings = np.diff(result.peak_indices)
            axes[1].hist(spacings, bins=min(30, len(spacings)), edgecolor='black')
            axes[1].set_title(f"Peak Spacing Distribution (mean={result.mean_spacing:.1f})")
            axes[1].set_xlabel("Spacing")
            axes[1].set_ylabel("Count")
        else:
            axes[1].text(0.5, 0.5, "Not enough peaks for spacing analysis",
                        ha='center', va='center')

        plt.tight_layout()

        # Summary
        summary = f"""
### Peak Detection Results

- **Total Peaks Found**: {result.count}
- **Mean Peak Height**: {result.mean_height:.4f}
- **Mean Peak Spacing**: {result.mean_spacing:.1f} samples
- **Spacing Std Dev**: {result.std_spacing:.1f}
- **Regularity**: {result.regularity}
"""

        # Details table
        details_data = []
        for i in range(min(20, result.count)):
            details_data.append([
                i + 1,
                result.peak_indices[i],
                f"{result.peak_values[i]:.4f}"
            ])

        details_df = pd.DataFrame(
            details_data,
            columns=["Peak #", "Index", "Value"]
        )

        return fig, summary, details_df

    def run_matrix_profile(
        series: np.ndarray,
        subsequence_length: int,
        n_motifs: int,
        n_discords: int
    ) -> Tuple[Any, str, pd.DataFrame]:
        """Run matrix profile analysis."""
        import matplotlib.pyplot as plt
        from ...core.patterns import analyze_matrix_profile

        result = analyze_matrix_profile(
            series,
            m=subsequence_length,
            max_motifs=n_motifs,
            max_discords=n_discords
        )

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot series
        axes[0].plot(series, 'b-', alpha=0.7)
        # Mark motifs
        for motif in result.motifs:
            axes[0].axvspan(motif.index, motif.index + subsequence_length,
                           alpha=0.3, color='green')
        # Mark discords
        for discord in result.discords:
            axes[0].axvspan(discord.index, discord.index + subsequence_length,
                           alpha=0.3, color='red')
        axes[0].set_title("Series with Motifs (green) and Discords (red)")

        # Plot matrix profile
        axes[1].plot(result.mp_values, 'purple', alpha=0.7)
        axes[1].set_title("Matrix Profile")
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Distance")

        # Plot top motif comparison
        if result.motifs:
            motif = result.motifs[0]
            subseq1 = series[motif.index:motif.index + subsequence_length]
            subseq2 = series[motif.neighbor_index:motif.neighbor_index + subsequence_length]
            axes[2].plot(subseq1, 'g-', label=f'Motif @ {motif.index}', linewidth=2)
            axes[2].plot(subseq2, 'g--', label=f'Match @ {motif.neighbor_index}', linewidth=2)
            axes[2].set_title(f"Top Motif (distance={motif.distance:.4f})")
            axes[2].legend()

        plt.tight_layout()

        # Summary
        summary = f"""
### Matrix Profile Results

- **Subsequence Length**: {result.subsequence_length}
- **MP Range**: [{result.mp_min:.4f}, {result.mp_max:.4f}]
- **Motifs Found**: {len(result.motifs)}
- **Discords Found**: {len(result.discords)}

#### Top Motifs (recurring patterns)
"""
        for i, motif in enumerate(result.motifs):
            summary += f"- Motif {i+1}: Index {motif.index}, matches {motif.neighbor_index} (dist={motif.distance:.4f})\n"

        summary += "\n#### Top Discords (anomalies)\n"
        for i, discord in enumerate(result.discords):
            summary += f"- Discord {i+1}: Index {discord.index} (dist={discord.distance:.4f})\n"

        # Details table
        details_data = []
        for i, motif in enumerate(result.motifs):
            details_data.append(["Motif", i+1, motif.index, motif.neighbor_index, f"{motif.distance:.4f}"])
        for i, discord in enumerate(result.discords):
            details_data.append(["Discord", i+1, discord.index, "-", f"{discord.distance:.4f}"])

        details_df = pd.DataFrame(
            details_data,
            columns=["Type", "#", "Index", "Match Index", "Distance"]
        )

        return fig, summary, details_df

    def run_recurrence_analysis(
        series: np.ndarray,
        embedding_dim: int,
        time_delay: int,
        threshold: float
    ) -> Tuple[Any, str, pd.DataFrame]:
        """Run recurrence quantification analysis."""
        import matplotlib.pyplot as plt
        from ...core.patterns import analyze_recurrence

        result = analyze_recurrence(
            series,
            threshold=threshold
        )

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Recurrence plot
        im = axes[0].imshow(result.recurrence_matrix, cmap='binary', origin='lower')
        axes[0].set_title(f"Recurrence Plot (threshold={result.threshold:.3f})")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Time")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Original series
        axes[1].plot(series, 'b-', alpha=0.7)
        axes[1].set_title("Time Series")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Value")

        plt.tight_layout()

        # Summary
        summary = f"""
### Recurrence Quantification Analysis

- **Embedding Dimension**: {embedding_dim}
- **Time Delay**: {time_delay}
- **Threshold**: {result.threshold:.4f}

#### RQA Metrics
- **Recurrence Rate (RR)**: {result.recurrence_rate:.4f}
  - Percentage of recurrent points
- **Determinism (DET)**: {result.determinism:.4f}
  - Fraction of recurrent points in diagonal lines (predictability)
- **Laminarity (LAM)**: {result.laminarity:.4f}
  - Fraction of recurrent points in vertical lines (intermittency)

#### Interpretation
- High RR: System revisits similar states frequently
- High DET: System behavior is deterministic/predictable
- High LAM: System exhibits laminar (stable) phases
"""

        # Details table
        details_df = pd.DataFrame([
            ["Recurrence Rate", f"{result.recurrence_rate:.4f}"],
            ["Determinism", f"{result.determinism:.4f}"],
            ["Laminarity", f"{result.laminarity:.4f}"],
            ["Threshold", f"{result.threshold:.4f}"],
        ], columns=["Metric", "Value"])

        return fig, summary, details_df

    def run_changepoint_detection(
        series: np.ndarray,
        n_changepoints: int,
        method: str
    ) -> Tuple[Any, str, pd.DataFrame]:
        """Run changepoint detection."""
        import matplotlib.pyplot as plt
        from ...core.patterns import segment_changepoint, segment_fluss

        if method == "fluss":
            result = segment_fluss(series, n_segments=n_changepoints + 1)
        else:
            result = segment_changepoint(series, n_segments=n_changepoints + 1)

        # Create visualization
        # Close any existing figures to prevent memory leaks in long-running sessions
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(series, 'b-', alpha=0.7, label='Series')

        # Mark changepoints
        colors = plt.cm.tab10.colors
        for i, cp in enumerate(result.changepoints):
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.8,
                      label='Changepoint' if i == 0 else '')

        # Shade segments
        boundaries = [0] + result.changepoints + [len(series)]
        for i in range(len(boundaries) - 1):
            ax.axvspan(boundaries[i], boundaries[i+1],
                      alpha=0.1, color=colors[i % len(colors)])

        ax.set_title(f"Changepoint Detection ({method.upper()})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

        plt.tight_layout()

        # Summary
        summary = f"""
### Changepoint Detection Results

- **Method**: {method.upper()}
- **Changepoints Found**: {len(result.changepoints)}
- **Number of Segments**: {result.n_segments}

#### Changepoint Locations
"""
        for i, cp in enumerate(result.changepoints):
            summary += f"- Changepoint {i+1}: Index {cp}\n"

        if result.segment_stats:
            summary += "\n#### Segment Statistics\n"
            for i, stats in enumerate(result.segment_stats):
                summary += f"- Segment {i+1}: mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f}\n"

        # Details table
        details_data = []
        for i, cp in enumerate(result.changepoints):
            details_data.append([i+1, cp])

        details_df = pd.DataFrame(
            details_data,
            columns=["Changepoint #", "Index"]
        )

        return fig, summary, details_df

    def run_analysis(
        run_id: str,
        variable: str,
        analysis_type: str,
        # Peak params
        peak_height: float,
        peak_distance: int,
        peak_prominence: float,
        # MP params
        subsequence_length: int,
        n_motifs: int,
        n_discords: int,
        # RQA params
        embedding_dim: int,
        time_delay: int,
        threshold: float,
        # CP params
        n_changepoints: int,
        cp_method: str,
        # State
        ui_state: UIState
    ) -> Tuple[Any, str, pd.DataFrame, str, UIState]:
        """Run the selected analysis."""
        # Load data
        series = load_series_data(run_id, variable)
        if series is None:
            return None, "", pd.DataFrame(), f"Failed to load data", ui_state

        try:
            if analysis_type == "Peak Detection":
                fig, summary, details = run_peak_detection(
                    series, peak_height, peak_distance, peak_prominence
                )
            elif analysis_type == "Matrix Profile":
                fig, summary, details = run_matrix_profile(
                    series, subsequence_length, n_motifs, n_discords
                )
            elif analysis_type == "Recurrence Plot":
                fig, summary, details = run_recurrence_analysis(
                    series, embedding_dim, time_delay, threshold
                )
            elif analysis_type == "Changepoint Detection":
                fig, summary, details = run_changepoint_detection(
                    series, n_changepoints, cp_method
                )
            else:
                return None, "", pd.DataFrame(), f"Unknown analysis type: {analysis_type}", ui_state

            # Update UI state
            if ui_state is not None:
                ui_state.add_analysis(
                    f"patterns_{analysis_type.lower().replace(' ', '_')}",
                    {"run_id": run_id, "variable": variable, "analysis_type": analysis_type},
                    f"Completed {analysis_type}"
                )

            status = f"Completed {analysis_type}"
            return fig, summary, details, status, ui_state

        except Exception as e:
            return None, "", pd.DataFrame(), f"Analysis failed: {e}", ui_state

    # Update parameter visibility based on analysis type
    def update_param_visibility(analysis_type: str):
        return (
            gr.update(visible=(analysis_type == "Peak Detection")),
            gr.update(visible=(analysis_type == "Matrix Profile")),
            gr.update(visible=(analysis_type == "Recurrence Plot")),
            gr.update(visible=(analysis_type == "Changepoint Detection"))
        )

    analysis_type.change(
        update_param_visibility,
        inputs=[analysis_type],
        outputs=[peak_params, mp_params, rqa_params, cp_params]
    )

    # Connect run button
    run_btn.click(
        run_analysis,
        inputs=[
            run_id, variable, analysis_type,
            peak_height, peak_distance, peak_prominence,
            subsequence_length, n_motifs, n_discords,
            embedding_dim, time_delay, threshold,
            n_changepoints, cp_method,
            state
        ],
        outputs=[plot_output, summary_output, details_output, status_text, state]
    )
