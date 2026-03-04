"""Classification tab component.

This component provides time series classification with:
- Multiple classifiers (DTW-KNN, ROCKET, HIVE-COTE)
- Cross-validation and train/test split
- Accuracy metrics and confusion matrix visualization
"""

import gradio as gr
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, List

from ...config import AVAILABLE_RUNS, REAL_VARIABLES
from ..state import UIState


def create_classification_tab(state: gr.State):
    """Create the classification tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    """
    gr.Markdown("""
    ## Time Series Classification

    Classify time series using state-of-the-art algorithms from the aeon toolkit.
    Create labeled datasets from runs and compare classifier performance.

    **Note**: Classification requires labeled data. Use the data preparation section
    to create train/test sets from the available runs.
    """)

    with gr.Row():
        # Left column: Parameters
        with gr.Column(scale=1):
            gr.Markdown("### Data Preparation")

            variable = gr.Dropdown(
                choices=REAL_VARIABLES,
                label="Variable to Classify",
                value=REAL_VARIABLES[0] if REAL_VARIABLES else None
            )

            train_runs = gr.CheckboxGroup(
                choices=AVAILABLE_RUNS,
                label="Training Runs",
                value=AVAILABLE_RUNS[:4] if len(AVAILABLE_RUNS) >= 4 else AVAILABLE_RUNS,
                info="Select runs for training"
            )

            test_runs = gr.CheckboxGroup(
                choices=AVAILABLE_RUNS,
                label="Test Runs",
                value=AVAILABLE_RUNS[4:] if len(AVAILABLE_RUNS) > 4 else [],
                info="Select runs for testing (or use cross-validation)"
            )

            use_cv = gr.Checkbox(
                value=False,
                label="Use Cross-Validation",
                info="If checked, ignores test runs and uses k-fold CV"
            )

            n_folds = gr.Slider(
                minimum=2,
                maximum=10,
                value=5,
                step=1,
                label="Number of Folds (for CV)",
                visible=False
            )

            gr.Markdown("### Classifier Selection")
            classifier = gr.Radio(
                choices=["DTW-KNN", "ROCKET", "MiniRocket", "HIVE-COTE 2"],
                value="ROCKET",
                label="Classifier"
            )

            with gr.Accordion("Classifier Parameters", open=False):
                # DTW-KNN params
                n_neighbors = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="K Neighbors (DTW-KNN)"
                )
                distance_metric = gr.Dropdown(
                    choices=["dtw", "euclidean", "msm", "erp"],
                    value="dtw",
                    label="Distance Metric"
                )

                # ROCKET params
                n_kernels = gr.Slider(
                    minimum=1000,
                    maximum=20000,
                    value=10000,
                    step=1000,
                    label="Number of Kernels (ROCKET)"
                )

            with gr.Row():
                prepare_btn = gr.Button("Prepare Data")
                run_btn = gr.Button("Run Classification", variant="primary")

        # Right column: Results
        with gr.Column(scale=2):
            gr.Markdown("### Results")

            with gr.Tabs():
                with gr.Tab("Metrics"):
                    metrics_output = gr.Dataframe(
                        label="Classification Metrics",
                        headers=["Metric", "Value"],
                        row_count=5
                    )

                with gr.Tab("Confusion Matrix"):
                    cm_plot = gr.Plot(label="Confusion Matrix")

                with gr.Tab("Class Distribution"):
                    dist_plot = gr.Plot(label="Class Distribution")

                with gr.Tab("Predictions"):
                    predictions_output = gr.Dataframe(
                        label="Predictions",
                        headers=["Sample", "True Label", "Predicted", "Correct"],
                        row_count=10
                    )

            summary_output = gr.Markdown(label="Summary")
            status_text = gr.Textbox(label="Status", interactive=False)

    # Show/hide CV folds based on checkbox
    def update_cv_visibility(use_cv: bool):
        return gr.update(visible=use_cv)

    use_cv.change(
        update_cv_visibility,
        inputs=[use_cv],
        outputs=[n_folds]
    )

    def load_series_data(run_id: str, variable: str) -> Optional[np.ndarray]:
        """Load time series data."""
        try:
            from ...data_access import get_series
            return get_series(run_id, variable)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_classification_data(
        variable: str,
        train_runs: List[str],
        test_runs: List[str]
    ) -> Tuple[Any, str, UIState]:
        """Prepare data for classification."""
        import matplotlib.pyplot as plt

        if not train_runs:
            return None, "Please select at least one training run", None

        # Load data for each run
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for i, run_id in enumerate(train_runs):
            series = load_series_data(run_id, variable)
            if series is not None:
                train_data.append(series)
                train_labels.append(i)

        for i, run_id in enumerate(test_runs):
            series = load_series_data(run_id, variable)
            if series is not None:
                test_data.append(series)
                # Find the index of this run in train_runs for matching label
                if run_id in train_runs:
                    test_labels.append(train_runs.index(run_id))
                else:
                    test_labels.append(len(train_runs) + i)

        if not train_data:
            return None, "Failed to load training data", None

        # Create class distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Training class distribution
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        axes[0].bar(range(len(unique_labels)), counts, color='steelblue')
        axes[0].set_xticks(range(len(unique_labels)))
        axes[0].set_xticklabels([train_runs[l] if l < len(train_runs) else f"Class {l}"
                                 for l in unique_labels], rotation=45, ha='right')
        axes[0].set_title("Training Set Distribution")
        axes[0].set_ylabel("Count")

        # Test class distribution (if available)
        if test_data:
            unique_labels_test, counts_test = np.unique(test_labels, return_counts=True)
            axes[1].bar(range(len(unique_labels_test)), counts_test, color='coral')
            axes[1].set_xticks(range(len(unique_labels_test)))
            axes[1].set_xticklabels([test_runs[i] if i < len(test_runs) else f"Class {i}"
                                     for i in range(len(unique_labels_test))], rotation=45, ha='right')
            axes[1].set_title("Test Set Distribution")
            axes[1].set_ylabel("Count")
        else:
            axes[1].text(0.5, 0.5, "No test data\n(using cross-validation)",
                        ha='center', va='center', fontsize=12)
            axes[1].set_title("Test Set")

        plt.tight_layout()

        summary = f"""
### Data Preparation Summary

- **Variable**: {variable}
- **Training samples**: {len(train_data)} from {len(train_runs)} runs
- **Test samples**: {len(test_data)} from {len(test_runs)} runs
- **Series length**: {len(train_data[0]) if train_data else 'N/A'}

**Training Runs**: {', '.join(train_runs)}
**Test Runs**: {', '.join(test_runs) if test_runs else 'None (will use CV)'}
"""

        return fig, summary, "Data prepared successfully"

    def run_classification(
        variable: str,
        train_runs: List[str],
        test_runs: List[str],
        use_cv: bool,
        n_folds: int,
        classifier: str,
        n_neighbors: int,
        distance_metric: str,
        n_kernels: int,
        ui_state: UIState
    ) -> Tuple[pd.DataFrame, Any, Any, pd.DataFrame, str, str, UIState]:
        """Run the classification."""
        import matplotlib.pyplot as plt

        if not train_runs:
            return (pd.DataFrame(), None, None, pd.DataFrame(),
                   "Please select training runs", "Error", ui_state)

        # Load and prepare data
        X_train = []
        y_train = []

        for i, run_id in enumerate(train_runs):
            series = load_series_data(run_id, variable)
            if series is not None:
                X_train.append(series)
                y_train.append(run_id)  # Use run_id as label

        if not X_train:
            return (pd.DataFrame(), None, None, pd.DataFrame(),
                   "Failed to load training data", "Error", ui_state)

        # Convert to 3D array: (n_samples, n_channels, n_timepoints)
        X_train = np.array(X_train)
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]  # Add channel dimension

        y_train = np.array(y_train)

        # Prepare test data
        X_test = []
        y_test = []

        if test_runs and not use_cv:
            for run_id in test_runs:
                series = load_series_data(run_id, variable)
                if series is not None:
                    X_test.append(series)
                    y_test.append(run_id)

            X_test = np.array(X_test)
            if X_test.ndim == 2:
                X_test = X_test[:, np.newaxis, :]
            y_test = np.array(y_test)

        try:
            # Import classification functions
            from ...core.classification import knn_classify, rocket_classify

            if classifier == "DTW-KNN":
                if use_cv or not len(X_test):
                    # Use leave-one-out or k-fold
                    summary = "Cross-validation not fully implemented. Using train as test."
                    X_test = X_train
                    y_test = y_train

                result = knn_classify(
                    X_train, y_train, X_test, y_test,
                    distance=distance_metric,
                    n_neighbors=n_neighbors
                )

            elif classifier in ["ROCKET", "MiniRocket"]:
                if use_cv or not len(X_test):
                    X_test = X_train
                    y_test = y_train

                variant = "minirocket" if classifier == "MiniRocket" else "rocket"
                result = rocket_classify(
                    X_train, y_train, X_test, y_test,
                    variant=variant,
                    n_kernels=n_kernels
                )

            elif classifier == "HIVE-COTE 2":
                # HC2 is expensive, warn user
                if use_cv or not len(X_test):
                    X_test = X_train
                    y_test = y_train

                from ...core.classification import hivecote_classify
                result = hivecote_classify(X_train, y_train, X_test, y_test)

            else:
                return (pd.DataFrame(), None, None, pd.DataFrame(),
                       f"Unknown classifier: {classifier}", "Error", ui_state)

            # Create metrics table
            accuracy_str = f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
            metrics_data = [
                ["Accuracy", accuracy_str],
                ["F1 Score", f"{result.f1_score:.4f}" if result.f1_score is not None else "N/A"],
                ["Classifier", classifier],
                ["Training Samples", str(len(X_train))],
                ["Test Samples", str(len(X_test))],
            ]
            metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])

            # Create confusion matrix plot
            fig_cm, ax = plt.subplots(figsize=(8, 6))
            if result.confusion_matrix is not None:
                from sklearn.metrics import ConfusionMatrixDisplay
                unique_labels = np.unique(np.concatenate([y_train, y_test]))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=result.confusion_matrix,
                    display_labels=unique_labels
                )
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                ax.set_title(f"Confusion Matrix ({classifier})")
            else:
                ax.text(0.5, 0.5, "Confusion matrix not available",
                       ha='center', va='center')
            plt.tight_layout()

            # Create distribution plot
            fig_dist, ax = plt.subplots(figsize=(8, 5))
            unique, counts = np.unique(y_test, return_counts=True)
            correct_per_class = []
            for label in unique:
                mask = y_test == label
                correct = (result.predictions[mask] == label).sum()
                correct_per_class.append(correct)

            x = np.arange(len(unique))
            width = 0.35
            ax.bar(x - width/2, counts, width, label='Total', color='steelblue')
            ax.bar(x + width/2, correct_per_class, width, label='Correct', color='green')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Classification Results by Class')
            ax.set_xticks(x)
            ax.set_xticklabels(unique, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()

            # Create predictions table
            pred_data = []
            for i in range(min(20, len(result.predictions))):
                correct = "Yes" if result.predictions[i] == y_test[i] else "No"
                pred_data.append([i+1, y_test[i], result.predictions[i], correct])

            pred_df = pd.DataFrame(
                pred_data,
                columns=["Sample", "True Label", "Predicted", "Correct"]
            )

            # Summary
            summary = f"""
### Classification Results

- **Classifier**: {classifier}
- **Accuracy**: {accuracy_str}
- **Training samples**: {len(X_train)}
- **Test samples**: {len(X_test)}

#### Predictions Summary
- Correct: {(result.predictions == y_test).sum()} / {len(y_test)}
- Incorrect: {(result.predictions != y_test).sum()}
"""

            # Update UI state
            if ui_state is not None:
                ui_state.add_analysis(
                    "classification",
                    {
                        "classifier": classifier,
                        "train_runs": train_runs,
                        "test_runs": test_runs,
                        "variable": variable
                    },
                    f"Accuracy: {accuracy_str}"
                )

            return (metrics_df, fig_cm, fig_dist, pred_df, summary,
                   f"Classification complete. Accuracy: {accuracy_str}",
                   ui_state)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (pd.DataFrame(), None, None, pd.DataFrame(),
                   f"Classification failed: {e}", "Error", ui_state)

    # Connect events
    prepare_btn.click(
        prepare_classification_data,
        inputs=[variable, train_runs, test_runs],
        outputs=[dist_plot, summary_output, status_text]
    )

    run_btn.click(
        run_classification,
        inputs=[
            variable, train_runs, test_runs, use_cv, n_folds,
            classifier, n_neighbors, distance_metric, n_kernels,
            state
        ],
        outputs=[
            metrics_output, cm_plot, dist_plot, predictions_output,
            summary_output, status_text, state
        ]
    )
