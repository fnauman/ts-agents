"""Activity-recognition workflow for labeled multivariate streams."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

from ts_agents.cli.input_parsing import LabeledStreamInput
from ts_agents.cli.output import to_jsonable
from ts_agents.contracts import ToolPayload

from .common import ensure_output_dir, write_json_artifact, write_plot_artifact, write_text_artifact

_SUPPORTED_CLASSIFIERS = {"auto", "minirocket", "rocket", "knn"}
_SUPPORTED_METRICS = {"accuracy", "balanced_accuracy", "f1_macro"}
_SUPPORTED_BALANCE = {"none", "undersample", "segment_cap"}


def run_activity_recognition_workflow(
    stream_input: LabeledStreamInput,
    *,
    output_dir: str,
    window_sizes: Optional[Iterable[int]] = None,
    metric: str = "balanced_accuracy",
    classifier: str = "auto",
    balance: str = "segment_cap",
    max_windows_per_segment: int = 25,
    test_size: float = 0.25,
    seed: int = 1337,
    skip_plots: bool = False,
) -> ToolPayload:
    """Run window-size selection and evaluation on a labeled sensor stream."""
    from ts_agents.core.windowing import evaluate_windowed_classifier, select_window_size

    workflow_name = "activity-recognition"
    output_path = ensure_output_dir(output_dir)
    selected_classifier = _normalize_classifier(classifier)
    selected_metric = _normalize_metric(metric)
    selected_balance = _normalize_balance(balance)
    candidate_windows = _normalize_window_sizes(window_sizes)

    selection = select_window_size(
        stream_input.values,
        stream_input.labels,
        window_sizes=candidate_windows,
        metric=selected_metric,
        classifier=selected_classifier,
        balance=selected_balance,
        max_windows_per_segment=max_windows_per_segment,
        test_size=test_size,
        seed=seed,
    )
    evaluation = evaluate_windowed_classifier(
        stream_input.values,
        stream_input.labels,
        window_size=int(selection.best_window_size),
        metric=selected_metric,
        classifier=selected_classifier,
        balance=selected_balance,
        max_windows_per_segment=max_windows_per_segment,
        test_size=test_size,
        seed=seed,
    )

    selection_payload = to_jsonable(selection)
    evaluation_payload = to_jsonable(evaluation)
    score = evaluation_payload.get("score")
    summary_data = {
        "workflow": workflow_name,
        "source": stream_input.provenance.get("stream_ref", {}),
        "value_columns": list(stream_input.value_columns),
        "label_column": stream_input.label_column,
        "classifier_requested": classifier,
        "classifier_used": selected_classifier,
        "metric": selected_metric,
        "balance": selected_balance,
        "seed": int(seed),
        "best_window_size": int(selection.best_window_size),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "window_selection": selection_payload,
        "evaluation": evaluation_payload,
        "output_dir": str(output_path),
    }

    artifacts = [
        write_json_artifact(
            data=selection_payload,
            path=output_path / "window_selection.json",
            description="Window-size selection metrics.",
            created_by=workflow_name,
        ),
        write_json_artifact(
            data=evaluation_payload,
            path=output_path / "eval.json",
            description="Windowed classifier evaluation.",
            created_by=workflow_name,
        ),
    ]
    warnings: List[str] = []

    if not skip_plots:
        try:
            selection_fig = _plot_window_selection(selection_payload)
            artifacts.append(
                write_plot_artifact(
                    figure=selection_fig,
                    path=output_path / "window_scores.png",
                    description="Window-size search scores.",
                    created_by=workflow_name,
                )
            )

            confusion_fig = _plot_confusion_matrix(evaluation_payload)
            artifacts.append(
                write_plot_artifact(
                    figure=confusion_fig,
                    path=output_path / "confusion_matrix.png",
                    description="Confusion matrix for the selected window/classifier.",
                    created_by=workflow_name,
                )
            )

            import matplotlib.pyplot as plt

            plt.close(selection_fig)
            plt.close(confusion_fig)
        except ImportError:
            warnings.append("matplotlib is not installed; skipping activity-recognition plots.")

    report = _build_report(
        stream_input=stream_input,
        classifier_used=selected_classifier,
        metric=selected_metric,
        selection_payload=selection_payload,
        evaluation_payload=evaluation_payload,
    )
    artifacts.append(
        write_text_artifact(
            content=report,
            path=output_path / "report.md",
            description="Activity-recognition workflow markdown report.",
            created_by=workflow_name,
        )
    )

    score_text = _format_metric(summary_data["score"])
    summary = (
        f"Activity-recognition workflow completed for {stream_input.label}. "
        f"Best window size: {summary_data['best_window_size']}; "
        f"{selected_metric}: {score_text}."
    )
    return ToolPayload(
        kind="workflow",
        summary=summary,
        data=summary_data,
        artifacts=artifacts,
        warnings=warnings,
        provenance=stream_input.provenance,
    )


def _normalize_classifier(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier '{raw}'. Supported: {', '.join(sorted(_SUPPORTED_CLASSIFIERS))}."
        )
    return "minirocket" if normalized == "auto" else normalized


def _normalize_metric(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{raw}'. Supported: {', '.join(sorted(_SUPPORTED_METRICS))}."
        )
    return normalized


def _normalize_balance(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _SUPPORTED_BALANCE:
        raise ValueError(
            f"Unsupported balance strategy '{raw}'. Supported: {', '.join(sorted(_SUPPORTED_BALANCE))}."
        )
    return normalized


def _normalize_window_sizes(window_sizes: Optional[Iterable[int]]) -> Optional[List[int]]:
    if window_sizes is None:
        return None

    normalized = sorted({int(size) for size in window_sizes if size is not None})
    if not normalized:
        return None
    if normalized[0] < 2:
        raise ValueError("Window sizes must be >= 2.")
    return normalized


def _plot_window_selection(selection_payload: dict[str, Any]):
    import matplotlib.pyplot as plt

    scores = selection_payload.get("scores_by_window") or {}
    if not scores:
        raise ValueError("No scores_by_window found in selection payload.")

    items = sorted(((int(key), float(value)) for key, value in scores.items()), key=lambda item: item[0])
    xs = [window for window, _ in items]
    ys = [score for _, score in items]
    best_window = selection_payload.get("best_window_size")
    metric = selection_payload.get("metric", "score")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Window size")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Window-size selection (best={best_window})"
        if best_window is not None
        else "Window-size selection"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_confusion_matrix(evaluation_payload: dict[str, Any]):
    import matplotlib.pyplot as plt

    classification = evaluation_payload.get("classification") or {}
    confusion_matrix = classification.get("confusion_matrix")
    if confusion_matrix is None:
        raise ValueError("No classification.confusion_matrix found in evaluation payload.")

    confusion_array = np.asarray(confusion_matrix, dtype=float)
    class_counts = evaluation_payload.get("class_counts") or {}
    labels = sorted(str(label) for label in class_counts.keys())
    if labels and len(labels) != confusion_array.shape[0]:
        labels = []

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion_array)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    for row_idx in range(confusion_array.shape[0]):
        for col_idx in range(confusion_array.shape[1]):
            ax.text(col_idx, row_idx, str(int(confusion_array[row_idx, col_idx])), ha="center", va="center")

    fig.tight_layout()
    return fig


def _build_report(
    *,
    stream_input: LabeledStreamInput,
    classifier_used: str,
    metric: str,
    selection_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> str:
    classification = evaluation_payload.get("classification") or {}
    note = _build_quality_note(evaluation_payload)
    return "\n".join(
        [
            "### Report on Activity-Recognition Workflow",
            "",
            f"- **Source**: `{stream_input.label}`",
            f"- **Value Columns**: {', '.join(stream_input.value_columns)}",
            f"- **Label Column**: `{stream_input.label_column}`",
            f"- **Classifier**: `{classifier_used}`",
            f"- **Best Window Size**: {selection_payload.get('best_window_size')}",
            f"- **Metric**: `{metric}` = {_format_metric(evaluation_payload.get('score'))}",
            f"- **Accuracy**: {_format_metric(classification.get('accuracy'))}",
            f"- **Macro F1**: {_format_metric(classification.get('f1_score'))}",
            "",
            "#### Note",
            note,
        ]
    )


def _build_quality_note(evaluation_payload: dict[str, Any]) -> str:
    classification = evaluation_payload.get("classification") or {}
    confusion = classification.get("confusion_matrix")
    if isinstance(confusion, list):
        active_rows = 0
        for row in confusion:
            if isinstance(row, list) and any(value != 0 for value in row):
                active_rows += 1
        if active_rows <= 1:
            return "Test split appears single-class; treat classifier metrics with caution."

    score = evaluation_payload.get("score")
    accuracy = classification.get("accuracy")
    f1_macro = classification.get("f1_score")
    if all(
        isinstance(value, (int, float)) and float(value) == 1.0
        for value in (score, accuracy, f1_macro)
    ):
        return "All reported metrics are perfect; verify split balance and leakage assumptions."

    return "No obvious pathologies detected in the reported classification metrics."


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"
