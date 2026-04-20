"""Activity-recognition workflow for labeled multivariate streams."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

from ts_agents.cli.input_parsing import LabeledStreamInput
from ts_agents.cli.output import to_jsonable
from ts_agents.contracts import ToolPayload

from .common import (
    attach_workflow_run_metadata,
    ensure_output_dir,
    write_json_artifact,
    write_plot_artifact,
    write_text_artifact,
)

_SUPPORTED_CLASSIFIERS = {"auto", "minirocket", "rocket", "knn"}
_SUPPORTED_METRICS = {"accuracy", "balanced_accuracy", "f1_macro"}
_SUPPORTED_LABELING = {"strict", "majority"}
_SUPPORTED_BALANCE = {"none", "undersample", "segment_cap"}


def run_activity_recognition_workflow(
    stream_input: LabeledStreamInput,
    *,
    output_dir: str,
    window_sizes: Optional[Iterable[int]] = None,
    metric: str = "balanced_accuracy",
    classifier: str = "auto",
    labeling: str = "strict",
    balance: str = "segment_cap",
    max_windows_per_segment: int = 25,
    stride: Optional[int] = None,
    n_splits: int = 3,
    test_size: float = 0.25,
    seed: int = 1337,
    skip_plots: bool = False,
    run_id: Optional[str] = None,
    resumed: bool = False,
    output_dir_mode: str = "explicit",
) -> ToolPayload:
    """Run window-size selection and evaluation on a labeled sensor stream."""
    from ts_agents.core.windowing import evaluate_windowed_classifier, select_window_size

    workflow_name = "activity-recognition"
    output_path = ensure_output_dir(output_dir)
    selected_classifier = _normalize_classifier(classifier)
    selected_metric = _normalize_metric(metric)
    selected_labeling = _normalize_labeling(labeling)
    selected_balance = _normalize_balance(balance)
    candidate_windows = _normalize_window_sizes(window_sizes)
    preparation = (stream_input.provenance.get("stream_ref") or {}).get("preparation")
    label_source = (stream_input.provenance.get("stream_ref") or {}).get("label_source")

    selection = select_window_size(
        stream_input.values,
        stream_input.labels,
        window_sizes=candidate_windows,
        metric=selected_metric,
        classifier=selected_classifier,
        stride=stride,
        labeling=selected_labeling,
        balance=selected_balance,
        max_windows_per_segment=max_windows_per_segment,
        n_splits=n_splits,
        test_size=test_size,
        seed=seed,
    )
    evaluation = evaluate_windowed_classifier(
        stream_input.values,
        stream_input.labels,
        window_size=int(selection.best_window_size),
        metric=selected_metric,
        classifier=selected_classifier,
        labeling=selected_labeling,
        balance=selected_balance,
        max_windows_per_segment=max_windows_per_segment,
        stride=stride,
        n_splits=n_splits,
        test_size=test_size,
        seed=seed,
    )

    selection_payload = to_jsonable(selection)
    evaluation_payload = to_jsonable(evaluation)
    classification = evaluation_payload.get("classification") or {}
    evaluation_classifier = evaluation_payload.get("classifier")
    classification_method = classification.get("method")
    effective_backend = classification_method or evaluation_classifier or selected_classifier
    warnings: List[str] = []
    for warning in classification.get("warnings") or []:
        warning_text = str(warning)
        if warning_text not in warnings:
            warnings.append(warning_text)

    quality_flags = _activity_quality_flags(
        selection_payload=selection_payload,
        evaluation_payload=evaluation_payload,
    )
    if warnings and "classifier_backend_warning" not in quality_flags:
        quality_flags.append("classifier_backend_warning")
    score = evaluation_payload.get("score")
    summary_data = {
        "workflow": workflow_name,
        "source": stream_input.provenance.get("stream_ref", {}),
        "value_columns": list(stream_input.value_columns),
        "label_column": stream_input.label_column,
        "classifier_requested": classifier,
        "classifier_resolved": selected_classifier,
        "classifier_evaluation": evaluation_classifier,
        "classifier_effective_backend": effective_backend,
        "classification_method": classification_method,
        "classifier_used": effective_backend,
        "metric": selected_metric,
        "labeling": selected_labeling,
        "balance": selected_balance,
        "stride_requested": stride,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "best_window_size": int(selection.best_window_size),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "preparation": preparation,
        "label_source": label_source,
        "quality_flags": quality_flags,
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

    if not skip_plots:
        selection_fig = None
        confusion_fig = None
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
        except ImportError:
            warnings.append("matplotlib is not installed; skipping activity-recognition plots.")
        except ValueError as exc:
            warnings.append(f"Skipping activity-recognition plots: {exc}")
        finally:
            _close_plots(selection_fig, confusion_fig)

    report = _build_report(
        stream_input=stream_input,
        classifier_requested=classifier,
        classifier_resolved=selected_classifier,
        effective_backend=effective_backend,
        metric=selected_metric,
        labeling=selected_labeling,
        balance=selected_balance,
        stride=stride,
        n_splits=n_splits,
        quality_flags=quality_flags,
        warnings=warnings,
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
    payload = ToolPayload(
        kind="workflow",
        summary=summary,
        status="degraded" if warnings or quality_flags else "ok",
        data=summary_data,
        artifacts=artifacts,
        warnings=warnings,
        provenance=stream_input.provenance,
    )
    return attach_workflow_run_metadata(
        payload,
        workflow_name=workflow_name,
        output_dir=output_path,
        run_id=run_id,
        source=summary_data["source"],
        options={
            "window_sizes": candidate_windows,
            "metric": selected_metric,
            "classifier": classifier,
            "labeling": selected_labeling,
            "balance": selected_balance,
            "max_windows_per_segment": max_windows_per_segment,
            "stride": stride,
            "n_splits": n_splits,
            "test_size": test_size,
            "seed": seed,
            "skip_plots": skip_plots,
        },
        resumed=resumed,
        output_dir_mode=output_dir_mode,
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


def _normalize_labeling(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _SUPPORTED_LABELING:
        raise ValueError(
            f"Unsupported labeling strategy '{raw}'. Supported: {', '.join(sorted(_SUPPORTED_LABELING))}."
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


def _activity_quality_flags(
    *,
    selection_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> List[str]:
    flags: List[str] = []

    scores = (selection_payload.get("scores_by_window") or {}).values()
    if any(score is None for score in scores):
        flags.append("nan_window_scores")

    n_windows_by_window = selection_payload.get("n_windows_by_window") or {}
    if any(isinstance(count, int) and count < 10 for count in n_windows_by_window.values()):
        flags.append("too_few_windows")

    best_window = selection_payload.get("best_window_size")
    best_window_diagnostics = _window_diagnostic_for(
        selection_payload,
        best_window,
    )
    dropped_in_best_window = best_window_diagnostics.get("dropped_classes") or []
    if dropped_in_best_window:
        flags.append("best_window_dropped_classes")

    dropped_in_evaluation = evaluation_payload.get("dropped_classes") or []
    if dropped_in_evaluation:
        flags.append("evaluation_dropped_classes")

    selection_requested_splits = ((selection_payload.get("details") or {}).get("n_splits"))
    evaluation_valid_splits = evaluation_payload.get("n_splits")
    if (
        isinstance(selection_requested_splits, int)
        and isinstance(evaluation_valid_splits, int)
        and evaluation_valid_splits < selection_requested_splits
    ):
        flags.append("fewer_valid_splits_than_requested")

    classification = evaluation_payload.get("classification") or {}
    confusion = classification.get("confusion_matrix")
    active_rows = _count_active_confusion_rows(confusion)
    class_counts = evaluation_payload.get("class_counts") or {}
    if active_rows is not None:
        if active_rows <= 1:
            flags.append("single_class_test_split")
        if class_counts and active_rows < len(class_counts):
            flags.append("metric_not_comparable")

    score = evaluation_payload.get("score")
    accuracy = classification.get("accuracy")
    f1_macro = classification.get("f1_score")
    if all(
        isinstance(value, (int, float)) and float(value) == 1.0
        for value in (score, accuracy, f1_macro)
    ):
        flags.append("perfect_metrics")

    return flags


def _count_active_confusion_rows(confusion: Any) -> Optional[int]:
    if not isinstance(confusion, list):
        return None

    active_rows = 0
    for row in confusion:
        if isinstance(row, list) and any(value != 0 for value in row):
            active_rows += 1
    return active_rows


def _plot_window_selection(selection_payload: dict[str, Any]):
    import matplotlib.pyplot as plt

    scores = selection_payload.get("scores_by_window") or {}
    if not scores:
        raise ValueError("No scores_by_window found in selection payload.")

    items = sorted(
        (
            (int(key), float(value))
            for key, value in scores.items()
            if value is not None
        ),
        key=lambda item: item[0],
    )
    if not items:
        raise ValueError("No finite scores_by_window found in selection payload.")

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


def _close_plots(*figures: Any) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for figure in figures:
        if figure is not None:
            plt.close(figure)


def _build_report(
    *,
    stream_input: LabeledStreamInput,
    classifier_requested: str,
    classifier_resolved: str,
    effective_backend: str,
    metric: str,
    labeling: str,
    balance: str,
    stride: Optional[int],
    n_splits: int,
    quality_flags: list[str],
    warnings: list[str],
    selection_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> str:
    classification = evaluation_payload.get("classification") or {}
    retained_classes = _format_label_list(evaluation_payload.get("retained_classes") or [])
    dropped_classes = _format_label_list(evaluation_payload.get("dropped_classes") or [])
    flag_text = ", ".join(quality_flags) if quality_flags else "none"
    warning_text = "; ".join(warnings) if warnings else "none"
    note = _build_quality_note(
        selection_payload=selection_payload,
        evaluation_payload=evaluation_payload,
        quality_flags=quality_flags,
        warnings=warnings,
    )
    sections = [
        "### Report on Activity-Recognition Workflow",
        "",
        "#### Inputs",
        f"- **Source**: `{stream_input.label}`",
        f"- **Value Columns**: {', '.join(stream_input.value_columns)}",
        f"- **Label Column**: `{stream_input.label_column}`",
        f"- **Preparation**: {_describe_preparation(stream_input)}",
        "",
        "#### Controls",
        f"- **Classifier Requested**: `{classifier_requested}`",
        f"- **Classifier Resolved**: `{classifier_resolved}`",
        f"- **Effective Backend**: `{effective_backend}`",
        f"- **Metric**: `{metric}`",
        f"- **Labeling**: `{labeling}`",
        f"- **Balance**: `{balance}`",
        f"- **Stride**: {stride if stride is not None else 'auto (window_size // 2)' }",
        f"- **Grouped Splits**: {n_splits}",
        "",
        "#### Results",
        f"- **Best Window Size**: {selection_payload.get('best_window_size')}",
        f"- **Workflow Score**: {_format_metric(evaluation_payload.get('score'))}",
        f"- **Accuracy**: {_format_metric(classification.get('accuracy'))}",
        f"- **Macro F1**: {_format_metric(classification.get('f1_score'))}",
        f"- **Retained Classes**: {retained_classes}",
        f"- **Dropped Classes**: {dropped_classes}",
        f"- **Quality Flags**: {flag_text}",
        f"- **Warnings**: {warning_text}",
        "",
        "#### Window Sweep",
        _build_window_sweep_table(selection_payload),
        "",
        "#### Note",
        note,
    ]
    return "\n".join(sections)


def _build_quality_note(
    *,
    selection_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
    quality_flags: list[str],
    warnings: list[str],
) -> str:
    notes: List[str] = []
    if warnings:
        notes.append("Warnings were emitted during artifact generation.")

    best_window = selection_payload.get("best_window_size")
    best_window_diagnostics = _window_diagnostic_for(selection_payload, best_window)
    best_window_dropped = best_window_diagnostics.get("dropped_classes") or []
    if best_window_dropped:
        notes.append(
            "Best-scoring window drops class coverage "
            f"({', '.join(str(label) for label in best_window_dropped)})."
        )

    dropped_classes = evaluation_payload.get("dropped_classes") or []
    if dropped_classes:
        notes.append(
            "Final evaluation excludes one or more classes "
            f"({', '.join(str(label) for label in dropped_classes)})."
        )

    if "fewer_valid_splits_than_requested" in quality_flags:
        requested = (selection_payload.get("details") or {}).get("n_splits")
        observed = evaluation_payload.get("n_splits")
        notes.append(
            f"Requested {requested} grouped split(s), but only {observed} valid evaluation split(s) were available."
        )

    classification = evaluation_payload.get("classification") or {}
    confusion = classification.get("confusion_matrix")
    active_rows = _count_active_confusion_rows(confusion)
    if active_rows is not None and active_rows <= 1:
        notes.append("Test split appears single-class; treat classifier metrics with caution.")

    score = evaluation_payload.get("score")
    accuracy = classification.get("accuracy")
    f1_macro = classification.get("f1_score")
    if all(
        isinstance(value, (int, float)) and float(value) == 1.0
        for value in (score, accuracy, f1_macro)
    ):
        notes.append("All reported metrics are perfect; verify split balance and leakage assumptions.")

    if notes:
        return " ".join(notes)
    return "No obvious pathologies detected in the reported classification metrics."


def _describe_preparation(stream_input: LabeledStreamInput) -> str:
    stream_ref = stream_input.provenance.get("stream_ref") or {}
    preparation = stream_ref.get("preparation") or {}
    label_source = stream_ref.get("label_source") or {}
    mode = preparation.get("mode")
    label_path = label_source.get("path") or label_source.get("label")

    if not mode:
        return "Used labels already present in the primary input table."
    if mode == "row_aligned_labels":
        return f"Merged labels from `{label_path}` by row alignment."
    if mode == "time_joined_labels":
        return (
            f"Merged labels from `{label_path}` on `{preparation.get('signal_time_column')}` "
            f"using labels column `{preparation.get('labels_time_column')}`."
        )
    if mode in {"segment_time_labels", "segment_index_labels"}:
        basis = (
            f"time column `{preparation.get('signal_time_column')}`"
            if preparation.get("signal_time_column")
            else "row offsets"
        )
        return (
            f"Expanded segment labels from `{label_path}` using {basis}; "
            "intervals are start-inclusive and end-exclusive."
        )
    return "Prepared labels from a separate labels input."


def _build_window_sweep_table(selection_payload: dict[str, Any]) -> str:
    window_diagnostics = (selection_payload.get("details") or {}).get("window_diagnostics") or {}
    if not window_diagnostics:
        return "_Window sweep diagnostics were not recorded._"

    lines = [
        "| Window | Score | Windows | Retained | Dropped | Splits |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    scores = selection_payload.get("scores_by_window") or {}
    for window, diagnostic in _sorted_window_diagnostics(window_diagnostics):
        score = scores.get(window)
        if score is None and str(window) in scores:
            score = scores[str(window)]
        lines.append(
            "| "
            f"{window} | {_format_metric(score)} | {diagnostic.get('n_windows', 'n/a')} | "
            f"{_format_label_list(diagnostic.get('retained_classes') or [])} | "
            f"{_format_label_list(diagnostic.get('dropped_classes') or [])} | "
            f"{diagnostic.get('valid_split_count', 0)} |"
        )
    return "\n".join(lines)


def _sorted_window_diagnostics(window_diagnostics: dict[Any, Any]) -> list[tuple[int, dict[str, Any]]]:
    items: list[tuple[int, dict[str, Any]]] = []
    for key, value in window_diagnostics.items():
        try:
            window = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            items.append((window, value))
    return sorted(items, key=lambda item: item[0])


def _window_diagnostic_for(selection_payload: dict[str, Any], window_size: Any) -> dict[str, Any]:
    if window_size is None:
        return {}
    window_diagnostics = (selection_payload.get("details") or {}).get("window_diagnostics") or {}
    return window_diagnostics.get(window_size) or window_diagnostics.get(str(window_size)) or {}


def _format_label_list(labels: list[Any]) -> str:
    if not labels:
        return "none"
    return ", ".join(f"`{label}`" for label in labels)


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"
