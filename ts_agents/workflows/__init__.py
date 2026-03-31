"""Workflow registry for higher-level CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ts_agents.cli.input_parsing import load_labeled_stream_input, load_series_input

from .activity import run_activity_recognition_workflow
from .forecast import run_forecast_series_workflow
from .inspect import run_inspect_series_workflow


@dataclass(frozen=True)
class WorkflowDefinition:
    """Metadata for a public workflow command."""

    name: str
    description: str
    runner: Callable
    load_input: Callable[[Any], Any]
    build_runner_kwargs: Callable[[Any], Dict[str, Any]]


def _load_series_workflow_input(args: Any):
    return load_series_input(
        input_path=getattr(args, "input", None),
        input_json=getattr(args, "input_json", None),
        use_stdin=getattr(args, "stdin", False),
        run_id=getattr(args, "run_id", None),
        variable_name=getattr(args, "variable", None),
        time_col=getattr(args, "time_col", None),
        value_col=getattr(args, "value_col", None),
        use_test_data=getattr(args, "use_test_data_resolved", None),
    )


def _load_activity_workflow_input(args: Any):
    return load_labeled_stream_input(
        input_path=getattr(args, "input", None),
        input_json=getattr(args, "input_json", None),
        use_stdin=getattr(args, "stdin", False),
        time_col=getattr(args, "time_col", None),
        value_cols=getattr(args, "value_cols", None),
        label_col=getattr(args, "label_col", "label"),
    )


def _build_inspect_runner_kwargs(args: Any) -> Dict[str, Any]:
    return {
        "output_dir": args.output_dir,
        "max_lag": args.max_lag,
        "skip_plots": args.skip_plots,
    }


def _build_forecast_runner_kwargs(args: Any) -> Dict[str, Any]:
    methods = [
        method.strip()
        for method in getattr(args, "methods", "").split(",")
        if method.strip()
    ]
    return {
        "output_dir": args.output_dir,
        "horizon": args.horizon,
        "methods": methods,
        "validation_size": args.validation_size,
        "skip_plots": args.skip_plots,
    }


def _build_activity_runner_kwargs(args: Any) -> Dict[str, Any]:
    return {
        "output_dir": args.output_dir,
        "window_sizes": getattr(args, "window_sizes", None),
        "metric": args.metric,
        "classifier": args.classifier,
        "balance": args.balance,
        "max_windows_per_segment": args.max_windows_per_segment,
        "test_size": args.test_size,
        "seed": args.seed,
        "skip_plots": args.skip_plots,
    }


_WORKFLOWS = {
    "inspect-series": WorkflowDefinition(
        name="inspect-series",
        description="Run quick diagnostics on a series and write summary/report artifacts.",
        runner=run_inspect_series_workflow,
        load_input=_load_series_workflow_input,
        build_runner_kwargs=_build_inspect_runner_kwargs,
    ),
    "forecast-series": WorkflowDefinition(
        name="forecast-series",
        description="Compare baseline forecasting methods and write forecast artifacts.",
        runner=run_forecast_series_workflow,
        load_input=_load_series_workflow_input,
        build_runner_kwargs=_build_forecast_runner_kwargs,
    ),
    "activity-recognition": WorkflowDefinition(
        name="activity-recognition",
        description="Select a window size, evaluate a classifier, and write activity-recognition artifacts.",
        runner=run_activity_recognition_workflow,
        load_input=_load_activity_workflow_input,
        build_runner_kwargs=_build_activity_runner_kwargs,
    ),
}


def list_workflows() -> List[WorkflowDefinition]:
    """List the public workflow definitions."""
    return [definition for _, definition in sorted(_WORKFLOWS.items())]


def get_workflow(name: str) -> WorkflowDefinition:
    """Resolve a workflow by name."""
    try:
        return _WORKFLOWS[name]
    except KeyError as exc:
        raise KeyError(name) from exc
