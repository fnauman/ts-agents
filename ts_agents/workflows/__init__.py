"""Workflow registry for higher-level CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .forecast import run_forecast_series_workflow
from .inspect import run_inspect_series_workflow


@dataclass(frozen=True)
class WorkflowDefinition:
    """Metadata for a public workflow command."""

    name: str
    description: str
    runner: Callable
    build_runner_kwargs: Callable[[Any], Dict[str, Any]]


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


_WORKFLOWS = {
    "inspect-series": WorkflowDefinition(
        name="inspect-series",
        description="Run quick diagnostics on a series and write summary/report artifacts.",
        runner=run_inspect_series_workflow,
        build_runner_kwargs=_build_inspect_runner_kwargs,
    ),
    "forecast-series": WorkflowDefinition(
        name="forecast-series",
        description="Compare baseline forecasting methods and write forecast artifacts.",
        runner=run_forecast_series_workflow,
        build_runner_kwargs=_build_forecast_runner_kwargs,
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
