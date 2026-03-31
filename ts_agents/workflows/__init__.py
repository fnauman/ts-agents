"""Workflow registry for higher-level CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .forecast import run_forecast_series_workflow
from .inspect import run_inspect_series_workflow


@dataclass(frozen=True)
class WorkflowDefinition:
    """Metadata for a public workflow command."""

    name: str
    description: str
    runner: Callable


_WORKFLOWS = {
    "inspect-series": WorkflowDefinition(
        name="inspect-series",
        description="Run quick diagnostics on a series and write summary/report artifacts.",
        runner=run_inspect_series_workflow,
    ),
    "forecast-series": WorkflowDefinition(
        name="forecast-series",
        description="Compare baseline forecasting methods and write forecast artifacts.",
        runner=run_forecast_series_workflow,
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
