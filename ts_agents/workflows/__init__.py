"""Workflow registry for higher-level CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional

from ts_agents.cli.input_parsing import load_labeled_stream_input, load_series_input

from .activity import run_activity_recognition_workflow
from .forecast import run_forecast_series_workflow
from .inspect import run_inspect_series_workflow


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _optional_feature(
    *,
    name: str,
    available: bool,
    required_extras: Optional[List[str]] = None,
    missing_dependencies: Optional[List[str]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "available": available,
        "required_extras": required_extras or [],
        "missing_dependencies": missing_dependencies or [],
        "note": note,
    }


@dataclass(frozen=True)
class WorkflowOption:
    """Structured description of a workflow CLI option."""

    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    choices: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowArtifact:
    """Expected workflow artifact."""

    filename: str
    kind: str
    description: str
    required: bool = True
    condition: Optional[str] = None


@dataclass(frozen=True)
class WorkflowDefinition:
    """Metadata for a public workflow command."""

    name: str
    description: str
    runner: Callable
    load_input: Callable[[Any], Any]
    build_runner_kwargs: Callable[[Any], Dict[str, Any]]
    source_requirement: str
    supported_input_modes: List[str] = field(default_factory=list)
    options: List[WorkflowOption] = field(default_factory=list)
    artifacts: List[WorkflowArtifact] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    required_extras: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    availability_fn: Optional[Callable[[], Dict[str, Any]]] = None

    def availability(self) -> Dict[str, Any]:
        if self.availability_fn is None:
            return {
                "status": "available",
                "available": True,
                "missing_dependencies": [],
                "required_extras": list(self.required_extras),
                "optional_features": [],
                "install_hint": None,
            }
        return self.availability_fn()


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
        "season_length": getattr(args, "season_length", None),
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


def _inspect_workflow_availability() -> Dict[str, Any]:
    has_matplotlib = _module_available("matplotlib")
    optional_features = [
        _optional_feature(
            name="plots",
            available=has_matplotlib,
            required_extras=["viz"],
            missing_dependencies=[] if has_matplotlib else ["matplotlib"],
            note="Autocorrelation plots require matplotlib.",
        )
    ]
    return {
        "status": "available",
        "available": True,
        "missing_dependencies": [],
        "required_extras": [],
        "optional_features": optional_features,
        "install_hint": None,
    }


def _forecast_workflow_availability() -> Dict[str, Any]:
    has_statsforecast = _module_available("statsforecast")
    has_matplotlib = _module_available("matplotlib")
    available_methods = ["seasonal_naive"]
    unavailable_methods: List[str] = []
    missing_dependencies: List[str] = []
    status = "available"
    install_hint = None
    if has_statsforecast:
        available_methods.extend(["arima", "ets", "theta"])
    else:
        unavailable_methods.extend(["arima", "ets", "theta"])
        missing_dependencies.append("statsforecast")
        status = "degraded"
        install_hint = 'Install `ts-agents[forecasting]` or `ts-agents[recommended]` to enable ARIMA/ETS/Theta.'

    optional_features = [
        _optional_feature(
            name="plots",
            available=has_matplotlib,
            required_extras=["viz"],
            missing_dependencies=[] if has_matplotlib else ["matplotlib"],
            note="Forecast comparison plots require matplotlib.",
        )
    ]

    return {
        "status": status,
        "available": True,
        "available_methods": available_methods,
        "unavailable_methods": unavailable_methods,
        "missing_dependencies": missing_dependencies,
        "required_extras": ["forecasting"],
        "optional_features": optional_features,
        "install_hint": install_hint,
    }


def _activity_workflow_availability() -> Dict[str, Any]:
    has_aeon = _module_available("aeon")
    has_sklearn = _module_available("sklearn")
    missing_dependencies = [] if has_sklearn else ["scikit-learn"]
    has_matplotlib = _module_available("matplotlib")
    optional_features = [
        _optional_feature(
            name="rocket_backends",
            available=has_aeon,
            required_extras=["classification"],
            missing_dependencies=[] if has_aeon else ["aeon"],
            note="aeon enables ROCKET-family classifier backends; sklearn fallback remains available without it.",
        ),
        _optional_feature(
            name="plots",
            available=has_matplotlib,
            required_extras=["viz"],
            missing_dependencies=[] if has_matplotlib else ["matplotlib"],
            note="Window-score and confusion-matrix plots require matplotlib.",
        )
    ]
    available = has_sklearn
    status = "available" if has_aeon and available else "degraded" if available else "unavailable"
    return {
        "status": status,
        "available": available,
        "missing_dependencies": missing_dependencies,
        "required_extras": ["classification"],
        "optional_features": optional_features,
        "install_hint": None
        if available
        else 'Install `ts-agents[classification]` or `ts-agents[recommended]` to enable activity recognition.',
    }


_WORKFLOWS = {
    "inspect-series": WorkflowDefinition(
        name="inspect-series",
        description="Run quick diagnostics on a series and write summary/report artifacts.",
        runner=run_inspect_series_workflow,
        load_input=_load_series_workflow_input,
        build_runner_kwargs=_build_inspect_runner_kwargs,
        source_requirement="Exactly one of --input, --input-json, --stdin, or --run-id/--variable is required.",
        supported_input_modes=["input_file", "input_json", "stdin", "bundled_run"],
        options=[
            WorkflowOption("output_dir", "string", "Output directory for workflow artifacts.", default="outputs/inspect"),
            WorkflowOption("max_lag", "integer", "Maximum lag for autocorrelation output.", default=None),
            WorkflowOption("skip_plots", "boolean", "Skip plot generation.", default=False),
        ],
        artifacts=[
            WorkflowArtifact("summary.json", "json", "Inspection summary JSON."),
            WorkflowArtifact("report.md", "markdown", "Inspection workflow markdown report."),
            WorkflowArtifact(
                "autocorrelation.png",
                "image",
                "Autocorrelation plot generated by the inspect-series workflow.",
                required=False,
                condition="Written when matplotlib is installed and --skip-plots is false.",
            ),
        ],
        input_schema={
            "type": "series_input",
            "one_of": [
                {
                    "type": "object",
                    "required": ["series"],
                    "properties": {
                        "series": {"type": "array", "items": {"type": "number"}},
                        "time": {"type": "array"},
                        "name": {"type": "string"},
                    },
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {
                    "type": "tabular",
                    "description": "CSV/parquet/JSON data with a value column and optional time column.",
                },
            ]
        },
        examples=[
            "uv run ts-agents workflow run inspect-series --input-json '{\"series\":[1,2,3,4]}'",
            "uv run ts-agents workflow run inspect-series --input data.csv --time-col ds --value-col y",
            "uv run ts-agents workflow run inspect-series --run-id Re200Rm200 --variable bx001_real",
        ],
        capabilities={
            "writes_artifacts": ["summary.json", "report.md", "autocorrelation.png"],
        },
        availability_fn=_inspect_workflow_availability,
    ),
    "forecast-series": WorkflowDefinition(
        name="forecast-series",
        description="Compare baseline forecasting methods and write forecast artifacts.",
        runner=run_forecast_series_workflow,
        load_input=_load_series_workflow_input,
        build_runner_kwargs=_build_forecast_runner_kwargs,
        source_requirement="Exactly one of --input, --input-json, --stdin, or --run-id/--variable is required.",
        supported_input_modes=["input_file", "input_json", "stdin", "bundled_run"],
        options=[
            WorkflowOption("output_dir", "string", "Output directory for workflow artifacts.", default="outputs/forecast"),
            WorkflowOption("horizon", "integer", "Forecast horizon.", default=12),
            WorkflowOption(
                "season_length",
                "integer",
                "Optional seasonal period for seasonal methods.",
                default=None,
            ),
            WorkflowOption(
                "methods",
                "string",
                "Comma-separated methods to compare.",
                default="seasonal_naive,arima,theta",
            ),
            WorkflowOption("validation_size", "integer", "Holdout size for comparison.", default=None),
            WorkflowOption("skip_plots", "boolean", "Skip plot generation.", default=False),
        ],
        artifacts=[
            WorkflowArtifact("forecast_comparison.json", "json", "Forecast comparison metrics and rankings."),
            WorkflowArtifact("forecast.json", "json", "Best-model forecast in JSON form."),
            WorkflowArtifact("forecast.csv", "csv", "Best-model forecast as CSV."),
            WorkflowArtifact("report.md", "markdown", "Forecast workflow markdown report."),
            WorkflowArtifact(
                "forecast_comparison.png",
                "image",
                "Forecast comparison plot.",
                required=False,
                condition="Written when matplotlib is installed and --skip-plots is false.",
            ),
        ],
        input_schema={
            "type": "series_input",
            "one_of": [
                {
                    "type": "object",
                    "required": ["series"],
                    "properties": {
                        "series": {"type": "array", "items": {"type": "number"}},
                        "time": {"type": "array"},
                        "name": {"type": "string"},
                    },
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                },
                {
                    "type": "tabular",
                    "description": "CSV/parquet/JSON data with a value column and optional time column.",
                },
            ]
        },
        required_extras=["forecasting"],
        examples=[
            "uv run ts-agents workflow show forecast-series --json",
            "uv run ts-agents workflow run forecast-series --input-json '{\"series\":[1,2,3,4,5,6,7,8,9,10,11,12]}' --horizon 3 --methods seasonal_naive",
            "uv run ts-agents workflow run forecast-series --input data.csv --time-col ds --value-col y --horizon 24 --methods seasonal_naive,arima,theta",
        ],
        capabilities={
            "supported_methods": ["seasonal_naive", "arima", "ets", "theta"],
            "baseline_first": True,
        },
        availability_fn=_forecast_workflow_availability,
    ),
    "activity-recognition": WorkflowDefinition(
        name="activity-recognition",
        description="Select a window size, evaluate a classifier, and write activity-recognition artifacts.",
        runner=run_activity_recognition_workflow,
        load_input=_load_activity_workflow_input,
        build_runner_kwargs=_build_activity_runner_kwargs,
        source_requirement="Exactly one of --input, --input-json, or --stdin is required.",
        supported_input_modes=["input_file", "input_json", "stdin"],
        options=[
            WorkflowOption("output_dir", "string", "Output directory for workflow artifacts.", default="outputs/activity"),
            WorkflowOption("label_col", "string", "Label column containing activity labels.", default="label"),
            WorkflowOption("value_cols", "array", "Comma-separated value columns.", default=["x", "y", "z"]),
            WorkflowOption("window_sizes", "array", "Candidate window sizes.", default=[32, 64, 96, 128, 160]),
            WorkflowOption(
                "metric",
                "string",
                "Evaluation metric.",
                default="balanced_accuracy",
                choices=["accuracy", "balanced_accuracy", "f1_macro"],
            ),
            WorkflowOption(
                "classifier",
                "string",
                "Classifier backend.",
                default="auto",
                choices=["auto", "minirocket", "rocket", "knn"],
            ),
            WorkflowOption(
                "balance",
                "string",
                "Window balancing strategy.",
                default="segment_cap",
                choices=["none", "undersample", "segment_cap"],
            ),
            WorkflowOption("max_windows_per_segment", "integer", "Cap windows per segment when balance=segment_cap.", default=25),
            WorkflowOption("test_size", "number", "Test fraction per split.", default=0.25),
            WorkflowOption("seed", "integer", "Random seed for selection and evaluation.", default=1337),
            WorkflowOption("skip_plots", "boolean", "Skip plot generation.", default=False),
        ],
        artifacts=[
            WorkflowArtifact("window_selection.json", "json", "Window-size selection metrics."),
            WorkflowArtifact("eval.json", "json", "Windowed classifier evaluation."),
            WorkflowArtifact("report.md", "markdown", "Activity-recognition workflow markdown report."),
            WorkflowArtifact(
                "window_scores.png",
                "image",
                "Window-size search scores.",
                required=False,
                condition="Written when matplotlib is installed and --skip-plots is false.",
            ),
            WorkflowArtifact(
                "confusion_matrix.png",
                "image",
                "Confusion matrix for the selected window/classifier.",
                required=False,
                condition="Written when matplotlib is installed and --skip-plots is false.",
            ),
        ],
        input_schema={
            "type": "labeled_stream_input",
            "one_of": [
                {
                    "type": "object",
                    "required": ["records"],
                    "properties": {
                        "records": {"type": "array", "items": {"type": "object"}},
                        "name": {"type": "string"},
                    },
                },
                {
                    "type": "array",
                    "items": {"type": "object"},
                },
                {
                    "type": "tabular",
                    "description": "CSV/parquet/JSON data with feature columns and a label column.",
                },
            ]
        },
        required_extras=["classification"],
        examples=[
            "uv run ts-agents workflow show activity-recognition --json",
            "uv run ts-agents workflow run activity-recognition --input data/demo_labeled_stream.csv --label-col label --value-cols x,y,z",
        ],
        capabilities={
            "supported_metrics": ["accuracy", "balanced_accuracy", "f1_macro"],
            "supported_classifiers": ["auto", "minirocket", "rocket", "knn"],
            "supported_balance_strategies": ["none", "undersample", "segment_cap"],
        },
        availability_fn=_activity_workflow_availability,
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


def workflow_to_dict(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Convert a workflow definition into machine-readable metadata."""
    availability = workflow.availability()
    return {
        "name": workflow.name,
        "description": workflow.description,
        "supported_input_modes": list(workflow.supported_input_modes),
        "source_requirement": workflow.source_requirement,
        "options": [
            {
                "name": option.name,
                "type": option.type,
                "description": option.description,
                "required": option.required,
                "default": option.default,
                "choices": list(option.choices),
            }
            for option in workflow.options
        ],
        "required_options": [option.name for option in workflow.options if option.required],
        "artifacts": [
            {
                "filename": artifact.filename,
                "kind": artifact.kind,
                "description": artifact.description,
                "required": artifact.required,
                "condition": artifact.condition,
            }
            for artifact in workflow.artifacts
        ],
        "input_schema": workflow.input_schema,
        "required_extras": list(workflow.required_extras),
        "availability": availability,
        "examples": list(workflow.examples),
        "capabilities": workflow.capabilities,
    }
