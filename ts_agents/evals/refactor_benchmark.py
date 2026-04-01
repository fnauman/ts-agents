"""Deterministic contract benchmark for refactor workstreams 8 and 9.

This benchmark intentionally avoids live model calls. It compares four
"assist levels" by executing real ``ts-agents`` CLI commands and measuring:

- task success rate
- parse/schema failure rate
- invalid tool calls
- artifact completeness
- retries and recovery
- latency

The benchmark is internal to the repository and is designed to be easy to rerun
from a source checkout:

    uv run python -m ts_agents.evals.refactor_benchmark --output-dir benchmarks/results/latest
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from ts_agents.cli.main import run as run_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results" / "latest"


@dataclass(frozen=True)
class ScenarioDefinition:
    """Benchmark scenario definition."""

    name: str
    description: str
    expected_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class StepDefinition:
    """One step in a deterministic assist-level plan."""

    label: str
    argv_factory: Optional[Callable[[Dict[str, Any]], list[str]]] = None
    narrative: Optional[str] = None


@dataclass
class CommandExecution:
    """Captured result for one benchmark step."""

    label: str
    command: Optional[list[str]] = None
    command_text: Optional[str] = None
    narrative: Optional[str] = None
    executed: bool = False
    exit_code: Optional[int] = None
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    payload: Optional[dict[str, Any]] = None
    parse_failure: bool = False
    invalid_tool_call: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.executed and self.exit_code == 0 and self.payload and self.payload.get("ok") is True)


@dataclass
class ScenarioResult:
    """Benchmark result for one scenario at one assist level."""

    assist_level: str
    scenario: str
    description: str
    attempts: list[CommandExecution]
    expected_artifacts: list[str]
    observed_artifacts: list[str]
    artifact_completeness: float
    task_success: bool
    parse_failures: int
    invalid_tool_calls: int
    retries: int
    recovered: bool
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "assist_level": self.assist_level,
            "scenario": self.scenario,
            "description": self.description,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "expected_artifacts": self.expected_artifacts,
            "observed_artifacts": self.observed_artifacts,
            "artifact_completeness": self.artifact_completeness,
            "task_success": self.task_success,
            "parse_failures": self.parse_failures,
            "invalid_tool_calls": self.invalid_tool_calls,
            "retries": self.retries,
            "recovered": self.recovered,
            "duration_ms": self.duration_ms,
        }


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""

    output_dir: str
    scenarios: list[ScenarioResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "summary": self.summary,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        name="inspect_unknown_series",
        description="Inspect an arbitrary JSON series and produce a reusable artifact bundle.",
        expected_artifacts=("summary.json", "report.md"),
    ),
    ScenarioDefinition(
        name="compare_forecasting_baselines",
        description="Compare baseline forecasting methods and emit a full forecast artifact bundle.",
        expected_artifacts=("forecast_comparison.json", "forecast.json", "forecast.csv", "report.md"),
    ),
    ScenarioDefinition(
        name="activity_windowing_workflow",
        description="Select a window size and evaluate a labeled stream classifier with report artifacts.",
        expected_artifacts=("window_selection.json", "eval.json", "report.md"),
    ),
)


def _assist_levels() -> dict[str, Callable[[ScenarioDefinition, Dict[str, Any]], list[StepDefinition]]]:
    return {
        "plain_model": _plain_model_steps,
        "plain_tools": _plain_tools_steps,
        "structured_discovery": _structured_discovery_steps,
        "skills_workflows": _skills_workflow_steps,
    }


def _series_json_payload() -> str:
    values = [round(float(idx) + 0.25 * np.sin(idx / 2.0), 4) for idx in range(1, 21)]
    return json.dumps({"series": values})


def _prepare_activity_fixture(path: Path) -> None:
    rng = np.random.default_rng(1337)
    rows: list[tuple[float, float, float, str]] = []
    segments = [
        ("idle", 64),
        ("walk", 96),
        ("idle", 64),
        ("walk", 96),
        ("idle", 64),
        ("walk", 96),
    ]

    for label, length in segments:
        t = np.arange(length, dtype=float)
        if label == "idle":
            segment = rng.normal(0.0, 0.03, size=(length, 3))
        else:
            segment = np.column_stack(
                [
                    0.9 * np.sin(2 * np.pi * t / 16.0),
                    0.7 * np.sin(2 * np.pi * t / 16.0 + 0.8),
                    0.5 * np.sin(2 * np.pi * t / 8.0 + 1.2),
                ]
            )
            segment += rng.normal(0.0, 0.05, size=segment.shape)
        rows.extend((float(x), float(y), float(z), label) for x, y, z in segment)

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "x,y,z,label\n" + "\n".join(f"{x:.5f},{y:.5f},{z:.5f},{label}" for x, y, z, label in rows) + "\n"
    path.write_text(text)


def _scenario_context(output_dir: Path) -> dict[str, Any]:
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    activity_csv = workspace / "activity_fixture.csv"
    _prepare_activity_fixture(activity_csv)
    return {
        "series_input_json": _series_json_payload(),
        "activity_csv_path": str(activity_csv),
        "workspace": str(workspace),
    }


def _shell_join(argv: Iterable[str]) -> str:
    return "ts-agents " + " ".join(json.dumps(part) if " " in part else part for part in argv)


def _execute_step(step: StepDefinition, state: dict[str, Any]) -> CommandExecution:
    if step.argv_factory is None:
        execution = CommandExecution(
            label=step.label,
            narrative=step.narrative,
            parse_failure=True,
        )
        state["attempts"].append(execution)
        return execution

    argv = step.argv_factory(state)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exit_code = run_cli(argv)
    duration_ms = (time.perf_counter() - started) * 1000.0
    stdout_text = stdout_buffer.getvalue().strip()
    stderr_text = stderr_buffer.getvalue().strip()

    payload: Optional[dict[str, Any]] = None
    parse_failure = False
    if stdout_text:
        try:
            loaded = json.loads(stdout_text)
        except json.JSONDecodeError:
            parse_failure = True
        else:
            if isinstance(loaded, dict):
                payload = loaded
            else:
                parse_failure = True
    else:
        # Silent success means "no structured payload", not "malformed payload".
        parse_failure = exit_code != 0

    error_code = ((payload or {}).get("error") or {}).get("code")
    invalid_tool_call = bool(error_code == "validation_error")

    execution = CommandExecution(
        label=step.label,
        command=argv,
        command_text=_shell_join(argv),
        executed=True,
        exit_code=exit_code,
        duration_ms=duration_ms,
        stdout=stdout_text,
        stderr=stderr_text,
        payload=payload,
        parse_failure=parse_failure,
        invalid_tool_call=invalid_tool_call,
    )
    state["attempts"].append(execution)
    return execution


def _run_scenario(
    assist_level: str,
    scenario: ScenarioDefinition,
    steps: list[StepDefinition],
    context: dict[str, Any],
    output_dir: Path,
) -> ScenarioResult:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        **context,
        "output_dir": str(output_dir),
        "attempts": [],
    }

    for step in steps:
        _execute_step(step, state)

    attempts: list[CommandExecution] = list(state["attempts"])
    observed_artifacts = sorted(path.name for path in output_dir.iterdir()) if output_dir.exists() else []
    artifact_hits = len(set(observed_artifacts) & set(scenario.expected_artifacts))
    artifact_completeness = artifact_hits / len(scenario.expected_artifacts) if scenario.expected_artifacts else 1.0
    parse_failures = sum(1 for attempt in attempts if attempt.parse_failure)
    invalid_tool_calls = sum(1 for attempt in attempts if attempt.invalid_tool_call)
    retries = _retry_count(attempts)
    task_success = _scenario_success(scenario.name, attempts)
    recovered = retries > 0 and task_success
    duration_ms = sum(attempt.duration_ms for attempt in attempts)

    return ScenarioResult(
        assist_level=assist_level,
        scenario=scenario.name,
        description=scenario.description,
        attempts=attempts,
        expected_artifacts=list(scenario.expected_artifacts),
        observed_artifacts=observed_artifacts,
        artifact_completeness=artifact_completeness,
        task_success=task_success,
        parse_failures=parse_failures,
        invalid_tool_calls=invalid_tool_calls,
        retries=retries,
        recovered=recovered,
        duration_ms=duration_ms,
    )


def _scenario_success(scenario_name: str, attempts: list[CommandExecution]) -> bool:
    successful_payloads = [attempt.payload for attempt in attempts if attempt.ok and attempt.payload is not None]
    if scenario_name == "inspect_unknown_series":
        for payload in successful_payloads:
            result = payload.get("result") or {}
            if result.get("method") == "descriptive":
                return True
            if ((result.get("data") or {}).get("workflow")) == "inspect-series":
                return True
        return False

    if scenario_name == "compare_forecasting_baselines":
        for payload in successful_payloads:
            result = payload.get("result") or {}
            if ((result.get("data") or {}).get("workflow")) == "forecast-series":
                return True
            metrics = result.get("metrics")
            if isinstance(metrics, dict) and len(metrics) >= 2:
                return True
        return False

    if scenario_name == "activity_windowing_workflow":
        saw_selection = False
        saw_evaluation = False
        for payload in successful_payloads:
            result = payload.get("result") or {}
            if ((result.get("data") or {}).get("workflow")) == "activity-recognition":
                return True
            if result.get("method") == "window_size_selection":
                saw_selection = True
            if result.get("method") == "windowed_classification":
                saw_evaluation = True
        return saw_selection and saw_evaluation

    raise ValueError(f"Unknown scenario: {scenario_name}")


def _retry_count(attempts: list[CommandExecution]) -> int:
    retries = 0
    failure_seen = False
    for attempt in attempts:
        if attempt.parse_failure or attempt.invalid_tool_call or bool(attempt.executed and attempt.exit_code not in {0, None}):
            failure_seen = True
            continue
        if failure_seen and attempt.executed:
            retries += 1
    return retries


def _summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    by_level: dict[str, dict[str, Any]] = {}
    for assist_level in _assist_levels():
        level_results = [result for result in results if result.assist_level == assist_level]
        attempt_count = sum(len(result.attempts) for result in level_results)
        parse_failures = sum(result.parse_failures for result in level_results)
        invalid_tool_calls = sum(result.invalid_tool_calls for result in level_results)
        retries = sum(result.retries for result in level_results)
        recoveries = sum(1 for result in level_results if result.recovered)
        success_count = sum(1 for result in level_results if result.task_success)
        artifact_scores = [result.artifact_completeness for result in level_results]
        by_level[assist_level] = {
            "scenarios": len(level_results),
            "task_success_rate": success_count / len(level_results) if level_results else 0.0,
            "parse_failure_rate": parse_failures / attempt_count if attempt_count else 0.0,
            "invalid_tool_calls": invalid_tool_calls,
            "avg_artifact_completeness": sum(artifact_scores) / len(artifact_scores) if artifact_scores else 0.0,
            "avg_duration_ms": sum(result.duration_ms for result in level_results) / len(level_results)
            if level_results
            else 0.0,
            "total_retries": retries,
            "recovery_success_rate": recoveries / len(level_results) if level_results else 0.0,
        }

    return {
        "scenario_count": len(SCENARIOS),
        "assist_levels": by_level,
    }


def _write_summary_markdown(report: BenchmarkReport, path: Path) -> None:
    lines = [
        "# Refactor Benchmark Summary",
        "",
        "This deterministic benchmark compares four assist levels on three representative tasks.",
        "",
        "| Assist level | Success rate | Parse failure rate | Invalid tool calls | Avg artifact completeness | Avg duration (ms) | Retries | Recovery rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for assist_level, metrics in report.summary["assist_levels"].items():
        lines.append(
            f"| `{assist_level}` | "
            f"{metrics['task_success_rate']:.2f} | "
            f"{metrics['parse_failure_rate']:.2f} | "
            f"{metrics['invalid_tool_calls']} | "
            f"{metrics['avg_artifact_completeness']:.2f} | "
            f"{metrics['avg_duration_ms']:.1f} | "
            f"{metrics['total_retries']} | "
            f"{metrics['recovery_success_rate']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Notes",
            "",
            "- `plain_model` is deliberately freeform and non-machine-runnable, which gives it a schema/parse failure on every scenario.",
            "- `plain_tools` uses raw tool access with no discovery or workflow bundling, so it tends to under-produce artifacts and occasionally guess the wrong contract.",
            "- `structured_discovery` uses `tool search` / `tool show` before raw tool execution, which improves task completion but still lacks workflow artifact bundles.",
            "- `skills_workflows` inspects the policy layer and then runs the workflow layer, which is why it should dominate artifact completeness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_refactor_benchmark(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> BenchmarkReport:
    """Run the deterministic refactor benchmark and write JSON/Markdown outputs."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    context = _scenario_context(output_root)
    results: list[ScenarioResult] = []

    for assist_level, step_builder in _assist_levels().items():
        for scenario in SCENARIOS:
            scenario_output_dir = output_root / "artifacts" / assist_level / scenario.name
            steps = step_builder(scenario, context)
            results.append(_run_scenario(assist_level, scenario, steps, context, scenario_output_dir))

    report = BenchmarkReport(
        output_dir=str(output_root),
        scenarios=results,
        summary=_summarize(results),
    )

    (output_root / "results.json").write_text(json.dumps(report.to_dict(), indent=2))
    _write_summary_markdown(report, output_root / "summary.md")
    return report


def _plain_model_steps(scenario: ScenarioDefinition, context: dict[str, Any]) -> list[StepDefinition]:
    narratives = {
        "inspect_unknown_series": (
            "Inspect the series manually in Python or a notebook and describe the basic stats in prose."
        ),
        "compare_forecasting_baselines": (
            "Use statsforecast directly and compare a few baselines without relying on repo contracts."
        ),
        "activity_windowing_workflow": (
            "Load the CSV into pandas, hand-build windows, and evaluate a classifier in an ad hoc script."
        ),
    }
    return [StepDefinition(label="freeform_plan", narrative=narratives[scenario.name])]


def _plain_tools_steps(scenario: ScenarioDefinition, context: dict[str, Any]) -> list[StepDefinition]:
    series_payload = context["series_input_json"]
    activity_csv = context["activity_csv_path"]
    if scenario.name == "inspect_unknown_series":
        return [
            StepDefinition(
                label="guess_wrong_data_wrapper",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "describe_series_with_data",
                    "--input-json",
                    series_payload,
                    "--json",
                ],
            ),
            StepDefinition(
                label="retry_raw_describe_tool",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "describe_series",
                    "--input-json",
                    series_payload,
                    "--json",
                ],
            ),
        ]
    if scenario.name == "compare_forecasting_baselines":
        return [
            StepDefinition(
                label="compare_forecasts_raw_tool",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "compare_forecasts",
                    "--input-json",
                    json.dumps(
                        {
                            "series": json.loads(series_payload)["series"],
                            "horizon": 5,
                            "models": ["arima", "theta"],
                        }
                    ),
                    "--json",
                ],
            )
        ]
    if scenario.name == "activity_windowing_workflow":
        return [
            StepDefinition(
                label="window_selection_only",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "select_window_size_from_csv",
                    "--param",
                    f"csv_path={activity_csv}",
                    "--param",
                    "value_columns=x,y,z",
                    "--param",
                    "label_column=label",
                    "--param",
                    "window_sizes=8,16",
                    "--param",
                    "classifier=knn",
                    "--param",
                    "test_size=0.25",
                    "--json",
                ],
            )
        ]
    raise ValueError(f"Unknown scenario: {scenario.name}")


def _structured_discovery_steps(scenario: ScenarioDefinition, context: dict[str, Any]) -> list[StepDefinition]:
    series_payload = context["series_input_json"]
    activity_csv = context["activity_csv_path"]
    if scenario.name == "inspect_unknown_series":
        return [
            StepDefinition(
                label="search_describe_tools",
                argv_factory=lambda _state: ["tool", "search", "describe", "--json"],
            ),
            StepDefinition(
                label="show_describe_series",
                argv_factory=lambda _state: ["tool", "show", "describe_series", "--json"],
            ),
            StepDefinition(
                label="run_describe_series",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "describe_series",
                    "--input-json",
                    series_payload,
                    "--json",
                ],
            ),
        ]
    if scenario.name == "compare_forecasting_baselines":
        return [
            StepDefinition(
                label="search_forecast_tools",
                argv_factory=lambda _state: ["tool", "search", "forecast", "--json"],
            ),
            StepDefinition(
                label="show_compare_forecasts",
                argv_factory=lambda _state: ["tool", "show", "compare_forecasts", "--json"],
            ),
            StepDefinition(
                label="run_compare_forecasts",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "compare_forecasts",
                    "--input-json",
                    json.dumps(
                        {
                            "series": json.loads(series_payload)["series"],
                            "horizon": 5,
                            "models": ["arima", "theta"],
                        }
                    ),
                    "--json",
                ],
            ),
        ]
    if scenario.name == "activity_windowing_workflow":
        return [
            StepDefinition(
                label="search_window_tools",
                argv_factory=lambda _state: ["tool", "search", "window", "--json"],
            ),
            StepDefinition(
                label="show_window_selection_tool",
                argv_factory=lambda _state: ["tool", "show", "select_window_size_from_csv", "--json"],
            ),
            StepDefinition(
                label="show_window_eval_tool",
                argv_factory=lambda _state: ["tool", "show", "evaluate_windowed_classifier_from_csv", "--json"],
            ),
            StepDefinition(
                label="run_window_selection",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "select_window_size_from_csv",
                    "--param",
                    f"csv_path={activity_csv}",
                    "--param",
                    "value_columns=x,y,z",
                    "--param",
                    "label_column=label",
                    "--param",
                    "window_sizes=8,16",
                    "--param",
                    "classifier=knn",
                    "--param",
                    "test_size=0.25",
                    "--json",
                ],
            ),
            StepDefinition(
                label="run_window_evaluation",
                argv_factory=lambda _state: [
                    "tool",
                    "run",
                    "evaluate_windowed_classifier_from_csv",
                    "--param",
                    f"csv_path={activity_csv}",
                    "--param",
                    "window_size=8",
                    "--param",
                    "value_columns=x,y,z",
                    "--param",
                    "label_column=label",
                    "--param",
                    "classifier=knn",
                    "--param",
                    "test_size=0.25",
                    "--json",
                ],
            ),
        ]
    raise ValueError(f"Unknown scenario: {scenario.name}")


def _skills_workflow_steps(scenario: ScenarioDefinition, context: dict[str, Any]) -> list[StepDefinition]:
    series_payload = context["series_input_json"]
    activity_csv = context["activity_csv_path"]
    if scenario.name == "inspect_unknown_series":
        return [
            StepDefinition(
                label="inspect_skill_policy",
                argv_factory=lambda _state: ["skills", "show", "diagnostics", "--json"],
            ),
            StepDefinition(
                label="run_inspect_workflow",
                argv_factory=lambda state: [
                    "workflow",
                    "run",
                    "inspect-series",
                    "--input-json",
                    series_payload,
                    "--output-dir",
                    state["output_dir"],
                    "--skip-plots",
                    "--json",
                ],
            ),
        ]
    if scenario.name == "compare_forecasting_baselines":
        return [
            StepDefinition(
                label="inspect_forecasting_skill",
                argv_factory=lambda _state: ["skills", "show", "forecasting", "--json"],
            ),
            StepDefinition(
                label="run_forecast_workflow",
                argv_factory=lambda state: [
                    "workflow",
                    "run",
                    "forecast-series",
                    "--input-json",
                    series_payload,
                    "--horizon",
                    "5",
                    "--output-dir",
                    state["output_dir"],
                    "--skip-plots",
                    "--json",
                ],
            ),
        ]
    if scenario.name == "activity_windowing_workflow":
        return [
            StepDefinition(
                label="inspect_activity_skill",
                argv_factory=lambda _state: ["skills", "show", "activity-recognition", "--json"],
            ),
            StepDefinition(
                label="run_activity_workflow",
                argv_factory=lambda state: [
                    "workflow",
                    "run",
                    "activity-recognition",
                    "--input",
                    activity_csv,
                    "--value-cols",
                    "x,y,z",
                    "--label-col",
                    "label",
                    "--window-sizes",
                    "8,16",
                    "--classifier",
                    "knn",
                    "--output-dir",
                    state["output_dir"],
                    "--skip-plots",
                    "--json",
                ],
            ),
        ]
    raise ValueError(f"Unknown scenario: {scenario.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic refactor benchmark for workstreams 8 and 9."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON/Markdown outputs (default: benchmarks/results/latest).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_refactor_benchmark(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": report.output_dir,
                "summary": report.summary,
                "results_path": str(Path(report.output_dir) / "results.json"),
                "markdown_path": str(Path(report.output_dir) / "summary.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
