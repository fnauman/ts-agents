import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from ts_agents.cli.main import run
from ts_agents.core.comparison import ComparisonResult


def test_workflow_list_json_returns_envelope(capsys):
    code = run(["workflow", "list", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"
    assert payload["ok"] is True
    assert payload["command"] == "workflow list"
    workflow_names = [workflow["name"] for workflow in payload["result"]["workflows"]]
    assert "activity-recognition" in workflow_names
    assert "inspect-series" in workflow_names
    assert "forecast-series" in workflow_names


def test_workflow_run_inspect_series_accepts_stdin_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"series":[1,2,3,4,5]}'))

    output_dir = tmp_path / "inspect"
    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--stdin",
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "inspect-series"
    assert payload["result"]["data"]["workflow"] == "inspect-series"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "summary.json") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths
    assert payload["result"]["data"]["autocorrelation"]["max_lag"] == 4
    assert payload["result"]["data"]["autocorrelation"]["requested_max_lag"] == 8


def test_workflow_run_forecast_series_writes_expected_files(monkeypatch, capsys, tmp_path):
    import ts_agents.core.comparison as comparison_mod
    import ts_agents.workflows.forecast as forecast_workflow

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(series, horizon, methods, validation_size=None):
        return ComparisonResult(
            category="forecasting",
            methods=list(methods),
            results={},
            metrics={
                "arima": {"mae": 0.3, "rmse": 0.3, "mape": 10.0},
                "theta": {"mae": 0.1, "rmse": 0.1, "mape": 4.0},
            },
            rankings={"rmse": ["theta", "arima"], "mae": ["theta", "arima"], "mape": ["theta", "arima"]},
            recommendation="theta is the best baseline.",
            computation_times={"arima": 0.01, "theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)
    monkeypatch.setattr(
        forecast_workflow,
        "_forecast_with_method",
        lambda series, method, horizon: SimpleNamespace(forecast=np.array([1.4, 1.5])),
    )

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "forecast-series"
    assert payload["result"]["data"]["best_method"] == "theta"
    assert (output_dir / "forecast_comparison.json").exists()
    assert (output_dir / "forecast.json").exists()
    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "report.md").exists()
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "forecast_comparison.json") in artifact_paths
    assert str(output_dir / "forecast.json") in artifact_paths
    assert str(output_dir / "forecast.csv") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths


def test_workflow_run_forecast_series_reports_degraded_when_some_methods_fail(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.comparison as comparison_mod
    import ts_agents.workflows.forecast as forecast_workflow

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(series, horizon, methods, validation_size=None):
        return ComparisonResult(
            category="forecasting",
            methods=["theta"],
            results={},
            metrics={
                "arima": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
                "theta": {"mae": 0.1, "rmse": 0.1, "mape": 4.0},
            },
            rankings={"rmse": ["theta"], "mae": ["theta"], "mape": ["theta"]},
            recommendation="theta is the best baseline.",
            computation_times={"theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)
    monkeypatch.setattr(
        forecast_workflow,
        "_forecast_with_method",
        lambda series, method, horizon: SimpleNamespace(forecast=np.array([1.4, 1.5])),
    )

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["status"] == "degraded"
    assert payload["result"]["data"]["valid_methods"] == ["theta"]
    assert payload["result"]["data"]["failed_methods"] == ["arima"]
    assert "partial_method_failure" in payload["result"]["data"]["quality_flags"]
    assert "only_one_valid_method" in payload["result"]["data"]["quality_flags"]
    assert payload["result"]["data"]["best_method"] == "theta"


def test_workflow_run_forecast_series_fails_when_all_methods_fail(monkeypatch, capsys, tmp_path):
    import ts_agents.core.comparison as comparison_mod

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(series, horizon, methods, validation_size=None):
        return ComparisonResult(
            category="forecasting",
            methods=[],
            results={},
            metrics={
                "arima": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
                "theta": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
            },
            rankings={},
            recommendation="Unable to generate recommendation - no valid results.",
            computation_times={},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "dependency_error"
    assert "All forecast methods failed" in payload["error"]["message"]
    assert not (output_dir / "forecast.json").exists()


def test_workflow_run_activity_recognition_writes_expected_files(monkeypatch, capsys, tmp_path):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.66, 96: 0.84},
                "n_windows_by_window": {32: 180, 96: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.84

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.84,
                "n_windows": 120,
                "class_counts": {"idle": 62, "walk": 58},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.86,
                    "f1_score": 0.83,
                    "confusion_matrix": [[26, 4], [5, 25]],
                    "predictions": ["idle", "walk"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--window-sizes",
            "32,96",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "activity-recognition"
    assert payload["result"]["data"]["best_window_size"] == 96
    assert payload["result"]["data"]["classifier_used"] == "minirocket"
    assert (output_dir / "window_selection.json").exists()
    assert (output_dir / "eval.json").exists()
    assert (output_dir / "report.md").exists()
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "window_selection.json") in artifact_paths
    assert str(output_dir / "eval.json") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths


def test_workflow_run_activity_recognition_sanitizes_nan_scores_and_surfaces_quality_flags(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: float("nan"), 96: 1.0},
                "n_windows_by_window": {32: 7, 96: 12},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 1.0

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 1.0,
                "n_windows": 12,
                "class_counts": {"idle": 8, "walk": 4, "jog": 3},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[8, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "predictions": ["idle"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--window-sizes",
            "32,96",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    stdout = capsys.readouterr().out
    assert "NaN" not in stdout
    payload = json.loads(
        stdout,
        parse_constant=lambda constant: (_ for _ in ()).throw(AssertionError(constant)),
    )
    assert payload["result"]["status"] == "degraded"
    flags = set(payload["result"]["data"]["quality_flags"])
    assert "nan_window_scores" in flags
    assert "too_few_windows" in flags
    assert "single_class_test_split" in flags
    assert "metric_not_comparable" in flags
    assert "perfect_metrics" in flags
    assert payload["result"]["data"]["window_selection"]["scores_by_window"]["32"] is None


def test_activity_window_selection_plot_ignores_null_scores():
    import ts_agents.workflows.activity as activity_workflow

    fig = activity_workflow._plot_window_selection(
        {
            "best_window_size": 96,
            "metric": "balanced_accuracy",
            "scores_by_window": {"32": None, "96": 0.84},
        }
    )

    assert fig is not None
    activity_workflow._close_plots(fig)


def test_workflow_run_activity_recognition_continues_when_plot_generation_fails(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing
    import ts_agents.workflows.activity as activity_workflow

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.66, 96: 0.84},
                "n_windows_by_window": {32: 180, 96: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.84

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.84,
                "n_windows": 120,
                "class_counts": {"idle": 62, "walk": 58},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.86,
                    "f1_score": 0.83,
                    "confusion_matrix": [[26, 4], [5, 25]],
                    "predictions": ["idle", "walk"],
                },
            }

    def raise_plot_error(_payload):
        raise ValueError("plot payload incomplete")

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())
    monkeypatch.setattr(activity_workflow, "_plot_window_selection", raise_plot_error)

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["warnings"] == ["Skipping activity-recognition plots: plot payload incomplete"]
    assert (output_dir / "window_selection.json").exists()
    assert (output_dir / "eval.json").exists()
    assert (output_dir / "report.md").exists()
    assert not (output_dir / "window_scores.png").exists()
    assert not (output_dir / "confusion_matrix.png").exists()


def test_handle_workflow_command_uses_registry_runner(monkeypatch):
    import importlib
    import ts_agents.workflows as workflows

    cli_main = importlib.import_module("ts_agents.cli.main")
    observed = {}
    fake_series_input = SimpleNamespace(provenance={"series_ref": {"source_type": "inline_json"}})

    def fake_runner(series_input, **kwargs):
        observed["series_input"] = series_input
        observed["kwargs"] = kwargs
        return {"ok": True}

    def fake_build_runner_kwargs(args):
        observed["builder_args"] = args.workflow_name
        return {"output_dir": "outputs/custom", "skip_plots": True}

    def fake_load_input(args):
        observed["loader_args"] = args.workflow_name
        return fake_series_input

    monkeypatch.setattr(
        workflows,
        "get_workflow",
        lambda name: SimpleNamespace(
            name=name,
            runner=fake_runner,
            load_input=fake_load_input,
            build_runner_kwargs=fake_build_runner_kwargs,
        ),
    )

    args = argparse.Namespace(
        workflow_command="run",
        workflow_name="inspect-series",
        input=None,
        input_json='{"series":[1,2,3]}',
        stdin=False,
        run_id=None,
        variable=None,
        time_col=None,
        value_col=None,
        use_test_data=False,
        full_data=False,
        output_dir="outputs/inspect",
        max_lag=None,
        skip_plots=True,
    )

    result, text = cli_main._handle_workflow_command(args)

    assert text is None
    assert result == {"ok": True}
    assert observed["series_input"] is fake_series_input
    assert observed["loader_args"] == "inspect-series"
    assert observed["builder_args"] == "inspect-series"
    assert observed["kwargs"] == {
        "output_dir": "outputs/custom",
        "skip_plots": True,
    }
    assert args._ts_input_payload["options"] == {
        "output_dir": "outputs/custom",
        "skip_plots": True,
    }


def test_workflow_run_inspect_series_returns_absolute_paths_for_relative_output_dir(
    capsys,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            "outputs/inspect",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    expected_output_dir = str((tmp_path / "outputs" / "inspect").resolve())
    assert payload["result"]["data"]["output_dir"] == expected_output_dir
    for artifact in payload["result"]["artifacts"]:
        assert Path(artifact["path"]).is_absolute()
