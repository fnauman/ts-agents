import argparse
import json
from pathlib import Path

import numpy as np


def test_resolve_demo_report_path_uses_output_dir_for_default_report(tmp_path):
    from ts_agents.cli.main import _resolve_demo_report_path

    args = argparse.Namespace(
        output_dir=str(tmp_path / "demo_out"),
        report_path="outputs/demo/report.md",
    )

    resolved = _resolve_demo_report_path(args)
    assert resolved == Path(args.output_dir) / "report.md"


def test_resolve_demo_report_path_keeps_explicit_path(tmp_path):
    from ts_agents.cli.main import _resolve_demo_report_path

    explicit = tmp_path / "custom" / "report.md"
    args = argparse.Namespace(
        output_dir=str(tmp_path / "demo_out"),
        report_path=str(explicit),
    )

    resolved = _resolve_demo_report_path(args)
    assert resolved == explicit


def test_scripted_demo_writes_report(monkeypatch, tmp_path):
    from ts_agents.cli.main import _run_demo_window_classification_scripted
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 64
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 64,
                "metric": "balanced_accuracy",
                "scores_by_window": {64: 1.0},
                "n_windows_by_window": {64: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 1.0

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 64,
                "stride": 32,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 1.0,
                "n_windows": 120,
                "class_counts": {"idle": 70, "walk": 50},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[20, 0], [0, 30]],
                    "predictions": ["idle", "walk"],
                },
            }

    def fake_select_window_size(*args, **kwargs):
        return FakeSelection()

    def fake_evaluate_windowed_classifier(*args, **kwargs):
        return FakeEval()

    monkeypatch.setattr(windowing, "select_window_size", fake_select_window_size)
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", fake_evaluate_windowed_classifier)

    csv_path = tmp_path / "demo_labeled_stream.csv"
    csv_path.write_text("x,y,z,label\n0,0,0,idle\n")

    output_dir = tmp_path / "outputs"
    args = argparse.Namespace(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        no_generate=True,
        scenario="gait",
        hz=20,
        minutes=4,
        seed=1337,
        value_columns="x,y,z",
        window_sizes="32,64",
        classifier="auto",
        label_column="label",
        metric="balanced_accuracy",
        balance="segment_cap",
        max_windows_per_segment=25,
        test_size=0.25,
        skip_plots=True,
        report_path="outputs/demo/report.md",
        no_llm=True,
    )

    result = _run_demo_window_classification_scripted(args)

    report_path = output_dir / "report.md"
    assert report_path.exists()
    assert result["report_path"] == str(report_path)
    assert "Report on Windowed Classification" in report_path.read_text()


def test_demo_parser_includes_forecasting_subcommand():
    from ts_agents.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["demo", "forecasting", "--no-llm"])

    assert args.command == "demo"
    assert args.demo_command == "forecasting"
    assert args.no_llm is True


def test_demo_forecasting_default_methods_are_stable_for_tiny_data():
    from ts_agents.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["demo", "forecasting", "--no-llm"])

    assert args.methods == "arima,theta"


def test_demo_alias_emits_deprecation_warning(monkeypatch, capsys):
    import importlib

    cli_main = importlib.import_module("ts_agents.cli.main")

    monkeypatch.setattr(cli_main, "_handle_demo_command", lambda args: ({"ok": True}, "demo result"))

    code = cli_main.run(["demo", "forecasting", "--no-llm"])

    assert code == 0
    captured = capsys.readouterr()
    assert "demo result" in captured.out
    assert "`ts-agents demo` is a legacy compatibility surface" in captured.err


def test_demo_window_default_scenario_is_stairs():
    from ts_agents.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["demo", "window-classification", "--no-llm"])

    assert args.scenario == "stairs"


def test_demo_window_default_csv_path_exists():
    from ts_agents.cli.main import _default_demo_csv_path

    assert Path(_default_demo_csv_path()).exists()


def test_scripted_forecasting_demo_writes_report_and_json(monkeypatch, tmp_path):
    from ts_agents.cli.main import _run_demo_forecasting_scripted
    from ts_agents.core.comparison import ComparisonResult
    import ts_agents.data_access as data_access
    import ts_agents.core.comparison as comparison_mod

    def fake_get_series(run_id, variable_name, use_test_data=None, data_type=None):
        return np.array([0.10, 0.12, 0.15, 0.14, 0.16])

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=["arima", "theta"],
            results={},
            metrics={
                "arima": {"mae": 0.03, "rmse": 0.03, "mape": 18.0},
                "theta": {"mae": 0.01, "rmse": 0.01, "mape": 6.7},
            },
            rankings={"rmse": ["theta", "arima"], "mae": ["theta", "arima"], "mape": ["theta", "arima"]},
            recommendation="theta looks best",
            computation_times={"arima": 0.01, "theta": 0.01},
        )

    monkeypatch.setattr(data_access, "get_series", fake_get_series)
    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)

    output_dir = tmp_path / "forecast_demo"
    args = argparse.Namespace(
        run_id="Re200Rm200",
        variable="bx001_real",
        horizon=1,
        validation_size=None,
        methods="arima,theta",
        use_test_data=False,
        full_data=False,
        output_dir=str(output_dir),
        report_path="outputs/demo/forecasting_report.md",
        no_llm=True,
        model=None,
        print_report=False,
        skip_plots=True,
    )

    result = _run_demo_forecasting_scripted(args)

    comparison_path = output_dir / "forecast_comparison.json"
    report_path = output_dir / "forecasting_report.md"
    assert comparison_path.exists()
    assert report_path.exists()
    assert result["comparison_path"] == str(comparison_path)
    assert result["report_path"] == str(report_path)
    assert result["best_method"] == "theta"
    assert "Report on Forecasting Demo" in report_path.read_text()


def test_demo_window_classification_cli_scripted_integration(monkeypatch, tmp_path):
    from ts_agents.cli.main import run
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

    csv_path = tmp_path / "demo_labeled_stream.csv"
    csv_path.write_text("x,y,z,label\n0,0,0,idle\n1,1,1,walk\n")
    output_dir = tmp_path / "window_demo"

    code = run(
        [
            "demo",
            "window-classification",
            "--no-llm",
            "--no-generate",
            "--skip-plots",
            "--csv-path",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--window-sizes",
            "32,96",
        ]
    )

    assert code == 0
    selection = json.loads((output_dir / "window_selection.json").read_text())
    evaluation = json.loads((output_dir / "eval.json").read_text())
    assert selection["best_window_size"] == 96
    assert evaluation["metric"] == "balanced_accuracy"
    assert float(evaluation["score"]) > 0
    assert "classification" in evaluation
    assert (output_dir / "report.md").exists()


def test_demo_forecasting_cli_scripted_integration(monkeypatch, tmp_path):
    from ts_agents.cli.main import run
    from ts_agents.core.comparison import ComparisonResult
    import ts_agents.data_access as data_access
    import ts_agents.core.comparison as comparison_mod

    monkeypatch.setattr(
        data_access,
        "get_series",
        lambda run_id, variable_name, use_test_data=None, data_type=None: np.array([0.1, 0.2, 0.3, 0.4]),
    )

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=["arima", "theta"],
            results={},
            metrics={
                "arima": {"mae": 0.03, "rmse": 0.03, "mape": 9.0},
                "theta": {"mae": 0.02, "rmse": 0.02, "mape": 6.0},
            },
            rankings={
                "rmse": ["theta", "arima"],
                "mae": ["theta", "arima"],
                "mape": ["theta", "arima"],
            },
            recommendation="theta is preferred",
            computation_times={"arima": 0.01, "theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)

    output_dir = tmp_path / "forecast_demo"
    code = run(
        [
            "demo",
            "forecasting",
            "--no-llm",
            "--skip-plots",
            "--run-id",
            "Re200Rm200",
            "--variable",
            "bx001_real",
            "--methods",
            "arima,theta",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    comparison = json.loads((output_dir / "forecast_comparison.json").read_text())
    assert set(comparison["metrics"].keys()) == {"arima", "theta"}
    assert comparison["rankings"]["rmse"][0] == "theta"
    assert comparison["recommendation"]
    assert (output_dir / "forecasting_report.md").exists()
