import json
from pathlib import Path


def test_run_refactor_benchmark_writes_outputs(monkeypatch, tmp_path):
    import ts_agents.evals.refactor_benchmark as benchmark_mod

    def fake_run_cli(argv):
        command = tuple(argv[:3])

        if tuple(argv[:2]) == ("workflow", "run"):
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            workflow_name = argv[2]
            if workflow_name == "inspect-series":
                for name in ("summary.json", "report.md"):
                    (output_dir / name).write_text(name)
                payload = {"ok": True, "result": {"data": {"workflow": "inspect-series"}}}
            elif workflow_name == "forecast-series":
                for name in ("forecast_comparison.json", "forecast.json", "forecast.csv", "report.md"):
                    (output_dir / name).write_text(name)
                payload = {"ok": True, "result": {"data": {"workflow": "forecast-series"}}}
            else:
                for name in ("window_selection.json", "eval.json", "report.md"):
                    (output_dir / name).write_text(name)
                payload = {"ok": True, "result": {"data": {"workflow": "activity-recognition"}}}
            print(json.dumps(payload))
            return 0

        if command == ("tool", "run", "describe_series_with_data"):
            print(json.dumps({"ok": False, "error": {"code": "validation_error"}}))
            return 2
        if command == ("tool", "run", "describe_series"):
            print(json.dumps({"ok": True, "result": {"method": "descriptive"}}))
            return 0
        if command == ("tool", "run", "compare_forecasts"):
            print(json.dumps({"ok": True, "result": {"metrics": {"arima": {}, "theta": {}}}}))
            return 0
        if command == ("tool", "run", "select_window_size_from_csv"):
            print(json.dumps({"ok": True, "result": {"method": "window_size_selection", "best_window_size": 8}}))
            return 0
        if command == ("tool", "run", "evaluate_windowed_classifier_from_csv"):
            print(json.dumps({"ok": True, "result": {"method": "windowed_classification"}}))
            return 0

        if tuple(argv[:2]) in {
            ("tool", "search"),
            ("tool", "show"),
            ("skills", "show"),
            ("workflow", "list"),
        }:
            print(json.dumps({"ok": True, "result": {}}))
            return 0

        raise AssertionError(f"Unexpected argv: {argv}")

    monkeypatch.setattr(benchmark_mod, "run_cli", fake_run_cli)

    report = benchmark_mod.run_refactor_benchmark(tmp_path)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert report.summary["assist_levels"]["plain_model"]["task_success_rate"] == 0.0
    assert report.summary["assist_levels"]["plain_model"]["parse_failure_rate"] == 1.0
    assert report.summary["assist_levels"]["plain_tools"]["invalid_tool_calls"] == 1
    assert report.summary["assist_levels"]["structured_discovery"]["task_success_rate"] == 1.0
    assert report.summary["assist_levels"]["skills_workflows"]["avg_artifact_completeness"] == 1.0


def test_refactor_benchmark_main_prints_output_paths(monkeypatch, tmp_path, capsys):
    import ts_agents.evals.refactor_benchmark as benchmark_mod

    fake_report = benchmark_mod.BenchmarkReport(
        output_dir=str(tmp_path),
        scenarios=[],
        summary={"assist_levels": {}},
    )

    monkeypatch.setattr(benchmark_mod, "run_refactor_benchmark", lambda output_dir: fake_report)

    code = benchmark_mod.main(["--output-dir", str(tmp_path)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(tmp_path)
    assert payload["results_path"] == str(tmp_path / "results.json")
    assert payload["markdown_path"] == str(tmp_path / "summary.md")


def test_execute_step_silent_success_is_not_parse_failure(monkeypatch):
    import ts_agents.evals.refactor_benchmark as benchmark_mod

    monkeypatch.setattr(benchmark_mod, "run_cli", lambda argv: 0)

    step = benchmark_mod.StepDefinition(
        label="silent_success",
        argv_factory=lambda state: ["tool", "search", "forecast", "--json"],
    )
    state = {"attempts": []}

    execution = benchmark_mod._execute_step(step, state)

    assert execution.executed is True
    assert execution.exit_code == 0
    assert execution.stdout == ""
    assert execution.parse_failure is False
    assert execution.payload is None
