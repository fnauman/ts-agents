import argparse
import io
import json
import sys

import pytest

from ts_agents.cli.main import (
    _build_run_example_command,
    _parse_param_value,
    _raise_missing_required_error,
    _suggest_tool_names,
    build_parser,
    run,
)




def test_parse_list_ints_from_csv():
    assert _parse_param_value("1,2,3", "list[int]") == [1, 2, 3]


def test_parse_list_ints_from_json():
    assert _parse_param_value("[1,2,3]", "list[int]") == [1, 2, 3]


def test_parse_union_str_list_prefers_str():
    assert _parse_param_value("value", "str | list[str]") == "value"


def test_parse_union_str_list_from_csv():
    assert _parse_param_value("a,b", "str | list[str]") == ["a", "b"]


def test_parse_union_str_list_from_json():
    assert _parse_param_value("[\"a\",\"b\"]", "str | list[str]") == ["a", "b"]


def test_suggest_tool_names_returns_close_matches():
    suggestions = _suggest_tool_names(
        "forecast_thta",
        [
            "forecast_theta",
            "forecast_ets",
            "stl_decompose",
        ],
    )
    assert "forecast_theta" in suggestions


def test_build_run_example_command_uses_series_file_input():
    command = _build_run_example_command(
        "forecast_theta",
        required=["series"],
        param_types={"series": "np.ndarray"},
    )
    assert command.startswith("ts-agents tool run forecast_theta")
    assert "--input data.csv --value-col value" in command
    assert "--param" not in command


def test_build_run_example_command_quotes_dict_params_shell_safe():
    command = _build_run_example_command(
        "compare_series_stats",
        required=["series_dict"],
        param_types={"series_dict": "dict"},
    )
    assert "--param 'series_dict={\"key\":\"value\"}'" in command


def test_missing_required_error_includes_actionable_example():
    with pytest.raises(ValueError) as exc_info:
        _raise_missing_required_error(
            tool_name="compare_forecasts",
            param_types={
                "series": "np.ndarray",
                "horizon": "int",
                "models": "list",
            },
            required=["series", "horizon"],
            provided={"horizon": 3},
        )

    message = str(exc_info.value)
    assert "Missing required parameters for 'compare_forecasts'" in message
    assert "Required parameters:" in message
    assert "Example:" in message
    assert "ts-agents tool run compare_forecasts --input data.csv" in message


def test_tool_run_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tool", "run", "--help"])
    output = capsys.readouterr().out
    assert "Examples" in output
    assert "tool run detect_periodicity --input" in output
    assert "tool search anomaly" in output


@pytest.mark.parametrize("command", ["run", "demo"])
def test_removed_legacy_commands_are_rejected(command):
    parser = build_parser(exit_on_error=False)
    with pytest.raises(argparse.ArgumentError):
        parser.parse_args([command, "--help"])


def test_workflow_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["workflow", "run", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "workflow show forecast-series --json" in output
    assert "workflow run inspect-series" in output
    assert "workflow run forecast-series" in output
    assert "workflow run activity-recognition" in output
    assert "workflow run forecast-panel" in output
    assert "seasonal_naive,lightgbm,chronos2_small" in output


def test_forecast_panel_help_documents_every_flag(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["workflow", "run", "forecast-panel", "--help"])
    output = " ".join(capsys.readouterr().out.split())  # undo argparse line wrapping
    for expected in (
        "Series identifier column",
        "Target value column",
        "outputs/panel",
        "Forecast horizon per origin",
        "Rolling-validation origins",
        "Estimator random seed",
        "Skip comparison plot generation",
        "NHITS context; defaults to 2*horizon",
        "Foundation-model context (default 512, capped per model)",
        "NHITS/foundation-model device",
        "chronos2_small",
        "patchtst_fm",
    ):
        assert expected in output, expected


def test_forecast_series_help_lists_foundation_methods_and_flags(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["workflow", "run", "forecast-series", "--help"])
    output = " ".join(capsys.readouterr().out.split())
    assert "--context-length" in output
    assert "--accelerator" in output
    assert "chronos2_small" in output
    assert "timesfm2p5" in output


def test_forecast_series_builder_forwards_foundation_options_only_when_set():
    from ts_agents.workflows import get_workflow

    parser = build_parser()
    workflow = get_workflow("forecast-series")
    base_argv = ["workflow", "run", "forecast-series", "--input-json", '{"series":[1,2,3]}']

    args = parser.parse_args([*base_argv, "--methods", " seasonal_naive , chronos2_small ,"])
    assert args.context_length is None and args.accelerator is None
    kwargs = workflow.build_runner_kwargs(args)
    assert kwargs["methods"] == ["seasonal_naive", "chronos2_small"]
    # Unset options stay out so resume identities of older runs are unchanged.
    assert "context_length" not in kwargs
    assert "accelerator" not in kwargs

    args = parser.parse_args(
        [*base_argv, "--methods", "chronos2_small", "--context-length", "256", "--accelerator", "gpu"]
    )
    kwargs = workflow.build_runner_kwargs(args)
    assert kwargs["context_length"] == 256
    assert kwargs["accelerator"] == "gpu"


def test_forecast_panel_builder_forwards_context_length_only_when_set():
    from ts_agents.workflows import get_workflow

    parser = build_parser()
    workflow = get_workflow("forecast-panel")
    base_argv = ["workflow", "run", "forecast-panel", "--input", "panel.csv", "--freq", "MS"]

    args = parser.parse_args([*base_argv, "--methods", "seasonal_naive, chronos2_small"])
    kwargs = workflow.build_runner_kwargs(args)
    assert kwargs["methods"] == ["seasonal_naive", "chronos2_small"]
    assert "context_length" not in kwargs
    assert kwargs["accelerator"] == "cpu"

    args = parser.parse_args([*base_argv, "--methods", "chronos2_small", "--context-length", "128"])
    kwargs = workflow.build_runner_kwargs(args)
    assert kwargs["context_length"] == 128


def test_autoresearch_help_uses_foundation_smoke_loop(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["autoresearch", "run", "--help"])
    output = capsys.readouterr().out
    assert "autoresearch run foundation-smoke --dry-run --json" in output
    assert "foundation-smoke --models timesfm2p5" in output
    assert "foundation-chronos-smoke" not in output


def test_workflow_run_parser_accepts_explicit_fallback_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3]}',
            "--sandbox",
            "docker",
            "--allow-fallback",
            "--fallback-backend",
            "local",
        ]
    )
    assert args.allow_fallback is True
    assert args.fallback_backend == "local"


def test_sandbox_help_includes_commands(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sandbox", "--help"])
    output = capsys.readouterr().out
    assert "list" in output
    assert "doctor" in output


def test_unknown_tool_error_includes_suggestions(capsys):
    code = run(["tool", "run", "forecast_thta", "--param", "horizon=3"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Did you mean:" in err
    assert "forecast_theta" in err


def test_missing_required_error_is_actionable(capsys):
    code = run(["tool", "run", "forecast_theta", "--param", "horizon=3"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Missing required parameters for 'forecast_theta'" in err
    assert "Example:" in err


def test_parser_accepts_extract_images_with_save():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tool",
            "run",
            "describe_series",
            "--param",
            "series=[1,2,3]",
            "--save",
            "outputs/result.txt",
            "--extract-images",
            "outputs/assets",
        ]
    )
    assert args.extract_images == "outputs/assets"
    assert args.save == "outputs/result.txt"


def test_tool_run_parser_accepts_explicit_fallback_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tool",
            "run",
            "describe_series",
            "--param",
            "series=[1,2,3]",
            "--sandbox",
            "docker",
            "--allow-fallback",
            "--fallback-backend",
            "local",
        ]
    )
    assert args.allow_fallback is True
    assert args.fallback_backend == "local"


def test_extract_images_requires_save(capsys):
    code = run(["data", "vars", "--extract-images", "outputs/assets"])
    assert code == 2
    err = capsys.readouterr().err
    assert "--extract-images requires --save" in err


def test_data_vars_command_does_not_require_runs_attribute(monkeypatch, capsys):
    import importlib
    import ts_agents.data_access as data_access

    cli_main = importlib.import_module("ts_agents.cli.main")

    monkeypatch.setattr(data_access, "list_runs", lambda **kwargs: ["Re200Rm200"])
    monkeypatch.setattr(data_access, "list_variables", lambda **kwargs: ["bx001_real", "by001_real"])

    code = cli_main.run(["data", "vars"])

    assert code == 0
    out = capsys.readouterr().out
    assert "Variables:" in out
    assert "- bx001_real" in out
    assert "- by001_real" in out


def test_run_extracts_images_when_saving(monkeypatch, tmp_path):
    import importlib

    cli_main = importlib.import_module("ts_agents.cli.main")

    def fake_handle_data_command(args):
        return {"ok": True}, "Result\n[IMAGE_DATA:ZmFrZQ==]"

    monkeypatch.setattr(cli_main, "_handle_data_command", fake_handle_data_command)

    output_path = tmp_path / "result.txt"
    images_dir = tmp_path / "assets"
    code = cli_main.run(
        [
            "data",
            "vars",
            "--save",
            str(output_path),
            "--extract-images",
            str(images_dir),
        ]
    )

    assert code == 0
    saved = output_path.read_text()
    assert "[IMAGE_DATA:" not in saved
    assert "[IMAGE_FILE:" in saved
    assert len(list(images_dir.glob("*.png"))) == 1


def test_run_extracts_images_in_json_without_breaking_json(monkeypatch, tmp_path, capsys):
    import importlib

    cli_main = importlib.import_module("ts_agents.cli.main")

    def fake_handle_data_command(args):
        return {"message": "Result [IMAGE_DATA:ZmFrZQ==]"}, None

    monkeypatch.setattr(cli_main, "_handle_data_command", fake_handle_data_command)

    output_path = tmp_path / "result.json"
    images_dir = tmp_path / "assets\\Udir"
    code = cli_main.run(
        [
            "data",
            "vars",
            "--json",
            "--save",
            str(output_path),
            "--extract-images",
            str(images_dir),
        ]
    )

    assert code == 0
    saved = output_path.read_text()
    payload = json.loads(saved)
    assert payload["ok"] is True
    assert "[IMAGE_DATA:" not in payload["result"]["message"]
    assert "[IMAGE_FILE:" in payload["result"]["message"]
    assert len(list(images_dir.glob("*.png"))) == 1
    assert json.loads(capsys.readouterr().out) == payload

    # An output failure must replace success, rather than concatenate a second
    # JSON object after an already published successful response.
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli_main, "write_output", fail_save)
    assert cli_main.run(["data", "vars", "--json", "--save", str(output_path)]) == 6
    failure = json.loads(capsys.readouterr().out)
    assert failure["ok"] is False
    assert "disk full" in failure["error"]["message"]


def test_cli_compare_forecasts_surfaces_tool_failure(monkeypatch, capsys, tmp_path):
    import importlib
    import ts_agents.core.forecasting as forecasting

    cli_main = importlib.import_module("ts_agents.cli.main")
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n0.1\n0.12\n0.15\n0.14\n0.16\n")

    def fake_compare_forecasts(series, horizon=10, test_size=None, models=None, **kwargs):
        raise ValueError("number sections must be larger than 0")

    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    code = cli_main.run(
        [
            "tool",
            "run",
            "compare_forecasts",
            "--input",
            str(csv_path),
            "--param",
            "models=arima,theta",
            "--param",
            "horizon=12",
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "number sections must be larger than 0" in err


def test_tool_show_json_returns_envelope(capsys):
    code = run(["tool", "show", "forecast_theta", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"
    assert payload["ok"] is True
    assert payload["command"] == "tool show"
    assert payload["name"] == "forecast_theta"
    assert payload["result"]["name"] == "forecast_theta"
    assert "input_schema" in payload["result"]
    assert "availability" in payload["result"]
    assert "input_modes" in payload["result"]
    assert "cli_templates" in payload["result"]
    assert "status_contract" in payload["result"]
    assert payload["result"]["writes_artifacts"] is False
    assert "series_file" in payload["result"]["input_modes"]
    assert payload["result"]["cli_templates"][1].startswith(
        "ts-agents tool run forecast_theta --input data.csv --value-col value"
    )


def test_tool_show_json_reports_optional_backend_for_seasonal_naive(capsys):
    code = run(["tool", "show", "forecast_seasonal_naive", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["result"]
    assert result["dependencies"] == []
    assert result["optional_dependencies"] == ["statsforecast"]
    assert result["required_extras"] == []
    assert result["availability"]["available"] is True
    assert result["availability"]["optional_features"][0]["name"] == "statsforecast_backend"


def test_tool_show_json_foundation_extras_match_availability(capsys):
    # torch belongs to both neural and foundation; the FM tool needs only foundation.
    code = run(["tool", "show", "forecast_foundation", "--json"])

    assert code == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["required_extras"] == ["foundation"]
    assert result["availability"]["required_extras"] == ["foundation"]


def test_tool_show_json_includes_prior_diagnostics(capsys):
    code = run(["tool", "show", "forecast_arima", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    prior_diagnostics = payload["result"]["prior_diagnostics"]
    assert len(prior_diagnostics) == 2
    assert prior_diagnostics[0]["kind"] == "workflow"
    assert prior_diagnostics[0]["name"] == "inspect-series"
    assert "Inspect period, lag structure" in prior_diagnostics[0]["reason"]
    assert prior_diagnostics[1]["kind"] == "tool"
    assert prior_diagnostics[1]["name"] == "detect_periodicity"
    assert "Estimate season_length" in prior_diagnostics[1]["reason"]


def test_tool_search_json_includes_prior_diagnostics_in_summary(capsys):
    code = run(["tool", "search", "arima", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    forecast_arima = next(
        tool for tool in payload["result"]["tools"] if tool["name"] == "forecast_arima"
    )
    assert forecast_arima["prior_diagnostics"]
    assert forecast_arima["prior_diagnostics"][0]["name"] == "inspect-series"


def test_tool_show_text_reports_optional_backend_without_install_hint_when_backend_missing(
    monkeypatch,
    capsys,
):
    import ts_agents.tools.registry as registry_mod

    real_available = registry_mod._module_available

    def fake_available(module_name: str) -> bool:
        if module_name == "statsforecast":
            return False
        return real_available(module_name)

    monkeypatch.setattr(registry_mod, "_module_available", fake_available)

    code = run(["tool", "show", "forecast_seasonal_naive"])

    assert code == 0
    output = capsys.readouterr().out
    assert "Install hint:" not in output
    assert "Optional features:" in output
    assert "statsforecast_backend [unavailable] extras=forecasting" in output


def test_tool_show_json_omits_install_hint_for_optional_backend_when_backend_missing(
    monkeypatch,
    capsys,
):
    import ts_agents.tools.registry as registry_mod

    real_available = registry_mod._module_available

    def fake_available(module_name: str) -> bool:
        if module_name == "statsforecast":
            return False
        return real_available(module_name)

    monkeypatch.setattr(registry_mod, "_module_available", fake_available)

    code = run(["tool", "show", "forecast_seasonal_naive", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    availability = payload["result"]["availability"]
    assert availability["available"] is True
    assert availability["install_hint"] is None
    assert availability["optional_features"][0]["name"] == "statsforecast_backend"
    assert availability["optional_features"][0]["available"] is False


def test_tool_show_json_unknown_tool_is_typed(capsys):
    code = run(["tool", "show", "forecast_tetha", "--json"])

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["command"] == "tool show"
    assert payload["name"] == "forecast_tetha"
    assert payload["error"]["code"] == "validation_error"
    assert "Tool 'forecast_tetha' not found." in payload["error"]["message"]
    assert "Did you mean:" in payload["error"]["message"]
    assert "forecast_theta" in payload["error"]["message"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [("anomaly", "find_discords"), ("regime", "segment_changepoint"), ("spectrum", "compute_psd")],
)
def test_tool_search_matches_task_words(capsys, query, expected):
    code = run(["tool", "search", query, "--json"])

    assert code == 0
    names = [tool["name"] for tool in json.loads(capsys.readouterr().out)["result"]["tools"]]
    assert expected in names


def test_tool_search_json_returns_envelope(capsys):
    code = run(["tool", "search", "forecast", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "tool search"
    assert payload["name"] == "forecast"
    assert "tools" in payload["result"]


def test_tool_run_json_returns_envelope(capsys):
    code = run(["tool", "run", "describe_series", "--param", "series=[1,2,3]", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "tool run"
    assert payload["name"] == "describe_series"
    assert payload["result"]["method"] == "descriptive"


def test_tool_run_accepts_input_json(capsys):
    code = run(
        [
            "tool",
            "run",
            "describe_series",
            "--input-json",
            '{"series":[1,2,3,4]}',
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["input"]["input_source"] == "inline_json"
    assert payload["result"]["length"] == 4


def test_tool_run_input_json_rejects_unknown_keys(capsys):
    code = run(
        [
            "tool",
            "run",
            "describe_series",
            "--input-json",
            '{"series":[1,2,3,4],"name":"demo"}',
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "Unknown parameter(s) in JSON input: name" in payload["error"]["message"]


def test_tool_run_accepts_stdin_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"series":[1,2,3,4]}'))

    code = run(
        [
            "tool",
            "run",
            "describe_series",
            "--stdin",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["input"]["input_source"] == "stdin_json"
    assert payload["result"]["length"] == 4


def test_tool_run_rejects_fallback_backend_without_allow_fallback(capsys):
    code = run(
        [
            "tool",
            "run",
            "describe_series",
            "--param",
            "series=[1,2,3,4]",
            "--fallback-backend",
            "local",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "--fallback-backend requires --allow-fallback" in payload["error"]["message"]


def test_workflow_parser_rejects_conflicting_primary_sources():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "workflow",
                "run",
                "inspect-series",
                "--input",
                "data.csv",
                "--run-id",
                "Re200Rm200",
            ]
        )


def test_activity_workflow_parser_accepts_separate_labels_and_windowing_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            "signals.csv",
            "--labels-input",
            "segments.csv",
            "--labels-start-col",
            "start",
            "--labels-end-col",
            "end",
            "--label-col",
            "activity",
            "--value-cols",
            "x,y,z",
            "--labeling",
            "majority",
            "--stride",
            "16",
            "--n-splits",
            "5",
        ]
    )

    assert args.labels_input == "segments.csv"
    assert args.labels_start_col == "start"
    assert args.labels_end_col == "end"
    assert args.labeling == "majority"
    assert args.stride == 16
    assert args.n_splits == 5


def test_tool_run_series_file_derives_sample_rate_and_summarizes_input(capsys, tmp_path):
    import numpy as np

    t = np.arange(400) / 50.0
    values = np.sin(2 * np.pi * 5.0 * t)
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("t,value\n" + "\n".join(f"{a},{b}" for a, b in zip(t, values)))

    code = run(
        ["tool", "run", "detect_periodicity", "--input", str(csv_path), "--time-col", "t", "--json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "ok"
    assert payload["input"]["series"]["sample_rate_source"] == "time_column"
    assert payload["input"]["series"]["time_axis"]["regular"] is True
    # Long arrays are echoed as summaries, not raw values.
    assert payload["input"]["params"]["series"]["shape"] == [400]
    assert "sha256" in payload["input"]["params"]["series"]
    # A 5 Hz sine sampled at 50 Hz has a 0.2 s period in time-column units.
    assert payload["result"]["detected"] is True
    assert payload["result"]["dominant_period"] == pytest.approx(0.2, rel=1e-3)


def test_tool_run_series_file_flags_gaps_for_review(capsys, tmp_path):
    rows = [f"{index},{index % 7}" for index in list(range(0, 50)) + list(range(80, 130))]
    csv_path = tmp_path / "gapped.csv"
    csv_path.write_text("t,value\n" + "\n".join(rows))

    code = run(
        ["tool", "run", "describe_series", "--input", str(csv_path), "--time-col", "t", "--json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["quality_status"] == "review"
    assert payload["requires_review"] is True
    assert payload["input"]["series"]["time_axis"]["gap_count"] == 1
    assert "Irregular sampling" in payload["input"]["warnings"][0]


def test_tool_run_rejects_series_file_for_tools_without_series(capsys, tmp_path):
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("value\n1\n2\n3\n")

    code = run(["tool", "run", "compute_coherence", "--input", str(csv_path), "--json"])

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert "has no `series` parameter" in payload["error"]["message"]


def test_tool_run_json_dependency_error_is_typed(monkeypatch, capsys, tmp_path):
    import importlib
    from ts_agents.tools.registry import ToolRegistry

    cli_main = importlib.import_module("ts_agents.cli.main")
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\n2\n3\n4\n")

    def fake_forecast_theta(series, horizon=10, level=None, season_length=None):
        raise ImportError(
            'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
        )

    monkeypatch.setattr(ToolRegistry.get("forecast_theta"), "core_function", fake_forecast_theta)

    code = cli_main.run(
        [
            "tool",
            "run",
            "forecast_theta",
            "--input",
            str(csv_path),
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["command"] == "tool run"
    assert payload["name"] == "forecast_theta"
    assert payload["error"]["code"] == "dependency_error"


def test_tool_run_parse_failure_with_json_returns_typed_envelope(capsys):
    code = run(["tool", "run", "describe_series", "--badflag", "--json"])

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["schema_version"] == "1.0"
    assert payload["command"] == "tool run"
    assert payload["name"] == "describe_series"
    assert payload["error"]["code"] == "usage_error"
    assert "unrecognized arguments: --badflag" in payload["error"]["message"]
    assert payload["input"]["argv"] == [
        "tool",
        "run",
        "describe_series",
        "--badflag",
        "--json",
    ]


def test_time_axis_preserves_large_integer_steps_and_gaps():
    from ts_agents.cli.input_parsing import describe_time_axis

    origin = 10**18
    regular = describe_time_axis([origin, origin + 1, origin + 2])
    assert regular["median_step"] == 1
    assert regular["sample_rate"] == 1
    gapped = describe_time_axis([origin + x for x in (0, 1, 2, 12)])
    assert gapped["regular"] is False and gapped["gap_count"] == 1


def test_parameter_echo_bounds_ragged_and_object_arrays():
    from ts_agents.cli.main import _describe_params

    ragged = [list(range(n)) for n in range(100)]
    description = _describe_params(ragged)
    assert len(json.dumps(description)) < 2500
    assert description["sha256"] != _describe_params(ragged + [[1]])["sha256"]
    import numpy as np
    objects = np.empty(2, dtype=object)
    objects[:] = [list(range(1000)), list(range(500))]
    assert len(json.dumps(_describe_params(objects))) < 2500
