import io
import json
import sys
from types import SimpleNamespace

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
        "forecast_thta_with_data",
        [
            "forecast_theta_with_data",
            "forecast_ets_with_data",
            "stl_decompose_with_data",
        ],
    )
    assert "forecast_theta_with_data" in suggestions


def test_build_run_example_command_uses_run_var_shorthands():
    command = _build_run_example_command(
        "forecast_theta_with_data",
        required=["variable_name", "unique_id"],
        param_types={"variable_name": "str", "unique_id": "str"},
    )
    assert "uv run ts-agents tool run forecast_theta_with_data" in command
    assert "--run Re200Rm200" in command
    assert "--var bx001_real" in command
    assert "--param" not in command


def test_build_run_example_command_quotes_dict_params_shell_safe():
    command = _build_run_example_command(
        "compare_series_stats_with_data",
        required=["series_dict"],
        param_types={"series_dict": "dict"},
    )
    assert "--param 'series_dict={\"key\":\"value\"}'" in command


def test_missing_required_error_includes_actionable_example():
    with pytest.raises(ValueError) as exc_info:
        _raise_missing_required_error(
            tool_name="compare_forecasts_with_data",
            param_types={
                "variable_name": "str",
                "unique_id": "str",
                "models": "list",
            },
            required=["variable_name", "unique_id"],
            provided={"variable_name": "bx001_real"},
        )

    message = str(exc_info.value)
    assert "Missing required parameters for 'compare_forecasts_with_data'" in message
    assert "Required parameters:" in message
    assert "Example:" in message
    assert "uv run ts-agents tool run compare_forecasts_with_data" in message


def test_run_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "tool run forecast_theta_with_data" in output


def test_tool_run_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tool", "run", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "tool run forecast_theta_with_data" in output


def test_agent_run_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "run", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "Compare forecasting methods for bx001_real" in output


def test_demo_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "demo forecasting --no-llm" in output


def test_workflow_help_includes_examples(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["workflow", "run", "--help"])
    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "workflow run inspect-series" in output
    assert "workflow run forecast-series" in output
    assert "workflow run activity-recognition" in output


def test_sandbox_help_includes_commands(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sandbox", "--help"])
    output = capsys.readouterr().out
    assert "list" in output
    assert "doctor" in output


def test_unknown_tool_error_includes_suggestions(capsys):
    code = run(["run", "forecast_thta_with_data", "--run", "Re200Rm200", "--var", "bx001_real"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Did you mean:" in err
    assert "forecast_theta_with_data" in err


def test_missing_required_error_is_actionable(capsys):
    code = run(["run", "forecast_theta_with_data", "--var", "bx001_real"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Missing required parameters for 'forecast_theta_with_data'" in err
    assert "Example:" in err


def test_parser_accepts_extract_images_with_save():
    parser = build_parser()
    args = parser.parse_args(
        [
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


def test_run_extracts_images_in_json_without_breaking_json(monkeypatch, tmp_path):
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


def test_cli_compare_forecasts_accepts_methods_alias(monkeypatch, capsys):
    import importlib
    import numpy as np
    import ts_agents.core.forecasting as forecasting
    import ts_agents.tools.agent_tools as agent_tools

    cli_main = importlib.import_module("ts_agents.cli.main")
    observed = {}

    monkeypatch.setattr(
        agent_tools,
        "_get_series_data",
        lambda variable_name, unique_id: np.array([1.0, 2.0, 3.0, 4.0]),
    )

    def fake_compare_forecasts(series, horizon=10, test_size=None, models=None):
        observed["horizon"] = horizon
        observed["models"] = models
        return {"best_model": "theta", "metrics": {"theta": {"rmse": 0.1}}}

    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    code = cli_main.run(
        [
            "run",
            "compare_forecasts_with_data",
            "--run",
            "Re200Rm200",
            "--var",
            "bx001_real",
            "--param",
            "methods=arima,theta",
            "--param",
            "horizon=12",
        ]
    )

    assert code == 0
    assert observed["horizon"] == 12
    assert observed["models"] == ["arima", "theta"]
    output = capsys.readouterr().out
    assert "Error in Compare Forecasts" not in output


def test_cli_compare_forecasts_surfaces_tool_failure(monkeypatch, capsys):
    import importlib
    import numpy as np
    import ts_agents.core.forecasting as forecasting
    import ts_agents.tools.agent_tools as agent_tools

    cli_main = importlib.import_module("ts_agents.cli.main")

    monkeypatch.setattr(
        agent_tools,
        "_get_series_data",
        lambda variable_name, unique_id: np.array([0.1, 0.12, 0.15, 0.14, 0.16]),
    )

    def fake_compare_forecasts(series, horizon=10, test_size=None, models=None):
        raise ValueError("number sections must be larger than 0")

    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    code = cli_main.run(
        [
            "run",
            "compare_forecasts_with_data",
            "--run",
            "Re200Rm200",
            "--var",
            "bx001_real",
            "--param",
            "methods=arima,theta",
            "--param",
            "horizon=12",
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "number sections must be larger than 0" in err


def test_tool_show_json_returns_envelope(capsys):
    code = run(["tool", "show", "forecast_theta_with_data", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"
    assert payload["ok"] is True
    assert payload["command"] == "tool show"
    assert payload["name"] == "forecast_theta_with_data"
    assert payload["result"]["name"] == "forecast_theta_with_data"
    assert "input_schema" in payload["result"]


def test_tool_show_json_unknown_tool_is_typed(capsys):
    code = run(["tool", "show", "forecast_tetha_with_data", "--json"])

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["command"] == "tool show"
    assert payload["name"] == "forecast_tetha_with_data"
    assert payload["error"]["code"] == "validation_error"
    assert "Tool 'forecast_tetha_with_data' not found." in payload["error"]["message"]
    assert "Did you mean:" in payload["error"]["message"]
    assert "forecast_theta_with_data" in payload["error"]["message"]


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


def test_run_alias_emits_deprecation_warning(monkeypatch, capsys):
    import importlib

    cli_main = importlib.import_module("ts_agents.cli.main")

    monkeypatch.setattr(cli_main, "_handle_run_command", lambda args: ({"ok": True}, "compat result"))

    code = cli_main.run(["run", "describe_series", "--param", "series=[1,2,3]"])

    assert code == 0
    captured = capsys.readouterr()
    assert "compat result" in captured.out
    assert "`ts-agents run` is deprecated" in captured.err


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


def test_tool_run_json_includes_artifact_refs_for_payload_wrappers(
    monkeypatch,
    capsys,
    tmp_path,
):
    import importlib
    import numpy as np
    import ts_agents.core.spectral as spectral
    import ts_agents.tools.agent_tools as agent_tools

    cli_main = importlib.import_module("ts_agents.cli.main")

    class _DummyAxis:
        def plot(self, *args, **kwargs):
            return None

        def loglog(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

    class _DummyPlotLib:
        def subplots(self, *args, **kwargs):
            return object(), _DummyAxis()

        def tight_layout(self):
            return None

        def savefig(self, buf, format="png"):
            buf.write(b"png")

        def close(self, fig):
            return None

    monkeypatch.setenv("TS_AGENTS_TOOL_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(
        agent_tools,
        "_get_series_data",
        lambda variable_name, unique_id: np.array([1.0, 0.5, 0.25, 0.125]),
    )
    monkeypatch.setattr(agent_tools, "_get_plt", lambda: _DummyPlotLib())
    monkeypatch.setattr(
        spectral,
        "compute_psd",
        lambda series, sample_rate=1.0, method="welch", nperseg=None: SimpleNamespace(
            frequencies=np.array([0.25, 0.5]),
            psd=np.array([1.0, 0.5]),
            spectral_slope=-1.75,
        ),
    )

    code = cli_main.run(
        [
            "tool",
            "run",
            "compute_psd_with_data",
            "--run",
            "Re200Rm200",
            "--var",
            "bx001_real",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "compute_psd_with_data"
    assert payload["result"]["artifacts"][0]["path"].startswith(str(tmp_path))
    assert payload["result"]["artifacts"][0]["mime_type"] == "image/png"


def test_tool_run_json_dependency_error_is_typed(monkeypatch, capsys):
    import importlib
    import numpy as np
    import ts_agents.core.forecasting as forecasting
    import ts_agents.tools.agent_tools as agent_tools

    cli_main = importlib.import_module("ts_agents.cli.main")

    monkeypatch.setattr(
        agent_tools,
        "_get_series_data",
        lambda variable_name, unique_id: np.array([1.0, 2.0, 3.0, 4.0]),
    )

    def fake_forecast_theta(series, horizon=10, level=None, season_length=None):
        raise ImportError(
            'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
        )

    monkeypatch.setattr(forecasting, "forecast_theta", fake_forecast_theta)

    code = cli_main.run(
        [
            "tool",
            "run",
            "forecast_theta_with_data",
            "--run",
            "Re200Rm200",
            "--var",
            "bx001_real",
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["command"] == "tool run"
    assert payload["name"] == "forecast_theta_with_data"
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
