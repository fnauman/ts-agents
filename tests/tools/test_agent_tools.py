import numpy as np
import pytest

from ts_agents.contracts import ToolPayload


def _fake_forecast_arrays(observed):
    def fake(arrays, *, model, horizon, context_length=None, accelerator="cpu", seed=0):
        observed.update(
            model=model,
            horizon=horizon,
            context_length=context_length,
            accelerator=accelerator,
            n_arrays=len(arrays),
        )
        return [np.full(horizon, float(np.asarray(arrays[0])[-1]))]

    return fake


def test_forecast_foundation_series_tool_forwards_model_and_accelerator(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting.foundation as foundation

    observed = {}
    monkeypatch.setattr(foundation, "forecast_arrays", _fake_forecast_arrays(observed))

    result = agent_tools.forecast_foundation(
        [1.0, 2.0, 3.0, 4.0], horizon=3, model="timesfm2p5", accelerator="gpu"
    )

    assert result.method == "timesfm2p5"
    assert list(result.forecast) == [4.0, 4.0, 4.0]
    assert observed["model"] == "timesfm2p5"
    assert observed["accelerator"] == "gpu"


def test_forecast_foundation_tool_rejects_non_foundation_models():
    from ts_agents.tools import agent_tools

    with pytest.raises(ValueError, match="chronos2_small"):
        agent_tools.forecast_foundation([1.0, 2.0, 3.0], model="arima")
    with pytest.raises(ValueError, match="Unknown foundation model"):
        agent_tools.forecast_foundation([1.0, 2.0, 3.0], model="lightgbm")


def test_forecast_foundation_missing_extra_names_install(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting.catalog as catalog

    monkeypatch.setattr(catalog, "missing_modules", lambda name: ["darts"])

    with pytest.raises(ImportError, match=r"ts-agents\[foundation\]"):
        agent_tools.forecast_foundation(np.arange(40.0))


def _write_panel_csv(path, n_series=2, length=24):
    import pandas as pd

    frames = []
    for index in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "series": f"s{index}",
                    "date": pd.date_range("2024-01-01", periods=length, freq="D"),
                    "value": np.sin(np.arange(length) * 2 * np.pi / 4) + index + 5.0,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def test_forecast_panel_from_csv_forwards_options_and_returns_payload(monkeypatch, tmp_path):
    from ts_agents.tools import agent_tools
    import ts_agents.workflows.panel as panel_workflow

    observed = {}
    sentinel = ToolPayload(kind="workflow", summary="ok")

    def fake_run(panel_input, **kwargs):
        observed["n_records"] = len(panel_input.records)
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(panel_workflow, "run_forecast_panel_workflow", fake_run)
    monkeypatch.setenv("TS_AGENTS_TOOL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    csv_path = _write_panel_csv(tmp_path / "panel.csv")

    output = agent_tools.forecast_panel_from_csv(
        input_path=str(csv_path),
        freq="D",
        horizon=3,
        methods=" seasonal_naive, chronos2_small ",
        season_length=4,
        n_windows=2,
        id_col="series",
        time_col="date",
        value_col="value",
    )

    assert output is sentinel
    assert observed["n_records"] == 48
    kwargs = observed["kwargs"]
    assert kwargs["methods"] == ["seasonal_naive", "chronos2_small"]
    assert kwargs["freq"] == "D"
    assert (kwargs["horizon"], kwargs["season_length"], kwargs["n_windows"]) == (3, 4, 2)
    assert kwargs["input_size"] is None
    # Unset options stay out so the workflow keeps its own defaults.
    assert "context_length" not in kwargs
    assert kwargs["output_dir"].startswith(str(tmp_path / "artifacts" / "forecast_panel_"))

    agent_tools.forecast_panel_from_csv(
        input_path=str(csv_path),
        freq="D",
        methods=["chronos2_small"],
        context_length=64,
        id_col="series",
        time_col="date",
        value_col="value",
        output_dir=str(tmp_path / "explicit"),
    )
    assert observed["kwargs"]["context_length"] == 64
    assert observed["kwargs"]["output_dir"] == str(tmp_path / "explicit")

    with pytest.raises(ValueError, match="at least one"):
        agent_tools.forecast_panel_from_csv(input_path=str(csv_path), freq="D", methods=" , ")


def test_forecast_panel_from_csv_runs_seasonal_naive_workflow(tmp_path):
    from pathlib import Path
    from ts_agents.tools import agent_tools

    csv_path = _write_panel_csv(tmp_path / "panel.csv")

    output = agent_tools.forecast_panel_from_csv(
        input_path=str(csv_path),
        freq="D",
        horizon=3,
        season_length=4,
        n_windows=2,
        id_col="series",
        time_col="date",
        value_col="value",
        output_dir=str(tmp_path / "out"),
    )

    assert isinstance(output, ToolPayload)
    assert output.kind == "workflow"
    assert output.data["best_method"] == "seasonal_naive"
    names = {Path(artifact.path).name for artifact in output.artifacts}
    assert {"metrics.json", "forecast.csv", "backtest_predictions.csv", "report.md"} <= names
    for artifact in output.artifacts:
        assert Path(artifact.path).is_file()
        assert Path(artifact.path).resolve().is_relative_to((tmp_path / "out").resolve())
