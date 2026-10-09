import numpy as np
import pytest
from types import SimpleNamespace

from ts_agents.contracts import ToolPayload


def _patch_plotting(monkeypatch, agent_tools):
    class _DummyAxis:
        def plot(self, *args, **kwargs):
            return None

        def imshow(self, *args, **kwargs):
            return None

        def bar(self, *args, **kwargs):
            return None

        def loglog(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
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

    monkeypatch.setattr(agent_tools, "_get_plt", lambda: _DummyPlotLib())


def test_compare_forecasts_with_data_forwards_models(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_compare_forecasts(
        series,
        horizon=10,
        test_size=None,
        models=None,
        **kwargs,
    ):
        observed["series"] = series
        observed["horizon"] = horizon
        observed["models"] = models
        observed["extra_kwargs"] = kwargs
        return {"best_model": "theta", "metrics": {}}

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    output = agent_tools.compare_forecasts_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=12,
        models=["theta"],
    )

    assert isinstance(output, ToolPayload)
    assert observed["horizon"] == 12
    assert observed["models"] == ["theta"]
    assert observed["extra_kwargs"] == {}


def test_compare_forecasts_with_data_accepts_methods_alias(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_compare_forecasts(
        series,
        horizon=10,
        test_size=None,
        models=None,
        season_length=None,
    ):
        observed["models"] = models
        return {"best_model": "arima", "metrics": {}}

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    output = agent_tools.compare_forecasts_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        methods=["arima", "ets"],
    )

    assert isinstance(output, ToolPayload)
    assert observed["models"] == ["arima", "ets"]


def test_compare_forecasts_with_data_models_take_precedence(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_compare_forecasts(
        series,
        horizon=10,
        test_size=None,
        models=None,
        season_length=None,
    ):
        observed["models"] = models
        return {"best_model": "theta", "metrics": {}}

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    output = agent_tools.compare_forecasts_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        models=["theta"],
        methods=["arima"],
    )

    assert isinstance(output, ToolPayload)
    assert observed["models"] == ["theta"]


def test_compare_forecasts_with_data_propagates_errors(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_compare_forecasts(
        series,
        horizon=10,
        test_size=None,
        models=None,
        season_length=None,
    ):
        raise ValueError("number sections must be larger than 0")

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    with pytest.raises(ValueError, match="number sections must be larger than 0"):
        agent_tools.compare_forecasts_with_data(
            variable_name="bx001_real",
            unique_id="Re200Rm200",
            horizon=12,
            models=["arima", "theta"],
        )


def test_compare_forecasts_with_data_forwards_season_length(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_compare_forecasts(
        series,
        horizon=10,
        test_size=None,
        models=None,
        season_length=None,
    ):
        observed["season_length"] = season_length
        return {"best_model": "seasonal_naive", "metrics": {}}

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    output = agent_tools.compare_forecasts_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        models=["seasonal_naive"],
        season_length=12,
    )

    assert isinstance(output, ToolPayload)
    assert observed["season_length"] == 12


def test_forecast_seasonal_naive_with_data_forwards_season_length(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    def fake_forecast_seasonal_naive(series, horizon=10, level=None, season_length=None):
        observed["horizon"] = horizon
        observed["season_length"] = season_length
        return SimpleNamespace(forecast=np.array([3.0, 4.0]))

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "forecast_seasonal_naive", fake_forecast_seasonal_naive)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.forecast_seasonal_naive_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=2,
        season_length=12,
    )

    assert isinstance(output, ToolPayload)
    assert output.kind == "forecast"
    assert observed["horizon"] == 2
    assert observed["season_length"] == 12


def test_forecast_ensemble_with_data_uses_get_ensemble(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    class FakeEnsembleResult:
        def get_ensemble(self):
            return np.array([4.5, 5.5])

    def fake_forecast_ensemble(series, horizon=10, models=None, season_length=None):
        observed["horizon"] = horizon
        observed["models"] = models
        observed["season_length"] = season_length
        return FakeEnsembleResult()

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "forecast_ensemble", fake_forecast_ensemble)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.forecast_ensemble_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=2,
        models=["seasonal_naive", "theta"],
        season_length=12,
    )

    assert isinstance(output, ToolPayload)
    assert output.kind == "forecast_comparison"
    assert output.summary == (
        "Ensemble forecast completed for bx001_real "
        "(run Re200Rm200) with 2 model forecasts."
    )
    assert observed["horizon"] == 2
    assert observed["models"] == ["seasonal_naive", "theta"]
    assert observed["season_length"] == 12


def test_forecast_ensemble_with_data_omits_zero_count_when_models_absent(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting as forecasting

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 2.0, 3.0, 4.0])

    class FakeEnsembleResult:
        def get_ensemble(self):
            return np.array([4.5, 5.5])

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(
        forecasting,
        "forecast_ensemble",
        lambda series, horizon=10, models=None, season_length=None: FakeEnsembleResult(),
    )
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.forecast_ensemble_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=2,
    )

    assert isinstance(output, ToolPayload)
    assert output.summary == "Ensemble forecast completed for bx001_real (run Re200Rm200)."


def test_segment_changepoint_with_data_maps_n_changepoints_alias(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.patterns as patterns

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([0.0, 0.0, 3.0, 3.0, 1.0, 1.0])

    def fake_segment_changepoint(
        series,
        n_segments=None,
        algorithm="pelt",
        cost_model="rbf",
        penalty=None,
        min_size=5,
    ):
        observed["n_segments"] = n_segments
        observed["algorithm"] = algorithm
        observed["cost_model"] = cost_model
        observed["penalty"] = penalty
        observed["min_size"] = min_size
        return SimpleNamespace(changepoints=[2, 4])

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(patterns, "segment_changepoint", fake_segment_changepoint)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.segment_changepoint_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        n_changepoints=2,
        algorithm="binseg",
        cost_model="l2",
        penalty=0.7,
        min_size=3,
    )

    assert isinstance(output, ToolPayload)
    assert output.summary == (
        "Changepoint detection completed for bx001_real "
        "(run Re200Rm200). Changepoints: [2, 4]."
    )
    assert len(output.artifacts) == 1
    assert observed["n_segments"] == 3
    assert observed["algorithm"] == "binseg"
    assert observed["cost_model"] == "l2"
    assert observed["penalty"] == 0.7
    assert observed["min_size"] == 3


def test_segment_changepoint_with_data_prefers_n_segments(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.patterns as patterns

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def fake_segment_changepoint(
        series,
        n_segments=None,
        algorithm="pelt",
        cost_model="rbf",
        penalty=None,
        min_size=5,
    ):
        observed["n_segments"] = n_segments
        return SimpleNamespace(changepoints=[3])

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(patterns, "segment_changepoint", fake_segment_changepoint)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.segment_changepoint_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        n_segments=5,
        n_changepoints=1,
    )

    assert isinstance(output, ToolPayload)
    assert observed["n_segments"] == 5


def test_segment_fluss_with_data_formats_segment_result(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.patterns as patterns

    def fake_get_series_data(variable_name, unique_id):
        return np.array([0.0, 0.1, 1.0, 1.1, -0.2, -0.1])

    def fake_segment_fluss(series, m=50, n_segments=3, n_regimes=None):
        return SimpleNamespace(
            changepoints=[2, 4],
            n_segments=3,
            segment_stats=[
                {"start": 0, "end": 2, "length": 2, "mean": 0.05, "std": 0.05},
                {"start": 2, "end": 4, "length": 2, "mean": 1.05, "std": 0.05},
                {"start": 4, "end": 6, "length": 2, "mean": -0.15, "std": 0.05},
            ],
        )

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(patterns, "segment_fluss", fake_segment_fluss)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.segment_fluss_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        window_size=12,
        n_segments=3,
    )

    assert isinstance(output, ToolPayload)
    assert output.summary == (
        "FLUSS segmentation completed for bx001_real "
        "(run Re200Rm200). Changepoints: [2, 4]; segments: 3."
    )
    assert output.data.changepoints == [2, 4]
    assert output.data.n_segments == 3
    assert len(output.artifacts) == 1


def test_compute_psd_with_data_aliases_spectrum_name(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.spectral as spectral

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 0.5, 0.25, 0.125])

    def fake_compute_psd(series, sample_rate=1.0, method="welch", nperseg=None):
        return SimpleNamespace(
            frequencies=np.array([0.25, 0.5]),
            psd=np.array([1.0, 0.5]),
            spectral_slope=-1.75,
        )

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(spectral, "compute_psd", fake_compute_psd)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.compute_psd_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        sampling_rate=2.0,
    )
    alias_output = agent_tools.compute_spectrum_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        sampling_rate=2.0,
    )

    assert isinstance(output, ToolPayload)
    assert output.summary == (
        "Power spectral density computed for bx001_real "
        "(run Re200Rm200). Dominant frequency: 0.2500; spectral slope: -1.7500."
    )
    assert output.kind == "spectral"
    assert len(output.artifacts) == 1
    assert alias_output.summary == output.summary


def test_compute_coherence_with_data_accepts_sampling_aliases(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.spectral as spectral

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def fake_compute_coherence(series1, series2, sample_rate=1.0, nperseg=None):
        observed["sample_rate"] = sample_rate
        return SimpleNamespace(
            frequencies=np.array([0.0, 0.5]),
            coherence=np.array([0.2, 0.8]),
        )

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(spectral, "compute_coherence", fake_compute_coherence)
    _patch_plotting(monkeypatch, agent_tools)

    output = agent_tools.compute_coherence_with_data(
        variable1="bx001_real",
        unique_id1="Re200Rm200",
        variable2="by001_real",
        unique_id2="Re200Rm200",
        fs=2.5,
    )

    assert isinstance(output, ToolPayload)
    assert observed["sample_rate"] == 2.5

    output = agent_tools.compute_coherence_with_data(
        variable1="bx001_real",
        unique_id1="Re200Rm200",
        variable2="by001_real",
        unique_id2="Re200Rm200",
        sampling_rate=3.5,
    )

    assert isinstance(output, ToolPayload)
    assert observed["sample_rate"] == 3.5

    output = agent_tools.compute_coherence_with_data(
        variable1="bx001_real",
        unique_id1="Re200Rm200",
        variable2="by001_real",
        unique_id2="Re200Rm200",
        sample_rate=4.5,
        sampling_rate=3.5,
        fs=2.5,
    )

    assert isinstance(output, ToolPayload)
    assert output.kind == "spectral"
    assert len(output.artifacts) == 1
    assert observed["sample_rate"] == 4.5


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


def test_forecast_foundation_with_data_returns_forecast_with_checkpoint(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting.foundation as foundation

    observed = {}
    monkeypatch.setattr(
        agent_tools, "_get_series_data", lambda variable_name, unique_id: np.arange(40.0)
    )
    monkeypatch.setattr(foundation, "forecast_arrays", _fake_forecast_arrays(observed))

    output = agent_tools.forecast_foundation_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=6,
        context_length=32,
    )

    assert isinstance(output, ToolPayload)
    assert output.kind == "forecast"
    assert output.summary.startswith("Chronos-2 small (Darts, zero-shot) forecast completed")
    assert observed == {
        "model": "chronos2_small",
        "horizon": 6,
        "context_length": 32,
        "accelerator": "cpu",
        "n_arrays": 1,
    }
    assert list(output.data["forecast"]) == [39.0] * 6
    spec = output.data["foundation_model"]
    assert spec["method"] == "chronos2_small"
    assert spec["hub_model_name"] == "autogluon/chronos-2-small"
    assert spec["hub_model_revision"] == "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"
    assert spec["input_chunk_length"] == [1, 32]
    assert spec["weights_saved"] is False
    assert output.provenance["series_ref"]["run_id"] == "Re200Rm200"


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


def test_forecast_foundation_tools_reject_non_foundation_models(monkeypatch):
    from ts_agents.tools import agent_tools

    monkeypatch.setattr(
        agent_tools, "_get_series_data", lambda variable_name, unique_id: np.arange(40.0)
    )

    with pytest.raises(ValueError, match="chronos2_small"):
        agent_tools.forecast_foundation([1.0, 2.0, 3.0], model="arima")
    with pytest.raises(ValueError, match="Unknown foundation model"):
        agent_tools.forecast_foundation_with_data("bx001_real", "Re200Rm200", model="lightgbm")


def test_forecast_foundation_with_data_missing_extra_names_install(monkeypatch):
    from ts_agents.tools import agent_tools
    import ts_agents.core.forecasting.catalog as catalog

    monkeypatch.setattr(
        agent_tools, "_get_series_data", lambda variable_name, unique_id: np.arange(40.0)
    )
    monkeypatch.setattr(catalog, "missing_modules", lambda name: ["darts"])

    with pytest.raises(ImportError, match=r"ts-agents\[foundation\]"):
        agent_tools.forecast_foundation_with_data("bx001_real", "Re200Rm200")


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
