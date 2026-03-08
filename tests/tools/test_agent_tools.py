import numpy as np
import pytest
from types import SimpleNamespace


def _patch_plotting(monkeypatch, agent_tools):
    class _DummyAxis:
        def plot(self, *args, **kwargs):
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
    monkeypatch.setattr(agent_tools, "_create_plot_response", lambda buf: "")


def test_compare_forecasts_with_data_forwards_models(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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
        observed["series"] = series
        observed["horizon"] = horizon
        observed["models"] = models
        observed["season_length"] = season_length
        return {"best_model": "theta", "metrics": {}}

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(forecasting, "compare_forecasts", fake_compare_forecasts)

    output = agent_tools.compare_forecasts_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        horizon=12,
        models=["theta"],
    )

    assert "Error in Compare Forecasts" not in output
    assert observed["horizon"] == 12
    assert observed["models"] == ["theta"]
    assert observed["season_length"] is None


def test_compare_forecasts_with_data_accepts_methods_alias(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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

    assert "Error in Compare Forecasts" not in output
    assert observed["models"] == ["arima", "ets"]


def test_compare_forecasts_with_data_models_take_precedence(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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

    assert "Error in Compare Forecasts" not in output
    assert observed["models"] == ["theta"]


def test_compare_forecasts_with_data_propagates_errors(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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

    assert "Error in Compare Forecasts" not in output
    assert observed["season_length"] == 12


def test_forecast_seasonal_naive_with_data_forwards_season_length(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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

    assert "Error in Seasonal Naive" not in output
    assert observed["horizon"] == 2
    assert observed["season_length"] == 12


def test_forecast_ensemble_with_data_uses_get_ensemble(monkeypatch):
    from src.tools import agent_tools
    import src.core.forecasting as forecasting

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

    assert "Error in Ensemble" not in output
    assert observed["horizon"] == 2
    assert observed["models"] == ["seasonal_naive", "theta"]
    assert observed["season_length"] == 12


def test_segment_changepoint_with_data_maps_n_changepoints_alias(monkeypatch):
    from src.tools import agent_tools
    import src.core.patterns as patterns

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

    assert "Error in Changepoint" not in output
    assert observed["n_segments"] == 3
    assert observed["algorithm"] == "binseg"
    assert observed["cost_model"] == "l2"
    assert observed["penalty"] == 0.7
    assert observed["min_size"] == 3


def test_segment_changepoint_with_data_prefers_n_segments(monkeypatch):
    from src.tools import agent_tools
    import src.core.patterns as patterns

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

    assert "Error in Changepoint" not in output
    assert observed["n_segments"] == 5


def test_segment_fluss_with_data_formats_segment_result(monkeypatch):
    from src.tools import agent_tools
    import src.core.patterns as patterns

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

    assert "Error in FLUSS segmentation" not in output
    assert "Changepoints detected at indices: [2, 4]" in output
    assert "Segments: 3" in output
    assert "Segment 1: start=0, end=2, length=2, mean=0.0500, std=0.0500" in output


def test_compute_psd_with_data_aliases_spectrum_name(monkeypatch):
    from src.tools import agent_tools
    import src.core.spectral as spectral

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

    assert "Error in PSD" not in output
    assert "Power Spectral Density for bx001_real:" in output
    assert "Dominant Frequency: 0.2500" in output
    assert "Spectral Slope: -1.7500" in output
    assert alias_output == output


def test_compute_coherence_with_data_accepts_sampling_aliases(monkeypatch):
    from src.tools import agent_tools
    import src.core.spectral as spectral

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

    assert "Error in Coherence" not in output
    assert observed["sample_rate"] == 2.5

    output = agent_tools.compute_coherence_with_data(
        variable1="bx001_real",
        unique_id1="Re200Rm200",
        variable2="by001_real",
        unique_id2="Re200Rm200",
        sampling_rate=3.5,
    )

    assert "Error in Coherence" not in output
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

    assert "Error in Coherence" not in output
    assert observed["sample_rate"] == 4.5


def test_hurst_exponent_with_data_forwards_max_window(monkeypatch):
    from src.tools import agent_tools
    import src.core.complexity as complexity

    observed = {}

    def fake_get_series_data(variable_name, unique_id):
        return np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.4, 1.2])

    def fake_hurst_exponent(series, min_window=10, max_window=None):
        observed["min_window"] = min_window
        observed["max_window"] = max_window
        return 0.61

    monkeypatch.setattr(agent_tools, "_get_series_data", fake_get_series_data)
    monkeypatch.setattr(complexity, "hurst_exponent", fake_hurst_exponent)

    output = agent_tools.hurst_exponent_with_data(
        variable_name="bx001_real",
        unique_id="Re200Rm200",
        min_window=8,
        max_window=64,
    )

    assert "Error in Hurst" not in output
    assert observed["min_window"] == 8
    assert observed["max_window"] == 64
