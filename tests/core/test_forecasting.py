"""Tests for the forecasting module."""

import importlib.util

import numpy as np
import pytest
import pandas as pd


HAS_STATSFORECAST = importlib.util.find_spec("statsforecast") is not None
requires_statsforecast = pytest.mark.skipif(
    not HAS_STATSFORECAST,
    reason="statsforecast not installed",
)


class TestForecasting:
    """Tests for forecasting functions."""

    @requires_statsforecast
    def test_forecast_arima(self):
        """Test ARIMA forecasting."""
        from ts_agents.core.forecasting import forecast_arima

        # Create a simple series
        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_arima(x, horizon=20)

        assert result.method == "auto_arima"
        assert len(result.forecast) == 20
        assert result.horizon == 20

    @requires_statsforecast
    def test_forecast_ets(self):
        """Test ETS forecasting."""
        from ts_agents.core.forecasting import forecast_ets

        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_ets(x, horizon=20)

        assert result.method == "auto_ets"
        assert len(result.forecast) == 20

    @requires_statsforecast
    def test_forecast_theta(self):
        """Test Theta forecasting."""
        from ts_agents.core.forecasting import forecast_theta

        x = np.linspace(0, 10, 200) + 0.5 * np.random.randn(200)

        result = forecast_theta(x, horizon=20)

        assert result.method == "auto_theta"
        assert len(result.forecast) == 20

    def test_forecast_seasonal_naive(self):
        """Test seasonal naive repeats the last observed season."""
        from ts_agents.core.forecasting import forecast_seasonal_naive

        x = np.tile(np.arange(1, 13, dtype=float), 2)

        result = forecast_seasonal_naive(x, horizon=18, season_length=12)
        expected = np.concatenate([
            np.arange(1, 13, dtype=float),
            np.arange(1, 7, dtype=float),
        ])

        assert result.method == "seasonal_naive"
        np.testing.assert_allclose(result.forecast, expected)

    def test_forecast_seasonal_naive_falls_back_without_statsforecast(self, monkeypatch):
        """Test seasonal naive still works without the forecasting extra."""
        import ts_agents.core.forecasting.statistical as statistical

        monkeypatch.setattr(
            statistical,
            "_get_statsforecast_components",
            lambda: (_ for _ in ()).throw(
                ImportError("Statistical forecasting requires optional dependencies.")
            ),
        )

        x = np.tile(np.arange(1, 5, dtype=float), 2)
        result = statistical.forecast_seasonal_naive(x, horizon=6, season_length=4)

        assert result.method == "seasonal_naive"
        np.testing.assert_allclose(result.forecast, np.array([1, 2, 3, 4, 1, 2], dtype=float))

    @requires_statsforecast
    def test_forecast_ensemble(self):
        """Test ensemble forecasting."""
        from ts_agents.core.forecasting import forecast_ensemble

        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_ensemble(x, horizon=20, models=['arima', 'ets'])

        assert 'arima' in result.forecasts or 'ets' in result.forecasts
        assert result.horizon == 20

        # Test ensemble average
        ensemble = result.get_ensemble()
        assert len(ensemble) == 20

    def test_forecast_ensemble_supports_seasonal_naive(self):
        """Test ensemble forecasting can include seasonal naive."""
        from ts_agents.core.forecasting import forecast_ensemble

        x = np.tile(np.arange(1, 13, dtype=float), 2)

        result = forecast_ensemble(
            x,
            horizon=12,
            models=["seasonal_naive"],
            season_length=12,
        )

        assert "seasonal_naive" in result.forecasts
        np.testing.assert_allclose(
            result.forecasts["seasonal_naive"],
            np.arange(1, 13, dtype=float),
        )

    def test_forecast_ensemble_supports_seasonal_naive_without_statsforecast(
        self,
        monkeypatch,
    ):
        """Test ensemble fallback works when only seasonal naive is requested."""
        import ts_agents.core.forecasting.statistical as statistical

        monkeypatch.setattr(
            statistical,
            "_get_statsforecast_components",
            lambda: (_ for _ in ()).throw(
                ImportError("Statistical forecasting requires optional dependencies.")
            ),
        )

        x = np.tile(np.arange(1, 5, dtype=float), 2)
        result = statistical.forecast_ensemble(
            x,
            horizon=4,
            models=["seasonal_naive"],
            season_length=4,
        )

        assert list(result.forecasts.keys()) == ["seasonal_naive"]
        np.testing.assert_allclose(
            result.forecasts["seasonal_naive"],
            np.array([1, 2, 3, 4], dtype=float),
        )

    @requires_statsforecast
    def test_forecast_arima_with_season_length(self):
        """Test ARIMA forecasting with season_length parameter."""
        from ts_agents.core.forecasting import forecast_arima

        # Create seasonal series
        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(200)

        result = forecast_arima(x, horizon=12, season_length=12)

        assert result.method == "auto_arima"
        assert len(result.forecast) == 12

    @requires_statsforecast
    def test_forecast_ensemble_with_season_length(self):
        """Test ensemble forecasting with season_length parameter."""
        from ts_agents.core.forecasting import forecast_ensemble

        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(200)

        result = forecast_ensemble(x, horizon=12, models=['arima', 'ets'], season_length=12)

        assert result.horizon == 12
        assert 'arima' in result.forecasts or 'ets' in result.forecasts

    def test_numpy_integer_season_length_uses_unit_step_freq(self, monkeypatch):
        """Test NumPy integer season_length keeps StatsForecast freq at unit steps."""
        import ts_agents.core.forecasting.statistical as statistical

        captured_freqs = []

        class DummyModel:
            def __init__(self, season_length=None):
                self.season_length = season_length

        class AutoARIMA(DummyModel):
            pass

        class AutoETS(DummyModel):
            pass

        class AutoTheta(DummyModel):
            pass

        class SeasonalNaive(DummyModel):
            pass

        class DummyStatsForecast:
            def __init__(self, models, freq, n_jobs):
                captured_freqs.append(freq)
                self.models = models

            def fit(self, df):
                return self

            def predict(self, h, level=None):
                data = {}
                for model in self.models:
                    name = type(model).__name__
                    data[name] = np.zeros(h)
                    if level:
                        data[f"{name}-lo-{level[0]}"] = np.zeros(h)
                        data[f"{name}-hi-{level[0]}"] = np.zeros(h)
                return pd.DataFrame(data)

        monkeypatch.setattr(
            statistical,
            "_get_statsforecast_components",
            lambda: (
                DummyStatsForecast,
                AutoARIMA,
                AutoETS,
                AutoTheta,
                SeasonalNaive,
            ),
        )

        t = np.arange(120)
        x = np.sin(2 * np.pi * t / 12)
        season_length = np.int64(12)

        statistical.forecast_arima(x, horizon=5, season_length=season_length)
        statistical.forecast_ets(x, horizon=5, season_length=season_length)
        statistical.forecast_theta(x, horizon=5, season_length=season_length)
        statistical.forecast_seasonal_naive(x, horizon=5, season_length=season_length)
        statistical.forecast_ensemble(
            x,
            horizon=5,
            models=['arima', 'ets', 'theta', 'seasonal_naive'],
            season_length=season_length,
        )

        assert captured_freqs == [1, 1, 1, 1, 1]
        assert all(isinstance(freq, int) for freq in captured_freqs)

    @requires_statsforecast
    def test_compare_forecasts(self):
        """Test forecast comparison."""
        from ts_agents.core.forecasting import compare_forecasts

        x = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * np.random.randn(500)

        results = compare_forecasts(x, horizon=20, test_size=50, models=['arima', 'ets'])

        assert 'metrics' in results
        assert 'rankings' in results
        assert 'best_model' in results

    def test_compare_forecasts_supports_seasonal_naive(self):
        """Test forecast comparison can score the seasonal naive baseline."""
        from ts_agents.core.forecasting import compare_forecasts

        x = np.tile(np.arange(1, 13, dtype=float), 3)

        results = compare_forecasts(
            x,
            horizon=12,
            test_size=12,
            models=["seasonal_naive"],
            season_length=12,
        )

        assert results["best_model"] == "seasonal_naive"
        assert results["metrics"]["seasonal_naive"]["mae"] == pytest.approx(0.0)
        assert results["metrics"]["seasonal_naive"]["rmse"] == pytest.approx(0.0)
        assert results["metrics"]["seasonal_naive"]["mape"] == pytest.approx(0.0)
