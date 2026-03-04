"""Tests for the forecasting module."""

import numpy as np
import pytest
import pandas as pd


class TestForecasting:
    """Tests for forecasting functions."""

    def test_forecast_arima(self):
        """Test ARIMA forecasting."""
        from src.core.forecasting import forecast_arima

        # Create a simple series
        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_arima(x, horizon=20)

        assert result.method == "auto_arima"
        assert len(result.forecast) == 20
        assert result.horizon == 20

    def test_forecast_ets(self):
        """Test ETS forecasting."""
        from src.core.forecasting import forecast_ets

        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_ets(x, horizon=20)

        assert result.method == "auto_ets"
        assert len(result.forecast) == 20

    def test_forecast_theta(self):
        """Test Theta forecasting."""
        from src.core.forecasting import forecast_theta

        x = np.linspace(0, 10, 200) + 0.5 * np.random.randn(200)

        result = forecast_theta(x, horizon=20)

        assert result.method == "auto_theta"
        assert len(result.forecast) == 20

    def test_forecast_ensemble(self):
        """Test ensemble forecasting."""
        from src.core.forecasting import forecast_ensemble

        x = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.1 * np.random.randn(200)

        result = forecast_ensemble(x, horizon=20, models=['arima', 'ets'])

        assert 'arima' in result.forecasts or 'ets' in result.forecasts
        assert result.horizon == 20

        # Test ensemble average
        ensemble = result.get_ensemble()
        assert len(ensemble) == 20

    def test_forecast_arima_with_season_length(self):
        """Test ARIMA forecasting with season_length parameter."""
        from src.core.forecasting import forecast_arima

        # Create seasonal series
        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(200)

        result = forecast_arima(x, horizon=12, season_length=12)

        assert result.method == "auto_arima"
        assert len(result.forecast) == 12

    def test_forecast_ensemble_with_season_length(self):
        """Test ensemble forecasting with season_length parameter."""
        from src.core.forecasting import forecast_ensemble

        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(200)

        result = forecast_ensemble(x, horizon=12, models=['arima', 'ets'], season_length=12)

        assert result.horizon == 12
        assert 'arima' in result.forecasts or 'ets' in result.forecasts

    def test_numpy_integer_season_length_uses_unit_step_freq(self, monkeypatch):
        """Test NumPy integer season_length keeps StatsForecast freq at unit steps."""
        import src.core.forecasting.statistical as statistical

        captured_freqs = []

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

        monkeypatch.setattr(statistical, "StatsForecast", DummyStatsForecast)

        t = np.arange(120)
        x = np.sin(2 * np.pi * t / 12)
        season_length = np.int64(12)

        statistical.forecast_arima(x, horizon=5, season_length=season_length)
        statistical.forecast_ets(x, horizon=5, season_length=season_length)
        statistical.forecast_theta(x, horizon=5, season_length=season_length)
        statistical.forecast_ensemble(
            x,
            horizon=5,
            models=['arima', 'ets', 'theta'],
            season_length=season_length,
        )

        assert captured_freqs == [1, 1, 1, 1]
        assert all(isinstance(freq, int) for freq in captured_freqs)

    def test_compare_forecasts(self):
        """Test forecast comparison."""
        from src.core.forecasting import compare_forecasts

        x = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * np.random.randn(500)

        results = compare_forecasts(x, horizon=20, test_size=50, models=['arima', 'ets'])

        assert 'metrics' in results
        assert 'rankings' in results
        assert 'best_model' in results
