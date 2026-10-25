"""Tests for the decomposition module."""

import builtins
import importlib.util
import sys

import numpy as np
import pytest

requires_statsmodels = pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None,
    reason="statsmodels not installed",
)

@requires_statsmodels
class TestSTL:
    """Tests for STL decomposition."""

    def test_stl_decompose_basic(self):
        """Test basic STL decomposition."""
        from ts_agents.core.decomposition import stl_decompose

        # Create series with trend and seasonality
        t = np.linspace(0, 10, 1000)
        trend = 0.5 * t
        seasonal = np.sin(2 * np.pi * t)
        noise = 0.1 * np.random.randn(1000)
        x = trend + seasonal + noise

        result = stl_decompose(x, period=100)

        assert result.method == "stl"
        assert len(result.trend) == 1000
        assert len(result.seasonal) == 1000
        assert len(result.residual) == 1000
        assert result.period == 100

    def test_stl_decompose_auto_period(self):
        """Test STL with auto period detection."""
        from ts_agents.core.decomposition import stl_decompose

        # Create series with clear periodicity
        x = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

        result = stl_decompose(x, period=None)

        # Should detect a reasonable period
        assert result.period >= 2

    def test_stl_metrics(self):
        """Test that STL metrics are computed."""
        from ts_agents.core.decomposition import stl_decompose

        x = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.05 * np.random.randn(2000)
        result = stl_decompose(x, period=100)

        assert result.residual_variance > 0
        assert result.seasonal_strength >= 0
        assert result.seasonal_strength <= 1


class TestMSTL:
    """Tests for MSTL decomposition."""

    @requires_statsmodels
    def test_mstl_decompose(self):
        """Test MSTL decomposition."""
        from ts_agents.core.decomposition import mstl_decompose

        # Create series with multiple seasonalities
        t = np.linspace(0, 100, 10000)
        s1 = np.sin(2 * np.pi * t)  # Period ~6283
        s2 = 0.5 * np.sin(2 * np.pi * t / 7)  # Period ~7 times slower
        x = 0.1 * t + s1 + s2 + 0.1 * np.random.randn(10000)

        result = mstl_decompose(x, periods=[100, 700])

        assert len(result.trend) == 10000
        assert len(result.seasonal) == 10000

        # Auto detection commonly returns just one period (a Series, not DataFrame).
        single = np.sin(2 * np.pi * np.arange(240) / 24)
        for periods in ([24], None):
            result = mstl_decompose(single, periods=periods)
            assert result.period == 24
            np.testing.assert_allclose(result.trend + result.seasonal + result.residual, single, atol=1e-12)

    def test_mstl_missing_statsmodels_reports_mstl_specific_error(self, monkeypatch):
        """Test MSTL missing-dependency path keeps an MSTL-specific error message."""
        from ts_agents.core.decomposition import mstl_decompose

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] == "statsmodels":
                raise ModuleNotFoundError(f"blocked optional import: {name}", name=name)
            return real_import(name, globals, locals, fromlist, level)

        for module_name in list(sys.modules):
            if module_name.split(".")[0] == "statsmodels":
                sys.modules.pop(module_name, None)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        x = np.sin(np.linspace(0, 10 * np.pi, 1000))

        with pytest.raises(
            ImportError,
            match=r'MSTL decomposition requires optional dependencies',
        ):
            mstl_decompose(x, periods=[100, 250])


@requires_statsmodels
class TestHoltWinters:
    """Tests for Holt-Winters decomposition."""

    def test_holt_winters_decompose(self):
        """Additive and multiplicative fits return additive contributions in data units."""
        from ts_agents.core.decomposition import holt_winters_decompose
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        t = np.arange(48)
        x = 10 + 0.05 * t + np.sin(2 * np.pi * t / 6)
        for trend, seasonal, damped in (("add", "add", False), ("mul", "mul", True), (None, None, False)):
            result = holt_winters_decompose(x, period=6, trend=trend, seasonal=seasonal, damped_trend=damped)
            reference = ExponentialSmoothing(
                x, seasonal_periods=6, trend=trend, seasonal=seasonal, damped_trend=damped,
            ).fit(optimized=True)
            assert result.method == "holt_winters"
            np.testing.assert_allclose(result.trend + result.seasonal + result.residual, x, atol=1e-12)
            np.testing.assert_allclose(result.residual, x - reference.fittedvalues, atol=1e-12)
            if seasonal:
                lagged_season = np.r_[reference.params["initial_seasons"], reference.season[:-6]]
                expected_trend = (
                    reference.fittedvalues - lagged_season if seasonal == "add"
                    else reference.fittedvalues / lagged_season
                )
                np.testing.assert_allclose(result.trend, expected_trend, atol=1e-12)
            else:
                np.testing.assert_array_equal(result.seasonal, np.zeros_like(x))
