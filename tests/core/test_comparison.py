"""Tests for the comparison module."""

import importlib.util

import numpy as np
import pytest


HAS_STATSFORECAST = importlib.util.find_spec("statsforecast") is not None
requires_statsforecast = pytest.mark.skipif(
    not HAS_STATSFORECAST,
    reason="statsforecast not installed",
)
requires_statsmodels = pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None,
    reason="statsmodels not installed",
)


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult."""
        from ts_agents.core.comparison import ComparisonResult

        result = ComparisonResult(
            category="test",
            methods=["a", "b"],
            results={},
            metrics={
                "a": {"rmse": 0.1, "mae": 0.08},
                "b": {"rmse": 0.2, "mae": 0.15},
            },
            rankings={"rmse": ["a", "b"], "mae": ["a", "b"]},
        )

        assert result.category == "test"
        assert len(result.methods) == 2

    def test_get_best_method(self):
        """Test getting best method for a metric."""
        from ts_agents.core.comparison import ComparisonResult

        result = ComparisonResult(
            category="test",
            methods=["a", "b"],
            results={},
            metrics={
                "a": {"rmse": 0.1},
                "b": {"rmse": 0.2},
            },
            rankings={"rmse": ["a", "b"]},
        )

        assert result.get_best_method("rmse") == "a"

    def test_get_overall_best(self):
        """Test getting overall best method."""
        from ts_agents.core.comparison import ComparisonResult

        result = ComparisonResult(
            category="test",
            methods=["a", "b", "c"],
            results={},
            metrics={},
            rankings={
                "metric1": ["a", "b", "c"],
                "metric2": ["a", "c", "b"],
                "metric3": ["b", "a", "c"],
            },
        )

        # 'a' wins in 2 out of 3 metrics
        assert result.get_overall_best() == "a"

    def test_to_table(self):
        """Test converting comparison to markdown table."""
        from ts_agents.core.comparison import ComparisonResult

        result = ComparisonResult(
            category="test",
            methods=["a", "b"],
            results={},
            metrics={
                "a": {"rmse": 0.1, "mae": 0.08},
                "b": {"rmse": 0.2, "mae": 0.15},
            },
        )

        table = result.to_table()

        assert "Method" in table
        assert "rmse" in table
        assert "mae" in table
        assert "0.1" in table or "0.10" in table
        from ts_agents.core.comparison import compare_forecasting_methods

        zero_targets = compare_forecasting_methods(
            np.zeros(8), horizon=2, methods=["seasonal_naive"],
        )
        table = zero_targets.to_table()
        assert "mape" in table
        assert "N/A" in table
        assert "mape_unavailable_reason" not in table
        assert "error" not in table

    def test_to_dict(self):
        """Test converting comparison to dictionary."""
        from ts_agents.core.comparison import ComparisonResult

        result = ComparisonResult(
            category="test",
            methods=["a", "b"],
            results={},
            metrics={"a": {"rmse": 0.1}},
            recommendation="Use method a",
        )

        d = result.to_dict()

        assert d["category"] == "test"
        assert d["methods"] == ["a", "b"]
        assert d["recommendation"] == "Use method a"


class TestDecompositionComparison:
    """Tests for decomposition method comparison."""

    @requires_statsmodels
    def test_compare_decomposition_methods(self):
        """Test comparing decomposition methods."""
        from ts_agents.core.comparison import compare_decomposition_methods

        # Create test data with trend and seasonality
        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_decomposition_methods(
            x,
            methods=["stl", "holt_winters"],
            period=100,
        )

        assert result.category == "decomposition"
        assert len(result.methods) >= 1
        assert len(result.metrics) >= 1
        assert result.recommendation is not None

    @requires_statsmodels
    def test_compare_decomposition_with_all_methods(self):
        """Test comparing all decomposition methods."""
        from ts_agents.core.comparison import compare_decomposition_methods

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_decomposition_methods(
            x,
            methods=None,  # All methods
            period=100,
        )

        # Should include multiple methods
        assert len(result.methods) >= 2

    def test_decomposition_metrics(self, monkeypatch):
        """Metric directions use known values, independent of any optional fitter."""
        from types import SimpleNamespace
        from ts_agents.core import decomposition
        from ts_agents.core.comparison import compare_decomposition_methods

        monkeypatch.setattr(
            decomposition, "stl_decompose",
            lambda *args, **kwargs: SimpleNamespace(
                residual_variance=1.0, trend_smoothness=2.0, seasonal_strength=0.8,
            ),
        )
        monkeypatch.setattr(
            decomposition, "holt_winters_decompose",
            lambda *args, **kwargs: SimpleNamespace(
                residual_variance=3.0, trend_smoothness=1.0, seasonal_strength=0.6,
            ),
        )
        result = compare_decomposition_methods(np.arange(12.0), methods=["stl", "holt_winters"], period=4)
        assert result.methods == ["stl", "holt_winters"]
        assert result.rankings == {
            "residual_variance": ["stl", "holt_winters"],
            "trend_smoothness": ["holt_winters", "stl"],
            "seasonal_strength": ["stl", "holt_winters"],
        }
        assert "Lowest residual variance: **stl**" in result.recommendation
        assert "Smoothest trend: **holt_winters**" in result.recommendation


class TestForecastingComparison:
    """Tests for forecasting method comparison."""

    @requires_statsforecast
    def test_compare_forecasting_methods(self):
        """Test comparing forecasting methods."""
        from ts_agents.core.comparison import compare_forecasting_methods

        # Create test data
        x = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * np.random.randn(500)

        result = compare_forecasting_methods(
            x,
            horizon=20,
            methods=["theta"],  # Quick method for testing
            validation_size=20,
        )

        assert result.category == "forecasting"
        assert len(result.methods) >= 1
        assert result.recommendation is not None

    @requires_statsforecast
    def test_forecasting_metrics(self):
        """Test that forecasting comparison includes error metrics."""
        from ts_agents.core.comparison import compare_forecasting_methods

        x = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * np.random.randn(500)

        result = compare_forecasting_methods(
            x,
            horizon=20,
            methods=["theta"],
            validation_size=20,
        )

        assert result.methods == ["theta"]
        metrics = result.metrics["theta"]
        assert {"mae", "rmse", "mape"} <= set(metrics)
        assert np.isfinite([metrics["mae"], metrics["rmse"], metrics["mape"]]).all()

    def test_compare_forecasting_methods_supports_seasonal_naive_without_statsforecast(
        self,
        monkeypatch,
    ):
        """Test comparison can still score the baseline without statsforecast."""
        import ts_agents.core.forecasting.statistical as statistical
        from ts_agents.core.comparison import compare_forecasting_methods

        monkeypatch.setattr(
            statistical,
            "_get_statsforecast_components",
            lambda: (_ for _ in ()).throw(
                ImportError("Statistical forecasting requires optional dependencies.")
            ),
        )

        x = np.tile(np.arange(1, 5, dtype=float), 3)
        result = compare_forecasting_methods(
            x,
            horizon=4,
            methods=["seasonal_naive"],
            validation_size=4,
            season_length=4,
        )

        assert result.methods == ["seasonal_naive"]
        assert result.rankings["rmse"] == ["seasonal_naive"]
        assert result.metrics["seasonal_naive"]["rmse"] == pytest.approx(0.0)
        # Zero targets make MAPE undefined, but MAE/RMSE remain valid rankings.
        result = compare_forecasting_methods(np.zeros(12), horizon=4, methods=["seasonal_naive"])
        assert result.metrics["seasonal_naive"]["mape"] is None
        assert result.metrics["seasonal_naive"]["mape_unavailable_reason"] == "zero validation targets"
        assert result.rankings == {"mae": ["seasonal_naive"], "rmse": ["seasonal_naive"]}
        for bad in (np.inf, np.nan, -np.inf):
            with pytest.raises(ValueError, match="finite"):
                compare_forecasting_methods(np.array([1.0, 2.0, bad, 3.0]), horizon=1)

    @pytest.mark.parametrize("method", ["not_a_method", "lightgbm"])
    def test_unknown_or_panel_only_method_yields_error_entry(self, method):
        """Unknown methods are reported instead of silently skipped."""
        from ts_agents.core.comparison import compare_forecasting_methods

        x = np.tile(np.arange(1, 5, dtype=float), 3)
        result = compare_forecasting_methods(
            x,
            horizon=4,
            methods=["seasonal_naive", method],
            validation_size=4,
            season_length=4,
        )

        assert result.methods == ["seasonal_naive"]
        assert result.metrics[method]["error_type"] == "ValueError"
        assert result.metrics[method]["error"].startswith("Unsupported forecasting method")
        assert result.rankings["rmse"] == ["seasonal_naive"]

    def test_mocked_foundation_method_is_scored(self, monkeypatch):
        """Foundation models go through the catalog forecaster with their own options."""
        from ts_agents.core.comparison import compare_forecasting_methods
        from ts_agents.core.forecasting import foundation

        calls = []

        def fake_forecast_arrays(arrays, *, model, horizon, context_length=None, accelerator="cpu", seed=0):
            calls.append((model, horizon, context_length, len(arrays[0])))
            return [np.full(horizon, float(arrays[0][-1]))]

        monkeypatch.setattr(foundation, "forecast_arrays", fake_forecast_arrays)
        x = np.tile(np.arange(1, 5, dtype=float), 3)
        result = compare_forecasting_methods(
            x,
            horizon=4,
            methods=["seasonal_naive", "chronos2_small"],
            validation_size=4,
            season_length=4,
            context_length=6,
        )

        # season_length is dropped for the foundation model; context_length reaches it.
        assert calls == [("chronos2_small", 4, 6, 8)]
        assert result.methods == ["seasonal_naive", "chronos2_small"]
        assert result.metrics["chronos2_small"]["mae"] == pytest.approx(1.5)
        assert result.rankings["rmse"][0] == "seasonal_naive"
        assert "Zero-shot foundation model" in result.recommendation
        # Backend output must cover the entire holdout without broadcasting,
        # truncation, or nonfinite primary metrics masquerading as success.
        for values in ([1.0], [1.0, 2.0, 3.0, np.inf], [1.0] * 5, [1e308] * 4):
            monkeypatch.setattr(
                foundation, "forecast_arrays", lambda *args, **kwargs: [np.array(values)],
            )
            rejected = compare_forecasting_methods(x, horizon=4, methods=["chronos2_small"])
            assert rejected.methods == []
            assert rejected.rankings == {}
            assert rejected.metrics["chronos2_small"]["error_type"] == "ValueError"


class TestGenericCompare:
    """Tests for the generic compare_methods function."""

    @requires_statsmodels
    def test_compare_methods_decomposition(self):
        """Test compare_methods with decomposition category."""
        from ts_agents.core.comparison import compare_methods

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_methods(
            x,
            category="decomposition",
            methods=["stl", "holt_winters"],
            period=100,
        )

        assert result.category == "decomposition"

    def test_compare_methods_invalid_category(self):
        """Test compare_methods with invalid category."""
        from ts_agents.core.comparison import compare_methods

        x = np.random.randn(100)

        with pytest.raises(ValueError):
            compare_methods(x, category="invalid_category")


class TestRankingComputation:
    """Tests for ranking computation."""

    def test_rankings_lower_is_better(self):
        """Test that rankings correctly handle lower-is-better metrics."""
        from ts_agents.core.comparison import _compute_rankings

        metrics = {
            "a": {"rmse": 0.1, "mae": 0.05},
            "b": {"rmse": 0.2, "mae": 0.10},
            "c": {"rmse": 0.15, "mae": 0.08},
            "invalid": {"rmse": np.inf, "mae": np.nan},
        }

        rankings = _compute_rankings(metrics, lower_is_better=["rmse", "mae"])

        # For lower-is-better, 'a' should be first
        assert rankings["rmse"][0] == "a"
        assert rankings["mae"][0] == "a"
        assert "invalid" not in rankings["rmse"]
        assert "invalid" not in rankings["mae"]

    def test_rankings_higher_is_better(self):
        """Test that rankings correctly handle higher-is-better metrics."""
        from ts_agents.core.comparison import _compute_rankings

        metrics = {
            "a": {"accuracy": 0.9},
            "b": {"accuracy": 0.95},
            "c": {"accuracy": 0.85},
        }

        rankings = _compute_rankings(metrics, lower_is_better=[])

        # For higher-is-better, 'b' should be first
        assert rankings["accuracy"][0] == "b"
