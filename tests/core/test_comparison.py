"""Tests for the comparison module."""

import numpy as np
import pytest


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult."""
        from src.core.comparison import ComparisonResult

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
        from src.core.comparison import ComparisonResult

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
        from src.core.comparison import ComparisonResult

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
        from src.core.comparison import ComparisonResult

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

    def test_to_dict(self):
        """Test converting comparison to dictionary."""
        from src.core.comparison import ComparisonResult

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

    def test_compare_decomposition_methods(self):
        """Test comparing decomposition methods."""
        from src.core.comparison import compare_decomposition_methods

        # Create test data with trend and seasonality
        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_decomposition_methods(
            x,
            methods=["stl", "hp_filter"],
            period=100,
        )

        assert result.category == "decomposition"
        assert len(result.methods) >= 1
        assert len(result.metrics) >= 1
        assert result.recommendation is not None

    def test_compare_decomposition_with_all_methods(self):
        """Test comparing all decomposition methods."""
        from src.core.comparison import compare_decomposition_methods

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_decomposition_methods(
            x,
            methods=None,  # All methods
            period=100,
        )

        # Should include multiple methods
        assert len(result.methods) >= 2

    def test_decomposition_metrics(self):
        """Test that decomposition comparison includes expected metrics."""
        from src.core.comparison import compare_decomposition_methods

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_decomposition_methods(
            x,
            methods=["stl"],
            period=100,
        )

        if "stl" in result.metrics:
            metrics = result.metrics["stl"]
            assert "residual_variance" in metrics
            assert "trend_smoothness" in metrics
            assert "seasonal_strength" in metrics


class TestForecastingComparison:
    """Tests for forecasting method comparison."""

    def test_compare_forecasting_methods(self):
        """Test comparing forecasting methods."""
        from src.core.comparison import compare_forecasting_methods

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

    def test_forecasting_metrics(self):
        """Test that forecasting comparison includes error metrics."""
        from src.core.comparison import compare_forecasting_methods

        x = np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.1 * np.random.randn(500)

        result = compare_forecasting_methods(
            x,
            horizon=20,
            methods=["theta"],
            validation_size=20,
        )

        if "theta" in result.metrics:
            metrics = result.metrics["theta"]
            # Should have error metrics
            assert "mae" in metrics or "rmse" in metrics


class TestGenericCompare:
    """Tests for the generic compare_methods function."""

    def test_compare_methods_decomposition(self):
        """Test compare_methods with decomposition category."""
        from src.core.comparison import compare_methods

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = compare_methods(
            x,
            category="decomposition",
            methods=["stl", "hp_filter"],
            period=100,
        )

        assert result.category == "decomposition"

    def test_compare_methods_invalid_category(self):
        """Test compare_methods with invalid category."""
        from src.core.comparison import compare_methods

        x = np.random.randn(100)

        with pytest.raises(ValueError):
            compare_methods(x, category="invalid_category")


class TestRankingComputation:
    """Tests for ranking computation."""

    def test_rankings_lower_is_better(self):
        """Test that rankings correctly handle lower-is-better metrics."""
        from src.core.comparison import _compute_rankings

        metrics = {
            "a": {"rmse": 0.1, "mae": 0.05},
            "b": {"rmse": 0.2, "mae": 0.10},
            "c": {"rmse": 0.15, "mae": 0.08},
        }

        rankings = _compute_rankings(metrics, lower_is_better=["rmse", "mae"])

        # For lower-is-better, 'a' should be first
        assert rankings["rmse"][0] == "a"
        assert rankings["mae"][0] == "a"

    def test_rankings_higher_is_better(self):
        """Test that rankings correctly handle higher-is-better metrics."""
        from src.core.comparison import _compute_rankings

        metrics = {
            "a": {"accuracy": 0.9},
            "b": {"accuracy": 0.95},
            "c": {"accuracy": 0.85},
        }

        rankings = _compute_rankings(metrics, lower_is_better=[])

        # For higher-is-better, 'b' should be first
        assert rankings["accuracy"][0] == "b"
