"""Tests for the decomposition module."""

import numpy as np
import pytest


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


class TestHoltWinters:
    """Tests for Holt-Winters decomposition."""

    def test_holt_winters_decompose(self):
        """Test Holt-Winters decomposition."""
        from ts_agents.core.decomposition import holt_winters_decompose

        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = holt_winters_decompose(x, period=100)

        assert len(result.trend) == 1000
        assert len(result.seasonal) == 1000
        assert len(result.residual) == 1000
