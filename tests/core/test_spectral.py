"""Tests for the spectral module."""

import numpy as np
import pytest


class TestPSD:
    """Tests for power spectral density."""

    def test_compute_psd_welch(self):
        """Test PSD computation with Welch method."""
        from ts_agents.core.spectral import compute_psd

        x = np.sin(np.linspace(0, 20 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

        result = compute_psd(x, sample_rate=1.0, method='welch')

        assert len(result.frequencies) == len(result.psd)
        assert result.frequencies[0] == 0
        assert all(result.psd >= 0)

    def test_compute_psd_periodogram(self):
        """Test PSD with periodogram method."""
        from ts_agents.core.spectral import compute_psd

        x = np.random.randn(500)

        result = compute_psd(x, method='periodogram')

        assert len(result.frequencies) > 0
        assert len(result.psd) > 0

    def test_spectral_slope(self):
        """Test spectral slope computation."""
        from ts_agents.core.spectral import compute_psd

        # Random walk has 1/f^2 spectrum (slope ~ -2)
        x = np.cumsum(np.random.randn(2000))

        result = compute_psd(x)

        # Slope should be negative for random walk
        assert result.spectral_slope < 0

    def test_compute_psd_short_series(self):
        """Short series should raise a clear error."""
        from ts_agents.core.spectral import compute_psd

        x = np.array([1.0])
        with pytest.raises(ValueError):
            compute_psd(x)

    def test_detect_periodicity(self):
        """Test periodicity detection."""
        from ts_agents.core.spectral import detect_periodicity

        # Create series with known period
        period = 50
        x = np.sin(2 * np.pi * np.arange(500) / period)

        result = detect_periodicity(x)

        # Should detect the correct period
        assert result.detected
        assert result.dominant_period == pytest.approx(50, rel=0.01)

    def test_detect_periodicity_ignores_trend_and_reports_time_units(self):
        """A trend must not become the "period"; periods use sample_rate units."""
        from ts_agents.core.spectral import detect_periodicity

        rng = np.random.default_rng(0)
        t = np.arange(1000)
        trended = 0.02 * t + np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(1000)
        assert detect_periodicity(trended).dominant_period == pytest.approx(24, rel=0.01)

        seconds = np.arange(1000) / 50.0
        five_hz = np.sin(2 * np.pi * 5.0 * seconds) + 0.2 * rng.standard_normal(1000)
        assert detect_periodicity(five_hz, sample_rate=50.0).dominant_period == pytest.approx(0.2, rel=0.01)

    @pytest.mark.parametrize("kind", ["random_walk", "level_shift", "white_noise"])
    def test_detect_periodicity_reports_none_for_aperiodic_series(self, kind):
        """Random walks, level shifts, and noise have no significant period."""
        from ts_agents.core.spectral import detect_periodicity

        rng = np.random.default_rng(1)
        noise = rng.standard_normal(1000)
        series = {
            "random_walk": np.cumsum(noise),
            "level_shift": noise + 3.0 * (np.arange(1000) >= 500),
            "white_noise": noise,
        }[kind]

        result = detect_periodicity(series)

        assert result.detected is False
        assert result.dominant_period is None
        assert result.candidate_period is not None


class TestCoherence:
    """Tests for coherence analysis."""

    def test_compute_coherence(self):
        """Test coherence computation."""
        from ts_agents.core.spectral import compute_coherence

        # Two correlated signals
        t = np.linspace(0, 10, 1000)
        x = np.sin(2 * np.pi * t)
        y = np.sin(2 * np.pi * t + 0.5) + 0.1 * np.random.randn(1000)

        result = compute_coherence(x, y)

        assert len(result.frequencies) == len(result.coherence)
        assert all(0 <= c <= 1 for c in result.coherence)
        assert result.mean_coherence > 0.5  # Should be highly coherent

    def test_compute_coherence_uncorrelated(self):
        """Test coherence of uncorrelated signals."""
        from ts_agents.core.spectral import compute_coherence

        x = np.random.randn(1000)
        y = np.random.randn(1000)

        result = compute_coherence(x, y)

        # Uncorrelated signals should have low coherence
        assert result.mean_coherence < 0.5

    def test_compute_coherence_short_series(self):
        """Short series should raise a clear error."""
        from ts_agents.core.spectral import compute_coherence

        x = np.array([1.0])
        with pytest.raises(ValueError):
            compute_coherence(x, x)
