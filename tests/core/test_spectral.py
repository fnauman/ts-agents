"""Tests for the spectral module."""

import numpy as np
import pytest


class TestPSD:
    """Tests for power spectral density."""

    def test_compute_psd_welch(self):
        """Test PSD computation with Welch method."""
        from src.core.spectral import compute_psd

        x = np.sin(np.linspace(0, 20 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

        result = compute_psd(x, sample_rate=1.0, method='welch')

        assert len(result.frequencies) == len(result.psd)
        assert result.frequencies[0] == 0
        assert all(result.psd >= 0)

    def test_compute_psd_periodogram(self):
        """Test PSD with periodogram method."""
        from src.core.spectral import compute_psd

        x = np.random.randn(500)

        result = compute_psd(x, method='periodogram')

        assert len(result.frequencies) > 0
        assert len(result.psd) > 0

    def test_spectral_slope(self):
        """Test spectral slope computation."""
        from src.core.spectral import compute_psd

        # Random walk has 1/f^2 spectrum (slope ~ -2)
        x = np.cumsum(np.random.randn(2000))

        result = compute_psd(x)

        # Slope should be negative for random walk
        assert result.spectral_slope < 0

    def test_compute_psd_short_series(self):
        """Short series should raise a clear error."""
        from src.core.spectral import compute_psd

        x = np.array([1.0])
        with pytest.raises(ValueError):
            compute_psd(x)

    def test_detect_periodicity(self):
        """Test periodicity detection."""
        from src.core.spectral import detect_periodicity

        # Create series with known period
        period = 50
        x = np.sin(2 * np.pi * np.arange(500) / period)

        result = detect_periodicity(x)

        # Should detect the correct period
        assert 40 < result.dominant_period < 60


class TestCoherence:
    """Tests for coherence analysis."""

    def test_compute_coherence(self):
        """Test coherence computation."""
        from src.core.spectral import compute_coherence

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
        from src.core.spectral import compute_coherence

        x = np.random.randn(1000)
        y = np.random.randn(1000)

        result = compute_coherence(x, y)

        # Uncorrelated signals should have low coherence
        assert result.mean_coherence < 0.5

    def test_compute_coherence_short_series(self):
        """Short series should raise a clear error."""
        from src.core.spectral import compute_coherence

        x = np.array([1.0])
        with pytest.raises(ValueError):
            compute_coherence(x, x)
