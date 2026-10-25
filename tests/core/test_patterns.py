"""Tests for the patterns module."""

import numpy as np
import pytest


class TestPeaks:
    """Tests for peak detection."""

    def test_detect_peaks_basic(self):
        """Test basic peak detection on a sine wave."""
        from ts_agents.core.patterns import detect_peaks

        # Create a sine wave with clear peaks
        x = np.sin(np.linspace(0, 10 * np.pi, 1000))
        result = detect_peaks(x, distance=50)

        # Should find approximately 5 peaks
        assert 4 <= result.count <= 6
        assert len(result.peak_indices) == result.count
        assert len(result.peak_values) == result.count

    def test_detect_peaks_with_prominence(self):
        """Test peak detection with prominence threshold."""
        from ts_agents.core.patterns import detect_peaks

        # Sine wave with noise
        x = np.sin(np.linspace(0, 4 * np.pi, 400)) + 0.1 * np.random.randn(400)
        result = detect_peaks(x, prominence=0.5)

        # Should find major peaks, not noise peaks
        assert result.count >= 2
        assert all(result.peak_values > 0.5)

    def test_count_peaks(self):
        """Test simple peak count function."""
        from ts_agents.core.patterns import count_peaks

        x = np.array([0, 1, 0, 2, 0, 1.5, 0])
        count = count_peaks(x)

        assert count == 3

    def test_peak_spacing_statistics(self):
        """Test that peak spacing statistics are computed."""
        from ts_agents.core.patterns import detect_peaks

        # Regular sine wave should have low spacing CV
        x = np.sin(np.linspace(0, 20 * np.pi, 2000))
        result = detect_peaks(x, distance=100)

        assert result.mean_spacing > 0
        assert result.spacing_cv < 0.3  # Should be regular


class TestRecurrence:
    """Tests for recurrence analysis."""

    def test_compute_recurrence_matrix(self):
        """Test recurrence matrix computation."""
        from ts_agents.core.patterns import compute_recurrence_matrix

        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        R, threshold = compute_recurrence_matrix(x)

        assert R.shape == (200, 200)
        assert R.dtype == np.int8
        assert np.all((R == 0) | (R == 1))
        # Diagonal should be all 1s
        assert np.all(np.diag(R) == 1)

    def test_analyze_recurrence(self):
        """Test full recurrence analysis."""
        from ts_agents.core.patterns import analyze_recurrence

        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        result = analyze_recurrence(x)

        assert 0 <= result.recurrence_rate <= 1
        assert 0 <= result.determinism <= 1
        assert 0 <= result.laminarity <= 1


class TestMatrixProfile:
    """Tests for matrix profile analysis."""

    def test_compute_matrix_profile(self):
        """Test basic matrix profile computation."""
        from ts_agents.core.patterns import compute_matrix_profile

        x = np.random.randn(500)
        mp = compute_matrix_profile(x, m=50)

        assert mp.shape[0] == 500 - 50 + 1
        assert mp.shape[1] == 4  # distance, index, left, right

    def test_find_motifs(self):
        """Test motif discovery."""
        from ts_agents.core.patterns import find_motifs

        # Create a series with a repeating pattern
        pattern = np.sin(np.linspace(0, 2 * np.pi, 50))
        x = np.random.default_rng(3).standard_normal(500)
        x[100:150] = pattern
        x[300:350] = pattern * 1.1

        result = find_motifs(x, m=50, max_motifs=3)

        best = result.motifs[0]
        assert {best.index, best.neighbor_index} == {100, 300}
        assert result.exclusion_zone == 50
        # The planted pair is reported once, not again as (300, 100).
        members = [index for motif in result.motifs for index in (motif.index, motif.neighbor_index)]
        assert sum(abs(index - 300) < 50 for index in members) == 1

        # A third occurrence can still point to an already excluded neighbor.
        from ts_agents.core.patterns.matrix_profile import _top_motifs
        profile = np.array([[0., 3], [9., 6], [9., 6], [0., 0], [9., 0], [9., 0], [1., 0]])
        motifs = _top_motifs(np.arange(9), profile, 2, 3, 1, False)
        assert len(motifs) == 1

    def test_find_discords(self):
        """Test discord (anomaly) detection."""
        from ts_agents.core.patterns import find_discords

        # Two separate events; one event must not occupy several top-k slots.
        rng = np.random.default_rng(0)
        x = np.sin(2 * np.pi * np.arange(3000) / 100) + 0.05 * rng.standard_normal(3000)
        x[700:705] += 3.0
        x[2100:2130] = x[2100]

        result = find_discords(x, m=40, max_discords=2)

        assert result.exclusion_zone == 40
        assert len(result.discords) == 2
        assert any(abs(d.index - 700) < 40 for d in result.discords)
        assert any(abs(d.index - 2100) < 40 for d in result.discords)
        assert all(d.robust_z > 5 for d in result.discords)

    def test_analyze_matrix_profile(self):
        """Test comprehensive matrix profile analysis."""
        from ts_agents.core.patterns import analyze_matrix_profile

        x = np.random.randn(500) + np.sin(np.linspace(0, 10 * np.pi, 500))
        result = analyze_matrix_profile(x, m=50, max_motifs=3, max_discords=3)

        assert result.mp_min <= result.mp_max
        assert len(result.motifs) <= 3
        assert len(result.discords) <= 3


class TestSegmentation:
    """Tests for time series segmentation."""

    def test_segment_changepoint(self):
        """Test changepoint detection."""
        from ts_agents.core.patterns import segment_changepoint

        # Create series with clear regime change
        x = np.concatenate([
            np.random.randn(200),
            np.random.randn(200) + 5,
        ])

        result = segment_changepoint(x, n_segments=2)

        assert result.n_segments == 2
        assert len(result.changepoints) == 1
        # Changepoint should be near 200
        assert 150 < result.changepoints[0] < 250

    def test_segment_changepoint_default_is_scale_invariant(self):
        """Changing units must not change the detected regimes."""
        from ts_agents.core.patterns import segment_changepoint

        rng = np.random.default_rng(0)
        x = rng.standard_normal(900)
        x[300:600] += 3.0

        results = [segment_changepoint(x * scale) for scale in (1e-3, 1.0, 1e3)]

        assert all(result.changepoints == [300, 600] for result in results)
        assert results[0].settings["penalty_source"] == "default"

    def test_segment_changepoint_finds_no_regimes_in_random_walk(self):
        """Autocorrelation-adjusted penalties do not cut persistent noise into regimes."""
        from ts_agents.core.patterns import segment_changepoint

        walk = np.cumsum(np.random.default_rng(2).standard_normal(500))

        assert segment_changepoint(walk).changepoints == []

    def test_segment_changepoint_reports_substituted_algorithm(self):
        """PELT cannot target a segment count; the result names what ran."""
        from ts_agents.core.patterns import segment_changepoint

        x = np.concatenate([np.zeros(100), np.ones(100) * 5]) + np.random.default_rng(0).normal(0, 0.1, 200)
        result = segment_changepoint(x, n_segments=2, algorithm="pelt")

        assert result.method == "changepoint_binseg"
        assert result.requested_method == "changepoint_pelt"
        assert result.warnings

    @pytest.mark.parametrize(
        ("series", "kwargs", "message"),
        [
            (np.arange(100.0), {"algorithm": "bogus"}, "Unknown algorithm"),
            (np.arange(100.0), {"cost_model": "bogus"}, "Unknown cost_model"),
            (np.r_[np.arange(50.0), np.nan], {}, "non-finite"),
            (np.arange(8.0), {}, "too short"),
            (np.ones((100, 2)), {}, "one-dimensional"),
        ],
    )
    def test_segment_changepoint_rejects_invalid_input(self, series, kwargs, message):
        """Errors fail loudly instead of returning an empty 'no changepoints' result."""
        from ts_agents.core.patterns import segment_changepoint

        with pytest.raises(ValueError, match=message):
            segment_changepoint(series, **kwargs)

    def test_segment_fluss(self):
        """Test FLUSS segmentation."""
        from ts_agents.core.patterns import segment_fluss

        np.random.seed(0)
        x = np.concatenate([
            np.random.randn(300),
            np.random.randn(300) + 3,
            np.random.randn(300) - 2,
        ])

        result = segment_fluss(x, m=50, n_segments=3)

        assert result.n_segments >= 1
        assert len(result.segment_stats) == result.n_segments
        assert all(stats["length"] > 0 for stats in result.segment_stats)
        assert all("error" not in stats for stats in result.segment_stats)
        assert result.changepoints == sorted(result.changepoints)
        assert all(0 < cp < len(x) for cp in result.changepoints)

    def test_segment_fluss_accepts_n_regimes_alias(self):
        """FLUSS should support the backward-compatible n_regimes alias."""
        from ts_agents.core.patterns import segment_fluss

        x = np.sin(np.linspace(0, 40 * np.pi, 1000))
        result = segment_fluss(x, m=50, n_regimes=3)

        assert result.method == "fluss"
        assert result.n_segments >= 1

    def test_segment_changepoint_accepts_n_bkps_alias(self):
        """Changepoint segmentation should support the n_bkps alias."""
        from ts_agents.core.patterns import segment_changepoint

        x = np.concatenate([np.random.randn(200), np.random.randn(200) + 2])
        result = segment_changepoint(x, n_bkps=1)

        assert result.n_segments >= 1
        assert all(0 < cp < len(x) for cp in result.changepoints)

    def test_matrix_profile_accepts_legacy_parameter_names(self):
        """Matrix profile analysis should accept old UI argument names."""
        from ts_agents.core.patterns import analyze_matrix_profile

        x = np.random.randn(400)
        result = analyze_matrix_profile(
            x,
            subsequence_length=40,
            n_motifs=2,
            n_discords=2,
        )

        assert result.subsequence_length == 40
        assert len(result.motifs) <= 2
        assert len(result.discords) <= 2

    def test_recurrence_reports_decimation(self):
        """Decimated recurrence input is reported, not silently substituted."""
        from ts_agents.core.patterns import analyze_recurrence

        x = np.sin(np.linspace(0, 40 * np.pi, 2500))
        result = analyze_recurrence(x, max_points=1000, threshold=0.1)

        assert result.downsample_step == 3
        assert result.n_points_used == 834
        assert result.recurrence_matrix is None
        assert "decimated with step 3" in result.warnings[0]
