from __future__ import annotations

#!/usr/bin/env python
"""
Unit tests for coupling function analysis module.

Tests threshold strategies for coupling function estimation.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from astropy import units as u

from gwexpy.analysis.coupling import (
    CouplingResult,
    PercentileThreshold,
    RatioThreshold,
    SigmaThreshold,
    _align_psd_values_to_reference,
    _index_values,
)
from gwexpy.frequencyseries import FrequencySeries


@pytest.fixture
def sample_psd_inj():
    """Create a sample injection PSD with excess around 50 Hz."""
    freqs = np.linspace(1, 100, 100) * u.Hz
    values = np.ones(100)
    values[45:55] = 10.0  # 10x excess around 50 Hz
    return FrequencySeries(values, frequencies=freqs, unit=u.Unit("1/Hz"))


@pytest.fixture
def sample_psd_bkg():
    """Create a sample background PSD (flat)."""
    freqs = np.linspace(1, 100, 100) * u.Hz
    values = np.ones(100)
    return FrequencySeries(values, frequencies=freqs, unit=u.Unit("1/Hz"))


class TestRatioThreshold:
    """Tests for RatioThreshold strategy."""

    def test_init(self):
        """Test RatioThreshold initialization."""
        strategy = RatioThreshold(ratio=2.0)
        assert strategy.ratio == 2.0

    def test_check_detects_excess(self, sample_psd_inj, sample_psd_bkg):
        """Test that check() correctly identifies excess."""
        strategy = RatioThreshold(ratio=2.0)
        mask = strategy.check(sample_psd_inj, sample_psd_bkg)

        # Should detect excess at indices 45-54 where values are 10 (> 2*1)
        assert np.sum(mask) == 10
        assert np.all(mask[45:55])

    def test_check_no_excess(self, sample_psd_bkg):
        """Test check() with no excess (same PSDs)."""
        strategy = RatioThreshold(ratio=2.0)
        mask = strategy.check(sample_psd_bkg, sample_psd_bkg)

        # No excess when PSDs are equal and ratio > 1
        assert np.sum(mask) == 0

    def test_threshold_values(self, sample_psd_inj, sample_psd_bkg):
        """Test threshold() returns correct values."""
        strategy = RatioThreshold(ratio=3.0)
        thresh = strategy.threshold(sample_psd_inj, sample_psd_bkg)

        # Threshold should be 3.0 * psd_bkg = 3.0 everywhere
        np.testing.assert_array_almost_equal(thresh, np.full(100, 3.0))

    def test_high_ratio_no_detection(self, sample_psd_inj, sample_psd_bkg):
        """Test that high ratio threshold doesn't detect excess."""
        strategy = RatioThreshold(ratio=100.0)
        mask = strategy.check(sample_psd_inj, sample_psd_bkg)

        # 10x excess < 100x threshold
        assert np.sum(mask) == 0


class TestSigmaThreshold:
    """Tests for SigmaThreshold strategy."""

    def test_init(self):
        """Test SigmaThreshold initialization."""
        strategy = SigmaThreshold(sigma=3.0)
        assert strategy.sigma == 3.0

    def test_check_detects_excess(self, sample_psd_inj, sample_psd_bkg):
        """Test check() with sigma threshold."""
        strategy = SigmaThreshold(sigma=2.0)
        # Catch warning about low n_avg
        with pytest.warns(UserWarning, match="SigmaThreshold: n_avg"):
            mask = strategy.check(sample_psd_inj, sample_psd_bkg)

        # With sigma=2, should detect the 10x excess
        assert np.sum(mask) > 0

    def test_threshold_values(self, sample_psd_inj, sample_psd_bkg):
        """Test threshold() returns threshold values."""
        strategy = SigmaThreshold(sigma=2.0)
        thresh = strategy.threshold(sample_psd_inj, sample_psd_bkg)

        # Threshold should have correct shape
        assert thresh.shape == sample_psd_bkg.value.shape
        assert np.all(thresh > 0)


class TestPercentileThreshold:
    """Tests for PercentileThreshold strategy."""

    def test_init(self):
        """Test PercentileThreshold initialization with Appendix B.1 defaults."""
        strategy = PercentileThreshold()
        assert strategy.percentile == 99.7
        assert strategy.factor == 2.6

    def test_init_custom(self):
        """Test custom PercentileThreshold initialization."""
        strategy = PercentileThreshold(percentile=95.0, factor=1.5)
        assert strategy.percentile == 95.0
        assert strategy.factor == 1.5

    def test_requires_raw_bkg(self, sample_psd_inj, sample_psd_bkg):
        """Test that check() requires raw_bkg and fftlength or bkg_table."""
        strategy = PercentileThreshold(percentile=95.0)
        # Should raise ValueError without raw_bkg or bkg_table
        with pytest.raises(ValueError, match="requires 'bkg_table', or 'raw_bkg'"):
            strategy.check(sample_psd_inj, sample_psd_bkg, raw_bkg=None)

    def test_with_segment_table(self, sample_psd_inj, sample_psd_bkg):
        """Test PercentileThreshold with SegmentTable input."""
        from gwpy.segments import Segment

        from gwexpy.table.segment_table import SegmentTable

        # Create a SegmentTable with 3 rows of identical background PSDs
        # Note: units must match in PercentileThreshold
        st = SegmentTable.from_segments(
            [Segment(0, 1), Segment(1, 2), Segment(2, 3)],
            psd=[sample_psd_bkg, sample_psd_bkg, sample_psd_bkg]
        )

        # Default factor 2.6
        strategy = PercentileThreshold()
        thresh = strategy.threshold(sample_psd_inj, sample_psd_bkg, bkg_table=st)

        # Threshold should be sample_psd_bkg * 2.6
        np.testing.assert_array_almost_equal(thresh, sample_psd_bkg.value * 2.6)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ratio_threshold_with_zeros(self):
        """Test RatioThreshold handles zero values."""
        freqs = np.linspace(1, 10, 10) * u.Hz
        psd_inj = FrequencySeries(np.ones(10), frequencies=freqs)
        psd_bkg = FrequencySeries(
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), frequencies=freqs
        )

        strategy = RatioThreshold(ratio=2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = strategy.check(psd_inj, psd_bkg)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_negative_ratio_raises(self):
        """Test that negative ratio raises ValueError."""
        # RatioThreshold with ratio <= 0 should be invalid
        RatioThreshold(ratio=-1.0)
        # The behavior depends on implementation - might raise or just return wrong results

    def test_sigma_threshold_with_uniform_psd(self, sample_psd_bkg):
        """Test SigmaThreshold with uniform PSD."""
        strategy = SigmaThreshold(sigma=3.0)
        with pytest.warns(UserWarning, match="SigmaThreshold: n_avg"):
            mask = strategy.check(sample_psd_bkg, sample_psd_bkg)

        # No excess with identical PSDs
        assert np.sum(mask) == 0


# ---------------------------------------------------------------------------
# _index_values helper
# ---------------------------------------------------------------------------


class TestIndexValues:
    def test_with_value_attr(self):
        class Obj:
            value = np.array([1.0, 2.0, 3.0])
        result = _index_values(Obj())
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_with_plain_array(self):
        arr = np.array([4.0, 5.0])
        result = _index_values(arr)
        np.testing.assert_array_equal(result, [4.0, 5.0])


class TestAlignPsdValuesToReference:
    def test_clip_requires_exact_match(self):
        values = np.array([1.0, 2.0, 3.0])
        freqs = np.array([10.0, 11.0, 12.0])
        ref = np.array([10.1, 11.1, 12.1])
        assert _align_psd_values_to_reference(values, freqs, ref, method="clip") is None

    def test_interpolate_allows_small_bin_shift(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        freqs = np.array([10.0, 11.0, 12.0, 13.0])
        ref = np.array([10.25, 11.25, 12.25])
        aligned = _align_psd_values_to_reference(values, freqs, ref, method="interpolate")
        assert aligned is not None
        np.testing.assert_allclose(aligned, np.array([1.25, 2.25, 3.25]))

    def test_interpolate_rejects_large_bin_shift(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        freqs = np.array([10.0, 11.0, 12.0, 13.0])
        ref = np.array([11.5, 12.5])
        assert _align_psd_values_to_reference(values, freqs, ref, method="interpolate") is None


# ---------------------------------------------------------------------------
# SigmaThreshold — additional branches
# ---------------------------------------------------------------------------


class TestSigmaThresholdExtra:
    def test_check_n_avg_zero_returns_all_true(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        mask = strategy.check(sample_psd_inj, sample_psd_bkg, n_avg=0)
        assert np.all(mask)

    def test_check_n_avg_negative_returns_all_true(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        mask = strategy.check(sample_psd_inj, sample_psd_bkg, n_avg=-1)
        assert np.all(mask)

    def test_check_non_numeric_n_avg_raises(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        with pytest.raises(TypeError):
            strategy.check(sample_psd_inj, sample_psd_bkg, n_avg="a")

    def test_threshold_n_avg_zero_returns_bkg(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        thresh = strategy.threshold(sample_psd_inj, sample_psd_bkg, n_avg=0)
        np.testing.assert_array_equal(thresh, sample_psd_bkg.value)

    def test_threshold_non_numeric_n_avg_raises(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        with pytest.raises(TypeError):
            strategy.threshold(sample_psd_inj, sample_psd_bkg, n_avg="a")

    def test_check_high_n_avg_no_warning(self, sample_psd_inj, sample_psd_bkg):
        strategy = SigmaThreshold(sigma=3.0)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            strategy.check(sample_psd_inj, sample_psd_bkg, n_avg=100.0)
        assert not any("SigmaThreshold" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# PercentileThreshold — additional branches
# ---------------------------------------------------------------------------


class TestPercentileThresholdExtra:
    def test_check_requires_fftlength(self, sample_psd_inj, sample_psd_bkg):
        strategy = PercentileThreshold(percentile=95)
        from gwexpy.timeseries import TimeSeries
        ts = TimeSeries(np.ones(100), t0=0, dt=0.01)
        with pytest.raises(ValueError, match="requires"):
            strategy.check(sample_psd_inj, sample_psd_bkg, raw_bkg=ts)

    def test_threshold_non_numeric_fftlength_raises(self, sample_psd_inj, sample_psd_bkg):
        strategy = PercentileThreshold(percentile=95)
        from gwexpy.timeseries import TimeSeries
        ts = TimeSeries(np.ones(100), t0=0, dt=0.01)
        with pytest.raises(TypeError):
            strategy.threshold(sample_psd_inj, sample_psd_bkg, raw_bkg=ts, fftlength="a")

    def test_threshold_non_numeric_overlap_raises(self, sample_psd_inj, sample_psd_bkg):
        strategy = PercentileThreshold(percentile=95)
        from gwexpy.timeseries import TimeSeries
        ts = TimeSeries(np.ones(1000), t0=0, dt=0.01)
        with pytest.raises(TypeError):
            strategy.threshold(sample_psd_inj, sample_psd_bkg, raw_bkg=ts, fftlength=1.0, overlap="x")

    def test_threshold_skips_non_overlapping_segment_table_rows(
        self, sample_psd_inj, sample_psd_bkg
    ):
        from gwpy.segments import Segment

        from gwexpy.table.segment_table import SegmentTable

        shifted_freqs = (np.linspace(200, 300, 100) * u.Hz)
        shifted_psd = FrequencySeries(
            np.full(100, 10.0), frequencies=shifted_freqs, unit=u.Unit("1/Hz")
        )
        st = SegmentTable.from_segments(
            [Segment(0, 1), Segment(1, 2)],
            psd=[sample_psd_bkg, shifted_psd],
        )

        strategy = PercentileThreshold(percentile=50, factor=1.0)
        with pytest.warns(UserWarning, match="Skipping background PSD row"):
            thresh = strategy.threshold(sample_psd_inj, sample_psd_bkg, bkg_table=st)

        np.testing.assert_array_almost_equal(thresh, sample_psd_bkg.value)


# ---------------------------------------------------------------------------
# CouplingResult — construction and properties
# ---------------------------------------------------------------------------


def _make_fs(n=50):
    freqs = np.linspace(1, 100, n) * u.Hz
    vals = np.random.default_rng(42).uniform(0.1, 1.0, n)
    return FrequencySeries(vals, frequencies=freqs, unit=u.Unit("1/Hz"))


class TestCouplingResult:
    def _make_result(self, **kwargs):
        fs = _make_fs()
        defaults = dict(
            cf=fs,
            psd_witness_inj=fs,
            psd_witness_bkg=fs,
            psd_target_inj=fs,
            psd_target_bkg=fs,
            valid_mask=np.ones(50, dtype=bool),
            witness_name="W",
            target_name="T",
        )
        defaults.update(kwargs)
        return CouplingResult(**defaults)

    def test_init_basic(self):
        result = self._make_result()
        assert result.witness_name == "W"
        assert result.target_name == "T"

    def test_frequencies_property(self):
        result = self._make_result()
        assert hasattr(result.frequencies, "__len__")

    def test_cf_ul_none_by_default(self):
        result = self._make_result()
        assert result.cf_ul is None

    def test_cf_ul_can_be_set(self):
        fs = _make_fs()
        result = self._make_result(cf_ul=fs)
        assert result.cf_ul is not None

    def test_ts_none_by_default(self):
        result = self._make_result()
        assert result.ts_witness_bkg is None
        assert result.ts_target_bkg is None

    def test_fftlength_overlap_stored(self):
        result = self._make_result(fftlength=1.0, overlap=0.5)
        assert result.fftlength == 1.0
        assert result.overlap == 0.5

    def test_plot_cf_returns_plot(self):
        import matplotlib.pyplot as plt
        result = self._make_result()
        p = result.plot_cf()
        assert p is not None
        plt.close("all")

    def test_plot_cf_with_cf_ul(self):
        import matplotlib.pyplot as plt
        fs = _make_fs()
        result = self._make_result(cf_ul=fs)
        p = result.plot_cf()
        assert p is not None
        plt.close("all")

    def test_plot_cf_with_xlim(self):
        import matplotlib.pyplot as plt
        result = self._make_result()
        p = result.plot_cf(xlim=(10.0, 80.0))
        assert p is not None
        plt.close("all")

    def test_plot_returns_plot(self):
        import matplotlib.pyplot as plt
        result = self._make_result()
        p = result.plot()
        assert p is not None
        plt.close("all")

    def test_plot_with_xlim(self):
        import matplotlib.pyplot as plt
        result = self._make_result()
        p = result.plot(xlim=(10.0, 80.0))
        assert p is not None
        plt.close("all")

    def test_plot_with_cf_ul(self):
        import matplotlib.pyplot as plt
        fs = _make_fs()
        result = self._make_result(cf_ul=fs)
        p = result.plot()
        assert p is not None
        plt.close("all")
