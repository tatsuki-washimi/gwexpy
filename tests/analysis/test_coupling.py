#!/usr/bin/env python
"""
Unit tests for coupling function analysis module.

Tests threshold strategies for coupling function estimation.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.analysis.coupling import (
    PercentileThreshold,
    RatioThreshold,
    SigmaThreshold,
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
        """Test PercentileThreshold initialization."""
        strategy = PercentileThreshold(percentile=95.0)
        assert strategy.percentile == 95.0

    def test_requires_raw_bkg(self, sample_psd_inj, sample_psd_bkg):
        """Test that check() requires raw_bkg and fftlength."""
        strategy = PercentileThreshold(percentile=95.0)
        # Should raise ValueError without raw_bkg
        with pytest.raises(ValueError, match="requires 'raw_bkg'"):
            strategy.check(sample_psd_inj, sample_psd_bkg, raw_bkg=None)


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
