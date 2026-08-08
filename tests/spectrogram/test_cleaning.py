"""Tests for Spectrogram.clean() and cleaning algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram
from gwexpy.spectrogram.cleaning import (
    line_removal_clean,
    rolling_median_clean,
    threshold_clean,
)


@pytest.fixture
def base_spectrogram():
    """Create a clean spectrogram (20 time bins x 8 freq bins)."""
    np.random.seed(42)
    data = np.abs(np.random.randn(20, 8)) + 2.0
    return Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


@pytest.fixture
def glitchy_spectrogram():
    """Spectrogram with injected glitches."""
    np.random.seed(42)
    data = np.ones((20, 8)) + 0.1 * np.random.randn(20, 8)
    # Inject glitches
    data[5, 3] = 100.0
    data[10, 6] = 50.0
    data[15, 1] = 80.0
    return Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


@pytest.fixture
def line_spectrogram():
    """Spectrogram with persistent narrowband lines."""
    np.random.seed(42)
    data = np.ones((20, 8)) + 0.1 * np.random.randn(20, 8)
    # Inject persistent line at freq bin 2
    data[:, 2] = 10.0  # always elevated
    # Inject intermittent line at freq bin 5 (only 30% of time)
    data[:6, 5] = 10.0
    return Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


# ---------------------------------------------------------------------------
# Low-level algorithm tests
# ---------------------------------------------------------------------------


class TestThresholdClean:
    def test_removes_outliers(self):
        np.random.seed(0)
        data = np.ones((20, 5)) + 0.01 * np.random.randn(20, 5)
        data[3, 2] = 100.0  # obvious outlier
        cleaned, mask = threshold_clean(data, threshold=5.0)
        assert mask[3, 2]
        assert cleaned[3, 2] != 100.0

    def test_fill_nan(self):
        data = np.ones((10, 3))
        data[5, 1] = 100.0
        cleaned, mask = threshold_clean(data, threshold=3.0, fill="nan")
        assert np.isnan(cleaned[5, 1])

    def test_fill_zero(self):
        data = np.ones((10, 3))
        data[5, 1] = 100.0
        cleaned, mask = threshold_clean(data, threshold=3.0, fill="zero")
        assert cleaned[5, 1] == 0.0

    def test_fill_interpolate(self):
        data = np.ones((10, 3))
        data[5, 1] = 100.0
        cleaned, mask = threshold_clean(data, threshold=3.0, fill="interpolate")
        # Interpolated value should be close to 1.0
        assert abs(cleaned[5, 1] - 1.0) < 0.1

    def test_unknown_fill_raises(self):
        data = np.ones((5, 3))
        with pytest.raises(ValueError, match="Unknown fill"):
            threshold_clean(data, fill="bad")

    def test_no_outliers_no_changes(self):
        data = np.ones((10, 5))
        cleaned, mask = threshold_clean(data, threshold=5.0)
        assert not np.any(mask)
        np.testing.assert_array_equal(cleaned, data)


class TestRollingMedianClean:
    def test_removes_slow_trend(self):
        # Data with linear trend
        data = np.outer(np.linspace(1, 10, 20), np.ones(5))
        normalized = rolling_median_clean(data, window_size=5)
        # After removing trend, values should be close to 1
        assert np.nanstd(normalized) < 0.5

    def test_preserves_shape(self):
        data = np.random.randn(15, 6) + 5.0
        result = rolling_median_clean(data, window_size=5)
        assert result.shape == data.shape


class TestLineRemovalClean:
    def test_detects_persistent_line(self):
        data = np.ones((20, 8))
        data[:, 3] = 10.0  # persistent line
        cleaned, lines = line_removal_clean(data, persistence_threshold=0.8)
        assert 3 in lines

    def test_ignores_intermittent_line(self):
        data = np.ones((20, 8))
        data[:4, 5] = 10.0  # only 20% elevated
        cleaned, lines = line_removal_clean(data, persistence_threshold=0.8)
        assert 5 not in lines

    def test_replaces_with_median(self):
        data = np.ones((20, 5))
        data[:, 2] = 10.0
        cleaned, lines = line_removal_clean(data, persistence_threshold=0.5)
        assert 2 in lines
        # After cleaning, all values in column 2 should be the median (10.0)
        assert np.all(cleaned[:, 2] == np.median(data[:, 2]))


# ---------------------------------------------------------------------------
# Spectrogram.clean() method tests
# ---------------------------------------------------------------------------


class TestSpectrogramClean:
    def test_threshold_method(self, glitchy_spectrogram):
        result = glitchy_spectrogram.clean(method="threshold", threshold=5.0)
        assert isinstance(result, Spectrogram)
        # Glitches should be removed
        assert result.value[5, 3] < 50.0

    def test_rolling_median_method(self, base_spectrogram):
        result = base_spectrogram.clean(method="rolling_median", window_size=5)
        assert isinstance(result, Spectrogram)

    def test_line_removal_method(self, line_spectrogram):
        result = line_spectrogram.clean(method="line_removal")
        assert isinstance(result, Spectrogram)

    def test_combined_method(self, glitchy_spectrogram):
        result = glitchy_spectrogram.clean(method="combined", window_size=5)
        assert isinstance(result, Spectrogram)

    def test_unknown_method(self, base_spectrogram):
        with pytest.raises(ValueError, match="Unknown method"):
            base_spectrogram.clean(method="invalid")

    def test_return_mask(self, glitchy_spectrogram):
        result, mask = glitchy_spectrogram.clean(
            method="threshold", threshold=5.0, return_mask=True
        )
        assert isinstance(result, Spectrogram)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == glitchy_spectrogram.shape

    def test_no_mask_by_default(self, base_spectrogram):
        result = base_spectrogram.clean()
        assert isinstance(result, Spectrogram)

    def test_metadata_preserved(self, base_spectrogram):
        result = base_spectrogram.clean()
        assert result.unit == base_spectrogram.unit
        np.testing.assert_array_equal(
            result.frequencies.value, base_spectrogram.frequencies.value
        )

    def test_immutability(self, glitchy_spectrogram):
        original_data = glitchy_spectrogram.value.copy()
        glitchy_spectrogram.clean(method="threshold")
        np.testing.assert_array_equal(glitchy_spectrogram.value, original_data)

    def test_default_window_size(self, base_spectrogram):
        """Default window_size = max(3, ntimes // 4)."""
        result = base_spectrogram.clean(method="rolling_median")
        assert isinstance(result, Spectrogram)

    def test_combined_return_mask(self, glitchy_spectrogram):
        result, mask = glitchy_spectrogram.clean(
            method="combined", window_size=5, return_mask=True
        )
        assert mask.shape == glitchy_spectrogram.shape
        # At least the glitch pixels should be masked
        assert np.any(mask)
