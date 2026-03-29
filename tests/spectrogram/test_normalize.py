"""Tests for Spectrogram.normalize() method."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram


@pytest.fixture
def sample_spectrogram():
    """Create a simple spectrogram (10 time bins x 5 freq bins)."""
    np.random.seed(42)
    data = np.abs(np.random.randn(10, 5)) + 1.0  # positive values
    return Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


class TestNormalizeMethod:
    """Tests for Spectrogram.normalize()."""

    def test_snr_default(self, sample_spectrogram):
        """SNR normalization divides by median along time axis."""
        result = sample_spectrogram.normalize(method="snr")
        expected_ref = np.median(sample_spectrogram.value, axis=0)
        expected = sample_spectrogram.value / expected_ref[np.newaxis, :]
        np.testing.assert_allclose(result.value, expected)

    def test_median(self, sample_spectrogram):
        """Median normalization is equivalent to SNR."""
        snr = sample_spectrogram.normalize(method="snr")
        med = sample_spectrogram.normalize(method="median")
        np.testing.assert_allclose(snr.value, med.value)

    def test_mean(self, sample_spectrogram):
        """Mean normalization divides by mean along time axis."""
        result = sample_spectrogram.normalize(method="mean")
        expected_ref = np.mean(sample_spectrogram.value, axis=0)
        expected = sample_spectrogram.value / expected_ref[np.newaxis, :]
        np.testing.assert_allclose(result.value, expected)

    def test_percentile(self, sample_spectrogram):
        """Percentile normalization with custom percentile."""
        result = sample_spectrogram.normalize(method="percentile", percentile=75.0)
        expected_ref = np.percentile(sample_spectrogram.value, 75.0, axis=0)
        expected = sample_spectrogram.value / expected_ref[np.newaxis, :]
        np.testing.assert_allclose(result.value, expected)

    def test_reference(self, sample_spectrogram):
        """Reference normalization with user-provided spectrum."""
        ref = np.ones(5) * 2.0
        result = sample_spectrogram.normalize(method="reference", reference=ref)
        expected = sample_spectrogram.value / 2.0
        np.testing.assert_allclose(result.value, expected)

    def test_snr_with_reference(self, sample_spectrogram):
        """SNR with user-provided reference uses that instead of median."""
        ref = np.ones(5) * 3.0
        result = sample_spectrogram.normalize(method="snr", reference=ref)
        expected = sample_spectrogram.value / 3.0
        np.testing.assert_allclose(result.value, expected)

    def test_reference_required(self, sample_spectrogram):
        """Method='reference' without reference raises ValueError."""
        with pytest.raises(ValueError, match="reference must be provided"):
            sample_spectrogram.normalize(method="reference")

    def test_unknown_method(self, sample_spectrogram):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            sample_spectrogram.normalize(method="invalid")

    def test_result_is_new_object(self, sample_spectrogram):
        """Normalization returns a new Spectrogram (immutability)."""
        result = sample_spectrogram.normalize()
        assert result is not sample_spectrogram

    def test_unit_is_dimensionless(self, sample_spectrogram):
        """Normalized spectrogram has dimensionless unit."""
        result = sample_spectrogram.normalize()
        assert result.unit == u.dimensionless_unscaled

    def test_metadata_preserved(self, sample_spectrogram):
        """Times, frequencies, channel, epoch are preserved."""
        result = sample_spectrogram.normalize()
        np.testing.assert_array_equal(result.times.value, sample_spectrogram.times.value)
        np.testing.assert_array_equal(
            result.frequencies.value, sample_spectrogram.frequencies.value
        )

    def test_shape_preserved(self, sample_spectrogram):
        """Output shape matches input."""
        result = sample_spectrogram.normalize()
        assert result.shape == sample_spectrogram.shape

    def test_zero_column_becomes_nan(self):
        """Zero-valued frequency bins become NaN, not inf."""
        data = np.ones((5, 3))
        data[:, 1] = 0.0  # second freq bin is all zeros
        spec = Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)
        result = spec.normalize(method="median")
        assert np.all(np.isnan(result.value[:, 1]))
        assert np.all(np.isfinite(result.value[:, 0]))
        assert np.all(np.isfinite(result.value[:, 2]))

    def test_original_unchanged(self, sample_spectrogram):
        """Original spectrogram data is not modified."""
        original_data = sample_spectrogram.value.copy()
        sample_spectrogram.normalize()
        np.testing.assert_array_equal(sample_spectrogram.value, original_data)
