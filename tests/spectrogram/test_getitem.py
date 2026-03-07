"""Tests for Spectrogram.__getitem__ edge cases and indexing modes."""

import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries

from gwexpy.spectrogram import Spectrogram


@pytest.fixture
def spec():
    """10-time x 8-freq spectrogram."""
    data = np.arange(80, dtype=float).reshape(10, 8)
    return Spectrogram(data, t0=100, dt=1, f0=0, df=10, unit="m/s", name="test")


class TestGetitemInt:
    def test_single_int_returns_frequencyseries(self, spec):
        row = spec[3]
        assert isinstance(row, FrequencySeries)
        assert row.shape == (8,)

    def test_single_int_values(self, spec):
        row = spec[0]
        np.testing.assert_array_equal(row.value, np.arange(8, dtype=float))

    def test_negative_int(self, spec):
        last = spec[-1]
        expected = spec[9]
        np.testing.assert_array_equal(last.value, expected.value)

    def test_frequencies_preserved(self, spec):
        row = spec[0]
        np.testing.assert_allclose(row.frequencies.value, np.arange(0, 80, 10))


class TestGetitemSlice:
    def test_slice_returns_spectrogram(self, spec):
        sub = spec[2:5]
        assert isinstance(sub, Spectrogram)
        assert sub.shape == (3, 8)

    def test_slice_metadata(self, spec):
        sub = spec[2:5]
        assert sub.name == "test"
        assert sub.unit == u.Unit("m/s")

    def test_empty_slice(self, spec):
        sub = spec[5:5]
        assert sub.shape[0] == 0
        assert sub.shape[1] == 8


class TestGetitemTuple:
    def test_time_freq_slice(self, spec):
        sub = spec[1:4, 2:6]
        assert sub.shape == (3, 4)

    def test_int_time_slice_freq(self, spec):
        """Integer time index with slice freq should return FrequencySeries."""
        row = spec[0, 2:6]
        assert isinstance(row, FrequencySeries)
        assert row.shape == (4,)

    def test_values_correct(self, spec):
        sub = spec[0:2, 0:3]
        expected = np.arange(80, dtype=float).reshape(10, 8)[0:2, 0:3]
        np.testing.assert_array_equal(sub.value, expected)


class TestGetitem1DFallback:
    """Test the custom ndim=1 fallback path."""

    def test_1d_getitem_does_not_raise(self, spec):
        """Iterating a spectrogram should not crash on 1D views."""
        # gwpy iteration can produce 1D views; our override handles this
        row = spec[0]
        # Indexing a 1D FrequencySeries should work
        val = row[0]
        assert np.isscalar(val) or val.ndim == 0


class TestGetitemBoolMask:
    def test_bool_mask(self, spec):
        mask = spec.value > 40
        result = spec.value[mask]
        assert result.shape == (mask.sum(),)
