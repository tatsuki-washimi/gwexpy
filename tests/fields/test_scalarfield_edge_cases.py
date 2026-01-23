"""Boundary and edge case tests for ScalarField."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField


def test_singleton_axis_fft_raises():
    """Verify FFT on singleton axis raises ValueError."""
    data = np.ones((1, 4, 4, 4))
    field = ScalarField(data, axis0=np.array([0.0]) * u.s)

    with pytest.raises(ValueError, match="requires.*axis length >= 2"):
        field.fft_time()


def test_non_contiguous_slice_fft():
    """Verify FFT works on non-contiguous slices (via copy internally if needed)."""
    data = np.random.randn(16, 4, 4, 4)
    field = ScalarField(data, axis0=np.arange(16) * 0.1 * u.s)

    # Non-contiguous slice: every 2nd sample
    sliced = field[::2, :, :, :]
    assert sliced.shape[0] == 8

    # FFT should work
    freq = sliced.fft_time()
    assert freq.shape[0] == 8 // 2 + 1
    assert_allclose(freq._axis0_index.value, np.fft.rfftfreq(8, d=0.2))


def test_degenerate_fft_sizes():
    """Verify FFT on small prime sizes or minimal length 2."""
    for n in [2, 3, 5, 7]:
        data = np.random.randn(n, 2, 2, 2)
        field = ScalarField(data, axis0=np.arange(n) * 0.1 * u.s)
        freq = field.fft_time()
        assert freq.shape[0] == n // 2 + 1


def test_nan_inf_handling():
    """Verify FFT behavior with NaN/Inf."""
    data = np.ones((8, 2, 2, 2))
    data[0, 0, 0, 0] = np.nan
    field = ScalarField(data, axis0=np.arange(8) * 0.1 * u.s)

    freq = field.fft_time()
    # FFT of NaN data should result in all NaNs in frequency domain (standard behavior)
    assert np.all(np.isnan(freq.value[:, 0, 0, 0]))
    # Other spatial cells should be fine if no NaN entered them
    assert not np.any(np.isnan(freq.value[:, 1, 0, 0]))


def test_empty_minimal_metadata():
    """Verify construction with minimal metadata."""
    data = np.ones((2, 2, 2, 2))
    field = ScalarField(data)
    assert field.axis_names == ("t", "x", "y", "z")
    assert field.axis0_domain == "time"
    # Default unit may be dimensionless; accept either explicit None or dimensionless.
    assert field.unit is None or field.unit.is_equivalent(u.dimensionless_unscaled)
