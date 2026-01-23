"""Unit consistency and validation tests for ScalarField."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField


def make_basic_field():
    data = np.ones((8, 4, 4, 4))
    t = np.arange(8) * 0.1 * u.s
    x = np.arange(4) * 1.0 * u.m
    return ScalarField(
        data,
        unit=u.V,
        axis0=t,
        axis1=x,
        axis2=x.copy(),
        axis3=x.copy(),
        axis0_domain="time",
    )


def test_unit_propagation_fft_time():
    """Verify units are propagated and scaled during time FFT."""
    field = make_basic_field()
    freq = field.fft_time()

    # Data unit should be preserved (standard normalization preserves amplitude unit)
    assert freq.unit == u.V
    # Axis unit should be Hz
    assert freq._axis0_index.unit == u.Hz


def test_unit_propagation_fft_space():
    """Verify units are propagated during spatial FFT."""
    field = make_basic_field()
    k_field = field.fft_space(axes=["x"])

    assert k_field.unit == u.V
    assert k_field._axis1_index.unit == (1 / u.m)


def test_invalid_unit_combinations_construction():
    """Verify construction fails with physically inconsistent units."""
    data = np.ones((4, 2, 2, 2))

    # Time domain with distance units
    with pytest.raises(ValueError, match="domain 'time' expects units equivalent to"):
        ScalarField(data, axis0=np.arange(4) * u.m, axis0_domain="time")

    # Real space domain with frequency units
    with pytest.raises(ValueError, match="expects units equivalent to"):
        ScalarField(data, axis1=np.arange(2) * u.Hz, space_domain="real")

    # K space domain with distance units
    with pytest.raises(ValueError, match="expects units equivalent to"):
        ScalarField(data, axis1=np.arange(2) * u.m, space_domain="k")


def test_fft_time_complex_input_raises():
    """Verify complex input to rfft-based fft_time raises TypeError."""
    field = make_basic_field()
    complex_field = field + 1j * field

    with pytest.raises(TypeError, match="requires real-valued input"):
        complex_field.fft_time()


def test_arithmetic_unit_preservation():
    """Verify arithmetic operations preserve or check units."""
    f1 = make_basic_field()
    f2 = make_basic_field() * 2.0

    # Addition
    res = f1 + f2
    assert res.unit == u.V
    assert_allclose(res.value, 3.0)

    # Mismatched units addition
    f3 = make_basic_field()
    f3._unit = u.m  # Force hack for test
    with pytest.raises(u.UnitConversionError):
        f1 + f3


def test_scaling_normalization_fft_units():
    """Verify FFT scaling respects physical units (e.g., PSD-like scaling if requested).
    Note: fft_time matches GWpy's .fft() which is amplitude scaling.
    """
    field = make_basic_field() * (1.0 * u.V)
    freq = field.fft_time()

    # Amplitude of a constant 1.0 signal should be 1.0 at DC
    assert_allclose(freq.value[0, 0, 0, 0], 1.0)
    assert freq.unit == u.V**2
