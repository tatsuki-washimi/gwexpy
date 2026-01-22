"""Domain and unit propagation tests for ScalarField."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField


def make_scalar_field(shape=(8, 4, 4, 4)):
    data = np.ones(shape)
    t = np.arange(shape[0]) * 0.1 * u.s
    x = np.arange(shape[1]) * 1.0 * u.m
    y = np.arange(shape[2]) * 1.0 * u.m
    z = np.arange(shape[3]) * 1.0 * u.m
    return ScalarField(
        data,
        unit=u.V,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
    )


def test_invalid_axis0_units_rejected():
    """axis0_domain='time' must have time-equivalent units."""
    data = np.zeros((4, 2, 2, 2))
    with pytest.raises(ValueError, match="Axis0 domain 'time' expects units"):
        ScalarField(data, axis0=np.arange(4) * u.Hz, axis0_domain="time")


def test_invalid_space_units_rejected():
    """Spatial domains enforce position/k units."""
    data = np.zeros((4, 2, 2, 2))
    bad_x = np.arange(2) * u.Hz
    with pytest.raises(ValueError, match="expects units equivalent"):
        ScalarField(
            data,
            axis0=np.arange(4) * u.s,
            axis1=bad_x,
            axis2=np.arange(2) * u.m,
            axis3=np.arange(2) * u.m,
            space_domain="real",
        )


def test_fft_time_updates_domain_and_units():
    field = make_scalar_field()
    freq = field.fft_time()
    assert freq.axis0_domain == "frequency"
    assert freq._axis0_index.unit.is_equivalent(u.Hz)


def test_fft_space_partial_axis_domains():
    field = make_scalar_field()
    k_field = field.fft_space(axes=["x", "z"])
    # Only transformed axes become k-domain with k-prefixed names
    assert k_field.space_domains["kx"] == "k"
    assert k_field.space_domains["y"] == "real"
    assert k_field.space_domains["kz"] == "k"
    assert k_field.axis_names[1:] == ("kx", "y", "kz")


def test_ifft_space_restores_real_domains():
    field = make_scalar_field()
    k_field = field.fft_space(axes=["x"])
    restored = k_field.ifft_space(axes=["kx"])
    assert restored.space_domains["x"] == "real"
    assert restored.axis_names[1] == "x"
