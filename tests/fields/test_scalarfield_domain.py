"""Domain and unit propagation tests for ScalarField."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

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


def test_chained_operations_domain_propagation():
    """Verify domain metadata survives slice -> FFT -> slice."""
    field = make_scalar_field(shape=(16, 8, 8, 8))

    # Slice first
    sliced = field[4:12, 2:6, :, :]
    assert sliced.axis0_domain == "time"

    # FFT
    freq = sliced.fft_time()
    assert freq.axis0_domain == "frequency"
    assert freq.shape[0] == (12 - 4) // 2 + 1  # rfft of 8 samples -> 5

    # Slice frequency domain
    sub_freq = freq[1:4, :, :, :]
    assert sub_freq.axis0_domain == "frequency"
    assert sub_freq.shape[0] == 3
    assert_allclose(sub_freq._axis0_index.value, freq._axis0_index.value[1:4])


def test_repeated_fft_cycles_domain():
    """Verify repeated FFT/IFFT cycles don't corrupt domain state."""
    field = make_scalar_field()

    # Time -> Freq -> Time -> Freq
    f1 = field.fft_time()
    t1 = f1.ifft_time()
    f2 = t1.fft_time()

    assert f2.axis0_domain == "frequency"
    assert f2.axis_names[0] == "f"
    assert_allclose(f2._axis0_index.value, f1._axis0_index.value)


def test_grid_spacing_verification():
    """Explicitly assert grid spacing / resolution is correct after transforms."""
    field = make_scalar_field(shape=(10, 4, 4, 4))
    # Original dt = 0.1 s

    freq = field.fft_time()
    # df = 1 / (n * dt) = 1 / (10 * 0.1) = 1 Hz
    df = freq._axis0_index[1] - freq._axis0_index[0]
    assert_allclose(df.value, 1.0)
    assert df.unit == u.Hz

    k_field = field.fft_space(axes=["x"])
    # Original dx = 1.0 m
    # dk = 2pi / (n * dx) = 2pi / (4 * 1.0) = pi/2 rad/m
    dk = k_field._axis1_index[1] - k_field._axis1_index[0]
    assert_allclose(dk.value, np.pi / 2)


def test_axis_ordering_and_labels_integrity():
    """Verify axis ordering remains 4D and labels are correct."""
    field = make_scalar_field()

    # Transpose is not explicitly supported by ScalarField to maintain 4D axis semantics,
    # but we check that standard operations don't shuffle them.
    freq_k = field.fft_time().fft_space(axes=["x", "y", "z"])

    assert freq_k.axis_names == ("f", "kx", "ky", "kz")
    assert freq_k.ndim == 4
    assert freq_k.shape[0] == field.shape[0] // 2 + 1
    assert freq_k.shape[1:] == field.shape[1:]
