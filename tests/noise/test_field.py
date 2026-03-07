"""Tests for gwexpy.noise.field - 4D field noise generation."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.noise.field import gaussian, plane_wave


class TestGaussian:
    def test_basic_shape(self):
        sf = gaussian(shape=(10, 5, 5, 5), seed=42)
        assert sf.value.shape == (10, 5, 5, 5)

    def test_seed_reproducibility(self):
        sf1 = gaussian(shape=(10, 3, 3, 3), seed=123)
        sf2 = gaussian(shape=(10, 3, 3, 3), seed=123)
        np.testing.assert_array_equal(sf1.value, sf2.value)

    def test_different_seeds(self):
        sf1 = gaussian(shape=(10, 3, 3, 3), seed=1)
        sf2 = gaussian(shape=(10, 3, 3, 3), seed=2)
        assert not np.array_equal(sf1.value, sf2.value)

    def test_mean_and_std(self):
        sf = gaussian(shape=(1000, 10, 10, 10), mean=5.0, std=2.0, seed=42)
        assert sf.value.mean() == pytest.approx(5.0, abs=0.1)
        assert sf.value.std() == pytest.approx(2.0, abs=0.1)

    def test_unit(self):
        sf = gaussian(shape=(10, 3, 3, 3), unit=u.m, seed=42)
        assert sf.unit == u.m

    def test_custom_axes(self):
        axes = {
            "axis0": np.arange(5) * u.s,
            "axis1": np.arange(4) * u.m,
            "axis2": np.arange(3) * u.m,
            "axis3": np.arange(2) * u.m,
        }
        sf = gaussian(axes=axes, seed=42)
        assert sf.value.shape == (5, 4, 3, 2)


class TestPlaneWave:
    def test_basic_shape(self):
        sf = plane_wave(
            frequency=1.0 * u.Hz,
            k_vector=(1.0 / u.m, 0.0 / u.m, 0.0 / u.m),
            shape=(10, 5, 5, 5),
        )
        assert sf.value.shape == (10, 5, 5, 5)

    def test_amplitude(self):
        sf = plane_wave(
            frequency=1.0 * u.Hz,
            k_vector=(0.0 / u.m, 0.0 / u.m, 0.0 / u.m),
            amplitude=3.0,
            shape=(100, 3, 3, 3),
            sample_rate=10.0 * u.Hz,
        )
        # With k=0, the wave is cos(2*pi*f*t), amplitude bounded by 3.0
        assert sf.value.max() == pytest.approx(3.0, abs=0.1)

    def test_bounded(self):
        sf = plane_wave(
            frequency=2.0 * u.Hz,
            k_vector=(1.0 / u.m, 0.0 / u.m, 0.0 / u.m),
            amplitude=1.0,
            shape=(50, 5, 5, 5),
        )
        assert np.all(np.abs(sf.value) <= 1.0 + 1e-10)
