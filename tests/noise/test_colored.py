"""Tests for gwexpy.noise.colored - Power-law noise generation."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.noise.colored import pink_noise, power_law, red_noise, white_noise


@pytest.fixture
def frequencies():
    return np.arange(1.0, 101.0, 1.0)


class TestPowerLaw:
    def test_basic(self, frequencies):
        fs = power_law(1.0, amplitude=1.0, f_ref=1.0, frequencies=frequencies)
        assert len(fs) == len(frequencies)

    def test_exponent_zero_is_flat(self, frequencies):
        fs = power_law(0.0, amplitude=2.0, frequencies=frequencies)
        np.testing.assert_allclose(fs.value, 2.0)

    def test_exponent_positive_decreasing(self, frequencies):
        fs = power_law(1.0, amplitude=1.0, f_ref=1.0, frequencies=frequencies)
        assert fs.value[0] > fs.value[-1]

    def test_quantity_amplitude(self, frequencies):
        amp = 5.0 * u.Unit("1/Hz(1/2)")
        fs = power_law(0.0, amplitude=amp, frequencies=frequencies)
        assert fs.unit == amp.unit
        np.testing.assert_allclose(fs.value, 5.0)

    def test_quantity_frequencies(self):
        freqs = np.arange(1.0, 11.0) * u.Hz
        fs = power_law(0.0, amplitude=1.0, frequencies=freqs)
        assert len(fs) == 10

    def test_f_ref_quantity(self, frequencies):
        fs = power_law(1.0, amplitude=1.0, f_ref=10.0 * u.Hz, frequencies=frequencies)
        assert len(fs) == len(frequencies)

    def test_zero_frequency_positive_exponent(self):
        freqs = np.array([0.0, 1.0, 2.0])
        fs = power_law(1.0, amplitude=1.0, frequencies=freqs)
        assert np.isinf(fs.value[0])

    def test_zero_frequency_negative_exponent(self):
        freqs = np.array([0.0, 1.0, 2.0])
        fs = power_law(-1.0, amplitude=1.0, frequencies=freqs)
        assert fs.value[0] == 0.0

    def test_no_frequencies_with_kwargs(self):
        fs = power_law(0.0, amplitude=1.0, df=1.0, fmin=1.0, fmax=10.0)
        assert len(fs) == 10

    def test_no_frequencies_raises(self):
        with pytest.raises(ValueError):
            power_law(0.0, amplitude=1.0)


class TestWhiteNoise:
    def test_flat_spectrum(self, frequencies):
        fs = white_noise(3.0, frequencies=frequencies)
        np.testing.assert_allclose(fs.value, 3.0)


class TestPinkNoise:
    def test_decreasing(self, frequencies):
        fs = pink_noise(1.0, frequencies=frequencies)
        assert fs.value[0] > fs.value[-1]


class TestRedNoise:
    def test_steeper_than_pink(self, frequencies):
        pink = pink_noise(1.0, f_ref=1.0, frequencies=frequencies)
        red = red_noise(1.0, f_ref=1.0, frequencies=frequencies)
        # Red noise drops faster at high frequencies
        ratio_high = red.value[-1] / red.value[0]
        ratio_pink = pink.value[-1] / pink.value[0]
        assert ratio_high < ratio_pink
