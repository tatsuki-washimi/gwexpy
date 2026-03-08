"""Tests for gwexpy.noise.peaks - Peak generation functions."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.noise.peaks import gaussian_line, lorentzian_line, voigt_line


@pytest.fixture
def frequencies():
    return np.arange(0.1, 100.0, 0.1)


class TestLorentzianLine:
    def test_peak_at_f0(self, frequencies):
        fs = lorentzian_line(50.0, 1.0, Q=10.0, frequencies=frequencies)
        peak_idx = np.argmax(fs.value)
        assert abs(frequencies[peak_idx] - 50.0) < 0.2

    def test_peak_amplitude(self):
        freqs = np.arange(0.1, 200.0, 0.01)
        fs = lorentzian_line(100.0, 5.0, Q=10.0, frequencies=freqs)
        assert fs.value.max() == pytest.approx(5.0, rel=0.01)

    def test_gamma_parameter(self, frequencies):
        fs = lorentzian_line(50.0, 1.0, gamma=2.5, frequencies=frequencies)
        assert len(fs) == len(frequencies)

    def test_q_or_gamma_required(self, frequencies):
        with pytest.raises(ValueError, match="Either"):
            lorentzian_line(50.0, 1.0, frequencies=frequencies)

    def test_quantity_amplitude(self, frequencies):
        amp = 3.0 * u.Unit("pT/Hz(1/2)")
        fs = lorentzian_line(50.0, amp, Q=10.0, frequencies=frequencies)
        assert fs.unit == amp.unit

    def test_unit_kwarg(self, frequencies):
        fs = lorentzian_line(50.0, 1.0, Q=10.0, frequencies=frequencies, unit="m")
        assert str(fs.unit) == "m"


class TestGaussianLine:
    def test_peak_at_f0(self, frequencies):
        fs = gaussian_line(50.0, 1.0, sigma=2.0, frequencies=frequencies)
        peak_idx = np.argmax(fs.value)
        assert abs(frequencies[peak_idx] - 50.0) < 0.2

    def test_peak_amplitude(self):
        freqs = np.arange(0.1, 200.0, 0.01)
        fs = gaussian_line(100.0, 5.0, sigma=2.0, frequencies=freqs)
        assert fs.value.max() == pytest.approx(5.0, rel=0.01)

    def test_narrow_vs_wide(self, frequencies):
        narrow = gaussian_line(50.0, 1.0, sigma=1.0, frequencies=frequencies)
        wide = gaussian_line(50.0, 1.0, sigma=5.0, frequencies=frequencies)
        # Narrow should drop faster away from peak
        far_idx = np.argmin(np.abs(frequencies - 60.0))
        assert narrow.value[far_idx] < wide.value[far_idx]


class TestVoigtLine:
    def test_peak_at_f0(self, frequencies):
        fs = voigt_line(50.0, 1.0, sigma=2.0, gamma=1.0, frequencies=frequencies)
        peak_idx = np.argmax(fs.value)
        assert abs(frequencies[peak_idx] - 50.0) < 0.5

    def test_peak_amplitude(self):
        freqs = np.arange(0.1, 200.0, 0.01)
        fs = voigt_line(100.0, 5.0, sigma=2.0, gamma=1.0, frequencies=freqs)
        assert fs.value.max() == pytest.approx(5.0, rel=0.02)

    def test_returns_frequency_series(self, frequencies):
        fs = voigt_line(50.0, 1.0, sigma=2.0, gamma=1.0, frequencies=frequencies)
        from gwexpy.frequencyseries import FrequencySeries
        assert isinstance(fs, FrequencySeries)
