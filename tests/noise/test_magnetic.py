"""Tests for gwexpy.noise.magnetic - Geomagnetic noise models."""

import numpy as np
import pytest

from gwexpy.noise.magnetic import geomagnetic_background, schumann_resonance


@pytest.fixture
def frequencies():
    return np.arange(1.0, 50.0, 0.1)


class TestSchumannResonance:
    def test_basic(self, frequencies):
        fs = schumann_resonance(frequencies=frequencies)
        assert len(fs) == len(frequencies)
        assert np.all(fs.value >= 0)

    def test_peaks_near_modes(self, frequencies):
        fs = schumann_resonance(frequencies=frequencies)
        # First Schumann resonance is around 7.83 Hz
        idx_peak = np.argmax(fs.value)
        assert 5.0 < frequencies[idx_peak] < 10.0

    def test_custom_modes(self, frequencies):
        modes = [(10.0, 5.0, 2.0)]
        fs = schumann_resonance(frequencies=frequencies, modes=modes)
        peak_idx = np.argmax(fs.value)
        assert abs(frequencies[peak_idx] - 10.0) < 1.0

    def test_amplitude_scale(self, frequencies):
        fs1 = schumann_resonance(frequencies=frequencies, amplitude_scale=1.0)
        fs2 = schumann_resonance(frequencies=frequencies, amplitude_scale=2.0)
        assert fs2.value.max() > fs1.value.max()

    def test_no_frequencies_raises(self):
        with pytest.raises(ValueError, match="frequencies"):
            schumann_resonance()


class TestGeomagneticBackground:
    def test_basic(self, frequencies):
        fs = geomagnetic_background(frequencies=frequencies)
        assert len(fs) == len(frequencies)

    def test_decreasing(self, frequencies):
        fs = geomagnetic_background(frequencies=frequencies)
        # 1/f noise should decrease with frequency
        assert fs.value[0] > fs.value[-1]

    def test_custom_amplitude(self, frequencies):
        fs = geomagnetic_background(frequencies=frequencies, amplitude_1hz=100.0)
        assert fs.value[0] == pytest.approx(100.0, rel=0.1)
