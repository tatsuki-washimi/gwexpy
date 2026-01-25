"""Tests for colored noise, peaks, magnetic models, and wave synthesis."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.noise.colored import pink_noise, power_law, red_noise, white_noise
from gwexpy.noise.magnetic import geomagnetic_background, schumann_resonance
from gwexpy.noise.peaks import gaussian_line, lorentzian_line, voigt_line
from gwexpy.noise.wave import from_asd
from gwexpy.timeseries import TimeSeries


def test_power_law_exponent_zero_constant():
    freqs = np.array([1.0, 2.0, 4.0])
    asd = power_law(0.0, amplitude=2.5, frequencies=freqs)
    np.testing.assert_allclose(asd.value, 2.5)


def test_power_law_zero_frequency_rules():
    freqs = np.array([0.0, 1.0])
    pos = power_law(1.0, amplitude=1.0, frequencies=freqs)
    neg = power_law(-1.0, amplitude=1.0, frequencies=freqs)
    assert np.isinf(pos.value[0])
    assert neg.value[0] == 0.0


def test_colored_wrappers_match_power_law():
    freqs = np.array([1.0, 2.0, 4.0])
    amp = 3.0
    white = white_noise(amp, frequencies=freqs)
    pink = pink_noise(amp, frequencies=freqs)
    red = red_noise(amp, frequencies=freqs)

    np.testing.assert_allclose(
        white.value, power_law(0.0, amp, frequencies=freqs).value
    )
    np.testing.assert_allclose(pink.value, power_law(0.5, amp, frequencies=freqs).value)
    np.testing.assert_allclose(red.value, power_law(1.0, amp, frequencies=freqs).value)


def test_lorentzian_requires_q_or_gamma():
    with pytest.raises(ValueError):
        lorentzian_line(10.0, amplitude=1.0)


def test_lorentzian_peak_at_f0():
    f0 = 10.0
    amp = 2.0
    asd = lorentzian_line(f0, amplitude=amp, Q=5.0, frequencies=np.array([f0]))
    assert asd.value[0] == pytest.approx(amp)


def test_gaussian_peak_at_f0():
    f0 = 12.0
    amp = 1.5
    asd = gaussian_line(f0, amplitude=amp, sigma=0.5, frequencies=np.array([f0]))
    assert asd.value[0] == pytest.approx(amp)


def test_voigt_peak_at_f0():
    pytest.importorskip("scipy")
    f0 = 15.0
    amp = 4.0
    asd = voigt_line(
        f0, amplitude=amp, sigma=0.5, gamma=0.3, frequencies=np.array([f0])
    )
    assert asd.value[0] == pytest.approx(amp)


def test_schumann_requires_frequencies():
    with pytest.raises(ValueError):
        schumann_resonance()


def test_schumann_psd_sums_incoherently():
    freqs = np.array([10.0])
    modes = [(10.0, 10.0, 2.0), (10.0, 10.0, 2.0)]
    asd = schumann_resonance(frequencies=freqs, modes=modes)
    assert asd.value[0] == pytest.approx(2.0 * np.sqrt(2))


def test_geomagnetic_default_unit_and_tesla_conversion():
    freqs = np.array([1.0])
    # Default behavior treats small amplitudes as Tesla inputs and converts to pT.
    asd = geomagnetic_background(frequencies=freqs, amplitude_1hz=5e-12, exponent=0.0)
    assert asd.value[0] == pytest.approx(5.0)
    assert asd.unit.is_equivalent(u.Unit("pT / Hz^(1/2)"))


def test_from_asd_returns_timeseries_metadata():
    freqs = np.linspace(1.0, 4.0, 4)
    asd = FrequencySeries(
        np.ones_like(freqs),
        frequencies=freqs,
        unit=u.m / (u.Hz**0.5),
        name="TEST_ASD",
    )

    ts = from_asd(
        asd,
        duration=2.0,
        sample_rate=4.0,
        t0=1.25,
        rng=np.random.default_rng(0),
    )

    assert isinstance(ts, TimeSeries)
    assert len(ts) == 8
    assert ts.t0.value == pytest.approx(1.25)
    assert ts.unit == asd.unit * (u.Hz**0.5)
