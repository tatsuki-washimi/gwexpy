"""
Tests for TimeSeries.demodulate and extended lock_in functionality.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gwexpy.timeseries import TimeSeries


def test_demodulate_basic():
    """Test TimeSeries.demodulate for basic sinusoid recovery."""
    fs = 1024.0
    duration = 10.0
    t = np.arange(0, duration, 1 / fs)
    f0 = 30.0
    amp_in = 2.5
    phase_deg_in = 45.0
    phase_rad_in = np.radians(phase_deg_in)

    data = amp_in * np.cos(2 * np.pi * f0 * t + phase_rad_in)
    ts = TimeSeries(data, sample_rate=fs, t0=0, unit="V")

    # Test 1: mag/ph trends (default)
    mag, ph = ts.demodulate(f0, stride=1.0, deg=True)
    assert_allclose(mag.value, amp_in, rtol=1e-3)
    assert_allclose(ph.value, phase_deg_in, rtol=1e-3)
    assert ph.unit == "deg"

    # Test 2: complex exponential (exp=True)
    outc = ts.demodulate(f0, stride=1.0, exp=True)
    expected_complex = amp_in * np.exp(1j * phase_rad_in)
    assert_allclose(outc.value, expected_complex, rtol=1e-3)


def test_demodulate_phase_offset():
    """Test demodulate with an initial phase offset."""
    fs = 1024.0
    t = np.arange(0, 5, 1 / fs)
    f0 = 20.0
    # Signal: cos(wt + pi/4)
    # Demodulate with ref phase: pi/4
    # Result should have 0 phase.
    data = np.cos(2 * np.pi * f0 * t + np.pi / 4)
    ts = TimeSeries(data, sample_rate=fs)

    mag, ph = ts.demodulate(f0, stride=1.0, phase=np.pi / 4, deg=False)
    assert_allclose(mag.value, 1.0, rtol=1e-3)
    assert_allclose(ph.value, 0.0, atol=1e-7)


def test_lock_in_lpf_mode_recovery():
    """Test lock_in LPF mode recovers the signal correctly."""
    fs = 1024.0
    duration = 5.0
    t = np.arange(0, duration, 1 / fs)
    f0 = 50.0
    amp_in = 1.2
    phase_deg_in = -30.0

    data = amp_in * np.cos(2 * np.pi * f0 * t + np.radians(phase_deg_in))
    ts = TimeSeries(data, sample_rate=fs)

    # LPF mode (bandwidth=10 Hz)
    # Use output='complex'
    res = ts.lock_in(f0=f0, bandwidth=10.0, output="complex")

    # Ignore transients
    res_crop = res.crop(1.0, 4.0)
    expected_complex = amp_in * np.exp(1j * np.radians(phase_deg_in))

    assert_allclose(res_crop.value, expected_complex, rtol=1e-2)


def test_lock_in_singlesided_false():
    """Test lock_in with singlesided=False (GWpy alignment)."""
    fs = 1024.0
    t = np.arange(0, 1, 1 / fs)
    f0 = 10.0
    data = np.cos(2 * np.pi * f0 * t)
    ts = TimeSeries(data, sample_rate=fs)

    # singlesided=False should return half amplitude (0.5)
    mag, _ = ts.lock_in(f0=f0, stride=1.0, singlesided=False, output="amp_phase")
    assert_allclose(mag.value, 0.5, rtol=1e-3)
