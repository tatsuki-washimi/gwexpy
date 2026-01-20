import numpy as np
import pytest
from astropy import units as u
from scipy import signal as scipy_signal

from gwexpy.noise.wave import (
    chirp,
    colored,
    exponential,
    gaussian,
    impulse,
    pink_noise,
    red_noise,
    sawtooth,
    sine,
    square,
    step,
    triangle,
    uniform,
    white_noise,
)
from gwexpy.timeseries import TimeSeries

# Common parameters for tests
DURATION = 2.0
SAMPLE_RATE = 100.0
SIZE = int(DURATION * SAMPLE_RATE)

@pytest.fixture
def params():
    return {
        "duration": DURATION,
        "sample_rate": SAMPLE_RATE,
        "name": "TEST_WAVE",
        "channel": "X1:TEST",
        "t0": 10.0,
    }

def verify_metadata(ts, params):
    """Helper to verify TimeSeries metadata."""
    assert isinstance(ts, TimeSeries)
    assert len(ts) == int(params["duration"] * params["sample_rate"])
    assert ts.sample_rate.value == pytest.approx(params["sample_rate"])
    assert ts.t0.value == pytest.approx(params["t0"])
    assert ts.name == params["name"]
    # Check if channel matches (gwpy Channel or string)
    if hasattr(ts.channel, "name"):
        assert ts.channel.name == params["channel"]
    else:
        assert str(ts.channel) == params["channel"]

def test_sine(params):
    freq = 10.0
    amp = 2.0
    ts = sine(frequency=freq, amplitude=amp, **params)
    verify_metadata(ts, params)
    assert ts.max().value <= amp + 1e-9
    assert ts.min().value >= -amp - 1e-9
    # Basic check: value at t=0 should be 0 (phase=0, sin(0)=0)
    assert ts[0].value == pytest.approx(0.0)

def test_square(params):
    freq = 10.0
    amp = 1.5
    ts = square(frequency=freq, amplitude=amp, duty=0.5, **params)
    verify_metadata(ts, params)
    assert np.all(np.abs(ts.value) <= amp)

def test_sawtooth(params):
    freq = 5.0
    ts = sawtooth(frequency=freq, **params)
    verify_metadata(ts, params)

def test_triangle(params):
    freq = 5.0
    ts = triangle(frequency=freq, **params)
    verify_metadata(ts, params)

def test_chirp(params):
    f0 = 1.0
    f1 = 10.0
    t1 = DURATION
    ts = chirp(f0=f0, f1=f1, t1=t1, **params)
    verify_metadata(ts, params)

def test_gaussian(params):
    mean = 1.0
    std = 0.5
    ts = gaussian(mean=mean, std=std, seed=42, **params)
    verify_metadata(ts, params)
    # Statistical check
    assert np.mean(ts.value) == pytest.approx(mean, abs=0.2)
    assert np.std(ts.value) == pytest.approx(std, abs=0.2)

def test_uniform(params):
    low = -1.0
    high = 1.0
    ts = uniform(low=low, high=high, seed=42, **params)
    verify_metadata(ts, params)
    assert ts.min().value >= low
    assert ts.max().value <= high

def test_step(params):
    t_step = 0.5
    amp = 5.0
    ts = step(t_step=t_step, amplitude=amp, **params)
    verify_metadata(ts, params)
    # Check values before and after step
    t_idx = int(0.5 * params["sample_rate"])
    assert ts[0].value == 0.0
    assert ts[t_idx + 1].value == amp # +1 to be safe past step

def test_impulse(params):
    t_imp = 0.5
    amp = 10.0
    ts = impulse(t_impulse=t_imp, amplitude=amp, **params)
    verify_metadata(ts, params)
    # Check that sum/integral approximates amp? No, impulse is discrete here usually?
    # In wave.py implementation: impulse sets ONE sample to value 'amplitude'?
    # Or is it dirac delta approximation?
    # Let's assume one sample.
    assert np.count_nonzero(ts.value) == 1
    assert ts.max().value == amp

def test_exponential(params):
    tau = 0.5
    amp = 1.0
    ts = exponential(tau=tau, amplitude=amp, **params)
    verify_metadata(ts, params)
    assert ts[0].value == amp
    # Check decay
    t_tau_idx = int(tau * params["sample_rate"])
    expected = amp * np.exp(-1)
    if t_tau_idx < len(ts):
        assert ts[t_tau_idx].value == pytest.approx(expected, rel=0.1)

def test_colored(params):
    exponent = 1 # Pink
    ts = colored(exponent=exponent, **params)
    verify_metadata(ts, params)

def test_noise_aliases(params):
    ts_white = white_noise(**params)
    verify_metadata(ts_white, params)

    ts_pink = pink_noise(**params)
    verify_metadata(ts_pink, params)

    ts_red = red_noise(**params)
    verify_metadata(ts_red, params)

def test_unit_handling(params):
    # Verify unit is passed correctly
    unit = 'm'
    ts = sine(frequency=10, unit=unit, **params)
    assert ts.unit == u.m
