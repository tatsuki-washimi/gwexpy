
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

def test_lock_in_constant_freq():
    amp = 1.0
    phi0 = np.pi/4
    f0 = 30.0
    sample_rate = 4096.0
    duration = 600.0
    stride = 60.0

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    data = TimeSeries(amp * np.cos(2 * np.pi * f0 * t + phi0), unit='', times=t)

    mag, ph = data.lock_in(f0=f0, stride=stride, output='amp_phase', deg=False, singlesided=True)

    # Assert amplitude
    assert np.allclose(mag.value, amp, rtol=1e-4)

    # Assert phase unit and value
    assert ph.unit == 'rad'
    assert np.allclose(ph.value, phi0, rtol=1e-4)

def test_singlesided_scaling():
    amp = 1.0
    f0 = 30.0
    sample_rate = 4096.0
    duration = 10.0
    stride = 1.0

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    phases = 2 * np.pi * f0 * t
    data = TimeSeries(amp * np.cos(phases), unit='', times=t)

    out_double = data.lock_in(phase=phases, stride=stride, output='complex', singlesided=False)
    out_single = data.lock_in(phase=phases, stride=stride, output='complex', singlesided=True)

    # singlesided=False -> 0.5 * amp
    assert np.isclose(np.median(np.abs(out_double.value)), 0.5 * amp, rtol=1e-2)
    # singlesided=True -> amp
    assert np.isclose(np.median(np.abs(out_single.value)), amp, rtol=1e-2)

def test_fdot_phase_model():
    amp = 1.0
    phi0 = 0.5
    f0 = 30.0
    fdot = 1e-4
    sample_rate = 4096.0
    duration = 100.0
    stride = 10.0

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    # Phase = 2pi * (f0*t + 0.5*fdot*t^2) + phi0
    true_phase = 2 * np.pi * (f0 * t + 0.5 * fdot * t**2) + phi0
    data = TimeSeries(amp * np.cos(true_phase), unit='', times=t)

    # Demodulate with exact parameters
    out = data.lock_in(f0=f0, fdot=fdot, stride=stride, output='complex', singlesided=True)

    assert np.allclose(np.abs(out.value), amp, rtol=1e-2)
    assert np.allclose(np.angle(out.value), phi0, rtol=1e-2)

def test_phase_epoch():
    amp = 1.0
    f0 = 10.0
    sample_rate = 1024.0
    duration = 10.0
    stride = 1.0

    # Absolute times: shift by 1000s
    t_rel = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    t_abs = t_rel + 1000.0

    # Case 1: Simple aligned
    data = TimeSeries(amp * np.cos(2 * np.pi * f0 * t_rel), unit='', times=t_abs)
    # If phase_epoch defaults to t0 (1000), then t_rel_model = t_abs - 1000 = t_rel_signal. Match.
    out1 = data.lock_in(f0=f0, stride=stride, output='complex', singlesided=True)
    assert np.allclose(np.abs(out1.value), amp, rtol=1e-3)

    # Case 2: Explicit epoch
    out2 = data.lock_in(f0=f0, phase_epoch=1000.0, stride=stride, output='complex', singlesided=True)
    assert np.allclose(np.abs(out2.value), amp, rtol=1e-3)

    # Case 3: Shifted epoch with phase sensitivity
    # f=10.5 Hz (integer half-cycles in 1s stride) to avoid leakage
    f_test = 10.5
    data_test = TimeSeries(amp * np.cos(2 * np.pi * f_test * t_rel), unit='', times=t_abs)

    # Aligned: epoch=1000
    out_aligned = data_test.lock_in(f0=f_test, phase_epoch=1000.0, stride=stride, output='complex', singlesided=True)
    assert np.allclose(np.angle(out_aligned.value), 0.0, atol=1e-3)

    # Shifted: epoch=999.0
    # t_rel_model = t_abs - 999 = (t_abs - 1000) + 1 = t_rel_signal + 1
    # Model phase leads signal phase by 2*pi*f_test * 1 = 21*pi = pi (mod 2pi)
    # Mixed = Signal * exp(-j Model) = exp(j phi_s) * exp(-j (phi_s + pi)) = exp(-j pi) = -1
    # Angle should be pi or -pi
    out_shifted = data_test.lock_in(f0=f_test, phase_epoch=999.0, stride=stride, output='complex', singlesided=True)
    angle_shifted = np.angle(out_shifted.value)
    assert np.allclose(np.abs(angle_shifted), np.pi, atol=1e-3)

def test_dict_methods():
    ts1 = TimeSeries(np.ones(100), dt=1/100, name='a')
    ts2 = TimeSeries(np.ones(100)*2, dt=1/100, name='b')
    d = TimeSeriesDict({'a': ts1, 'b': ts2})

    di, dq = d.lock_in(f0=0.0, stride=1.0, singlesided=False, output='iq')

    assert isinstance(di, TimeSeriesDict)
    assert di['a'].value[0] == 1.0
    assert di['b'].value[0] == 2.0
    assert dq['a'].value[0] == 0.0

def test_irregular_mix_down():
    t = np.array([0.0, 0.1, 0.5, 0.6])
    data = TimeSeries(np.ones(4), times=t)

    # f0=10.0
    # phase should be 2pi * 10.0 * t (since epoch defaults to t[0]=0)
    out = data.mix_down(f0=10.0)

    # Check if out has times
    assert np.allclose(out.times.value, t)

    # Check value: 1.0 * exp(-j * 2pi * 10 * t)
    expected = np.exp(-1j * 2 * np.pi * 10.0 * t)
    assert np.allclose(out.value, expected)
