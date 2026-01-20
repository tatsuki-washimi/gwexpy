"""
Regression tests for D1 signal processing functions.

D1 = mechanically verifiable against reference implementations (np.angle, scipy.signal, etc.)

Target functions in TimeSeriesSignalMixin:
- radian / degree
- unwrap_phase
- _build_phase_series
- mix_down
- xcorr
- heterodyne / lock_in (conditional)
"""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries

# =============================================================================
# radian / degree tests
# =============================================================================

class TestRadianDegree:
    """Tests for radian() and degree() methods."""

    def test_radian_matches_np_angle(self):
        """radian() should match np.angle(x, deg=False)."""
        # Complex array with various phases
        phases = np.array([0, np.pi/4, np.pi/2, np.pi, -np.pi/2, -np.pi/4])
        data = np.exp(1j * phases)
        ts = TimeSeries(data, dt=0.01, unit='V')

        result = ts.radian()
        expected = np.angle(data, deg=False)

        assert result.unit == 'rad'
        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_degree_matches_np_angle(self):
        """degree() should match np.angle(x, deg=True)."""
        phases = np.array([0, 45, 90, 180, -90, -45]) * np.pi / 180
        data = np.exp(1j * phases)
        ts = TimeSeries(data, dt=0.01, unit='V')

        result = ts.degree()
        expected = np.angle(data, deg=True)

        assert result.unit == 'deg'
        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_radian_unwrap_matches_np_unwrap(self):
        """radian(unwrap=True) should match np.unwrap(np.angle(x))."""
        # Phase that wraps around: 0, pi/2, pi, -pi/2 (wraps), 0, pi/2, ...
        t = np.linspace(0, 4*np.pi, 100)
        data = np.exp(1j * t)
        ts = TimeSeries(data, dt=0.01, unit='V')

        result = ts.radian(unwrap=True)
        expected = np.unwrap(np.angle(data), period=2*np.pi)

        assert result.unit == 'rad'
        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_degree_unwrap_matches_np_unwrap(self):
        """degree(unwrap=True) should match np.unwrap(np.angle(x, deg=True), period=360)."""
        t = np.linspace(0, 720, 100) * np.pi / 180  # 0 to 720 degrees
        data = np.exp(1j * t)
        ts = TimeSeries(data, dt=0.01, unit='V')

        result = ts.degree(unwrap=True)
        expected = np.unwrap(np.angle(data, deg=True), period=360.0)

        assert result.unit == 'deg'
        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_radian_real_signal(self):
        """radian() on real signals returns 0 or pi based on sign."""
        data = np.array([1.0, -1.0, 2.5, -0.5, 0.0])
        ts = TimeSeries(data, dt=0.01, unit='V')

        result = ts.radian()
        expected = np.angle(data)  # [0, pi, 0, pi, 0]

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)


# =============================================================================
# _build_phase_series tests
# =============================================================================

class TestBuildPhaseSeries:
    """Tests for _build_phase_series() internal method."""

    def test_constant_frequency(self):
        """phase(t) = phase0 + 2π·f·t for constant frequency."""
        f0 = 10.0  # Hz
        phase0 = 0.5  # radians
        dt = 0.001
        n_samples = 1000

        ts = TimeSeries(np.zeros(n_samples), dt=dt, unit='V')
        result = ts._build_phase_series(f0=f0, phase0=phase0)

        t_rel = dt * np.arange(n_samples)
        expected = phase0 + 2 * np.pi * f0 * t_rel

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_linear_chirp_fdot(self):
        """phase(t) = phase0 + 2π·(f·t + 0.5·fdot·t²) for linear chirp."""
        f0 = 10.0
        fdot = 1.0  # Hz/s
        phase0 = 0.0
        dt = 0.001
        n_samples = 1000

        ts = TimeSeries(np.zeros(n_samples), dt=dt, unit='V')
        result = ts._build_phase_series(f0=f0, fdot=fdot, phase0=phase0)

        t_rel = dt * np.arange(n_samples)
        expected = 2 * np.pi * (f0 * t_rel + 0.5 * fdot * t_rel**2)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_quadratic_chirp_fddot(self):
        """phase(t) = phase0 + 2π·(f·t + 0.5·fdot·t² + (1/6)·fddot·t³)."""
        f0 = 10.0
        fdot = 1.0
        fddot = 0.1  # Hz/s²
        phase0 = np.pi / 4
        dt = 0.001
        n_samples = 500

        ts = TimeSeries(np.zeros(n_samples), dt=dt, unit='V')
        result = ts._build_phase_series(f0=f0, fdot=fdot, fddot=fddot, phase0=phase0)

        t_rel = dt * np.arange(n_samples)
        expected = phase0 + 2 * np.pi * (f0 * t_rel + 0.5 * fdot * t_rel**2 + (1/6) * fddot * t_rel**3)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_phase_array_passthrough(self):
        """When phase array is provided, it should be used directly (+ phase0)."""
        phase_input = np.linspace(0, 10, 100)
        phase0 = 0.5

        ts = TimeSeries(np.zeros(100), dt=0.01, unit='V')
        result = ts._build_phase_series(phase=phase_input, phase0=phase0)

        expected = phase_input + phase0
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_phase_length_mismatch_raises(self):
        """ValueError when phase array length doesn't match TimeSeries length."""
        ts = TimeSeries(np.zeros(100), dt=0.01, unit='V')
        phase_wrong_len = np.zeros(50)

        with pytest.raises(ValueError, match="Length of phase"):
            ts._build_phase_series(phase=phase_wrong_len)

    def test_must_provide_f0_or_phase(self):
        """ValueError when neither f0 nor phase is provided."""
        ts = TimeSeries(np.zeros(100), dt=0.01, unit='V')

        with pytest.raises(ValueError, match="Exactly one of"):
            ts._build_phase_series()

    def test_cannot_provide_both_f0_and_phase(self):
        """ValueError when both f0 and phase are provided."""
        ts = TimeSeries(np.zeros(100), dt=0.01, unit='V')

        with pytest.raises(ValueError, match="Exactly one of"):
            ts._build_phase_series(f0=10.0, phase=np.zeros(100))


# =============================================================================
# mix_down tests
# =============================================================================

class TestMixDown:
    """Tests for mix_down() method."""

    def test_basic_demodulation(self):
        """x = exp(1j*(ωt+φ)), phase=ωt → mix_down → exp(1j*φ)."""
        f0 = 20.0
        phi = np.pi / 3
        dt = 0.001
        duration = 1.0
        n_samples = int(duration / dt)

        t = dt * np.arange(n_samples)
        omega_t = 2 * np.pi * f0 * t
        signal = np.exp(1j * (omega_t + phi))

        ts = TimeSeries(signal, dt=dt, unit='V')
        result = ts.mix_down(phase=omega_t)

        # Result should be exp(1j*φ) (constant)
        expected = np.exp(1j * phi)
        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_singlesided_doubles_amplitude(self):
        """singlesided=True should multiply result by 2."""
        f0 = 20.0
        phi = np.pi / 6
        dt = 0.001
        n_samples = 1000

        t = dt * np.arange(n_samples)
        omega_t = 2 * np.pi * f0 * t
        signal = np.exp(1j * (omega_t + phi))

        ts = TimeSeries(signal, dt=dt, unit='V')

        result_single = ts.mix_down(phase=omega_t, singlesided=False)
        result_double = ts.mix_down(phase=omega_t, singlesided=True)

        # singlesided=True should be 2x
        np.testing.assert_allclose(result_double.value, 2.0 * result_single.value, rtol=1e-10)

    def test_mix_down_with_f0(self):
        """mix_down with f0 parameter builds phase internally."""
        f0 = 30.0
        phi = 0.0
        dt = 0.001
        n_samples = 500

        t = dt * np.arange(n_samples)
        signal = np.cos(2 * np.pi * f0 * t + phi)  # Real cosine

        ts = TimeSeries(signal, dt=dt, unit='V')
        result = ts.mix_down(f0=f0)

        # cos(ωt) * exp(-1j*ωt) = 0.5 * (exp(1j*ωt) + exp(-1j*ωt)) * exp(-1j*ωt)
        #                       = 0.5 * (1 + exp(-2j*ωt))
        # Real part oscillates around 0.5
        assert np.isclose(np.mean(np.real(result.value)), 0.5, rtol=0.01)

    def test_mix_down_preserves_metadata(self):
        """mix_down should preserve unit, channel, name."""
        ts = TimeSeries(np.ones(100), dt=0.01, unit='m', channel='TEST', name='test_signal')
        result = ts.mix_down(f0=10.0)

        assert result.unit == 'm'
        # Channel may be a Channel object, compare by name
        assert getattr(result.channel, 'name', result.channel) == 'TEST'
        assert result.name == 'test_signal'


# =============================================================================
# xcorr tests
# =============================================================================

class TestXcorr:
    """Tests for xcorr() method."""

    def test_xcorr_matches_scipy(self):
        """xcorr should match scipy.signal.correlate and correlation_lags."""
        from scipy import signal

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 1, 0, 1, 2], dtype=float)

        ts_x = TimeSeries(x, dt=0.1, unit='V')
        ts_y = TimeSeries(y, dt=0.1, unit='V')

        result = ts_x.xcorr(ts_y, demean=False)

        # Reference
        corr_ref = signal.correlate(x, y, mode='full', method='auto')
        lags_ref = signal.correlation_lags(len(x), len(y), mode='full')

        np.testing.assert_allclose(result.value, corr_ref, rtol=1e-10)
        np.testing.assert_allclose(result.times.value / 0.1, lags_ref, rtol=1e-10)

    def test_xcorr_normalize_biased(self):
        """normalize='biased' divides by N."""
        from scipy import signal

        x = np.random.randn(50)
        y = np.random.randn(50)

        ts_x = TimeSeries(x, dt=0.01, unit='V')
        ts_y = TimeSeries(y, dt=0.01, unit='V')

        result = ts_x.xcorr(ts_y, normalize='biased', demean=False)
        corr_unnorm = signal.correlate(x, y, mode='full')
        expected = corr_unnorm / len(x)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_xcorr_normalize_unbiased(self):
        """normalize='unbiased' divides by (N - |lag|)."""
        from scipy import signal

        x = np.random.randn(30)
        y = np.random.randn(30)

        ts_x = TimeSeries(x, dt=0.01, unit='V')
        ts_y = TimeSeries(y, dt=0.01, unit='V')

        result = ts_x.xcorr(ts_y, normalize='unbiased', demean=False)

        corr_unnorm = signal.correlate(x, y, mode='full')
        lags = signal.correlation_lags(len(x), len(y), mode='full')
        expected = corr_unnorm / (len(x) - np.abs(lags))

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_xcorr_normalize_coeff(self):
        """normalize='coeff' divides by sqrt(sum(x²) * sum(y²))."""
        from scipy import signal

        x = np.random.randn(40)
        y = np.random.randn(40)

        ts_x = TimeSeries(x, dt=0.01, unit='V')
        ts_y = TimeSeries(y, dt=0.01, unit='V')

        result = ts_x.xcorr(ts_y, normalize='coeff', demean=False)

        corr_unnorm = signal.correlate(x, y, mode='full')
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
        expected = corr_unnorm / denom

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_xcorr_maxlag(self):
        """maxlag should trim lags to |lag| <= maxlag."""
        from scipy import signal

        x = np.random.randn(100)
        y = np.random.randn(100)

        ts_x = TimeSeries(x, dt=0.01, unit='V')
        ts_y = TimeSeries(y, dt=0.01, unit='V')

        maxlag = 20  # samples
        result = ts_x.xcorr(ts_y, maxlag=maxlag, demean=False)

        # Check lag range
        lags_samples = result.times.value / 0.01
        assert np.all(np.abs(lags_samples) <= maxlag)
        assert len(result.value) == 2 * maxlag + 1

    def test_xcorr_demean(self):
        """demean=True should subtract mean before correlation."""
        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0])  # Mean = 12
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])      # Mean = 3

        ts_x = TimeSeries(x, dt=0.1, unit='V')
        ts_y = TimeSeries(y, dt=0.1, unit='V')

        result_demean = ts_x.xcorr(ts_y, demean=True)
        result_no_demean = ts_x.xcorr(ts_y, demean=False)

        # With demean, x-mean and y-mean are the same relative pattern
        # [-2,-1,0,1,2] for both, so autocorrelation-like
        assert not np.allclose(result_demean.value, result_no_demean.value)


# =============================================================================
# heterodyne / lock_in tests (regression)
# =============================================================================

class TestHeterodyneLockIn:
    """Regression tests for heterodyne() and lock_in()."""

    def test_heterodyne_default_singlesided(self):
        """heterodyne should default to singlesided=False (GWpy compatibility)."""
        f0 = 50.0
        amp = 2.0
        sample_rate = 1024.0
        duration = 1.0
        stride = 1.0

        t = np.arange(0, duration, 1/sample_rate)
        signal = amp * np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(signal, dt=1/sample_rate, unit='V')

        phase = 2 * np.pi * f0 * t

        # Default (should be False)
        res_default = ts.heterodyne(phase, stride=stride)
        # Explicit False
        res_false = ts.heterodyne(phase, stride=stride, singlesided=False)
        # Explicit True
        res_true = ts.heterodyne(phase, stride=stride, singlesided=True)

        # Default should match False
        np.testing.assert_allclose(res_default.value, res_false.value)
        # True should be 2x False
        np.testing.assert_allclose(res_true.value, 2.0 * res_false.value)
        # For cos(wt) = 0.5(exp(iwt)+exp(-iwt)), mixing with exp(-iwt) gives 0.5.
        # So amp * 0.5 = 2.0 * 0.5 = 1.0.
        np.testing.assert_allclose(np.abs(res_false.value), 1.0, rtol=0.01)

    def test_heterodyne_dc_peak(self):
        """heterodyne(f0) should move f0 component to DC."""
        f0 = 50.0
        amp = 2.0
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0

        t = np.arange(0, duration, 1/sample_rate)
        signal = amp * np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(signal, dt=1/sample_rate, unit='V')

        # Build phase for heterodyne
        phase = 2 * np.pi * f0 * t
        result = ts.heterodyne(phase, stride=stride, singlesided=True)

        # DC component should be close to amplitude
        # For cos signal with singlesided=True, expect amp
        assert np.allclose(np.abs(result.value), amp, rtol=0.05)

    def test_lock_in_amplitude_recovery(self):
        """lock_in should recover amplitude of sinusoidal signal."""
        f0 = 30.0
        amp = 1.5
        phi0 = np.pi / 6
        sample_rate = 4096.0
        duration = 60.0
        stride = 10.0

        t = np.arange(0, duration, 1/sample_rate)
        signal = amp * np.cos(2 * np.pi * f0 * t + phi0)
        ts = TimeSeries(signal, dt=1/sample_rate, unit='V')

        mag, phase = ts.lock_in(f0=f0, stride=stride, singlesided=True, output='amp_phase', deg=False)

        # Amplitude should be recovered
        np.testing.assert_allclose(mag.value, amp, rtol=0.01)
        # Phase should match initial phase
        np.testing.assert_allclose(phase.value, phi0, atol=0.01)

    def test_lock_in_singlesided_scaling(self):
        """singlesided=True should double the amplitude vs singlesided=False."""
        f0 = 20.0
        amp = 1.0
        sample_rate = 2048.0
        duration = 10.0
        stride = 1.0

        t = np.arange(0, duration, 1/sample_rate)
        signal = amp * np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(signal, dt=1/sample_rate, unit='V')

        out_single = ts.lock_in(f0=f0, stride=stride, singlesided=False, output='complex')
        out_double = ts.lock_in(f0=f0, stride=stride, singlesided=True, output='complex')

        # singlesided=True should give 2x amplitude
        ratio = np.median(np.abs(out_double.value)) / np.median(np.abs(out_single.value))
        np.testing.assert_allclose(ratio, 2.0, rtol=0.01)
