"""
GWpy-compatible regression tests for heterodyne/lock_in.

This test module verifies that:
1. `heterodyne()` implements the exact GWpy algorithm
2. `lock_in()` enforces phase precedence and mode selection rules
3. Output formats and validation work correctly
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gwexpy.timeseries import TimeSeries


class TestHeterodyneGWpyCompatibility:
    """Test heterodyne() for GWpy-identical algorithm."""

    def test_sinusoid_demodulation_doublesided(self):
        """Test demodulation of sinusoid returns expected complex amplitude (doublesided)."""
        # Parameters
        A = 2.5
        f0 = 30.0
        phi0 = np.pi / 4
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0

        # Create sinusoid: x = A * cos(2*pi*f0*t + phi0)
        # Make sure duration is exact multiple of stride
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = A * np.cos(2 * np.pi * f0 * t + phi0)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        # Phase for heterodyning
        phase = 2 * np.pi * f0 * t

        # Heterodyne with doublesided (singlesided=False)
        het = ts.heterodyne(phase, stride=stride, singlesided=False)

        # Expected: (A/2) * exp(1j * phi0) for doublesided
        expected_complex = (A / 2) * np.exp(1j * phi0)

        # Check all stride outputs are close to expected
        assert_allclose(het.value, expected_complex, rtol=1e-3)

        # Check sample rate
        assert_allclose(het.sample_rate.value, 1 / stride, rtol=1e-6)

    def test_sinusoid_demodulation_singlesided(self):
        """Test singlesided=True doubles the amplitude."""
        A = 2.5
        f0 = 30.0
        phi0 = np.pi / 4
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = A * np.cos(2 * np.pi * f0 * t + phi0)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')
        phase = 2 * np.pi * f0 * t

        # Heterodyne with singlesided=True
        het = ts.heterodyne(phase, stride=stride, singlesided=True)

        # Expected: A * exp(1j * phi0) for singlesided
        expected_complex = A * np.exp(1j * phi0)

        assert_allclose(het.value, expected_complex, rtol=1e-3)

    def test_floor_truncation_stride_samples(self):
        """Test that stride samples are floor-truncated."""
        sample_rate = 100.0
        duration = 10.0
        stride = 0.33  # 33 samples (not exact)

        n_samples = int(duration * sample_rate)
        data = np.ones(n_samples)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')
        phase = np.zeros(n_samples)

        het = ts.heterodyne(phase, stride=stride, singlesided=False)

        # Expected nsteps = floor(1000 / int(0.33 * 100)) = floor(1000 / 33) = 30
        stridesamp = int(stride * sample_rate)
        expected_nsteps = int(n_samples // stridesamp)
        assert len(het) == expected_nsteps

    def test_trailing_samples_discarded(self):
        """Test that trailing samples are discarded when not divisible."""
        sample_rate = 100.0
        n_samples = 1005  # Not divisible by stride
        stride = 1.0

        data = np.ones(n_samples)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')
        phase = np.zeros(n_samples)

        het = ts.heterodyne(phase, stride=stride, singlesided=False)

        # stridesamp = 100, nsteps = floor(1005 / 100) = 10
        assert len(het) == 10

    def test_phase_not_array_like_raises_typeerror(self):
        """Test that non-array-like phase raises TypeError."""
        ts = TimeSeries(np.ones(100), dt=0.01, unit='V')

        # Scalar float
        with pytest.raises(TypeError, match="Phase is not array_like"):
            ts.heterodyne(phase=3.14, stride=1.0)

        # None-like that can't be converted
        with pytest.raises((TypeError, ValueError)):
            ts.heterodyne(phase=None, stride=1.0)

    def test_phase_length_mismatch_raises_valueerror(self):
        """Test that phase length != TimeSeries length raises ValueError."""
        ts = TimeSeries(np.ones(100), dt=0.01, unit='V')
        wrong_phase = np.zeros(50)  # Wrong length

        with pytest.raises(ValueError, match="same length as the TimeSeries"):
            ts.heterodyne(wrong_phase, stride=1.0)

    def test_complex_input_handled(self):
        """Test that complex input TimeSeries is handled correctly."""
        sample_rate = 100.0
        n_samples = 1000
        t = np.arange(n_samples) / sample_rate

        # Complex sinusoid
        data = np.exp(1j * 2 * np.pi * 10 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')
        phase = 2 * np.pi * 10 * t

        het = ts.heterodyne(phase, stride=1.0, singlesided=False)

        # exp(1j*phi) * exp(-1j*phi) = 1
        assert_allclose(het.value, 1.0, rtol=1e-6)


class TestLockInValidation:
    """Test lock_in() parameter validation rules."""

    def test_phase_and_f0_mutual_exclusivity(self):
        """Test that specifying both phase and f0 raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')
        phase = np.zeros(1000)

        with pytest.raises(ValueError, match="Cannot specify both 'phase' and any of"):
            ts.lock_in(f0=10.0, phase=phase, stride=1.0)

    def test_phase_and_fdot_raises_valueerror(self):
        """Test that specifying phase with fdot != 0 raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')
        phase = np.zeros(1000)

        with pytest.raises(ValueError, match="Cannot specify both 'phase' and any of"):
            ts.lock_in(phase=phase, fdot=0.1, stride=1.0)

    def test_phase_and_phase0_raises_valueerror(self):
        """Test that specifying phase with phase0 != 0 raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')
        phase = np.zeros(1000)

        with pytest.raises(ValueError, match="Cannot specify both 'phase' and any of"):
            ts.lock_in(phase=phase, phase0=0.5, stride=1.0)

    def test_neither_phase_nor_f0_raises_valueerror(self):
        """Test that specifying neither phase nor f0 raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')

        with pytest.raises(ValueError, match="Either 'phase' or 'f0' must be specified"):
            ts.lock_in(stride=1.0)

    def test_bandwidth_and_stride_mutual_exclusivity(self):
        """Test that specifying both bandwidth and stride raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')

        with pytest.raises(ValueError, match="Cannot specify both 'bandwidth' and 'stride'"):
            ts.lock_in(f0=10.0, bandwidth=5.0, stride=1.0)

    def test_neither_bandwidth_nor_stride_raises_valueerror(self):
        """Test that specifying neither bandwidth nor stride raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')

        with pytest.raises(ValueError, match="Either 'bandwidth' or 'stride' must be specified"):
            ts.lock_in(f0=10.0)

    def test_invalid_output_format_raises_valueerror(self):
        """Test that invalid output format raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')

        with pytest.raises(ValueError, match="Unknown output format"):
            ts.lock_in(f0=10.0, stride=1.0, output="invalid")


class TestLockInStrideAverageMode:
    """Test lock_in() stride-average mode (bandwidth=None)."""

    def test_lock_in_equals_heterodyne(self):
        """Test that lock_in with stride mode gives same result as heterodyne."""
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0
        f0 = 30.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')
        phase = 2 * np.pi * f0 * t

        # Lock-in with explicit phase
        lockin_result = ts.lock_in(phase=phase, stride=stride, output='complex')

        # Direct heterodyne
        het_result = ts.heterodyne(phase, stride=stride, singlesided=True)

        assert_allclose(lockin_result.value, het_result.value, rtol=1e-10)

    def test_output_amp_phase_consistency(self):
        """Test that amp_phase output matches complex output."""
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0
        f0 = 30.0
        phi0 = np.pi / 4

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t + phi0)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        # Get complex and amp_phase
        complex_result = ts.lock_in(f0=f0, stride=stride, output='complex')
        amp, phase = ts.lock_in(f0=f0, stride=stride, output='amp_phase', deg=False)

        # Check consistency
        assert_allclose(amp.value, np.abs(complex_result.value), rtol=1e-10)
        assert_allclose(phase.value, np.angle(complex_result.value), rtol=1e-10)
        assert phase.unit == 'rad'

    def test_output_amp_phase_deg(self):
        """Test that deg=True gives phase in degrees."""
        sample_rate = 1024.0
        duration = 10.0
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        ts = TimeSeries(np.cos(2 * np.pi * 30 * t), dt=1 / sample_rate, unit='V')

        _, phase_deg = ts.lock_in(f0=30.0, stride=1.0, output='amp_phase', deg=True)
        _, phase_rad = ts.lock_in(f0=30.0, stride=1.0, output='amp_phase', deg=False)

        assert phase_deg.unit == 'deg'
        assert phase_rad.unit == 'rad'
        assert_allclose(phase_deg.value, np.rad2deg(phase_rad.value), rtol=1e-10)

    def test_output_iq_consistency(self):
        """Test that I/Q output matches complex real/imag."""
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0
        f0 = 30.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        complex_result = ts.lock_in(f0=f0, stride=stride, output='complex')
        i, q = ts.lock_in(f0=f0, stride=stride, output='iq')

        assert_allclose(i.value, complex_result.value.real, rtol=1e-10)
        assert_allclose(q.value, complex_result.value.imag, rtol=1e-10)


class TestLockInLPFMode:
    """Test lock_in() LPF mode (bandwidth is not None)."""

    def test_lpf_mode_returns_complex(self):
        """Test that LPF mode returns complex series."""
        sample_rate = 1024.0
        duration = 10.0
        f0 = 100.0
        bandwidth = 10.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        result = ts.lock_in(f0=f0, bandwidth=bandwidth, output='complex')

        # Check it's complex
        assert np.iscomplexobj(result.value)
        # Check it's a TimeSeries
        assert isinstance(result, TimeSeries)

    def test_lpf_mode_dc_dominant(self):
        """Test that LPF mode makes DC component dominant."""
        sample_rate = 1024.0
        duration = 10.0
        f0 = 100.0
        bandwidth = 5.0  # Narrow bandwidth to suppress high frequencies

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        # Pure tone at f0
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        result = ts.lock_in(f0=f0, bandwidth=bandwidth, output='complex')

        # After demodulation, DC should be dominant
        # The mean amplitude should be significant (not near zero)
        mean_amp = np.mean(np.abs(result.value))
        assert mean_amp > 0.3  # Loose check - just verify signal is present

    def test_lpf_mode_with_stride_raises(self):
        """Test that bandwidth + stride raises ValueError."""
        ts = TimeSeries(np.ones(1000), dt=0.001, unit='V')

        with pytest.raises(ValueError, match="Cannot specify both 'bandwidth' and 'stride'"):
            ts.lock_in(f0=10.0, bandwidth=5.0, stride=1.0)

    def test_lpf_mode_amp_phase_output(self):
        """Test that LPF mode works with amp_phase output."""
        sample_rate = 1024.0
        duration = 10.0
        f0 = 100.0
        bandwidth = 10.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        amp, phase = ts.lock_in(f0=f0, bandwidth=bandwidth, output='amp_phase')

        assert isinstance(amp, TimeSeries)
        assert isinstance(phase, TimeSeries)
        assert not np.iscomplexobj(amp.value)


class TestLockInPhaseOnly:
    """Test lock_in() with explicit phase (no f0)."""

    def test_phase_only_mode(self):
        """Test that phase-only mode works correctly."""
        sample_rate = 1024.0
        duration = 10.0
        stride = 1.0
        f0 = 30.0

        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=1 / sample_rate, unit='V')

        # Explicit phase
        phase = 2 * np.pi * f0 * t

        result = ts.lock_in(phase=phase, stride=stride, output='complex')

        # Should return complex TimeSeries
        assert np.iscomplexobj(result.value)
        # Amplitude should be close to 1 for singlesided
        assert_allclose(np.abs(result.value), 1.0, rtol=1e-3)
