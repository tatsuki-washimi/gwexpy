"""
Regression tests for D2 Hilbert transform signal processing functions.

D2 = Hilbert / phase / frequency semantics (specification-locked tests)

Target functions in TimeSeriesSignalMixin:
- hilbert
- instantaneous_phase
- instantaneous_frequency

Specification (fixed):
- hilbert: pad=0 default, raises ValueError on NaN/inf
- instantaneous_phase: matches np.angle(hilbert()), unwrap with correct period
- instantaneous_frequency: d/dt(unwrap(phase)) / (2π)
"""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries

# =============================================================================
# hilbert: NaN/inf exception tests
# =============================================================================

class TestHilbertNaNInf:
    """Tests for hilbert() NaN/inf exception behavior."""

    def test_hilbert_raises_on_nan(self):
        """hilbert() should raise ValueError when input contains NaN."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ts = TimeSeries(data, dt=0.01, unit='V')

        with pytest.raises(ValueError, match="NaN|infinite"):
            ts.hilbert()

    def test_hilbert_raises_on_inf(self):
        """hilbert() should raise ValueError when input contains inf."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        ts = TimeSeries(data, dt=0.01, unit='V')

        with pytest.raises(ValueError, match="NaN|infinite"):
            ts.hilbert()

    def test_hilbert_raises_on_negative_inf(self):
        """hilbert() should raise ValueError when input contains -inf."""
        data = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])
        ts = TimeSeries(data, dt=0.01, unit='V')

        with pytest.raises(ValueError, match="NaN|infinite"):
            ts.hilbert()

    def test_hilbert_raises_on_multiple_nan(self):
        """hilbert() should raise ValueError when input contains multiple NaNs."""
        data = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
        ts = TimeSeries(data, dt=0.01, unit='V')

        with pytest.raises(ValueError, match="NaN|infinite"):
            ts.hilbert()


# =============================================================================
# hilbert: basic functionality tests
# =============================================================================

class TestHilbertBasic:
    """Tests for hilbert() basic functionality."""

    def test_hilbert_returns_complex(self):
        """hilbert() should return complex analytic signal."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        ts = TimeSeries(data, dt=0.001, unit='V')

        result = ts.hilbert()

        assert np.iscomplexobj(result.value)

    def test_hilbert_preserves_length(self):
        """hilbert() should preserve signal length."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        ts = TimeSeries(data, dt=0.001, unit='V')

        result = ts.hilbert()

        assert len(result.value) == len(data)

    def test_hilbert_default_pad_is_zero(self):
        """hilbert() default pad should be 0 (no padding)."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
        ts = TimeSeries(data, dt=0.01, unit='V')

        # With pad=0 (default), endpoint artifacts may exist, but that's expected
        result = ts.hilbert()

        # Result should be complex and same length
        assert np.iscomplexobj(result.value)
        assert len(result.value) == len(data)

    def test_hilbert_uses_scipy(self):
        """hilbert() should produce results matching scipy.signal.hilbert."""
        import scipy.signal as sig

        data = np.sin(2 * np.pi * 5 * np.linspace(0, 2, 500))
        ts = TimeSeries(data, dt=0.004, unit='V')

        result = ts.hilbert()
        expected = sig.hilbert(data)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)


# =============================================================================
# instantaneous_phase: reference implementation equivalence tests
# =============================================================================

class TestInstantaneousPhase:
    """Tests for instantaneous_phase() reference implementation equivalence."""

    @pytest.fixture
    def sine_timeseries(self):
        """Create a simple sine wave TimeSeries for testing."""
        f0 = 10.0
        duration = 2.0
        sample_rate = 500.0
        dt = 1.0 / sample_rate
        t = np.arange(0, duration, dt)
        data = np.sin(2 * np.pi * f0 * t)
        return TimeSeries(data, dt=dt, unit='V')

    def test_phase_matches_np_angle_radian(self, sine_timeseries):
        """instantaneous_phase(deg=False, unwrap=False) should match np.angle(hilbert())."""
        ts = sine_timeseries

        analytic = ts.hilbert()
        expected = np.angle(analytic.value)

        result = ts.instantaneous_phase(deg=False, unwrap=False)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)
        assert result.unit == 'rad'

    def test_phase_matches_np_angle_degree(self, sine_timeseries):
        """instantaneous_phase(deg=True, unwrap=False) should match np.angle(hilbert(), deg=True)."""
        ts = sine_timeseries

        analytic = ts.hilbert()
        expected = np.angle(analytic.value, deg=True)

        result = ts.instantaneous_phase(deg=True, unwrap=False)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)
        assert result.unit == 'deg'

    def test_phase_unwrap_radian(self, sine_timeseries):
        """instantaneous_phase(unwrap=True, deg=False) should apply np.unwrap with period=2π."""
        ts = sine_timeseries

        analytic = ts.hilbert()
        phase_raw = np.angle(analytic.value)
        expected = np.unwrap(phase_raw, period=2 * np.pi)

        result = ts.instantaneous_phase(deg=False, unwrap=True)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)
        assert result.unit == 'rad'

    def test_phase_unwrap_degree(self, sine_timeseries):
        """instantaneous_phase(unwrap=True, deg=True) should apply np.unwrap with period=360."""
        ts = sine_timeseries

        analytic = ts.hilbert()
        phase_raw = np.angle(analytic.value, deg=True)
        expected = np.unwrap(phase_raw, period=360.0)

        result = ts.instantaneous_phase(deg=True, unwrap=True)

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)
        assert result.unit == 'deg'

    def test_phase_preserves_length(self, sine_timeseries):
        """instantaneous_phase() should preserve signal length (no endpoint trimming)."""
        ts = sine_timeseries

        result = ts.instantaneous_phase()

        assert len(result.value) == len(ts.value)


# =============================================================================
# instantaneous_frequency: single frequency recovery tests
# =============================================================================

class TestInstantaneousFrequency:
    """Tests for instantaneous_frequency() single frequency recovery."""

    def test_frequency_recovery_pure_cosine(self):
        """instantaneous_frequency() should recover f0 for cos(2π f0 t).

        Test evaluates only the central region (10%-90%) to avoid endpoint artifacts.
        The API does NOT trim endpoints; this is a test evaluation strategy.
        """
        f0 = 25.0  # Hz
        duration = 10.0  # seconds (long enough for stable measurement)
        sample_rate = 1000.0  # Hz
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=dt, unit='V')

        f_inst = ts.instantaneous_frequency()

        # Evaluate central 10%-90% region
        n = len(f_inst.value)
        start = int(n * 0.1)
        end = int(n * 0.9)
        central_region = f_inst.value[start:end]

        # Median should be close to f0
        median_freq = np.median(central_region)
        np.testing.assert_allclose(median_freq, f0, rtol=0.01)

        # Unit should be Hz
        assert f_inst.unit == 'Hz'

    def test_frequency_recovery_pure_sine(self):
        """instantaneous_frequency() should recover f0 for sin(2π f0 t)."""
        f0 = 17.5  # Hz
        duration = 8.0  # seconds
        sample_rate = 800.0  # Hz
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.sin(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=dt, unit='V')

        f_inst = ts.instantaneous_frequency()

        # Evaluate central region
        n = len(f_inst.value)
        start = int(n * 0.1)
        end = int(n * 0.9)
        central_region = f_inst.value[start:end]

        median_freq = np.median(central_region)
        np.testing.assert_allclose(median_freq, f0, rtol=0.01)

    def test_frequency_recovery_with_phase_offset(self):
        """instantaneous_frequency() should recover f0 regardless of initial phase."""
        f0 = 30.0  # Hz
        phi0 = np.pi / 3  # Initial phase offset
        duration = 5.0  # seconds
        sample_rate = 600.0  # Hz
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.cos(2 * np.pi * f0 * t + phi0)
        ts = TimeSeries(data, dt=dt, unit='V')

        f_inst = ts.instantaneous_frequency()

        # Evaluate central region
        n = len(f_inst.value)
        start = int(n * 0.15)
        end = int(n * 0.85)
        central_region = f_inst.value[start:end]

        median_freq = np.median(central_region)
        np.testing.assert_allclose(median_freq, f0, rtol=0.01)

    def test_frequency_preserves_length(self):
        """instantaneous_frequency() should preserve signal length (no endpoint trimming)."""
        f0 = 20.0
        duration = 2.0
        sample_rate = 500.0
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=dt, unit='V')

        f_inst = ts.instantaneous_frequency()

        assert len(f_inst.value) == len(ts.value)

    def test_frequency_definition_formula(self):
        """instantaneous_frequency() should match the definition: d/dt(unwrap(phase)) / (2π)."""
        f0 = 15.0
        duration = 3.0
        sample_rate = 400.0
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=dt, unit='V')

        # Manual calculation following the definition
        phase_ts = ts.instantaneous_phase(deg=False, unwrap=True)
        phase = phase_ts.value
        dphi_dt = np.gradient(phase, dt)
        expected = dphi_dt / (2 * np.pi)

        result = ts.instantaneous_frequency()

        np.testing.assert_allclose(result.value, expected, rtol=1e-10)

    def test_frequency_smooth_disabled_by_default(self):
        """instantaneous_frequency() should not apply smoothing by default."""
        f0 = 20.0
        duration = 2.0
        sample_rate = 500.0
        dt = 1.0 / sample_rate

        t = np.arange(0, duration, dt)
        data = np.cos(2 * np.pi * f0 * t)
        ts = TimeSeries(data, dt=dt, unit='V')

        # Without smooth
        result_default = ts.instantaneous_frequency()
        result_no_smooth = ts.instantaneous_frequency(smooth=None)

        np.testing.assert_allclose(result_default.value, result_no_smooth.value, rtol=1e-10)
