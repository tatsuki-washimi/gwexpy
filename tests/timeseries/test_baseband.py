"""
Regression tests for baseband() method.

Tests for:
- Mode A (lowpass specified): mix_down → lowpass → optional resample
- Mode B (resample only): mix_down → resample
- Input validation (ValueError conditions)
- kwargs passthrough to lowpass/resample
"""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries

# =============================================================================
# Helper functions
# =============================================================================


def make_cosine_signal(f0: float, duration: float, sample_rate: float) -> TimeSeries:
    """Create a cosine signal TimeSeries for testing."""
    dt = 1.0 / sample_rate
    t = np.arange(0, duration, dt)
    data = np.cos(2 * np.pi * f0 * t)
    return TimeSeries(data, dt=dt, unit="V")


# =============================================================================
# Mode A: lowpass specified
# =============================================================================


class TestBasebandModeA:
    """Tests for Mode A: mix_down → lowpass [→ optional resample]."""

    def test_mode_a_returns_complex(self):
        """baseband with lowpass should return complex signal."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 2.0  # seconds

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=10.0)

        assert np.iscomplexobj(z.value)

    def test_mode_a_dc_dominant(self):
        """After baseband, the DC component should be dominant."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 5.0  # seconds

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=10.0)

        # DC component (mean) should be significant
        # For cos signal with exact f0 match, the DC should be ~0.5
        dc_magnitude = np.abs(np.mean(z.value))
        assert dc_magnitude > 0.1, f"DC magnitude {dc_magnitude} too small"

    def test_mode_a_suppresses_high_frequency(self):
        """Lowpass should suppress the 2*fc component."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 5.0  # seconds
        lowpass_cutoff = 10.0  # Hz

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=lowpass_cutoff)

        # Compute FFT to check spectral content
        fft = np.fft.fft(z.value)
        freqs = np.fft.fftfreq(len(z), d=z.dt.to("s").value)

        # DC power (near 0 Hz)
        dc_idx = np.abs(freqs) < 5  # Within 5 Hz of DC
        dc_power = np.mean(np.abs(fft[dc_idx]) ** 2)

        # High frequency power (near 2*fc = 100 Hz should be suppressed,
        # but after mixing it appears at fc in baseband)
        # Actually, cos(2π fc t) * exp(-1j 2π fc t) = 0.5 * (1 + exp(-2j 2π fc t))
        # So we have DC and a component at -2*fc
        # After lowpass at 10 Hz, the 2*fc component should be cut
        high_freq_idx = np.abs(freqs) > 30  # Above 30 Hz
        if np.any(high_freq_idx):
            high_power = np.mean(np.abs(fft[high_freq_idx]) ** 2)

            # High frequency should be suppressed relative to DC
            # Use a relaxed threshold since filter rolloff varies
            assert high_power < dc_power * 0.1, (
                f"High frequency power {high_power} not suppressed relative to DC {dc_power}"
            )

    def test_mode_a_with_resample(self):
        """Mode A with both lowpass and output_rate."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 2.0  # seconds

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=10.0, output_rate=100.0)

        # Output sample rate should be 100 Hz
        output_sr = z.sample_rate.to("Hz").value
        np.testing.assert_allclose(output_sr, 100.0, rtol=0.01)

        # Should still be complex and DC-dominant
        assert np.iscomplexobj(z.value)
        dc_magnitude = np.abs(np.mean(z.value))
        assert dc_magnitude > 0.1


# =============================================================================
# Mode B: resample only
# =============================================================================


class TestBasebandModeB:
    """Tests for Mode B: mix_down → resample (no explicit lowpass)."""

    def test_mode_b_returns_complex(self):
        """baseband with output_rate only should return complex signal."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 2.0  # seconds

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=None, output_rate=200.0)

        assert np.iscomplexobj(z.value)

    def test_mode_b_output_sample_rate(self):
        """Output sample rate should match specified output_rate."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 5.0  # seconds
        output_rate = 200.0  # Hz

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=None, output_rate=output_rate)

        output_sr = z.sample_rate.to("Hz").value
        np.testing.assert_allclose(output_sr, output_rate, rtol=0.01)

    def test_mode_b_dc_dominant(self):
        """Mode B should also produce DC-dominant baseband signal."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 5.0  # seconds

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=None, output_rate=200.0)

        dc_magnitude = np.abs(np.mean(z.value))
        assert dc_magnitude > 0.1, f"DC magnitude {dc_magnitude} too small"

    def test_mode_b_output_length(self):
        """Output length should be approximately duration * output_rate."""
        fc = 50.0  # Hz
        sample_rate = 1000.0  # Hz
        duration = 5.0  # seconds
        output_rate = 100.0  # Hz

        ts = make_cosine_signal(fc, duration, sample_rate)
        z = ts.baseband(f0=fc, lowpass=None, output_rate=output_rate)

        expected_length = duration * output_rate
        # Allow some tolerance for rounding
        assert abs(len(z) - expected_length) < 10, (
            f"Expected ~{expected_length} samples, got {len(z)}"
        )


# =============================================================================
# Exception condition tests
# =============================================================================


class TestBasebandValidation:
    """Tests for input validation (ValueError conditions)."""

    def test_f0_zero_raises(self):
        """f0 = 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="f0 must be positive"):
            ts.baseband(f0=0, lowpass=10)

    def test_f0_negative_raises(self):
        """f0 < 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="f0 must be positive"):
            ts.baseband(f0=-10, lowpass=10)

    def test_f0_exceeds_nyquist_raises(self):
        """f0 >= Nyquist should raise ValueError."""
        sample_rate = 1000.0  # Nyquist = 500 Hz
        ts = make_cosine_signal(50.0, 1.0, sample_rate)

        with pytest.raises(ValueError, match="must be less than Nyquist"):
            ts.baseband(f0=500, lowpass=10)  # f0 = Nyquist

        with pytest.raises(ValueError, match="must be less than Nyquist"):
            ts.baseband(f0=600, lowpass=10)  # f0 > Nyquist

    def test_lowpass_zero_raises(self):
        """lowpass = 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="lowpass must be positive"):
            ts.baseband(f0=50, lowpass=0)

    def test_lowpass_negative_raises(self):
        """lowpass < 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="lowpass must be positive"):
            ts.baseband(f0=50, lowpass=-10)

    def test_lowpass_exceeds_nyquist_raises(self):
        """lowpass >= Nyquist should raise ValueError."""
        sample_rate = 1000.0  # Nyquist = 500 Hz
        ts = make_cosine_signal(50.0, 1.0, sample_rate)

        with pytest.raises(ValueError, match="must be less than Nyquist"):
            ts.baseband(f0=50, lowpass=500)  # lowpass = Nyquist

        with pytest.raises(ValueError, match="must be less than Nyquist"):
            ts.baseband(f0=50, lowpass=600)  # lowpass > Nyquist

    def test_output_rate_zero_raises(self):
        """output_rate = 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="output_rate must be positive"):
            ts.baseband(f0=50, lowpass=None, output_rate=0)

    def test_output_rate_negative_raises(self):
        """output_rate < 0 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="output_rate must be positive"):
            ts.baseband(f0=50, lowpass=None, output_rate=-100)

    def test_no_lowpass_no_output_rate_raises(self):
        """Neither lowpass nor output_rate specified should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        with pytest.raises(ValueError, match="At least one of"):
            ts.baseband(f0=50, lowpass=None, output_rate=None)

    def test_lowpass_exceeds_new_nyquist_raises(self):
        """lowpass >= output_rate/2 should raise ValueError."""
        ts = make_cosine_signal(50.0, 1.0, 1000.0)

        # output_rate = 100 Hz -> new Nyquist = 50 Hz
        # lowpass = 50 Hz >= new Nyquist -> error
        with pytest.raises(ValueError, match="must be less than output_rate/2"):
            ts.baseband(f0=40, lowpass=50, output_rate=100)

        # lowpass = 60 Hz > new Nyquist = 50 Hz -> error
        with pytest.raises(ValueError, match="must be less than output_rate/2"):
            ts.baseband(f0=40, lowpass=60, output_rate=100)


# =============================================================================
# kwargs passthrough tests
# =============================================================================


class TestBasebandKwargsPassthrough:
    """Tests for lowpass_kwargs and resample_kwargs passthrough."""

    def test_lowpass_kwargs_accepted(self):
        """lowpass_kwargs should be passed to lowpass without error."""
        ts = make_cosine_signal(50.0, 2.0, 1000.0)

        # These are typical GWpy lowpass kwargs
        # Just verify they don't cause errors (actual behavior depends on GWpy)
        z = ts.baseband(f0=50, lowpass=10, lowpass_kwargs={"filtfilt": True})

        assert np.iscomplexobj(z.value)

    def test_resample_kwargs_accepted(self):
        """resample_kwargs should be passed to resample without error."""
        ts = make_cosine_signal(50.0, 2.0, 1000.0)

        # Just verify resample_kwargs are accepted
        z = ts.baseband(
            f0=50,
            lowpass=None,
            output_rate=200,
            resample_kwargs={},  # Empty dict should work
        )

        assert np.iscomplexobj(z.value)


# =============================================================================
# Quantity input tests
# =============================================================================


class TestBasebandQuantityInputs:
    """Tests for Quantity-typed inputs."""

    def test_f0_as_quantity(self):
        """f0 can be specified as Quantity."""
        ts = make_cosine_signal(50.0, 2.0, 1000.0)

        z = ts.baseband(f0=50 * u.Hz, lowpass=10)

        assert np.iscomplexobj(z.value)

    def test_lowpass_as_quantity(self):
        """lowpass can be specified as Quantity."""
        ts = make_cosine_signal(50.0, 2.0, 1000.0)

        z = ts.baseband(f0=50, lowpass=10 * u.Hz)

        assert np.iscomplexobj(z.value)

    def test_output_rate_as_quantity(self):
        """output_rate can be specified as Quantity."""
        ts = make_cosine_signal(50.0, 2.0, 1000.0)

        z = ts.baseband(f0=50, lowpass=None, output_rate=200 * u.Hz)

        output_sr = z.sample_rate.to("Hz").value
        np.testing.assert_allclose(output_sr, 200.0, rtol=0.01)
