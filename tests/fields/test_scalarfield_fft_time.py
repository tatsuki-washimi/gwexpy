"""Tests for ScalarField time FFT - GWpy TimeSeries.fft compatible."""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField


class TestScalarFieldFftTimeBasic:
    """Test basic fft_time functionality."""

    @pytest.fixture
    def time_domain_field(self):
        """Create a time-domain ScalarField."""
        np.random.seed(42)
        data = np.random.randn(64, 4, 4, 4)
        t = np.arange(64) * 0.01 * u.s  # 100 Hz sample rate
        x = np.arange(4) * 1.0 * u.m
        return ScalarField(
            data,
            unit=u.V,
            axis0=t,
            axis1=x,
            axis2=x.copy(),
            axis3=x.copy(),
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
        )

    def test_fft_time_basic(self, time_domain_field):
        """Test basic fft_time execution."""
        result = time_domain_field.fft_time()

        assert isinstance(result, ScalarField)
        assert result.axis0_domain == "frequency"

    def test_fft_time_shape(self, time_domain_field):
        """Test fft_time output shape (rfft)."""
        result = time_domain_field.fft_time()

        # rfft output: n//2 + 1 = 64//2 + 1 = 33
        assert result.shape[0] == 33
        assert result.shape[1:] == time_domain_field.shape[1:]

    def test_fft_time_axis_name_changes(self, time_domain_field):
        """Test that axis0 name changes to 'f'."""
        result = time_domain_field.fft_time()

        assert result.axis_names[0] == "f"
        # Other axes unchanged
        assert result.axis_names[1:] == ("x", "y", "z")

    def test_fft_time_frequency_axis_correct(self, time_domain_field):
        """Test frequency axis values are correct."""
        result = time_domain_field.fft_time()

        expected_freqs = np.fft.rfftfreq(64, d=0.01)
        assert_allclose(result._axis0_index.value, expected_freqs)
        assert result._axis0_index.unit == u.Hz

    def test_fft_time_preserves_spatial_axes(self, time_domain_field):
        """Test that spatial axis indices are preserved."""
        result = time_domain_field.fft_time()

        assert_allclose(result._axis1_index.value, time_domain_field._axis1_index.value)
        assert result._axis1_index.unit == time_domain_field._axis1_index.unit

    def test_fft_time_unit_preserved(self, time_domain_field):
        """Test that data unit is preserved."""
        result = time_domain_field.fft_time()
        assert result.unit == u.V

    def test_fft_time_space_domains_preserved(self, time_domain_field):
        """Test that space_domains are preserved."""
        result = time_domain_field.fft_time()
        assert result.space_domains == time_domain_field.space_domains


class TestScalarFieldFftTimeNormalization:
    """Test fft_time normalization matches GWpy."""

    def test_dc_not_doubled(self):
        """Test that DC component is not doubled."""
        # Constant signal
        data = np.ones((64, 2, 2, 2)) * 3.0
        field = ScalarField(
            data,
            axis0=np.arange(64) * 0.01 * u.s,
            axis0_domain="time",
        )
        result = field.fft_time()

        # DC component should be 3.0 (mean value normalized by /nfft)
        assert_allclose(result.value[0, 0, 0, 0], 3.0)

    def test_single_frequency_amplitude(self):
        """Test amplitude of single frequency sine wave.

        Use a frequency that falls exactly on a bin to avoid spectral leakage.
        """
        n = 64
        dt = 0.01  # s
        # Use frequency that falls exactly on a bin: f = k / (n * dt)
        # k = 10: f = 10 / 0.64 = 15.625 Hz
        freq = 10.0 / (n * dt)  # = 15.625 Hz, exactly on bin
        amp = 2.5
        t = np.arange(n) * dt

        # Sine wave at exact bin frequency
        signal = amp * np.sin(2 * np.pi * freq * t)
        # Broadcast to 4D
        data = signal[:, np.newaxis, np.newaxis, np.newaxis] * np.ones((n, 1, 1, 1))

        field = ScalarField(data, axis0=t * u.s, axis0_domain="time")
        result = field.fft_time()

        # Find the bin closest to the target frequency
        freqs = result._axis0_index.value
        freq_idx = np.argmin(np.abs(freqs - freq))

        # Amplitude should be close to original amplitude (for rfft normalization)
        # GWpy normalization: |FFT|/N * 2 for non-DC
        measured_amp = np.abs(result.value[freq_idx, 0, 0, 0])
        assert_allclose(measured_amp, amp, rtol=1e-10)


class TestScalarFieldFftTimeErrors:
    """Test fft_time error conditions."""

    def test_fft_time_wrong_domain_raises(self):
        """Test fft_time raises if not in time domain."""
        field = ScalarField(
            np.zeros((10, 4, 4, 4)),
            axis0_domain="frequency",
        )

        with pytest.raises(ValueError, match="requires axis0_domain='time'"):
            field.fft_time()


class TestScalarFieldIfftTimeBasic:
    """Test basic ifft_time functionality."""

    @pytest.fixture
    def freq_domain_field(self):
        """Create a frequency-domain ScalarField."""
        # Create from fft_time
        np.random.seed(42)
        data = np.random.randn(64, 4, 4, 4)
        field = ScalarField(
            data,
            axis0=np.arange(64) * 0.01 * u.s,
            axis0_domain="time",
        )
        return field.fft_time()

    def test_ifft_time_basic(self, freq_domain_field):
        """Test basic ifft_time execution."""
        result = freq_domain_field.ifft_time()

        assert isinstance(result, ScalarField)
        assert result.axis0_domain == "time"

    def test_ifft_time_axis_name_changes(self, freq_domain_field):
        """Test that axis0 name changes to 't'."""
        result = freq_domain_field.ifft_time()
        assert result.axis_names[0] == "t"

    def test_ifft_time_shape(self, freq_domain_field):
        """Test ifft_time output shape."""
        result = freq_domain_field.ifft_time()

        # (n_freq - 1) * 2 = (33 - 1) * 2 = 64
        assert result.shape[0] == 64


class TestScalarFieldFftIfftReversibility:
    """Test FFT/IFFT reversibility."""

    def test_ifft_fft_reversible(self):
        """Test ifft_time(fft_time(field)) recovers original."""
        np.random.seed(123)
        data = np.random.randn(64, 4, 4, 4)
        times = np.arange(64) * 0.01 * u.s

        original = ScalarField(data, axis0=times, axis0_domain="time", unit=u.m)

        # Round trip
        freq = original.fft_time()
        recovered = freq.ifft_time()

        assert_allclose(recovered.value, original.value, rtol=1e-10)
        assert recovered.axis0_domain == "time"
        assert recovered.unit == original.unit

    def test_ifft_fft_reversible_different_sizes(self):
        """Test reversibility with different even-length array sizes.

        Note: rfft/irfft round-trip is exact only for even-length arrays
        when nout is not explicitly specified.
        """
        for n in [32, 64, 100, 128]:
            np.random.seed(n)
            data = np.random.randn(n, 2, 2, 2)
            field = ScalarField(
                data, axis0=np.arange(n) * 0.01 * u.s, axis0_domain="time"
            )

            recovered = field.fft_time().ifft_time()
            assert_allclose(recovered.value, data, rtol=1e-10)


class TestScalarFieldIfftTimeErrors:
    """Test ifft_time error conditions."""

    def test_ifft_time_wrong_domain_raises(self):
        """Test ifft_time raises if not in frequency domain."""
        field = ScalarField(np.zeros((10, 4, 4, 4)), axis0_domain="time")

        with pytest.raises(ValueError, match="requires axis0_domain='frequency'"):
            field.ifft_time()


class TestScalarFieldFftTimeWithNfft:
    """Test fft_time with explicit nfft parameter."""

    def test_fft_time_nfft_larger(self):
        """Test fft_time with nfft larger than data."""
        data = np.random.randn(32, 2, 2, 2)
        field = ScalarField(data, axis0=np.arange(32) * 0.01 * u.s, axis0_domain="time")

        result = field.fft_time(nfft=64)

        # rfft of 64 points: 33 output
        assert result.shape[0] == 33

    def test_fft_time_nfft_smaller(self):
        """Test fft_time with nfft smaller than data (truncation)."""
        data = np.random.randn(64, 2, 2, 2)
        field = ScalarField(data, axis0=np.arange(64) * 0.01 * u.s, axis0_domain="time")

        result = field.fft_time(nfft=32)

        # rfft of 32 points: 17 output
        assert result.shape[0] == 17


class TestScalarFieldFftTimeDomainTransition:
    """Test domain state transitions during FFT."""

    def test_time_to_frequency_domain(self):
        """Test transition from time to frequency domain."""
        field = ScalarField(
            np.zeros((16, 4, 4, 4)),
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
        )

        result = field.fft_time()

        assert result.axis0_domain == "frequency"
        assert result.axis_names[0] == "f"

    def test_frequency_to_time_domain(self):
        """Test transition from frequency to time domain."""
        field = ScalarField(
            np.zeros((16, 4, 4, 4)),
            axis_names=["t", "x", "y", "z"],
            axis0_domain="time",
        )

        freq = field.fft_time()
        result = freq.ifft_time()

        assert result.axis0_domain == "time"
        assert result.axis_names[0] == "t"
