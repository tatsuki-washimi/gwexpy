"""
Test spectral estimation semantics for gwexpy.spectral module.

These tests fix the specification for:
1. Unit consistency (density vs spectrum)
2. Output type (FrequencySeries)
3. Averaging methods and error handling
"""

import numpy as np
import pytest
from astropy import units as u

pytest.importorskip("gwpy")
pytest.importorskip("scipy")

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectral import bootstrap_spectrogram, estimate_psd
from gwexpy.timeseries import TimeSeries


class TestSpectralSemantics:
    """Test A2 semantics of spectral estimation."""

    @pytest.fixture
    def noise_ts(self):
        """10 seconds of white noise at 1024 Hz."""
        np.random.seed(42)
        data = np.random.randn(10240)
        return TimeSeries(data, sample_rate=1024, unit="V")

    def test_estimate_psd_returns_frequencyseries(self, noise_ts):
        """Test that estimate_psd returns FrequencySeries instance."""
        psd = estimate_psd(noise_ts, fftlength=1.0)
        assert isinstance(psd, FrequencySeries)

    def test_estimate_psd_density_units(self, noise_ts):
        """Test that PSD returns V^2/Hz for input in V (density normalization)."""
        psd = estimate_psd(noise_ts, fftlength=1.0)
        # PSD unit should be input_unit**2 / Hz
        expected_unit = u.V**2 / u.Hz
        assert psd.unit.is_equivalent(expected_unit), (
            f"Expected {expected_unit}, got {psd.unit}"
        )

    def test_estimate_psd_frequency_axis_consistency(self, noise_ts):
        """Test that frequency axis is consistent with fftlength."""
        fftlength = 1.0  # 1 second -> df = 1 Hz
        psd = estimate_psd(noise_ts, fftlength=fftlength)

        # df should be 1/fftlength
        expected_df = 1.0 / fftlength
        assert psd.df.value == pytest.approx(expected_df)

        # First frequency should be 0 Hz (DC)
        assert psd.frequencies[0].value == 0.0

        # Nyquist frequency should be sample_rate / 2
        nyquist = 1024 / 2
        assert psd.frequencies[-1].value == pytest.approx(nyquist)

    def test_estimate_psd_different_input_units(self):
        """Test that output units scale correctly with input units."""
        np.random.seed(123)
        data = np.random.randn(4096)

        # Test with different units
        for input_unit in ["m", "s", "V", "m/s"]:
            ts = TimeSeries(data, sample_rate=256, unit=input_unit)
            psd = estimate_psd(ts, fftlength=1.0)

            expected_unit = u.Unit(input_unit) ** 2 / u.Hz
            assert psd.unit.is_equivalent(expected_unit), (
                f"For input {input_unit}, expected {expected_unit}, got {psd.unit}"
            )

    def test_estimate_psd_methods(self, noise_ts):
        """Test different averaging methods produce valid results."""
        # Note: Only 'median' and 'welch' are registered in GWpy's PSD registry
        # 'mean' is not a valid method name
        for method in ["median", "welch"]:
            psd = estimate_psd(noise_ts, fftlength=1.0, method=method)
            assert isinstance(psd, FrequencySeries)
            assert not np.isnan(psd.value).any()
            assert not np.isinf(psd.value).any()
            # PSD should be positive
            assert (psd.value >= 0).all()

    def test_estimate_psd_spectrum_scaling(self, noise_ts):
        """Test that scaling='spectrum' returns V^2 (not V^2/Hz)."""
        # Scipy backend supports scaling='spectrum'
        spec = estimate_psd(noise_ts, fftlength=1.0, scaling="spectrum")
        # Unit should be V^2
        assert spec.unit == u.V**2
        assert not spec.unit.is_equivalent(u.V**2 / u.Hz)


class TestBootstrapSemantics:
    """Test bootstrap spectrogram semantics."""

    @pytest.fixture
    def dummy_spectrogram(self):
        """Create a dummy spectrogram for testing."""

        class DummySpectrogram:
            def __init__(self):
                np.random.seed(42)
                self.value = np.abs(np.random.randn(32, 64)) + 0.1  # Positive values
                self.frequencies = np.linspace(0, 512, 64) * u.Hz
                self.dt = 1.0 * u.s
                self.df = 8.0 * u.Hz
                self.unit = u.V**2 / u.Hz
                self.name = "test_spectrogram"

        return DummySpectrogram()

    def test_bootstrap_returns_frequencyseries(self, dummy_spectrogram):
        """Test that bootstrap returns FrequencySeries."""
        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50)
        assert isinstance(result, FrequencySeries)

    def test_bootstrap_preserves_units(self, dummy_spectrogram):
        """Test that bootstrap preserves input units."""
        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50)
        assert result.unit == dummy_spectrogram.unit

    def test_bootstrap_has_error_attributes(self, dummy_spectrogram):
        """Test that bootstrap result has error_low and error_high attributes."""
        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50)

        assert hasattr(result, "error_low")
        assert hasattr(result, "error_high")
        assert isinstance(result.error_low, FrequencySeries)
        assert isinstance(result.error_high, FrequencySeries)

    def test_bootstrap_error_units(self, dummy_spectrogram):
        """Test that error bars have same units as result."""
        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50)

        assert result.error_low.unit == result.unit
        assert result.error_high.unit == result.unit

    def test_bootstrap_frequency_consistency(self, dummy_spectrogram):
        """Test that output frequencies match input."""
        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50)

        np.testing.assert_array_equal(
            result.frequencies.value, dummy_spectrogram.frequencies.value
        )

    def test_bootstrap_methods_produce_different_results(self, dummy_spectrogram):
        """Test that median and mean methods produce different results."""
        np.random.seed(42)
        median_result = bootstrap_spectrogram(
            dummy_spectrogram, n_boot=100, method="median"
        )

        np.random.seed(42)
        mean_result = bootstrap_spectrogram(
            dummy_spectrogram, n_boot=100, method="mean"
        )

        # Results should be different (though may be close for normal-ish data)
        # At minimum, they should both be valid
        assert not np.allclose(median_result.value, mean_result.value)

    def test_bootstrap_rebin_reduces_frequency_bins(self, dummy_spectrogram):
        """Test that rebin_width reduces the number of frequency bins."""
        original_nfreq = len(dummy_spectrogram.frequencies)

        result = bootstrap_spectrogram(dummy_spectrogram, n_boot=50, rebin_width=32.0)

        # With rebinning, should have fewer frequency bins
        assert len(result.frequencies) < original_nfreq


class TestEstimatePsdEdgeCases:
    """Test edge cases and error handling for estimate_psd."""

    def test_rejects_nan_input(self):
        """Test that NaN in input raises ValueError."""
        data = np.ones(1024)
        data[100] = np.nan
        ts = TimeSeries(data, sample_rate=256)

        with pytest.raises(ValueError, match="NaN"):
            estimate_psd(ts, fftlength=1.0)

    def test_rejects_fftlength_exceeding_duration(self):
        """Test that fftlength > duration raises ValueError."""
        data = np.ones(256)  # 1 second at 256 Hz
        ts = TimeSeries(data, sample_rate=256)

        with pytest.raises(ValueError, match="fftlength"):
            estimate_psd(ts, fftlength=2.0)  # Request 2 seconds

    def test_dimensionless_input(self):
        """Test PSD of dimensionless input has correct units."""
        np.random.seed(42)
        data = np.random.randn(4096)
        ts = TimeSeries(data, sample_rate=256)  # No unit specified

        psd = estimate_psd(ts, fftlength=1.0)
        # Should be 1/Hz (dimensionless^2 / Hz)
        assert psd.unit.is_equivalent(1 / u.Hz)
