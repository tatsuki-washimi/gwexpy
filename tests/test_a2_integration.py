"""
Cross-module integration tests for Category A2 (Numerical Logic) in gwexpy.
These tests verify that the 'Contract' holds across different components and transformations.
"""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectral import estimate_psd
from gwexpy.timeseries import TimeSeries


class TestCrossModuleA2Integration:
    """Integration tests for numerical consistency."""

    def test_fft_ifft_scaling_contract(self):
        """
        Verify the FFT/IFFT scaling contract.
        FFT produces a Spectrum (V), IFFT should restore the original values.
        """
        # Create a basic sine wave
        fs = 1024
        t = np.arange(fs) / fs
        freq = 10.0
        amp = 2.0
        data = amp * np.sin(2 * np.pi * freq * t)
        ts = TimeSeries(data, dt=1.0 / fs * u.s, unit="V")

        # FFT (gwexpy FFT typically produces a Spectrum-like FrequencySeries)
        # We need to distinguish between 'steady' and 'transient' modes if applicable
        spec = ts.fft()
        assert isinstance(spec, FrequencySeries)
        assert spec.unit == u.V

        # IFFT back to time domain
        # Some implementations require explicit n or scaling
        ts_back = spec.ifft()

        # Verify amplitude restoration
        # Note: fft/ifft might have complex/real differences
        np.testing.assert_allclose(ts_back.value.real, ts.value, atol=1e-10)
        assert ts_back.unit == ts.unit
        assert ts_back.dt == ts.dt

    def test_psd_density_vs_spectrum_units(self):
        """
        Verify that scaling='density' and 'spectrum' return correct units.
        Contract:
          - density -> [Y]^2 / Hz
          - spectrum -> [Y]^2
        """
        fs = 1000
        data = np.random.normal(size=fs)
        ts = TimeSeries(data, dt=1.0 / fs * u.s, unit="V")

        # Density
        psd_density = estimate_psd(ts, scaling="density")
        assert psd_density.unit == u.V**2 / u.Hz

        # Spectrum
        psd_spectrum = estimate_psd(ts, scaling="spectrum")
        assert psd_spectrum.unit == u.V**2

    def test_f0_handling_at_dc(self):
        """
        Verify that DC component (f=0) doesn't cause errors and is handled correctly.
        """
        data = np.ones(1024)  # Constant offset
        ts = TimeSeries(data, dt=0.001 * u.s)

        spec = ts.fft()
        assert spec.frequencies[0] == 0 * u.Hz
        # DC component in FFT of ones should be non-zero (it's the sum)
        assert spec.value[0] != 0

        # Check if ifft restores it
        ts_back = spec.ifft()
        np.testing.assert_allclose(ts_back.value.real, 1.0, atol=1e-10)

    def test_cross_module_metadata_propagation(self):
        """
        TimeSeries -> Spectrogram (indirectly) -> FrequencySeries (PSD)
        Verify metadata propagation through a common analysis chain.
        """
        ts = TimeSeries(
            np.random.normal(size=2048),
            dt=1.0 / 1024 * u.s,
            epoch=1234567890,
            name="H1:TEST",
        )

        # Compute PSD (internally may go through overlapping windows)
        psd = ts.psd(fftlength=1)

        assert psd.epoch == ts.epoch
        assert psd.name == ts.name
        assert psd.unit == ts.unit**2 / u.Hz
        # FrequencySeries uses df = 1/fftlength
        assert psd.df == 1.0 * u.Hz

    def test_matrix_axis_mismatch_raises_error(self):
        """
        Verify that SeriesMatrix operations enforce axis alignment.
        No automatic interpolation for A2 operations.
        """
        from gwexpy.timeseries import TimeSeriesMatrix

        data1 = np.random.rand(1, 10)
        mat1 = TimeSeriesMatrix(data1, dt=1.0)

        data2 = np.random.rand(1, 10)
        mat2 = TimeSeriesMatrix(data2, dt=1.0001)  # Mismatched dt

        with pytest.raises(ValueError, match="mismatch"):
            _ = mat1 + mat2
