import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix


class TestFrequencySeriesCalculus:
    """Test A2-b calculus methods in FrequencySeries."""

    def test_integrate_time_dc_handling(self):
        """Test that time integration handles DC component (f=0) by setting it to 0."""
        data = np.array([1.0, 2.0, 3.0])
        freqs = np.array([0.0, 1.0, 2.0])
        fs = FrequencySeries(data, frequencies=freqs, unit="m")

        # Integrate time: should divide by (i * 2pi * f)
        # f=0 would be division by zero, expectation is 0j (DC blocking/stability)
        integ = fs.integrate_time()

        assert integ.value[0] == 0j, "DC component must be 0j to avoid singularity"
        assert not np.isnan(integ.value).any()
        assert not np.isinf(integ.value).any()
        assert integ.unit == u.m * u.s

    def test_differentiate_time_unit(self):
        """Test time differentiation unit propagation."""
        fs = FrequencySeries([1, 2, 3], df=1, f0=0, unit="m")
        diff = fs.differentiate_time()
        # d/dt -> * (i * 2pi * f) -> unit should be * Hz
        assert diff.unit == u.m * u.Hz

    def test_group_delay_unit(self):
        """Test group delay returns seconds."""
        # Use a frequency array with finer resolution and smaller phase slope
        # to ensure unwrap works correctly (phase changes less than pi per step)
        freqs = np.linspace(1, 10, 100)  # finer resolution
        delay = 0.05  # 50ms delay -> phase changes smoothly
        phase = -2.0 * np.pi * delay * freqs
        data = np.exp(1j * phase)
        fs = FrequencySeries(data, frequencies=freqs, unit="V")

        gd = fs.group_delay()

        assert gd.unit == u.s
        # Check value ignoring boundaries (gradient is sensitive at edges)
        np.testing.assert_allclose(gd.value[5:-5], delay, atol=1e-5)


class TestMatrixTransform:
    """Test A2-c transformation methods (IFFT)."""

    def test_matrix_ifft_scaling_roundtrip(self):
        """Verify amplitude conservation in Matrix IFFT vs GWpy FFT.
        GWpy's fft() scales amplitudes by 2 (one-sided).
        FrequencySeriesMatrix.ifft() must reverse this scaling (x0.5).
        """
        ts = TimeSeries([1, 0, -1, 0, 2, 0, -2, 0], sample_rate=8, unit="V")
        fs = ts.fft()

        matrix = FrequencySeriesMatrix([fs])
        reconstructed = matrix.ifft()

        # TimeSeriesMatrix returns at shape (1, 1, 8), extract first element
        # For unit check, use meta attribute; for data, use the sliced value
        recon_data = reconstructed.value[0, 0]
        recon_unit = reconstructed.meta[0, 0].unit

        assert recon_unit == ts.unit
        np.testing.assert_allclose(
            recon_data, ts.value, atol=1e-7, err_msg="IFFT amplitude mismatch"
        )
