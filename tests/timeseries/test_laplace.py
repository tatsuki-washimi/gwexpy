import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries


class TestLaplace:
    @pytest.fixture
    def exp_decay(self):
        # x(t) = exp(-a * t)
        # L(s) = 1 / (s + a)
        dt = 0.01
        t = np.arange(0, 10.0, dt)
        a = 2.0
        data = np.exp(-a * t)
        return TimeSeries(data, t0=0, dt=dt, unit="V", name="decay"), a

    def test_laplace_basic(self, exp_decay):
        ts, a = exp_decay
        sigma = 0.0

        # Calculate Laplace (Fourier mode since sigma=0) at f=0
        # Expected: integral_0^10 exp(-2t) dt = [-1/2 exp(-2t)]_0^10
        # = 0.5 * (1 - exp(-20)) approx 0.5

        frequencies = np.array([0.0]) * u.Hz
        fs = ts.laplace(sigma=sigma, frequencies=frequencies, normalize="integral")

        assert isinstance(fs, FrequencySeries)
        assert len(fs) == 1
        assert fs.unit == u.V * u.s
        assert np.isclose(fs.value[0].real, 0.5, atol=1e-2)
        assert hasattr(fs, "laplace_sigma")
        assert fs.laplace_sigma == 0.0

    def test_laplace_sigma(self, exp_decay):
        ts, a = exp_decay
        # Use sigma which cancels decay? No, s = sigma + iw
        # L(s) = integral x(t) exp(-s t) dt
        # If sigma = -1, then exp(-2t) * exp(-(-1)t) = exp(-t)
        # Integ exp(-t) from 0 to 10 = 1 - exp(-10) approx 1.0

        fs = ts.laplace(sigma=-1.0, frequencies=[0], normalize="integral")
        assert np.isclose(fs.value[0].real, 1.0, atol=1e-2)

    def test_laplace_default_freqs(self, exp_decay):
        ts, _ = exp_decay
        fs = ts.laplace()
        # Should match rfftfreq length
        n = len(ts)
        expected_len = n // 2 + 1
        assert len(fs) == expected_len
        assert fs.frequencies[0].value == 0

    def test_chunking(self, exp_decay):
        ts, _ = exp_decay
        frequencies = np.linspace(0, 10, 100) * u.Hz

        # All at once
        fs1 = ts.laplace(frequencies=frequencies)

        # Chunked
        fs2 = ts.laplace(frequencies=frequencies, chunk_size=10)

        assert np.allclose(fs1.value, fs2.value)

    def test_window_detrend(self, exp_decay):
        ts, _ = exp_decay
        # Just check it runs without error
        fs = ts.laplace(window="hann", detrend=True)
        assert len(fs) > 0

    def test_cropping(self, exp_decay):
        ts, _ = exp_decay
        # Crop 0 to 1 sec
        fs = ts.laplace(
            t_start=0 * u.s, t_stop=1 * u.s, frequencies=[0], normalize="integral"
        )

        # Integral exp(-2t) from 0 to 1 = 0.5 * (1 - exp(-2))
        expected = 0.5 * (1 - np.exp(-2))
        assert np.isclose(fs.value[0].real, expected, atol=1e-2)
