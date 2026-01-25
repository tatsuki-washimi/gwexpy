import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

SCIPY_AVAILABLE = False
try:
    import scipy  # noqa: F401 - availability check

    SCIPY_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available")
class TestTransforms:
    @pytest.fixture
    def sine_wave(self):
        # 1 second of 50Hz sine wave, 1024 Hz sampling
        dt = 1 / 1024.0
        t = np.arange(0, 1.0, dt)
        data = np.sin(2 * np.pi * 50 * t)
        return TimeSeries(data, t0=0, dt=dt, unit="V", name="sine")

    def test_dct_roundtrip(self, sine_wave):
        ts = sine_wave
        # DCT
        fs = ts.dct(type=2, norm="ortho")
        assert isinstance(fs, FrequencySeries)
        assert getattr(fs, "transform", "") == "dct"
        assert getattr(fs, "dct_type", 0) == 2
        assert getattr(fs, "dct_norm", "") == "ortho"
        assert getattr(fs, "original_n", 0) == len(ts)
        assert fs.dt == ts.dt

        # IDCT
        rec = fs.idct(type=2, norm="ortho")
        assert isinstance(rec, TimeSeries)
        assert np.allclose(rec.value, ts.value, atol=1e-7)
        assert rec.dt == ts.dt

    def test_dct_window(self, sine_wave):
        # Window
        ts = sine_wave
        fs = ts.dct(window="hann")
        assert len(fs) == len(ts)
        # Cannot invert directly to original because of window

    def test_dct_nondetrend(self, sine_wave):
        ts = sine_wave + 10 * sine_wave.unit  # Offset
        fs = ts.dct(detrend=True)
        # Check DC component roughly 0?
        # dct value at k=0 is related to sum.
        # If detrended, sum should be near 0.
        # Loose check if it runs and returns reasonable values
        assert isinstance(fs, FrequencySeries)

    def test_cepstrum_real(self, sine_wave):
        ts = sine_wave
        ceps = ts.cepstrum(kind="real")
        assert isinstance(ceps, FrequencySeries)
        assert getattr(ceps, "axis_type", "") == "quefrency"
        assert ceps.frequencies.unit == u.s
        assert len(ceps) == len(ts)

    def test_cepstrum_complex(self, sine_wave):
        ceps = sine_wave.cepstrum(kind="complex")
        assert len(ceps) == len(sine_wave)

    def test_cwt_ndarray(self, sine_wave):
        try:
            import pywt  # noqa: F401 - availability check
        except ImportError:
            pytest.skip("pywt (PyWavelets) not found")

        # Use scales expected by pywt.
        # For 'cmor1.5-1.0', central freq is approx 1?
        # Let's just run with some scales
        scales = np.arange(1, 31)
        cwt_mat, freqs = sine_wave.cwt(
            widths=scales, output="ndarray", wavelet="cmor1.5-1.0"
        )
        assert cwt_mat.shape == (len(scales), len(sine_wave))
        assert freqs.unit == u.Hz

    def test_cwt_spectrogram(self, sine_wave):
        try:
            import pywt  # noqa: F401 - availability check
        except ImportError:
            pytest.skip("pywt (PyWavelets) not found")

        frequencies = np.linspace(30, 70, 41)
        spec = sine_wave.cwt(
            frequencies=frequencies, output="spectrogram", wavelet="cmor1.5-1.0"
        )
        # Check type
        from gwpy.spectrogram import Spectrogram

        assert isinstance(spec, Spectrogram)
        assert spec.shape == (len(sine_wave), len(frequencies))

        # Check signal presence around 50Hz
        # Spectrogram values are complex CWT coefs.
        # Check magnitude
        mag = np.abs(spec.value)
        # Average over time (axis 0 is time for Spectrogram)
        mean_spec = mag.mean(axis=0)
        peak_idx = np.argmax(mean_spec)
        peak_freq = spec.frequencies[peak_idx].value
        assert 45 < peak_freq < 55
