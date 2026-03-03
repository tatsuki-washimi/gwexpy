import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def test_asd_accepts_nfft_noverlap_and_fftlength_overlap():
    rng = np.random.default_rng(2026)
    ts = TimeSeries(rng.standard_normal(8192), sample_rate=1024.0)

    from_samples = ts.asd(nfft=256, noverlap=128)
    from_seconds = ts.asd(fftlength=0.25, overlap=0.125)

    assert from_samples.size > 0
    assert from_seconds.size > 0
    assert np.isfinite(from_samples.value).all()
    assert np.isfinite(from_seconds.value).all()


def test_asd_nperseg_is_rejected():
    rng = np.random.default_rng(2027)
    ts = TimeSeries(rng.standard_normal(4096), sample_rate=1024.0)
    with pytest.raises(TypeError):
        ts.asd(nperseg=256)
