import numpy as np
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram


def test_spectrogram_bootstrap_asd_wrapper():
    rng = np.random.default_rng(0)
    data = rng.lognormal(mean=0.0, sigma=0.5, size=(32, 16))

    spec = Spectrogram(
        data,
        dt=1.0 * u.s,
        f0=10.0 * u.Hz,
        df=1.0 * u.Hz,
        unit="1",
        name="dummy",
    )

    np.random.seed(0)
    fs = spec.bootstrap_asd(
        n_boot=50,
        average="mean",
        ci=0.68,
        window="hann",
        fftlength=256.0,
        overlap=0.0,
    )

    assert isinstance(fs, FrequencySeries)
    assert fs.value.shape == (16,)
    assert fs.error_low.value.shape == (16,)
    assert fs.error_high.value.shape == (16,)
