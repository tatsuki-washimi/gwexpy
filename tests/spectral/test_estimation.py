import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectral import bootstrap_spectrogram, calculate_correlation_factor


class _DummySpectrogram:
    def __init__(self, data, *, dt, df, frequencies, unit="1", name="spec"):
        self.value = np.asarray(data)
        self.dt = dt
        self.df = df
        self.frequencies = frequencies
        self.unit = unit
        self.name = name


def test_calculate_correlation_factor_no_overlap_is_one():
    factor = calculate_correlation_factor("hann", nperseg=256, noverlap=0, n_blocks=50)
    assert factor == pytest.approx(1.0)


def test_calculate_correlation_factor_overlap_inflates():
    factor = calculate_correlation_factor("hann", nperseg=256, noverlap=128, n_blocks=50)
    assert factor > 1.0


def test_bootstrap_spectrogram_overlap_scales_errors():
    rng = np.random.default_rng(123)
    n_time = 64
    n_freq = 32

    # Positive-valued spectrogram (e.g., ASD-like); variability over time is required.
    data = rng.lognormal(mean=0.0, sigma=0.5, size=(n_time, n_freq))
    freqs = np.linspace(10, 1000, n_freq) * u.Hz

    spec = _DummySpectrogram(
        data,
        dt=1.0 * u.s,
        df=1.0 * u.Hz,
        frequencies=freqs,
        unit="1",
        name="dummy",
    )

    np.random.seed(0)
    fs_no = bootstrap_spectrogram(
        spec,
        n_boot=200,
        average="mean",
        ci=0.68,
        window="hann",
        nperseg=256,
        noverlap=0,
    )

    np.random.seed(0)
    fs_ov = bootstrap_spectrogram(
        spec,
        n_boot=200,
        average="mean",
        ci=0.68,
        window="hann",
        nperseg=256,
        noverlap=128,
    )

    factor = calculate_correlation_factor("hann", nperseg=256, noverlap=128, n_blocks=n_time)

    base = fs_no.error_low.value
    scaled = fs_ov.error_low.value
    assert np.mean(base) > 0

    mask = base > 0
    ratio = scaled[mask] / base[mask]
    assert np.allclose(ratio, factor, rtol=1e-12, atol=1e-12)

