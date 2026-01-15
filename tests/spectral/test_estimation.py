import numpy as np
import pytest
from astropy import units as u

pytest.importorskip("gwpy")
pytest.importorskip("scipy")

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectral import (
    bootstrap_spectrogram,
    calculate_correlation_factor,
    estimate_psd,
)
from gwexpy.timeseries import TimeSeries


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


def test_estimate_psd_white_noise_mean_matches_variance():
    rng = np.random.default_rng(123)
    n_samples = 16384
    dt = 0.5
    fftlength = 128.0

    data = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    ts = TimeSeries(data, dt=dt)

    psd = estimate_psd(
        ts,
        fftlength=fftlength,
        overlap=0.0,
        window="hann",
        method="welch",
    )

    expected = 1.0
    assert np.allclose(psd.mean().value, expected, rtol=0.1)


def test_estimate_psd_frequency_axis_definition():
    rng = np.random.default_rng(321)
    n_samples = 4096
    dt = 0.5
    fftlength = 128.0

    data = rng.normal(size=n_samples)
    ts = TimeSeries(data, dt=dt)

    psd = estimate_psd(
        ts,
        fftlength=fftlength,
        overlap=0.0,
        window="hann",
        method="welch",
    )

    df = 1.0 / fftlength
    assert psd.dx.value == pytest.approx(df)
    assert psd.xindex[0].value == 0.0
    assert np.allclose(np.diff(psd.xindex.value), df)
    assert psd.xindex[-1].value == pytest.approx(0.5 / dt)


def test_estimate_psd_rejects_nan():
    data = np.ones(1024)
    data[10] = np.nan
    ts = TimeSeries(data, dt=0.5)

    with pytest.raises(ValueError):
        estimate_psd(ts, fftlength=128.0, overlap=0.0, window="hann", method="welch")


def test_estimate_psd_rejects_short_input():
    data = np.ones(100)
    ts = TimeSeries(data, dt=0.5)

    with pytest.raises(ValueError):
        estimate_psd(ts, fftlength=128.0, overlap=0.0, window="hann", method="welch")


def test_estimate_psd_rejects_irregular_series():
    times = np.array([0.0, 1.0, 2.0, 4.0])
    ts = TimeSeries([1.0, 2.0, 3.0, 4.0], times=times, unit=u.m)

    with pytest.raises(ValueError):
        estimate_psd(ts, fftlength=2.0, overlap=0.0, window="hann", method="welch")


def test_estimate_psd_api_metadata():
    rng = np.random.default_rng(7)
    data = rng.normal(size=1024)

    ts = TimeSeries(
        data,
        dt=0.5,
        unit=u.m,
        name="test-psd",
        channel="X1:TEST",
        epoch=1234567890,
    )

    psd = estimate_psd(
        ts,
        fftlength=128.0,
        overlap=0.0,
        window="hann",
        method="welch",
    )

    assert isinstance(psd, FrequencySeries)
    assert psd.unit.is_equivalent(ts.unit**2 / u.Hz)
    assert psd.name == ts.name
    assert psd.channel == ts.channel
    assert psd.epoch == ts.epoch
