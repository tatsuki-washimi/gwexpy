from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


def _deterministic_timeseries() -> TimeSeries:
    sample_rate = 1024.0
    t = np.arange(1024) / sample_rate
    data = 0.5 * np.sin(2.0 * np.pi * 64.0 * t) + 0.25 * np.cos(2.0 * np.pi * 128.0 * t)
    return TimeSeries(
        data,
        sample_rate=sample_rate,
        unit=u.m,
        name="H1_TEST",
        channel="H1:TEST",
        epoch=0,
    )


def test_spectral_wrappers_preserve_density_spectrum_units_and_frequency_axes():
    ts = _deterministic_timeseries()

    psd_density = ts.psd(fftlength=0.25, overlap=0.0, scaling="density")
    psd_spectrum = ts.psd(fftlength=0.25, overlap=0.0, scaling="spectrum")
    asd = ts.asd(fftlength=0.25, overlap=0.0)
    csd = ts.csd(ts, fftlength=0.25, overlap=0.0)
    spectrogram = ts.spectrogram(stride=0.25, fftlength=0.25, overlap=0.0)

    assert psd_density.unit.is_equivalent(u.m**2 / u.Hz)
    assert psd_spectrum.unit.is_equivalent(u.m**2)
    assert asd.unit.is_equivalent(u.m / (u.Hz**0.5))
    assert csd.unit.is_equivalent(u.m**2 / u.Hz)
    assert spectrogram.unit.is_equivalent(u.m**2 / u.Hz)

    np.testing.assert_allclose(
        psd_density.frequencies.to_value(u.Hz),
        asd.frequencies.to_value(u.Hz),
    )
    np.testing.assert_allclose(
        psd_density.frequencies.to_value(u.Hz),
        csd.frequencies.to_value(u.Hz),
    )
    np.testing.assert_allclose(
        psd_density.frequencies.to_value(u.Hz),
        spectrogram.frequencies.to_value(u.Hz),
    )
    assert psd_density.frequencies[0].to_value(u.Hz) == pytest.approx(0.0)
    assert np.all(np.diff(psd_density.frequencies.to_value(u.Hz)) > 0.0)

    assert psd_density.name == ts.name
    assert str(psd_density.channel) == str(ts.channel)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Transient FFT currently passes dt.value directly to rfftfreq for "
        "non-second TimeSeries dt units; fix requires Wave 3 physics review."
    ),
)
def test_transient_fft_frequency_axis_uses_seconds_for_quantity_dt():
    data = np.ones(16)
    ts = TimeSeries(data, dt=1 * u.ms, unit=u.m)

    spectrum = ts.fft(mode="transient")

    expected = np.fft.rfftfreq(len(data), d=(1 * u.ms).to_value(u.s))
    np.testing.assert_allclose(spectrum.frequencies.to_value(u.Hz), expected)


def test_vectorized_matrix_fft_constructor_dt_default_axis_preserves_metadata():
    data = np.arange(32, dtype=float).reshape(2, 1, 16)
    matrix = TimeSeriesMatrix(
        data,
        dt=1 * u.ms,
        t0=0 * u.s,
        rows=["row_a", "row_b"],
        cols=["value"],
        units=[[u.m], [u.V]],
        names=[["a"], ["b"]],
        channels=[["H1:A"], ["H1:B"]],
    )

    spectrum = matrix._vectorized_fft()

    expected = np.fft.rfftfreq(data.shape[-1], d=(1 * u.ms).to_value(u.s))
    np.testing.assert_allclose(spectrum.frequencies.to_value(u.Hz), expected)
    assert list(spectrum.rows.keys()) == ["row_a", "row_b"]
    assert list(spectrum.cols.keys()) == ["value"]
    assert spectrum[0, 0].unit == u.m
    assert spectrum[1, 0].unit == u.V
    assert spectrum[0, 0].name == "a"
    assert str(spectrum[1, 0].channel) == "H1:B"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TimeSeriesMatrix vectorized FFT currently passes dt.value directly to "
        "rfftfreq when the matrix axis unit is explicitly non-second."
    ),
)
def test_vectorized_matrix_fft_frequency_axis_uses_seconds_for_explicit_xunit():
    data = np.arange(16, dtype=float).reshape(1, 1, 16)
    matrix = TimeSeriesMatrix(data, dt=1 * u.ms, t0=0 * u.ms, xunit="ms")

    spectrum = matrix._vectorized_fft()

    expected = np.fft.rfftfreq(data.shape[-1], d=(1 * u.ms).to_value(u.s))
    np.testing.assert_allclose(spectrum.frequencies.to_value(u.Hz), expected)


def test_vectorized_asd_current_contract_is_square_root_of_vectorized_psd():
    rng = np.random.default_rng(273)
    matrix = TimeSeriesMatrix(
        rng.normal(size=(2, 1, 256)),
        dt=1 * u.ms,
        t0=0 * u.s,
    )

    psd = matrix._vectorized_psd(fftlength=0.032, overlap=0.0)
    asd = matrix._vectorized_asd(fftlength=0.032, overlap=0.0)

    np.testing.assert_allclose(
        asd.frequencies.to_value(u.Hz),
        psd.frequencies.to_value(u.Hz),
    )
    np.testing.assert_allclose(asd.value**2, psd.value, rtol=1e-12, atol=1e-18)
    assert np.all(asd.value >= 0.0)
