from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix

pytestmark = pytest.mark.filterwarnings(
    "ignore:xindex was given to TimeSeries\\(\\), x0 will be ignored:UserWarning"
)


def test_transient_fft_one_sided_frequency_axis_even_and_odd_lengths():
    sample_rate = 16.0

    for n_samples in (15, 16):
        data = np.ones(n_samples)
        series = TimeSeries(data, sample_rate=sample_rate, unit=u.m)

        spectrum = series.fft(mode="transient")

        expected = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        np.testing.assert_allclose(spectrum.frequencies.to_value(u.Hz), expected)
        assert len(spectrum) == n_samples // 2 + 1
        assert spectrum.frequencies[0].to_value(u.Hz) == pytest.approx(0.0)
        if n_samples % 2 == 0:
            assert spectrum.frequencies[-1].to_value(u.Hz) == pytest.approx(
                sample_rate / 2.0
            )
        else:
            assert spectrum.frequencies[-1].to_value(u.Hz) < sample_rate / 2.0


@pytest.mark.parametrize("method_name", ["fft", "psd", "asd"])
def test_frequency_domain_transforms_reject_irregular_sampling(method_name):
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 1.0, 3.0] * u.s)

    method = getattr(series, method_name)

    with pytest.raises(ValueError, match="requires a regular sample rate"):
        method()


def _make_heterogeneous_matrix() -> TimeSeriesMatrix:
    sample_rate = 256.0
    t = np.arange(512) / sample_rate
    data = np.stack(
        [
            np.sin(2.0 * np.pi * 16.0 * t),
            0.5 * np.cos(2.0 * np.pi * 32.0 * t),
        ]
    ).reshape(2, 1, -1)
    return TimeSeriesMatrix(
        data,
        dt=(1.0 / sample_rate) * u.s,
        t0=0 * u.s,
        rows=["displacement", "voltage"],
        cols=["value"],
        units=[[u.m], [u.V]],
        names=[["DISP"], ["VOLT"]],
        channels=[["H1:DISP"], ["H1:VOLT"]],
    )


def test_public_matrix_fft_psd_asd_preserve_axis_and_per_element_metadata():
    matrix = _make_heterogeneous_matrix()

    fft = matrix.fft()
    psd = matrix.psd(fftlength=0.25, overlap=0.0)
    asd = matrix.asd(fftlength=0.25, overlap=0.0)

    np.testing.assert_allclose(fft.frequencies.to_value(u.Hz), np.arange(257) * 0.5)
    np.testing.assert_allclose(psd.frequencies.to_value(u.Hz), np.arange(33) * 4.0)
    np.testing.assert_allclose(
        psd.frequencies.to_value(u.Hz),
        asd.frequencies.to_value(u.Hz),
    )

    assert list(fft.rows.keys()) == ["displacement", "voltage"]
    assert list(psd.cols.keys()) == ["value"]
    assert fft[0, 0].unit == u.m
    assert fft[1, 0].unit == u.V
    assert psd[0, 0].unit.is_equivalent(u.m**2 / u.Hz)
    assert psd[1, 0].unit.is_equivalent(u.V**2 / u.Hz)
    assert asd[0, 0].unit.is_equivalent(u.m / (u.Hz**0.5))
    assert asd[1, 0].unit.is_equivalent(u.V / (u.Hz**0.5))
    assert fft[0, 0].name == "DISP"
    assert str(fft[1, 0].channel) == "H1:VOLT"


def test_public_matrix_csd_and_coherence_preserve_frequency_axis_and_bounds():
    matrix = _make_heterogeneous_matrix()

    csd = matrix.csd(matrix, fftlength=0.25, overlap=0.0)
    coherence = matrix.coherence(matrix, fftlength=0.25, overlap=0.0)

    np.testing.assert_allclose(
        csd.frequencies.to_value(u.Hz),
        coherence.frequencies.to_value(u.Hz),
    )
    assert list(csd.rows.keys()) == ["displacement", "voltage"]
    assert list(csd.cols.keys()) == ["value"]
    assert csd[0, 0].unit.is_equivalent(u.m**2 / u.Hz)
    assert csd[1, 0].unit.is_equivalent(u.V**2 / u.Hz)
    assert csd[0, 0].name == "DISP---DISP"
    assert str(csd[1, 0].channel) == "H1:VOLT"
    assert np.nanmin(coherence.value) >= -1e-12
    assert np.nanmax(coherence.value) <= 1.0 + 1e-12


def test_public_matrix_spectrogram_preserves_time_frequency_axes_and_metadata():
    sample_rate = 256.0
    t = np.arange(512) / sample_rate
    data = np.stack(
        [
            np.sin(2.0 * np.pi * 16.0 * t),
            np.cos(2.0 * np.pi * 32.0 * t),
        ]
    ).reshape(2, 1, -1)
    matrix = TimeSeriesMatrix(
        data,
        dt=(1.0 / sample_rate) * u.s,
        t0=0 * u.s,
        rows=["a", "b"],
        cols=["value"],
        unit=u.m,
        names=[["A"], ["B"]],
        channels=[["H1:A"], ["H1:B"]],
    )

    spectrogram = matrix.spectrogram(stride=0.25, fftlength=0.25, overlap=0.0)

    assert spectrogram.shape[:2] == (2, 1)
    np.testing.assert_allclose(spectrogram.times.to_value(u.s), np.arange(8) * 0.25)
    np.testing.assert_allclose(
        spectrogram.frequencies.to_value(u.Hz),
        np.arange(33) * 4.0,
    )
    assert spectrogram.unit.is_equivalent(u.m**2 / u.Hz)
    assert list(spectrogram.rows.keys()) == ["a", "b"]
    assert list(spectrogram.cols.keys()) == ["value"]
    assert spectrogram.meta[0, 0].unit.is_equivalent(u.m**2 / u.Hz)
    assert spectrogram.meta[0, 0].name == "A"
    assert str(spectrogram.meta[1, 0].channel) == "H1:B"
