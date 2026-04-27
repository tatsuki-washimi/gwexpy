"""Contract tests for Q-transform passthrough and gwexpy containers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram as GWPYSpectrogram

from gwexpy.spectrogram import (
    Spectrogram as GWEXSpectrogram,
)
from gwexpy.spectrogram import (
    SpectrogramDict,
    SpectrogramList,
    SpectrogramMatrix,
)
from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:xindex was given to TimeSeries\\(\\), x0 will be ignored:UserWarning"
)


def _q_kwargs() -> dict[str, object]:
    return {
        "qrange": (4, 8),
        "frange": (16, 64),
        "outseg": (0.5, 1.5),
        "tres": 0.1,
        "fres": 4,
        "whiten": False,
    }


def _series(
    freq: float = 32.0, *, name: str = "sig", channel: str = "H1:SIG"
) -> TimeSeries:
    sample_rate = 256.0
    times = np.arange(512) / sample_rate
    data = np.sin(2.0 * np.pi * freq * times)
    return TimeSeries(
        data,
        sample_rate=sample_rate * u.Hz,
        t0=0 * u.s,
        unit="V",
        name=name,
        channel=channel,
    )


def _assert_q_axes(qgram) -> None:
    assert qgram.shape == (10, 12)
    np.testing.assert_allclose(qgram.times.to_value(u.s), np.arange(0.5, 1.5, 0.1))
    np.testing.assert_allclose(qgram.frequencies.to_value(u.Hz), np.arange(16, 64, 4))
    assert qgram.unit == u.dimensionless_unscaled
    assert np.all(np.isfinite(qgram.value))


def _gps_seconds(epoch) -> float:
    return float(getattr(epoch, "gps", epoch))


def test_timeseries_q_transform_is_gwpy_passthrough_with_interpolated_axes():
    ts = _series()

    qgram = ts.q_transform(**_q_kwargs())

    assert isinstance(qgram, GWPYSpectrogram)
    assert not isinstance(qgram, GWEXSpectrogram)
    _assert_q_axes(qgram)
    assert _gps_seconds(qgram.epoch) == pytest.approx(0.5)
    assert qgram.name is None
    assert qgram.channel is None


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="TimeSeries.q_transform is currently GWpy passthrough and does not use gwexpy regularity errors.",
)
def test_timeseries_q_transform_irregular_sampling_contract_gap():
    irregular = TimeSeries([1.0, 2.0, 3.0, 4.0], times=[0.0, 1.0, 2.0, 4.0] * u.s)

    irregular.q_transform(**_q_kwargs())


def test_timeseries_collection_q_transform_returns_gwexpy_spectrogram_containers():
    first = _series(32.0, name="first", channel="H1:FIRST")
    second = _series(40.0, name="second", channel="H1:SECOND")

    qlist = TimeSeriesList([first, second]).q_transform(**_q_kwargs())
    qdict = TimeSeriesDict({"first": first, "second": second}).q_transform(
        **_q_kwargs()
    )

    assert isinstance(qlist, SpectrogramList)
    assert isinstance(qdict, SpectrogramDict)
    assert list(qdict.keys()) == ["first", "second"]

    for qgram in [*qlist, *qdict.values()]:
        assert isinstance(qgram, GWEXSpectrogram)
        _assert_q_axes(qgram)
        assert qgram.name is None
        assert qgram.channel is None


def test_timeseries_matrix_q_transform_combines_common_axes_and_records_metadata_loss():
    sample_rate = 256.0
    times = np.arange(512) / sample_rate
    data = np.stack(
        [
            np.sin(2.0 * np.pi * 32.0 * times),
            np.sin(2.0 * np.pi * 40.0 * times),
        ]
    ).reshape(2, 1, -1)
    matrix = TimeSeriesMatrix(
        data,
        sample_rate=sample_rate * u.Hz,
        t0=0 * u.s,
        rows=["a", "b"],
        cols=["value"],
        unit=u.V,
        names=[["A"], ["B"]],
        channels=[["H1:A"], ["H1:B"]],
    )

    qmatrix = matrix.q_transform(**_q_kwargs())

    assert isinstance(qmatrix, SpectrogramMatrix)
    assert qmatrix.shape == (2, 1, 10, 12)
    assert qmatrix.unit == u.dimensionless_unscaled
    assert _gps_seconds(qmatrix.epoch) == pytest.approx(0.5)
    assert list(qmatrix.rows.keys()) == ["a", "b"]
    assert list(qmatrix.cols.keys()) == ["value"]
    np.testing.assert_allclose(qmatrix.times.to_value(u.s), np.arange(0.5, 1.5, 0.1))
    np.testing.assert_allclose(qmatrix.frequencies.to_value(u.Hz), np.arange(16, 64, 4))
    assert np.all(np.isfinite(qmatrix.value))

    assert qmatrix.meta[0, 0].unit == u.dimensionless_unscaled
    assert qmatrix.meta[0, 0].name is None
    assert str(qmatrix.meta[0, 0].channel) == "None"
