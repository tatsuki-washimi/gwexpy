from __future__ import annotations

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError

import gwexpy
from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)

gwexpy.register_all()


def test_timeserieslist_hdf5_roundtrip(tmp_path):
    ts1 = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS1",
    )
    ts2 = TimeSeries(
        np.arange(4.0) * 2,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="L1:TS2",
    )
    tsl = TimeSeriesList(ts1, ts2)

    path = tmp_path / "tsl.h5"
    tsl.write(path, format="hdf5")
    tsl2 = TimeSeriesList.read(path, format="hdf5")

    assert len(tsl2) == 2
    assert tsl2[0].name == ts1.name
    np.testing.assert_allclose(tsl2[1].value, ts2.value)


def test_timeseriesmatrix_hdf5_roundtrip(tmp_path):
    tsm = TimeSeriesMatrix(
        np.arange(24.0).reshape(2, 3, 4),
        t0=10.0,
        dt=0.5,
    )

    path = tmp_path / "tsm.h5"
    tsm.write(path, format="hdf5")
    tsm2 = TimeSeriesMatrix.read(path, format="hdf5")

    assert tsm2.shape == tsm.shape
    np.testing.assert_allclose(tsm2.value, tsm.value)


def test_timeseries_hdf5_requires_explicit_format(tmp_path):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    path = tmp_path / "ts.h5"
    ts.write(path, format="hdf5")

    with pytest.raises((IORegistryError, ValueError), match="explicit format.*hdf5"):
        TimeSeries.read(path)


def test_timeseriesdict_hdf5_requires_explicit_format(tmp_path):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    tsd = TimeSeriesDict({"H1:TS": ts})
    path = tmp_path / "tsd.h5"
    tsd.write(path, format="hdf5")

    with pytest.raises((IORegistryError, ValueError), match="explicit format.*hdf5"):
        TimeSeriesDict.read(path)


@pytest.mark.parametrize(
    "reader", (TimeSeries, TimeSeriesDict), ids=lambda cls: cls.__name__
)
@pytest.mark.parametrize(
    "format_token",
    ("hdf5", "hdf.ndscope", "ndscope-hdf5", "ndscope_hdf5", "ndscopehdf5"),
)
def test_public_readers_accept_registered_positional_hdf5_formats(
    tmp_path, reader, format_token
):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    path = tmp_path / f"positional-{format_token}.h5"
    TimeSeriesDict({"H1:TS": ts}).write(path, format=format_token)

    result = reader.read(path, format_token)

    if reader is TimeSeries:
        assert result.name == "H1:TS"
        np.testing.assert_allclose(result.value, ts.value)
    else:
        assert list(result) == ["H1:TS"]
        np.testing.assert_allclose(result["H1:TS"].value, ts.value)


@pytest.mark.parametrize(
    "reader", (TimeSeries, TimeSeriesDict), ids=lambda cls: cls.__name__
)
def test_public_readers_reject_duplicate_positional_and_keyword_format(
    tmp_path, reader
):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    path = tmp_path / "duplicate-format.h5"
    ts.write(path, format="hdf5")

    with pytest.raises(TypeError, match="multiple values.*format"):
        reader.read(path, "hdf5", format="hdf5")


def test_timeseriesdict_ndscope_auto_identifies_single_path(tmp_path):
    ts = TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="H1:TS",
    )
    path = tmp_path / "tsd-ndscope.h5"
    TimeSeriesDict({"H1:TS": ts}).write(path, format="hdf.ndscope")

    actual = TimeSeriesDict.read(path)

    assert list(actual) == ["H1:TS"]
    np.testing.assert_allclose(actual["H1:TS"].value, ts.value)


def test_timeseriesdict_ndscope_auto_identifies_supported_path_list(tmp_path):
    paths = []
    for index, start in enumerate((1.0, 3.0)):
        ts = TimeSeries(
            np.arange(2.0) + index * 2.0,
            sample_rate=1.0,
            t0=start,
            unit="m",
            name="H1:TS",
        )
        path = tmp_path / f"tsd-ndscope-{index}.h5"
        TimeSeriesDict({"H1:TS": ts}).write(path, format="hdf.ndscope")
        paths.append(path)

    actual = TimeSeriesDict.read(paths)

    assert list(actual) == ["H1:TS"]
    np.testing.assert_allclose(actual["H1:TS"].value, np.arange(4.0))
