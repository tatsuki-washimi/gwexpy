from __future__ import annotations

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError

from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)


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

    with pytest.raises((IORegistryError, TypeError, ValueError)):
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

    with pytest.raises((IORegistryError, TypeError, ValueError)):
        TimeSeriesDict.read(path)
