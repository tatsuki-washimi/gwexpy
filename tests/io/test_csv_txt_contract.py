from __future__ import annotations

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _make_timeseries(name: str = "H1:TS") -> TimeSeries:
    return TimeSeries(
        np.arange(4.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name=name,
    )


def test_csv_timeseries_roundtrip_autodetect(tmp_path):
    ts = _make_timeseries()
    path = tmp_path / "single.csv"

    ts.write(path)
    ts2 = TimeSeries.read(path)

    np.testing.assert_allclose(ts2.value, ts.value)
    assert str(ts2.unit) == str(ts.unit)


def test_csv_timeseriesdict_reads_single_file(tmp_path):
    ts = _make_timeseries()
    path = tmp_path / "single.csv"
    ts.write(path)

    tsd = TimeSeriesDict.read(path)

    assert len(tsd) == 1
    only = next(iter(tsd.values()))
    np.testing.assert_allclose(only.value, ts.value)


def test_csv_timeseriesdict_collection_directory_roundtrip(tmp_path):
    tsd = TimeSeriesDict({
        "H1:TS": _make_timeseries("H1:TS"),
        "L1:TS": _make_timeseries("L1:TS"),
    })
    target = tmp_path / "dict.csv"

    tsd.write(target, format="csv")
    assert target.is_dir()

    tsd2 = TimeSeriesDict.read(target)

    assert list(tsd2.keys()) == list(tsd.keys())
    np.testing.assert_allclose(tsd2["L1:TS"].value, tsd["L1:TS"].value)


def test_txt_timeseries_requires_explicit_format(tmp_path):
    ts = _make_timeseries()
    path = tmp_path / "single.txt"
    ts.write(path, format="txt")

    with pytest.raises(IORegistryError):
        TimeSeries.read(path)

    read_back = TimeSeries.read(path, format="txt")
    np.testing.assert_allclose(read_back.value, ts.value)


def test_txt_timeseries_write_requires_explicit_format(tmp_path):
    ts = _make_timeseries()
    path = tmp_path / "single.txt"

    with pytest.raises(IORegistryError):
        ts.write(path)

    ts.write(path, format="txt")
    assert path.exists()


def test_txt_timeseriesdict_uses_collection_directory(tmp_path):
    tsd = TimeSeriesDict({
        "H1:TS": _make_timeseries("H1:TS"),
    })
    target = tmp_path / "dict.txt"

    tsd.write(target, format="txt")
    assert target.is_dir()

    tsd2 = TimeSeriesDict.read(target)
    assert list(tsd2.keys()) == ["H1:TS"]


def test_txt_timeseriesdict_file_read_is_not_public(tmp_path):
    ts = _make_timeseries()
    path = tmp_path / "single.txt"
    ts.write(path, format="txt")

    with pytest.raises(IORegistryError):
        TimeSeriesDict.read(path, format="txt")
