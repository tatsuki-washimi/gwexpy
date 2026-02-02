from __future__ import annotations

import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def test_timeseriesdict_group_layout_roundtrip(tmp_path):
    ts = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    tsd = TimeSeriesDict({"H1:TEST": ts})

    outfile = tmp_path / "tsd_group.h5"
    tsd.write(outfile, format="hdf5", layout="group")

    tsd2 = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(tsd2.keys()) == list(tsd.keys())
    np.testing.assert_allclose(tsd2["H1:TEST"].value, ts.value)


def test_timeserieslist_group_layout_roundtrip(tmp_path):
    ts1 = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    ts2 = TimeSeries(np.arange(3.0) * 2, sample_rate=1.0, t0=0, unit="m")
    tsl = TimeSeriesList(ts1, ts2)

    outfile = tmp_path / "tsl_group.h5"
    tsl.write(outfile, format="hdf5", layout="group")

    tsl2 = TimeSeriesList.read(outfile, format="hdf5")
    assert len(tsl2) == len(tsl)
    np.testing.assert_allclose(tsl2[0].value, ts1.value)
