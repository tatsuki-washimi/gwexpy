from __future__ import annotations

import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def test_timeseriesdict_csv_directory_roundtrip(tmp_path):
    a = TimeSeries(np.arange(5.0), sample_rate=2.0, t0=0, unit="m", name="a")
    b = TimeSeries(np.arange(5.0) * 2, sample_rate=2.0, t0=0, unit="m", name="b")
    tsd = TimeSeriesDict({"H1:TEST-CH": a, "L1:TEST-CH": b})

    outdir = tmp_path / "tsd_csv"
    tsd.write(outdir, format="csv")

    tsd2 = TimeSeriesDict.read(outdir, format="csv")
    assert list(tsd2.keys()) == list(tsd.keys())
    for k in tsd:
        np.testing.assert_allclose(tsd2[k].value, tsd[k].value)
        np.testing.assert_allclose(tsd2[k].times.value, tsd[k].times.value)
        assert str(tsd2[k].unit) == str(tsd[k].unit)


def test_timeserieslist_txt_directory_roundtrip(tmp_path):
    a = TimeSeries(np.arange(4.0), sample_rate=1.0, t0=10, unit="s", name="x/1")
    b = TimeSeries(np.arange(4.0) * 3, sample_rate=1.0, t0=10, unit="s", name="y:2")
    tsl = TimeSeriesList(a, b)

    outdir = tmp_path / "tsl_txt"
    tsl.write(outdir, format="txt")

    tsl2 = TimeSeriesList.read(outdir, format="txt")
    assert len(tsl2) == len(tsl)
    for i in range(len(tsl)):
        np.testing.assert_allclose(tsl2[i].value, tsl[i].value)
        np.testing.assert_allclose(tsl2[i].times.value, tsl[i].times.value)
        assert str(tsl2[i].unit) == str(tsl[i].unit)
