from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.collections import TimeSeriesDict, TimeSeriesList


def _make_series(values, *, t0=0, dt=1, unit=u.m):
    return TimeSeries(np.asarray(values, dtype=float), t0=t0, dt=dt, unit=unit)


def test_timeseriesdict_value_at_returns_series():
    pd = pytest.importorskip("pandas")
    ts1 = _make_series([0, 1, 2, 3, 4])
    ts2 = _make_series([10, 11, 12, 13, 14])
    td = TimeSeriesDict({"a": ts1, "b": ts2})

    res = td.value_at(2)
    assert isinstance(res, pd.Series)
    assert res["a"].to_value(u.m) == pytest.approx(2.0)
    assert res["b"].to_value(u.m) == pytest.approx(12.0)


def test_timeserieslist_value_at_returns_list():
    ts1 = _make_series([0, 1, 2, 3])
    ts2 = _make_series([4, 5, 6, 7])
    tl = TimeSeriesList()
    tl.append(ts1)
    tl.append(ts2)

    res = tl.value_at(1)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].to_value(u.m) == pytest.approx(1.0)
    assert res[1].to_value(u.m) == pytest.approx(5.0)


def test_timeseriesdict_to_matrix_shape():
    ts1 = _make_series([0, 1, 2, 3, 4])
    ts2 = _make_series([10, 11, 12, 13, 14])
    td = TimeSeriesDict({"a": ts1, "b": ts2})

    mat = td.to_matrix()
    assert mat.shape == (2, 1, 5)
