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


# --- TimeSeriesDict: statistical methods via to_matrix ---

def _make_dict():
    ts1 = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    ts2 = _make_series([2.0, 4.0, 6.0, 8.0, 10.0])
    return TimeSeriesDict({"a": ts1, "b": ts2})


def test_timeseriesdict_mean():
    td = _make_dict()
    result = td.mean()
    # Returns (n_channels, 1) numpy array ordered by dict keys
    assert isinstance(result, np.ndarray)
    assert result[0, 0] == pytest.approx(3.0)
    assert result[1, 0] == pytest.approx(6.0)


def test_timeseriesdict_std():
    td = _make_dict()
    result = td.std()
    assert isinstance(result, np.ndarray)
    assert result[0, 0] == pytest.approx(np.std([1, 2, 3, 4, 5]))
    assert result[1, 0] == pytest.approx(np.std([2, 4, 6, 8, 10]))


def test_timeseriesdict_min_max():
    td = _make_dict()
    mn = td.min()
    mx = td.max()
    assert isinstance(mn, np.ndarray)
    assert mn[0, 0] == pytest.approx(1.0)
    assert mx[0, 0] == pytest.approx(5.0)
    assert mn[1, 0] == pytest.approx(2.0)
    assert mx[1, 0] == pytest.approx(10.0)


def test_timeseriesdict_rms():
    td = _make_dict()
    result = td.rms()
    expected_a = np.sqrt(np.mean(np.array([1, 2, 3, 4, 5]) ** 2))
    assert isinstance(result, np.ndarray)
    assert result[0, 0] == pytest.approx(expected_a)


# --- TimeSeriesDict: crop ---

def test_timeseriesdict_crop():
    ts1 = _make_series(np.arange(10.0), t0=0, dt=1)
    ts2 = _make_series(np.arange(10.0) * 2, t0=0, dt=1)
    td = TimeSeriesDict({"a": ts1, "b": ts2})

    cropped = td.crop(start=2, end=7)
    assert cropped["a"].shape[0] == 5
    assert cropped["b"].shape[0] == 5


# --- TimeSeriesDict: rolling operations ---

def test_timeseriesdict_rolling_mean():
    ts1 = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    ts2 = _make_series([5.0, 4.0, 3.0, 2.0, 1.0])
    td = TimeSeriesDict({"a": ts1, "b": ts2})

    result = td.rolling_mean(3)
    assert isinstance(result, TimeSeriesDict)
    assert "a" in result and "b" in result
    assert result["a"].shape == ts1.shape


def test_timeseriesdict_rolling_std():
    ts1 = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    td = TimeSeriesDict({"a": ts1})
    result = td.rolling_std(3)
    assert isinstance(result, TimeSeriesDict)


def test_timeseriesdict_rolling_median():
    ts1 = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    td = TimeSeriesDict({"a": ts1})
    result = td.rolling_median(3)
    assert isinstance(result, TimeSeriesDict)


def test_timeseriesdict_rolling_min_max():
    ts1 = _make_series([3.0, 1.0, 4.0, 1.0, 5.0])
    td = TimeSeriesDict({"a": ts1})
    assert isinstance(td.rolling_min(3), TimeSeriesDict)
    assert isinstance(td.rolling_max(3), TimeSeriesDict)


# --- TimeSeriesDict: resample (time-bin path) ---

def test_timeseriesdict_resample_signal():
    ts1 = _make_series(np.random.default_rng(0).random(100), dt=1 / 100)
    ts2 = _make_series(np.random.default_rng(1).random(100), dt=1 / 100)
    td = TimeSeriesDict({"a": ts1, "b": ts2})
    result = td.resample(50)
    assert isinstance(result, TimeSeriesDict)


# --- TimeSeriesList: to_matrix ---

def test_timeserieslist_to_matrix_shape():
    tl = TimeSeriesList([
        _make_series([1.0, 2.0, 3.0]),
        _make_series([4.0, 5.0, 6.0]),
    ])
    mat = tl.to_matrix()
    assert mat.shape == (2, 1, 3)


# --- TimeSeriesList: crop ---

def test_timeserieslist_crop():
    tl = TimeSeriesList([
        _make_series(np.arange(10.0), t0=0, dt=1),
        _make_series(np.arange(10.0) * 2, t0=0, dt=1),
    ])
    cropped = tl.crop(start=3, end=7)
    assert cropped[0].shape[0] == 4
    assert cropped[1].shape[0] == 4


# --- TimeSeriesList: rolling operations ---

def test_timeserieslist_rolling_mean():
    tl = TimeSeriesList([
        _make_series([1.0, 2.0, 3.0, 4.0, 5.0]),
        _make_series([5.0, 4.0, 3.0, 2.0, 1.0]),
    ])
    result = tl.rolling_mean(3)
    assert isinstance(result, TimeSeriesList)
    assert len(result) == 2
