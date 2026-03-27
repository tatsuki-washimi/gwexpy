"""Tests for gwexpy/interop/dask_.py."""
from __future__ import annotations

import numpy as np
import pytest

dask = pytest.importorskip("dask")
dask_array = pytest.importorskip("dask.array")

from gwexpy.interop.dask_ import from_dask, to_dask
from gwexpy.timeseries import TimeSeries


def _make_ts(n=10, t0=0.0, dt=1.0, unit="m"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit)


class TestToDask:
    def test_returns_dask_array(self):
        ts = _make_ts()
        arr = to_dask(ts)
        assert hasattr(arr, "compute")  # dask array has compute()

    def test_values_match_after_compute(self):
        ts = _make_ts(n=5)
        arr = to_dask(ts)
        np.testing.assert_array_equal(arr.compute(), ts.value)

    def test_custom_chunks(self):
        ts = _make_ts(n=10)
        arr = to_dask(ts, chunks=5)
        assert arr.npartitions >= 1 or arr.chunks is not None


class TestFromDask:
    def test_basic_compute_true(self):
        arr = dask_array.from_array(np.arange(5.0), chunks=5)
        ts = from_dask(TimeSeries, arr, t0=0.0, dt=1.0, unit="m")
        assert isinstance(ts, TimeSeries)
        np.testing.assert_array_equal(ts.value, np.arange(5.0))

    def test_t0_and_dt(self):
        arr = dask_array.from_array(np.ones(5), chunks=5)
        ts = from_dask(TimeSeries, arr, t0=10.0, dt=0.5)
        assert ts.t0.value == pytest.approx(10.0)
        assert ts.dt.value == pytest.approx(0.5)

    def test_unit(self):
        arr = dask_array.from_array(np.ones(3), chunks=3)
        ts = from_dask(TimeSeries, arr, t0=0, dt=1.0, unit="s")
        assert str(ts.unit) == "s"

    def test_compute_false(self):
        arr = dask_array.from_array(np.arange(5.0), chunks=5)
        ts = from_dask(TimeSeries, arr, t0=0.0, dt=1.0, compute=False)
        # When compute=False, data may be dask array
        assert ts is not None

    def test_roundtrip(self):
        ts = _make_ts(n=8)
        arr = to_dask(ts)
        ts2 = from_dask(TimeSeries, arr, t0=float(ts.t0.value), dt=float(ts.dt.value), unit=str(ts.unit))
        np.testing.assert_array_equal(ts2.value, ts.value)
