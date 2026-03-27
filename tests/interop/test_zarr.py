"""Tests for gwexpy/interop/zarr_.py."""
from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from gwexpy.interop.zarr_ import from_zarr, to_zarr
from gwexpy.timeseries import TimeSeries


def _make_ts(n=8, t0=0.0, dt=1.0, unit="m", name="ch"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit, name=name)


@pytest.fixture
def mem_store():
    return zarr.storage.MemoryStore()


class TestToZarr:
    def test_basic_write(self, mem_store):
        ts = _make_ts()
        to_zarr(ts, mem_store, "ch")  # should not raise

    def test_attrs_stored(self, mem_store):
        ts = _make_ts(t0=100.0, dt=0.5, unit="s")
        to_zarr(ts, mem_store, "ch")
        arr = zarr.open_array(store=mem_store, mode="r", path="ch")
        assert arr.attrs["t0"] == pytest.approx(100.0)
        assert arr.attrs["dt"] == pytest.approx(0.5)
        assert arr.attrs["unit"] == "s"

    def test_name_attr_stored(self, mem_store):
        ts = _make_ts(name="myname")
        to_zarr(ts, mem_store, "ch")
        arr = zarr.open_array(store=mem_store, mode="r", path="ch")
        assert arr.attrs.get("name") == "myname"


class TestFromZarr:
    def test_basic_roundtrip(self, mem_store):
        ts = _make_ts(n=6)
        to_zarr(ts, mem_store, "ch")
        ts2 = from_zarr(TimeSeries, mem_store, "ch")
        np.testing.assert_array_equal(ts2.value, ts.value)

    def test_t0_restored(self, mem_store):
        ts = _make_ts(t0=50.0)
        to_zarr(ts, mem_store, "ch")
        ts2 = from_zarr(TimeSeries, mem_store, "ch")
        assert ts2.t0.value == pytest.approx(50.0)

    def test_dt_restored(self, mem_store):
        ts = _make_ts(dt=0.25)
        to_zarr(ts, mem_store, "ch")
        ts2 = from_zarr(TimeSeries, mem_store, "ch")
        assert ts2.dt.value == pytest.approx(0.25)

    def test_unit_restored(self, mem_store):
        ts = _make_ts(unit="km")
        to_zarr(ts, mem_store, "ch")
        ts2 = from_zarr(TimeSeries, mem_store, "ch")
        assert str(ts2.unit) == "km"
