"""Tests for gwexpy/interop/hdf5_.py."""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from gwexpy.interop.hdf5_ import from_hdf5, to_hdf5
from gwexpy.timeseries import TimeSeries


def _make_ts(n=8, t0=0.0, dt=1.0, unit="m", name="ch"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit, name=name)


class TestToHdf5:
    def test_basic_write(self, tmp_path):
        ts = _make_ts()
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "channel")
        with h5py.File(fp, "r") as h5f:
            assert "channel" in h5f

    def test_attrs_stored(self, tmp_path):
        ts = _make_ts(t0=100.0, dt=0.5, unit="s")
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
        with h5py.File(fp, "r") as h5f:
            assert h5f["ch"].attrs["t0"] == pytest.approx(100.0)
            assert h5f["ch"].attrs["dt"] == pytest.approx(0.5)
            assert h5f["ch"].attrs["unit"] == "s"

    def test_overwrite_false_raises(self, tmp_path):
        ts = _make_ts()
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            with pytest.raises(OSError, match="exists"):
                to_hdf5(ts, h5f, "ch", overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path):
        ts = _make_ts()
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            to_hdf5(ts, h5f, "ch", overwrite=True)  # should not raise

    def test_name_attr_stored(self, tmp_path):
        ts = _make_ts(name="myname")
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
        with h5py.File(fp, "r") as h5f:
            assert h5f["ch"].attrs.get("name") == "myname"


class TestFromHdf5:
    def test_basic_roundtrip(self, tmp_path):
        ts = _make_ts(n=5)
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            ts2 = from_hdf5(TimeSeries, h5f, "ch")
        np.testing.assert_array_equal(ts2.value, ts.value)

    def test_t0_restored(self, tmp_path):
        ts = _make_ts(t0=50.0)
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            ts2 = from_hdf5(TimeSeries, h5f, "ch")
        assert ts2.t0.value == pytest.approx(50.0)

    def test_dt_restored(self, tmp_path):
        ts = _make_ts(dt=0.25)
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            ts2 = from_hdf5(TimeSeries, h5f, "ch")
        assert ts2.dt.value == pytest.approx(0.25)

    def test_unit_restored(self, tmp_path):
        ts = _make_ts(unit="km")
        fp = tmp_path / "ts.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5(ts, h5f, "ch")
            ts2 = from_hdf5(TimeSeries, h5f, "ch")
        assert str(ts2.unit) == "km"
