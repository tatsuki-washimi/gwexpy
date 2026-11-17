"""Characterization of epoch handling for SeriesMatrix, Array, Array2D/3D/4D and Plane2D.

Pins behavior at origin/main (#725, v0.3.0 Step 6).  Tests named ``*_current``
record behavior that is not a desired contract and may change in Step 7.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types import Array, Array2D, Array3D, Array4D, Plane2D, SeriesMatrix

OMIT = "omitted"
NONFINITE = [np.nan, np.inf, -np.inf]
GPS = 1.4e9
ARRAYS = {
    "Array": (Array, (5,)),
    "Array2D": (Array2D, (4, 3)),
    "Array3D": (Array3D, (3, 2, 2)),
    "Array4D": (Array4D, (2, 2, 2, 2)),
}


def _kw(value):
    return {} if value is OMIT else {"epoch": value}


def _epoch_float(obj):
    ep = obj.epoch
    return None if ep is None else float(getattr(ep, "gps", getattr(ep, "value", ep)))


def _expected(value):
    return None if value in (OMIT, None) else float(value)


@pytest.mark.parametrize("name", list(ARRAYS))
class TestArrayFamilyConstructor:
    @pytest.mark.parametrize("value", [OMIT, None])
    def test_missing_epoch_is_none(self, name, value):
        cls, shape = ARRAYS[name]
        a = cls(np.ones(shape), unit="m", **_kw(value))
        assert a.epoch is None
        assert a.shape == shape
        assert a.unit == u.m

    @pytest.mark.parametrize("value", [0, GPS])
    def test_finite_epoch_is_gps_time(self, name, value):
        cls, shape = ARRAYS[name]
        a = cls(np.ones(shape), unit="m", **_kw(value))
        assert _epoch_float(a) == float(value)
        assert a.shape == shape
        assert a.unit == u.m

    @pytest.mark.parametrize("value", NONFINITE)
    def test_nonfinite_raises_runtimeerror_current(self, name, value):
        # characterizes current behavior, not a desired contract (#725):
        # RuntimeError from the GPS conversion rather than ValueError.
        cls, shape = ARRAYS[name]
        with pytest.raises(RuntimeError, match="Invalid argument"):
            cls(np.ones(shape), unit="m", **_kw(value))

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_copy_pickle_keep_epoch(self, name, value):
        cls, shape = ARRAYS[name]
        a = cls(
            np.arange(int(np.prod(shape)), dtype=float).reshape(shape),
            unit="m",
            **_kw(value),
        )
        assert _epoch_float(a.copy()) == _expected(value)
        back = pickle.loads(pickle.dumps(a))
        assert _epoch_float(back) == _expected(value)
        assert back.unit == u.m
        np.testing.assert_array_equal(np.asarray(back.value), np.asarray(a.value))


@pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
class TestArray2DSlicing:
    def test_slice_and_transpose_keep_epoch(self, value):
        a = Array2D(np.arange(12.0).reshape(4, 3), unit="m", **_kw(value))
        assert _epoch_float(a[1:3]) == _expected(value)
        assert a[1:3].shape == (2, 3)
        assert _epoch_float(a.T) == _expected(value)
        assert a.T.shape == (3, 4)
        assert a.unit == u.m


class TestPlane2D:
    @pytest.mark.parametrize("value", [OMIT, None])
    def test_missing_epoch_is_none(self, value):
        p = Plane2D(np.ones((4, 3)), unit="m", **_kw(value))
        assert p.epoch is None
        assert (p.x0.value, p.dx.value) == (0.0, 1.0)
        assert p.unit == u.m

    @pytest.mark.parametrize("value", [0, GPS])
    def test_finite_epoch(self, value):
        p = Plane2D(np.ones((4, 3)), unit="m", **_kw(value))
        assert _epoch_float(p) == float(value)

    @pytest.mark.parametrize("value", NONFINITE)
    def test_nonfinite_raises_current(self, value):
        # characterizes current behavior, not a desired contract (#725)
        with pytest.raises(RuntimeError, match="Invalid argument"):
            Plane2D(np.ones((4, 3)), unit="m", **_kw(value))

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_slice_pickle_keep_epoch(self, value):
        p = Plane2D(np.ones((4, 3)), unit="m", **_kw(value))
        assert _epoch_float(p[1:3]) == _expected(value)
        back = pickle.loads(pickle.dumps(p))
        assert _epoch_float(back) == _expected(value)
        assert back.unit == u.m


def _sm(value):
    return SeriesMatrix(np.ones((1, 1, 5)), dx=0.5, x0=2.0, xunit="s", **_kw(value))


class TestSeriesMatrix:
    def test_omitted_epoch_is_float_zero_unlike_array_current(self):
        # characterizes current behavior, not a desired contract (#725):
        # Array(omitted).epoch is None, SeriesMatrix(omitted).epoch is 0.0.
        m = _sm(OMIT)
        assert m.epoch == 0.0
        assert isinstance(m.epoch, float)
        assert Array(np.ones(5)).epoch is None

    def test_none_zero_gps_stored_raw(self):
        assert _sm(None).epoch is None
        z = _sm(0)
        assert z.epoch == 0
        assert isinstance(z.epoch, int)
        assert _sm(GPS).epoch == GPS

    def test_nan_epoch_is_relative_and_not_stored(self):
        # changed by U-1 (#725): NaN is "no absolute time" (relative) and is
        # never stored; the axis is independent (x0/dx) and stays finite.
        m = _sm(np.nan)
        assert m.epoch == 0.0
        assert m.time_reference == "relative"
        assert np.all(np.isfinite(np.asarray(m.xindex.value)))
        assert m.x0.value == 2.0
        assert m.dx.value == 0.5

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_epoch_raises_valueerror(self, value):
        # changed by U-1 (#725): +/-Inf used to be stored raw.
        with pytest.raises(ValueError, match="must not be infinite"):
            _sm(value)

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_slice_moves_x0_not_epoch_current(self, value):
        # characterizes current behavior, not a desired contract (#725)
        m = _sm(value)
        sl = m[:, :, 1:4]
        assert sl.x0.value == 2.5
        assert sl.epoch == m.epoch
        assert sl.dx.value == 0.5

    @pytest.mark.parametrize("value", [OMIT, None, 0, GPS])
    def test_copy_pickle_arithmetic_keep_epoch(self, value):
        m = _sm(value)
        assert m.copy().epoch == m.epoch
        assert (m + m).epoch == m.epoch
        back = pickle.loads(pickle.dumps(m))
        assert back.epoch == m.epoch
        assert back.x0.value == 2.0
        assert back.dx.value == 0.5

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(OMIT, 0.0), (None, 0.0), (0, 0.0), (GPS, GPS)],
        ids=["omitted", "none", "zero", "gps"],
    )
    def test_hdf5_roundtrip_epoch_current(self, value, expected, tmp_path):
        pytest.importorskip("h5py")
        # characterizes current behavior, not a desired contract (#725):
        # epoch=None does not survive HDF5 (it is read back as 0.0).
        m = _sm(value)
        p = tmp_path / "sm.h5"
        m.write(p, format="hdf5")
        back = SeriesMatrix.read(p, format="hdf5")
        assert float(back.epoch) == expected
        assert back.x0.value == 2.0
        assert back.dx.value == 0.5
        assert back.shape == (1, 1, 5)
