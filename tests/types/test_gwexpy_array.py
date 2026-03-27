"""Tests for gwexpy/types/array.py — Array class."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.array import Array


def _make_arr(shape=(2, 3), axis_names=None):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    if axis_names is not None:
        return Array(data * u.m, axis_names=axis_names)
    return Array(data * u.m)


class TestArrayConstruction:
    def test_default_axis_names(self):
        arr = _make_arr((2, 3))
        assert arr._axis_names == ["axis0", "axis1"]

    def test_custom_axis_names(self):
        # Line 26 — axis_names as list
        arr = _make_arr((2, 3), axis_names=["x", "y"])
        assert arr._axis_names == ["x", "y"]

    def test_ndim(self):
        arr = _make_arr((2, 3, 4))
        assert arr.ndim == 3


class TestArrayFinalize:
    def test_finalize_preserves_names(self):
        arr = _make_arr((2, 3), axis_names=["row", "col"])
        # Slicing triggers __array_finalize__
        sliced = arr[0:1, :]
        assert sliced._axis_names == ["row", "col"]

    def test_finalize_resets_names_for_wrong_ndim(self):
        # Line 37 — parent_names length != self.ndim
        arr = _make_arr((2, 3), axis_names=["row", "col"])
        flat = arr.reshape(6)
        # After reshape, _axis_names should be reset to length 1
        assert len(flat._axis_names) == flat.ndim


class TestAxesProperty:
    def test_axes_returns_tuple(self):
        arr = _make_arr((2, 3))
        axes = arr.axes
        assert len(axes) == 2

    def test_axes_resets_on_mismatch(self):
        # Lines 43-44 — if _axis_names length != ndim
        arr = _make_arr((2, 3))
        arr._axis_names = ["only_one"]  # force mismatch
        axes = arr.axes
        assert len(axes) == arr.ndim


class TestSetAxisName:
    def test_set_axis_name(self):
        # Line 51
        arr = _make_arr((2, 3), axis_names=["x", "y"])
        arr._set_axis_name(0, "new_x")
        assert arr._axis_names[0] == "new_x"


class TestIselTuple:
    def test_isel_tuple(self):
        # Line 54
        arr = _make_arr((2, 3))
        result = arr._isel_tuple((0, slice(None)))
        assert result.shape == (3,)


class TestSwapAxes:
    def test_swapaxes(self):
        # Lines 56-63
        arr = _make_arr((2, 3), axis_names=["row", "col"])
        result = arr._swapaxes_int(0, 1)
        assert result.shape == (3, 2)
        assert result._axis_names[0] == "col"
        assert result._axis_names[1] == "row"


class TestRms:
    def test_rms_basic(self):
        # Lines 65-68
        arr = Array(np.array([3.0, 4.0]) * u.m)
        result = arr.rms()
        expected = np.sqrt((9 + 16) / 2)
        assert float(result.value) == pytest.approx(expected)

    def test_rms_with_unit(self):
        arr = Array(np.array([1.0, 2.0, 3.0]) * u.V)
        result = arr.rms()
        assert result.unit == u.V

    def test_rms_ignore_nan_false(self):
        # Line 66 — ignore_nan=False
        arr = Array(np.array([1.0, 2.0, 3.0]) * u.m)
        result = arr.rms(ignore_nan=False)
        assert result is not None

    def test_rms_axis(self):
        arr = _make_arr((3, 4))
        result = arr.rms(axis=0)
        assert result.shape == (4,)
