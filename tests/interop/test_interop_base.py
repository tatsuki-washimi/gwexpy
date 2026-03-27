"""Tests for gwexpy/interop/base.py"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.units import Quantity

from gwexpy.interop.base import from_plain_array, to_plain_array


# ---------------------------------------------------------------------------
# to_plain_array
# ---------------------------------------------------------------------------

def test_to_plain_array_ndarray():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_plain_array(arr)
    np.testing.assert_array_equal(result, arr)


def test_to_plain_array_quantity():
    q = Quantity([1.0, 2.0], "m")
    result = to_plain_array(q)
    np.testing.assert_array_equal(result, [1.0, 2.0])
    assert isinstance(result, np.ndarray)


def test_to_plain_array_with_value_attr():
    class Wrapper:
        value = np.array([10.0, 20.0])

    result = to_plain_array(Wrapper())
    np.testing.assert_array_equal(result, [10.0, 20.0])


def test_to_plain_array_copy_false():
    arr = np.array([1.0, 2.0])
    result = to_plain_array(arr, copy=False)
    np.testing.assert_array_equal(result, arr)


def test_to_plain_array_copy_true():
    arr = np.array([1.0, 2.0])
    result = to_plain_array(arr, copy=True)
    np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# from_plain_array — torch/cupy branch coverage via duck typing
# ---------------------------------------------------------------------------

def test_from_plain_array_basic():
    from gwexpy.timeseries import TimeSeries
    from astropy import units as u

    arr = np.array([1.0, 2.0, 3.0])
    result = from_plain_array(TimeSeries, arr, t0=0.0, dt=0.01 * u.s)
    assert result is not None
    assert len(result) == 3


def test_from_plain_array_numpy_method():
    """Array with .numpy() method → calls .numpy() before passing to cls."""
    class TorchLike:
        def numpy(self):
            return np.array([5.0, 6.0, 7.0])

    from gwexpy.timeseries import TimeSeries
    from astropy import units as u

    result = from_plain_array(TimeSeries, TorchLike(), t0=0.0, dt=0.01 * u.s)
    assert result is not None


def test_from_plain_array_get_method():
    """Array with .get() method (cupy-like) → calls .get() before passing to cls."""
    class CupyLike:
        def get(self):
            return np.array([8.0, 9.0])

    from gwexpy.timeseries import TimeSeries
    from astropy import units as u

    result = from_plain_array(TimeSeries, CupyLike(), t0=0.0, dt=0.01 * u.s)
    assert result is not None
