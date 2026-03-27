"""Tests for gwexpy/time/core.py"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from gwexpy.time.core import _is_array, _is_numeric_array, _normalize_time_input, from_gps, tconvert, to_gps


# ---------------------------------------------------------------------------
# _is_array
# ---------------------------------------------------------------------------

def test_is_array_string_is_false():
    assert _is_array("2020-01-01") is False


def test_is_array_bytes_is_false():
    assert _is_array(b"abc") is False


def test_is_array_numpy_1d():
    assert _is_array(np.array([1.0, 2.0])) is True


def test_is_array_numpy_0d():
    assert _is_array(np.array(1.0)) is False


def test_is_array_list():
    assert _is_array([1, 2, 3]) is True


def test_is_array_tuple():
    assert _is_array((1, 2)) is True


def test_is_array_int():
    assert _is_array(42) is False


# ---------------------------------------------------------------------------
# _is_numeric_array
# ---------------------------------------------------------------------------

def test_is_numeric_array_float():
    assert _is_numeric_array(np.array([1.0, 2.0])) is True


def test_is_numeric_array_int():
    assert _is_numeric_array(np.array([1, 2])) is True


def test_is_numeric_array_datetime():
    assert _is_numeric_array(np.array(["2020-01-01"], dtype="datetime64")) is False


# ---------------------------------------------------------------------------
# _normalize_time_input
# ---------------------------------------------------------------------------

def test_normalize_time_input_passthrough():
    assert _normalize_time_input(1234567890.0) == 1234567890.0


def test_normalize_time_input_list_of_floats():
    result = _normalize_time_input([1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3.0]


def test_normalize_time_input_empty_list():
    result = _normalize_time_input([])
    assert result == []


# ---------------------------------------------------------------------------
# to_gps
# ---------------------------------------------------------------------------

def test_to_gps_scalar_float():
    result = to_gps(1234567890.0)
    assert float(result) == pytest.approx(1234567890.0)


def test_to_gps_scalar_int():
    result = to_gps(1000000000)
    assert float(result) == pytest.approx(1000000000.0)


def test_to_gps_astropy_time():
    t = Time(1234567890.0, format="gps")
    result = to_gps(t)
    assert float(result) == pytest.approx(1234567890.0)


def test_to_gps_numpy_array_numeric():
    arr = np.array([1000.0, 2000.0, 3000.0])
    result = to_gps(arr)
    np.testing.assert_allclose(result, [1000.0, 2000.0, 3000.0])


def test_to_gps_list_numeric():
    result = to_gps([1000.0, 2000.0])
    assert len(result) == 2


def test_to_gps_numpy_datetime64():
    dt = np.datetime64("2017-01-01T00:00:00")
    result = to_gps(dt)
    assert float(result) > 0


# ---------------------------------------------------------------------------
# from_gps
# ---------------------------------------------------------------------------

def test_from_gps_scalar():
    result = from_gps(1000000000)
    assert result is not None


def test_from_gps_array():
    arr = np.array([1000000000.0, 1000000001.0])
    result = from_gps(arr)
    assert len(result) == 2


def test_from_gps_astropy_time():
    t = Time(1000000000.0, format="gps")
    result = from_gps(t)
    assert result is not None


# ---------------------------------------------------------------------------
# tconvert
# ---------------------------------------------------------------------------

def test_tconvert_scalar_gps():
    result = tconvert(1000000000)
    assert result is not None


def test_tconvert_array_numeric():
    arr = np.array([1000000000.0, 1000000001.0])
    result = tconvert(arr)
    assert len(result) == 2


def test_tconvert_array_datetime_strings():
    arr = ["2017-01-01", "2018-01-01"]
    result = tconvert(arr)
    assert len(result) == 2


def test_tconvert_default_now():
    result = tconvert()
    assert result is not None
