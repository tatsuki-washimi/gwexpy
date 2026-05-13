"""Tests for gwexpy/time/core.py"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from gwexpy.time.core import (
    _is_array,
    _is_numeric_array,
    _normalize_time_input,
    from_gps,
    tconvert,
    to_gps,
)

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


def test_to_gps_astropy_time_honors_dtype():
    t = Time(1234567890.0, format="gps")

    float_result = to_gps(t, dtype=float)
    quantity_result = to_gps(t, dtype="quantity")

    assert isinstance(float_result, float)
    assert float_result == pytest.approx(1234567890.0)
    assert quantity_result.unit == u.s
    assert quantity_result.value == pytest.approx(1234567890.0)


def test_to_gps_numpy_array_numeric():
    arr = np.array([1000.0, 2000.0, 3000.0])
    result = to_gps(arr)
    np.testing.assert_allclose(result, [1000.0, 2000.0, 3000.0])


def test_to_gps_quantity_array_converts_to_seconds():
    result = to_gps(np.array([0, 1, 2]) * u.ms)
    np.testing.assert_allclose(result, [0.0, 0.001, 0.002])


def test_to_gps_dtype_float_returns_python_float_for_scalar():
    result = to_gps(1000000000.0, dtype=float)

    assert isinstance(result, float)
    assert result == pytest.approx(1000000000.0)


def test_to_gps_dtype_float_returns_float_array_for_vector():
    result = to_gps([1000000000, 1000000001], dtype="float")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [1000000000.0, 1000000001.0])


def test_to_gps_dtype_quantity_returns_seconds_quantity():
    scalar = to_gps(1000000000.0, dtype="quantity")
    vector = to_gps([1000000000, 1000000001], dtype="quantity")

    assert scalar.unit == u.s
    assert scalar.value == pytest.approx(1000000000.0)
    assert vector.unit == u.s
    np.testing.assert_allclose(vector.value, [1000000000.0, 1000000001.0])


def test_to_gps_invalid_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        to_gps(1000000000.0, dtype="datetime")


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
