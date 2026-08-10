"""Tests for gwexpy/time/core.py"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from gwpy import time as gwpy_time

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


def test_to_gps_quantity_array_rejects_non_time_units():
    with pytest.raises(u.UnitConversionError, match="not convertible"):
        to_gps(np.array([1, 2]) * u.m)


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


def test_to_gps_dtype_quantity_string_input():
    import astropy.units as u

    result = to_gps("2017-01-01T00:00:00", dtype="quantity")
    assert isinstance(result, u.Quantity)
    assert result.unit == u.s
    assert result.value > 0


def test_to_gps_dtype_quantity_astropy_time_input():
    import astropy.units as u

    t = Time(1234567890.0, format="gps")
    result = to_gps(t, dtype="quantity")
    assert isinstance(result, u.Quantity)
    assert result.unit == u.s
    assert float(result.value) == pytest.approx(1234567890.0)


def test_to_gps_invalid_dtype_raised_before_conversion():
    with pytest.raises(ValueError, match="Invalid dtype"):
        to_gps("2017-01-01T00:00:00", dtype="bad")


def test_to_gps_list_numeric():
    result = to_gps([1000.0, 2000.0])
    assert len(result) == 2


def test_to_gps_numpy_datetime64():
    dt = np.datetime64("2017-01-01T00:00:00")
    result = to_gps(dt)
    assert float(result) > 0


@pytest.mark.parametrize(
    ("resolution", "timestamp", "expected_seconds", "expected_nanoseconds"),
    [
        ("s", "2017-01-01T00:00:00", 1167264018, 0),
        ("ms", "2017-01-01T00:00:00.123", 1167264018, 123_000_000),
        ("us", "2017-01-01T00:00:00.123456", 1167264018, 123_456_000),
        ("ns", "2017-01-01T00:00:00.123456789", 1167264018, 123_456_789),
    ],
)
def test_to_gps_numpy_datetime64_preserves_represented_instant(
    resolution, timestamp, expected_seconds, expected_nanoseconds
):
    value = np.datetime64(timestamp, resolution)

    scalar = to_gps(value)
    vector = to_gps(np.asarray([value]))

    assert isinstance(scalar, gwpy_time.LIGOTimeGPS)
    assert (scalar.gpsSeconds, scalar.gpsNanoSeconds) == (
        expected_seconds,
        expected_nanoseconds,
    )
    assert vector.dtype == object
    assert (vector[0].gpsSeconds, vector[0].gpsNanoSeconds) == (
        expected_seconds,
        expected_nanoseconds,
    )

    float_vector = to_gps(np.asarray([value]), dtype=float)
    assert float_vector.dtype == np.float64
    assert float_vector[0] == float(scalar)


def test_to_gps_numpy_datetime64_common_instant_matches_all_resolutions():
    results = [
        to_gps(np.datetime64("2017-01-01T00:00:00", resolution))
        for resolution in ("s", "ms", "us", "ns")
    ]

    assert {(result.gpsSeconds, result.gpsNanoSeconds) for result in results} == {
        (1167264018, 0)
    }


def test_to_gps_datetime64_vector_preserves_ns_before_leap_second():
    value = np.datetime64("2016-12-31T23:59:59.999999999", "ns")

    scalar = to_gps(value)
    vector = to_gps(np.asarray([value]))

    expected = (1167264016, 999_999_999)
    assert (scalar.gpsSeconds, scalar.gpsNanoSeconds) == expected
    assert (vector[0].gpsSeconds, vector[0].gpsNanoSeconds) == expected
    assert str(vector[0]) == "1167264016.999999999"


@pytest.mark.parametrize("resolution", ["s", "ms", "us", "ns"])
def test_to_gps_numpy_datetime64_rejects_nat(resolution):
    with pytest.raises(ValueError, match="NaT|finite|valid"):
        to_gps(np.datetime64("NaT", resolution))

    with pytest.raises(ValueError, match="NaT|finite|valid"):
        to_gps(
            np.array([np.datetime64("2017-01-01", resolution), np.datetime64("NaT")])
        )


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


@pytest.mark.parametrize("resolution", ["s", "ms", "us", "ns"])
def test_tconvert_scalar_datetime64_uses_exact_to_gps_route(resolution):
    value = np.datetime64("2017-01-01T00:00:00.123456789", resolution)

    result = tconvert(value)
    expected = to_gps(value)

    assert isinstance(result, gwpy_time.LIGOTimeGPS)
    assert (result.gpsSeconds, result.gpsNanoSeconds) == (
        expected.gpsSeconds,
        expected.gpsNanoSeconds,
    )


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
