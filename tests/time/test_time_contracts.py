"""Contract tests for GWexPy time conversion boundaries."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from gwpy import time as gwpy_time

import gwexpy.time as gwexpy_time
import gwexpy.time.core as gwexpy_time_core
from gwexpy.interop._time import datetime_utc_to_gps, gps_to_datetime_utc

GPS_2017_UTC = 1167264018.0
ISO_UTC_TIMES = ["2017-01-01T00:00:00", "2017-01-01T00:00:01"]
UTC_DATETIMES = [
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=UTC),
    datetime(2017, 1, 1, 0, 0, 1, tzinfo=UTC),
]


def test_time_package_exposes_gwexpy_wrappers_and_gwpy_proxy_names():
    assert gwexpy_time.to_gps is gwexpy_time_core.to_gps
    assert gwexpy_time.from_gps is gwexpy_time_core.from_gps
    assert gwexpy_time.tconvert is gwexpy_time_core.tconvert

    assert "LIGOTimeGPS" in gwexpy_time.__all__
    assert "LIGOTimeGPS" in dir(gwexpy_time)
    assert gwexpy_time.LIGOTimeGPS is gwpy_time.LIGOTimeGPS


def test_to_gps_numeric_scalar_and_vectors_return_gps_equivalent_values():
    scalar = gwexpy_time.to_gps(1000000000.0)
    assert isinstance(scalar, gwpy_time.LIGOTimeGPS)
    assert float(scalar) == pytest.approx(1000000000.0)

    list_result = gwexpy_time.to_gps([1000000000, 1000000001])
    assert isinstance(list_result, np.ndarray)
    assert list_result.dtype == np.float64
    np.testing.assert_allclose(list_result, [1000000000.0, 1000000001.0])

    array_result = gwexpy_time.to_gps(np.array([1000000000, 1000000001]))
    assert isinstance(array_result, np.ndarray)
    assert array_result.dtype == np.float64
    np.testing.assert_allclose(array_result, [1000000000.0, 1000000001.0])

    quantity_result = gwexpy_time.to_gps(np.array([0, 1, 2]) * u.ms)
    assert isinstance(quantity_result, np.ndarray)
    np.testing.assert_allclose(quantity_result, [0.0, 0.001, 0.002])


def test_to_gps_dtype_output_modes_are_opt_in():
    default_scalar = gwexpy_time.to_gps(1000000000.0)
    assert isinstance(default_scalar, gwpy_time.LIGOTimeGPS)

    float_scalar = gwexpy_time.to_gps(1000000000.0, dtype=float)
    assert isinstance(float_scalar, float)
    assert float_scalar == pytest.approx(1000000000.0)

    quantity_scalar = gwexpy_time.to_gps(1000000000.0, dtype="quantity")
    assert quantity_scalar.unit == u.s
    assert quantity_scalar.value == pytest.approx(1000000000.0)


def test_to_gps_string_and_datetime_vectors_match_astropy_gps_values():
    string_result = gwexpy_time.to_gps(ISO_UTC_TIMES)
    assert isinstance(string_result, np.ndarray)
    np.testing.assert_allclose(string_result, Time(ISO_UTC_TIMES, scale="utc").gps)

    datetime_result = gwexpy_time.to_gps(UTC_DATETIMES)
    assert isinstance(datetime_result, np.ndarray)
    np.testing.assert_allclose(
        datetime_result,
        Time(UTC_DATETIMES, format="datetime", scale="utc").gps,
    )


def test_from_gps_scalar_and_vectors_record_timezone_and_type_contracts():
    scalar = gwexpy_time.from_gps(GPS_2017_UTC)
    assert scalar == gwpy_time.from_gps(GPS_2017_UTC)
    assert scalar.tzinfo == UTC

    gps_values = np.array([GPS_2017_UTC, GPS_2017_UTC + 1.0])
    expected_vector = Time(gps_values, format="gps").utc.to_datetime(timezone=UTC)

    array_result = gwexpy_time.from_gps(gps_values)
    assert isinstance(array_result, np.ndarray)
    assert array_result.dtype == object
    assert all(isinstance(item, datetime) for item in array_result)
    assert [item.tzinfo for item in array_result] == [UTC, UTC]
    assert array_result.tolist() == expected_vector.tolist()

    list_result = gwexpy_time.from_gps(gps_values.tolist())
    assert isinstance(list_result, np.ndarray)
    assert list_result.tolist() == expected_vector.tolist()


def test_from_gps_scalar_vector_and_astropy_routes_have_utc_parity():
    gps_values = np.array([GPS_2017_UTC + 0.25, GPS_2017_UTC + 1.25])
    expected = [
        datetime(2017, 1, 1, 0, 0, 0, 250_000, tzinfo=UTC),
        datetime(2017, 1, 1, 0, 0, 1, 250_000, tzinfo=UTC),
    ]

    scalar = gwexpy_time.from_gps(gps_values[0])
    vector = gwexpy_time.from_gps(gps_values)
    astropy_scalar = gwexpy_time.from_gps(Time(gps_values[0], format="gps"))
    astropy_vector = gwexpy_time.from_gps(Time(gps_values, format="gps"))

    assert scalar == astropy_scalar == expected[0]
    assert vector.tolist() == astropy_vector.tolist() == expected
    assert all(item.tzinfo is UTC for item in vector)


def test_from_gps_half_microsecond_rounding_has_route_parity():
    gps = 1167264102.1252985
    expected = gwpy_time.from_gps(gps)

    scalar = gwexpy_time.from_gps(gps)
    vector = gwexpy_time.from_gps(np.asarray([gps]))[0]

    assert scalar == vector == expected


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="host TZ switching unavailable")
def test_from_gps_is_independent_of_host_timezone():
    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Tokyo"
        time.tzset()
        result = gwexpy_time.from_gps(np.array([GPS_2017_UTC, GPS_2017_UTC + 1]))
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

    assert result.tolist() == [
        datetime(2017, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2017, 1, 1, 0, 0, 1, tzinfo=UTC),
    ]


def test_from_gps_leap_second_neighbors_and_unrepresentable_instant():
    before, leap, after = 1167264016, 1167264017, 1167264018

    assert gwexpy_time.from_gps(before) == datetime(
        2016, 12, 31, 23, 59, 59, tzinfo=UTC
    )
    assert gwexpy_time.from_gps(after) == datetime(2017, 1, 1, tzinfo=UTC)

    for value in (
        leap,
        np.array([before, leap, after]),
        Time(leap, format="gps"),
        Time([before, leap, after], format="gps"),
    ):
        with pytest.raises(ValueError, match="leap second"):
            gwexpy_time.from_gps(value)


def test_tconvert_dispatches_vectors_by_current_numeric_detection():
    gps_values = np.array([GPS_2017_UTC, GPS_2017_UTC + 1.0])
    numeric_result = gwexpy_time.tconvert(gps_values)
    assert numeric_result.tolist() == gwexpy_time.from_gps(gps_values).tolist()

    string_result = gwexpy_time.tconvert(ISO_UTC_TIMES)
    np.testing.assert_allclose(string_result, gwexpy_time.to_gps(ISO_UTC_TIMES))

    datetime_result = gwexpy_time.tconvert(UTC_DATETIMES)
    np.testing.assert_allclose(datetime_result, gwexpy_time.to_gps(UTC_DATETIMES))


def test_interop_datetime_to_gps_timezone_policy_matches_same_utc_instant():
    naive_utc = datetime(2017, 1, 1, 0, 0, 0)
    aware_utc = datetime(2017, 1, 1, 0, 0, 0, tzinfo=UTC)
    aware_eastern = datetime(
        2016,
        12,
        31,
        19,
        0,
        0,
        tzinfo=timezone(-timedelta(hours=5)),
    )

    aware_gps = float(datetime_utc_to_gps(aware_utc))
    assert float(datetime_utc_to_gps(naive_utc)) == pytest.approx(aware_gps)
    assert float(datetime_utc_to_gps(aware_eastern)) == pytest.approx(aware_gps)


def test_interop_gps_to_utc_datetime_returns_aware_utc_and_roundtrips():
    dt = gps_to_datetime_utc(GPS_2017_UTC)
    assert dt == datetime(2017, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert dt.tzinfo == UTC

    gps_back = datetime_utc_to_gps(dt)
    assert float(gps_back) == pytest.approx(GPS_2017_UTC, abs=1e-6)
