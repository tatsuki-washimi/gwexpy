"""Differential contracts for the Phase 2 time-conversion fixes."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.time import Time
from gwpy import time as gwpy_time

from gwexpy import time as gwexpy_time


def _assert_same_scalar(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, gwpy_time.LIGOTimeGPS):
        assert (actual.gpsSeconds, actual.gpsNanoSeconds) == (
            expected.gpsSeconds,
            expected.gpsNanoSeconds,
        )
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(Time(57754, format="mjd"), id="whole-mjd"),
        pytest.param(Time(57754.0001, format="mjd"), id="fractional-mjd"),
        pytest.param(
            Time(
                "2017-01-01T00:00:00.123456789",
                format="isot",
                scale="utc",
            ),
            id="sub-microsecond-isot",
        ),
    ],
)
def test_to_gps_scalar_time_matches_gwpy_default(value):
    _assert_same_scalar(gwexpy_time.to_gps(value), gwpy_time.to_gps(value))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param((2017, 1, 1), id="tuple-date"),
        pytest.param([2017, 1, 1], id="list-date"),
        pytest.param((2017, 1, 1, 1), id="hour"),
        pytest.param((2017, 1, 1, 1, 2), id="minute"),
        pytest.param((2017, 1, 1, 1, 2, 3), id="second"),
        pytest.param((2017, 1, 1, 1, 2, 3, 456789), id="microsecond"),
        pytest.param([2017, 1, 1, 1, 2, 3, 456789], id="list-microsecond"),
    ],
)
def test_to_gps_date_component_sequences_match_gwpy(value):
    _assert_same_scalar(gwexpy_time.to_gps(value), gwpy_time.to_gps(value))


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param((2017, 13, 1), ValueError, id="tuple-invalid-month"),
        pytest.param([2017, 13, 1], ValueError, id="list-invalid-month"),
        pytest.param((2017, 2, 29), ValueError, id="invalid-day"),
        pytest.param((2017.0, 1, 1), TypeError, id="float-component"),
        pytest.param([2017.0, 1, 1], TypeError, id="list-float-component"),
    ],
)
def test_to_gps_invalid_date_components_preserve_gwpy_failure(value, error):
    with pytest.raises(error):
        gwpy_time.to_gps(value)
    with pytest.raises(error):
        gwexpy_time.to_gps(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param([1000.0, 2000.0], [1000.0, 2000.0], id="short-numeric-list"),
        pytest.param(
            np.array([1000.0, 2000.0, 3000.0]),
            [1000.0, 2000.0, 3000.0],
            id="numeric-ndarray",
        ),
        pytest.param(
            ["2017-01-01", "2017-01-02"],
            [1167264018.0, 1167350418.0],
            id="date-string-list",
        ),
        pytest.param(
            [datetime(2017, 1, 1, tzinfo=UTC), datetime(2017, 1, 2, tzinfo=UTC)],
            [1167264018.0, 1167350418.0],
            id="datetime-list",
        ),
    ],
)
def test_to_gps_explicit_vector_extensions_remain(value, expected):
    result = gwexpy_time.to_gps(value)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


def test_tconvert_restores_canonical_parameter_name():
    parameters = inspect.signature(gwexpy_time.tconvert).parameters
    assert next(iter(parameters)) == "gpsordate"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1126259462, id="gps"),
        pytest.param("2017-01-01", id="date-string"),
        pytest.param((2017, 1, 1), id="date-components"),
    ],
)
def test_tconvert_gpsordate_keyword_matches_gwpy(value):
    _assert_same_scalar(
        gwexpy_time.tconvert(gpsordate=value),
        gwpy_time.tconvert(gpsordate=value),
    )


def test_tconvert_positional_date_components_match_gwpy():
    value = (2017, 1, 1, 1, 2, 3, 456789)
    _assert_same_scalar(gwexpy_time.tconvert(value), gwpy_time.tconvert(value))


def test_tconvert_invalid_date_components_do_not_fall_back_to_vector():
    value = (2017, 13, 1)
    with pytest.raises(ValueError):
        gwpy_time.tconvert(value)
    with pytest.raises(ValueError):
        gwexpy_time.tconvert(value)


def test_tconvert_preserves_documented_t_alias():
    _assert_same_scalar(
        gwexpy_time.tconvert(t=1126259462),
        gwpy_time.tconvert(gpsordate=1126259462),
    )


def test_tconvert_rejects_canonical_name_and_alias_together():
    with pytest.raises(TypeError, match="cannot specify both 'gpsordate' and 't'"):
        gwexpy_time.tconvert(gpsordate=1126259462, t=1126259463)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1167264018, id="integer"),
        pytest.param("1167264018", id="numeric-string"),
        pytest.param("1.13e9", id="scientific-string"),
        pytest.param(1126259462.391, id="fractional"),
        pytest.param(
            gwpy_time.LIGOTimeGPS(1126259462, 391000000),
            id="ligotimegps",
        ),
        pytest.param(1167264102.1252985, id="half-microsecond-float"),
        pytest.param(
            gwpy_time.LIGOTimeGPS(1167264102, 500),
            id="half-microsecond-ligotimegps",
        ),
        pytest.param(
            gwpy_time.LIGOTimeGPS(1167264102, 1500),
            id="one-and-half-microseconds",
        ),
    ],
)
def test_from_gps_scalar_matches_gwpy_rounding(value):
    _assert_same_scalar(gwexpy_time.from_gps(value), gwpy_time.from_gps(value))


def test_from_gps_vector_maps_the_gwpy_scalar_route():
    values = np.array(
        [
            gwpy_time.LIGOTimeGPS(1167264102, 500),
            gwpy_time.LIGOTimeGPS(1167264102, 1500),
            gwpy_time.LIGOTimeGPS(1167264102, 999999500),
        ],
        dtype=object,
    )
    result = gwexpy_time.from_gps(values)
    expected = [gwpy_time.from_gps(value) for value in values]

    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    assert result.shape == values.shape
    assert result.tolist() == expected


def test_from_gps_leap_second_preserves_gwpy_failure_class_for_scalar_and_vector():
    leap_second = 1167264017
    with pytest.raises(ValueError):
        gwpy_time.from_gps(leap_second)
    with pytest.raises(ValueError):
        gwexpy_time.from_gps(leap_second)
    with pytest.raises(ValueError):
        gwexpy_time.from_gps(np.array([1167264016, leap_second, 1167264018]))
