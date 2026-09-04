"""Differential contracts for the Phase 2 time-conversion fixes."""

from __future__ import annotations

import inspect
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction

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
        pytest.param((2017, 1, 32), ValueError, id="out-of-range-day"),
        pytest.param((2017.0, 1, 1), TypeError, id="float-component"),
        pytest.param([2017.0, 1, 1], TypeError, id="list-float-component"),
    ],
)
def test_to_gps_invalid_date_components_preserve_gwpy_failure(value, error):
    with pytest.raises(error):
        gwpy_time.to_gps(value)
    with pytest.raises(error):
        gwexpy_time.to_gps(value)


@pytest.mark.parametrize("function_name", ["to_gps", "tconvert"])
@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize(
    ("components", "error"),
    [
        pytest.param(
            [Decimal("2017"), Decimal("1"), Decimal("1")],
            TypeError,
            id="decimal",
        ),
        pytest.param(
            [Fraction(2017, 1), Fraction(1, 1), Fraction(1, 1)],
            TypeError,
            id="fraction",
        ),
        pytest.param([True, True, True], OverflowError, id="bool"),
        pytest.param(
            [np.bool_(True), np.bool_(True), np.bool_(True)],
            OverflowError,
            id="numpy-bool",
        ),
        pytest.param([2017 + 0j, 1 + 0j, 1 + 0j], TypeError, id="complex"),
        pytest.param(["2017", "1", "1"], TypeError, id="numeric-strings"),
    ],
)
def test_numeric_like_date_components_preserve_gwpy_failure(
    function_name,
    container,
    components,
    error,
):
    value = container(components)
    gwpy_function = getattr(gwpy_time, function_name)
    gwexpy_function = getattr(gwexpy_time, function_name)

    # Year 1 reaches ERFA before LAL rejects its negative GPS representation.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='ERFA function ".*dubious year')
        warnings.filterwarnings(
            "ignore",
            message="In future, it will be an error for 'np.bool_'",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
        with pytest.raises(error):
            gwpy_function(value)
        with pytest.raises(error):
            gwexpy_function(value)


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
            ["2017-01-01", "2017-01-02", "2017-01-03"],
            [1167264018.0, 1167350418.0, 1167436818.0],
            id="date-string-list",
        ),
        pytest.param(
            [
                datetime(2017, 1, 1, tzinfo=UTC),
                datetime(2017, 1, 2, tzinfo=UTC),
                datetime(2017, 1, 3, tzinfo=UTC),
            ],
            [1167264018.0, 1167350418.0, 1167436818.0],
            id="datetime-list",
        ),
    ],
)
def test_to_gps_explicit_vector_extensions_remain(value, expected):
    result = gwexpy_time.to_gps(value)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([0.0, 1.0, 2.0, 3.0], id="small-gps-vector"),
        pytest.param([1000.0, 2000.0, 3000.0], id="large-gps-vector"),
    ],
)
def test_to_gps_numeric_vectors_are_not_date_components(value):
    result = gwexpy_time.to_gps(value)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.asarray(value))


def test_nested_numeric_sequences_match_equivalent_ndarray_extensions():
    values = np.array(
        [
            [1000.0, 2000.0],
            [3000.0, 4000.0],
            [5000.0, 6000.0],
        ],
    )
    nested_values = (
        values.tolist(),
        tuple(tuple(row) for row in values),
        values,
    )

    for function_name in ("to_gps", "tconvert"):
        function = getattr(gwexpy_time, function_name)
        expected = function(values)
        for value in nested_values:
            result = function(value)
            assert isinstance(result, np.ndarray)
            assert result.shape == values.shape
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected)


def test_tconvert_first_parameter_matches_canonical_introspection():
    parameter = next(iter(inspect.signature(gwexpy_time.tconvert).parameters.values()))
    expected = next(iter(inspect.signature(gwpy_time.tconvert).parameters.values()))

    assert parameter.name == expected.name
    assert parameter.kind is expected.kind
    assert type(parameter.default) is type(expected.default)
    assert parameter.default == expected.default
    assert parameter.annotation == expected.annotation


def test_tconvert_no_argument_is_bounded_by_real_gwpy_calls():
    before = gwpy_time.tconvert()
    result = gwexpy_time.tconvert()
    after = gwpy_time.tconvert()

    assert type(result) is type(before)
    assert before <= result <= after


def test_tconvert_positional_gps_matches_gwpy():
    value = 1126259462
    _assert_same_scalar(gwexpy_time.tconvert(value), gwpy_time.tconvert(value))


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
    introspected_default = (
        inspect.signature(gwexpy_time.tconvert).parameters["gpsordate"].default
    )
    for canonical in (1126259462, introspected_default):
        with pytest.raises(TypeError, match="cannot specify both 'gpsordate' and 't'"):
            gwexpy_time.tconvert(gpsordate=canonical, t=1126259463)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            ["2017-01-01", "2017-01-02", "2017-01-03"],
            id="iso-string-list",
        ),
        pytest.param(
            [
                datetime(2017, 1, 1, tzinfo=UTC),
                datetime(2017, 1, 2, tzinfo=UTC),
                datetime(2017, 1, 3, tzinfo=UTC),
            ],
            id="datetime-list",
        ),
    ],
)
def test_tconvert_non_numeric_date_vectors_remain_extensions(value):
    np.testing.assert_array_equal(
        gwexpy_time.tconvert(value),
        gwexpy_time.to_gps(value),
    )


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


def test_from_gps_empty_vectors_preserve_rank_and_object_dtype():
    for values in (np.array([], dtype=float), np.empty((0, 2), dtype=float)):
        result = gwexpy_time.from_gps(values)

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape
        assert result.dtype == object
        assert result.size == 0


def test_from_gps_multidimensional_vectors_map_scalar_route():
    object_values = np.empty((2, 2), dtype=object)
    object_values[:] = [
        [
            gwpy_time.LIGOTimeGPS(1167264102, 500),
            gwpy_time.LIGOTimeGPS(1167264102, 1500),
        ],
        [
            gwpy_time.LIGOTimeGPS(1167264102, 999999500),
            gwpy_time.LIGOTimeGPS(1167264103, 500),
        ],
    ]
    numeric_values = np.array(
        [
            [1167264018.0, 1167264102.1252985],
            [1167264103.0, 1167264104.5],
        ],
    )

    for values in (numeric_values, object_values):
        expected = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            expected[index] = gwpy_time.from_gps(values[index])

        result = gwexpy_time.from_gps(values)

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape
        assert result.dtype == object
        np.testing.assert_array_equal(result, expected)


def test_from_gps_nonfinite_vectors_preserve_gwpy_failure_class():
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(RuntimeError):
            gwpy_time.from_gps(value)
        with pytest.raises(RuntimeError):
            gwexpy_time.from_gps(np.array([1167264018.0, value]))


def test_from_gps_leap_second_preserves_gwpy_failure_class_for_scalar_and_vector():
    leap_second = 1167264017
    with pytest.raises(ValueError):
        gwpy_time.from_gps(leap_second)
    with pytest.raises(ValueError):
        gwexpy_time.from_gps(leap_second)
    with pytest.raises(ValueError):
        gwexpy_time.from_gps(np.array([1167264016, leap_second, 1167264018]))
