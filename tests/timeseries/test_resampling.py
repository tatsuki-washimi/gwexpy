"""Tests for gwexpy/timeseries/_resampling.py"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries._resampling import (
    _determine_output_dtype,
    _generate_target_grid,
    _parse_rule_to_dt,
    _reindex_by_method,
    _resolve_origin,
    _validate_time_unit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ts(values=None, *, dt=0.1, unit=u.m, t0=0.0):
    """Create a simple TimeSeries for testing."""
    if values is None:
        values = np.arange(10, dtype=float)
    return TimeSeries(
        np.asarray(values, dtype=float),
        dt=dt * u.s,
        t0=t0 * u.s,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# _parse_rule_to_dt
# ---------------------------------------------------------------------------


def test_parse_rule_int():
    result = _parse_rule_to_dt(1)
    assert isinstance(result, u.Quantity)
    assert result.value == pytest.approx(1.0)


def test_parse_rule_float():
    result = _parse_rule_to_dt(0.5)
    assert result.value == pytest.approx(0.5)


def test_parse_rule_quantity():
    q = 0.1 * u.s
    result = _parse_rule_to_dt(q)
    assert result is q


def test_parse_rule_string_with_unit():
    result = _parse_rule_to_dt("1s")
    assert result.unit.is_equivalent(u.s)
    assert result.to(u.s).value == pytest.approx(1.0)


def test_parse_rule_string_number_only():
    result = _parse_rule_to_dt("0.5")
    assert result.value == pytest.approx(0.5)


def test_parse_rule_invalid_type():
    with pytest.raises(TypeError):
        _parse_rule_to_dt([1, 2, 3])  # type: ignore


def test_parse_rule_invalid_string():
    # A string that cannot be parsed as Quantity and has no numeric prefix
    with pytest.raises(ValueError):
        _parse_rule_to_dt("abc!")


# ---------------------------------------------------------------------------
# _validate_time_unit
# ---------------------------------------------------------------------------


def test_validate_time_unit_time():
    target_dt = 1.0 * u.s
    is_time, is_dimless = _validate_time_unit(target_dt)
    assert is_time is True
    assert is_dimless is False


def test_validate_time_unit_dimensionless():
    target_dt = u.Quantity(0.5)  # dimensionless
    is_time, is_dimless = _validate_time_unit(target_dt)
    assert is_time is False
    assert is_dimless is True


def test_validate_time_unit_invalid_raises():
    target_dt = 1.0 * u.m  # length — neither time nor dimensionless
    with pytest.raises(ValueError, match="time-like or dimensionless"):
        _validate_time_unit(target_dt)


# ---------------------------------------------------------------------------
# _resolve_origin
# ---------------------------------------------------------------------------


def test_resolve_origin_t0():
    val = _resolve_origin("t0", None, 5.0, u.s, "ceil")
    assert val == pytest.approx(5.0)


def test_resolve_origin_gps0():
    val = _resolve_origin("gps0", None, 5.0, u.s, "ceil")
    assert val == pytest.approx(0.0)


def test_resolve_origin_numeric():
    val = _resolve_origin(10.0, None, 5.0, u.s, "ceil")
    assert val == pytest.approx(10.0)


def test_resolve_origin_quantity():
    val = _resolve_origin(2.0 * u.s, None, 5.0, u.s, "ceil")
    assert val == pytest.approx(2.0)


def test_resolve_origin_with_offset():
    val = _resolve_origin("t0", 1.0 * u.s, 0.0, u.s, "ceil")
    assert val == pytest.approx(1.0)


def test_resolve_origin_unknown_fallback():
    # An unknown type falls back to 0.0
    val = _resolve_origin(object(), None, 5.0, u.s, "ceil")  # type: ignore
    assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _generate_target_grid
# ---------------------------------------------------------------------------


def test_generate_target_grid_ceil():
    times, grid_start = _generate_target_grid(0.0, 0.1, 0.0, 0.5, "ceil")
    assert len(times) == 5
    assert times[0] == pytest.approx(0.0)


def test_generate_target_grid_floor():
    times, grid_start = _generate_target_grid(0.0, 0.1, 0.0, 0.5, "floor")
    assert len(times) == 5


def test_generate_target_grid_empty_when_duration_zero():
    # ceil((0.4 - 0.0) / 1.0) = 1 → grid_start = 1.0 > t_end=0.5 → empty
    times, grid_start = _generate_target_grid(0.0, 1.0, 0.4, 0.5, "ceil")
    assert len(times) == 0


def test_generate_target_grid_invalid_align():
    with pytest.raises(ValueError, match="align"):
        _generate_target_grid(0.0, 0.1, 0.0, 1.0, "middle")  # type: ignore


# ---------------------------------------------------------------------------
# _determine_output_dtype
# ---------------------------------------------------------------------------


def test_determine_output_dtype_float_fill_nan():
    dtype, fill = _determine_output_dtype(None, np.nan, np.dtype(float))
    assert dtype == np.dtype(float)
    assert np.isnan(fill)


def test_determine_output_dtype_int_fill_none():
    # int dtype with None fill_value → out_dtype promoted to float64
    dtype, fill = _determine_output_dtype(None, None, np.dtype(int))
    assert dtype == np.dtype(np.float64)
    assert np.isnan(fill)


def test_determine_output_dtype_non_nan_fill():
    dtype, fill = _determine_output_dtype(None, 0.0, np.dtype(float))
    assert dtype == np.dtype(float)
    assert fill == pytest.approx(0.0)


def test_determine_output_dtype_int_with_nan_fill():
    # nan fill with int dtype → promotes to float64
    dtype, fill = _determine_output_dtype(None, np.nan, np.dtype(int))
    assert dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    "fill_value", [np.float32(0.5), np.complex64(1 + 2j), np.int16(7)]
)
def test_determine_output_dtype_boolean_fill_uses_result_type(fill_value):
    self_dtype = np.dtype(bool)
    dtype, fill = _determine_output_dtype(None, fill_value, self_dtype)
    assert dtype == np.result_type(self_dtype, np.asarray(fill_value).dtype)
    assert fill == fill_value


@pytest.mark.parametrize("self_dtype", [np.dtype("int16"), np.dtype("uint16")])
def test_determine_output_dtype_representable_integer_fill_keeps_source_dtype(
    self_dtype,
):
    dtype, fill = _determine_output_dtype(None, np.int16(7), self_dtype)
    assert dtype == self_dtype
    assert fill == 7


@pytest.mark.parametrize(
    "self_dtype", [np.dtype(bool), np.dtype("int16"), np.dtype("uint16")]
)
@pytest.mark.parametrize("fill_value", [np.inf, -np.inf])
def test_determine_output_dtype_nonfinite_fill_uses_result_type(self_dtype, fill_value):
    dtype, _ = _determine_output_dtype(None, fill_value, self_dtype)
    assert dtype == np.result_type(self_dtype, np.asarray(fill_value).dtype)


def test_determine_output_dtype_masked_fill_preserves_integer_dtype():
    dtype, _ = _determine_output_dtype(None, np.ma.masked, np.dtype("int16"))

    assert dtype == np.dtype("int16")


@pytest.mark.parametrize("fill_value", [0.5, np.nan])
def test_determine_output_dtype_float32_keeps_own_dtype(fill_value):
    dtype, _ = _determine_output_dtype(None, fill_value, np.dtype("float32"))
    assert dtype == np.dtype("float32")


def test_determine_output_dtype_uint64_rejects_negative_integer_fill():
    with pytest.raises(ValueError, match="cannot be represented"):
        _determine_output_dtype(None, -1, np.dtype("uint64"))


@pytest.mark.parametrize("fill_value", [0, np.int64(0)])
def test_determine_output_dtype_uint64_keeps_dtype_for_representable_integer_fill(
    fill_value,
):
    dtype, _ = _determine_output_dtype(None, fill_value, np.dtype("uint64"))

    assert dtype == np.dtype("uint64")


def test_determine_output_dtype_int64_keeps_dtype_for_representable_unsigned_fill():
    dtype, _ = _determine_output_dtype(None, np.uint64(1), np.dtype("int64"))

    assert dtype == np.dtype("int64")


@pytest.mark.parametrize(
    ("self_dtype", "fill_value"),
    [
        (np.dtype("uint64"), -1),
        (np.dtype("int8"), 128),
    ],
)
def test_determine_output_dtype_rejects_out_of_range_integer_fill(
    self_dtype, fill_value
):
    with pytest.raises(ValueError, match="cannot be represented"):
        _determine_output_dtype(None, fill_value, self_dtype)


@pytest.mark.parametrize(
    ("fill_value", "expected_dtype"),
    [
        (0.5, np.dtype("float64")),
        (1 + 2j, np.dtype("complex128")),
        (np.nan, np.dtype("float64")),
        (np.inf, np.dtype("float64")),
        (-np.inf, np.dtype("float64")),
    ],
)
def test_determine_output_dtype_uint64_promotes_noninteger_fill(
    fill_value, expected_dtype
):
    dtype, _ = _determine_output_dtype(None, fill_value, np.dtype("uint64"))

    assert dtype == expected_dtype


# ---------------------------------------------------------------------------
# _reindex_by_method
# ---------------------------------------------------------------------------


def _make_reindex_inputs():
    old_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    old_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    new_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    return old_times, old_values, new_times


def test_reindex_none_exact_match():
    old_t, old_v, new_t = _make_reindex_inputs()
    result = _reindex_by_method(
        old_t, old_v, new_t, None, np.nan, None, None, u.s, np.dtype(float)
    )
    np.testing.assert_allclose(result, old_v)


def test_reindex_none_no_exact_match():
    old_t = np.array([0.0, 0.2, 0.4])
    old_v = np.array([10.0, 30.0, 50.0])
    new_t = np.array([0.1, 0.3])
    result = _reindex_by_method(
        old_t, old_v, new_t, None, np.nan, None, None, u.s, np.dtype(float)
    )
    # No exact match → fill with NaN
    assert np.all(np.isnan(result))


def test_reindex_ffill():
    old_t = np.array([0.0, 0.2, 0.4])
    old_v = np.array([10.0, 30.0, 50.0])
    new_t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    result = _reindex_by_method(
        old_t, old_v, new_t, "ffill", np.nan, None, None, u.s, np.dtype(float)
    )
    # At t=0.1, ffill from t=0 → 10.0; at t=0.3, ffill from t=0.2 → 30.0
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(10.0)
    assert result[2] == pytest.approx(30.0)
    assert result[3] == pytest.approx(30.0)
    assert result[4] == pytest.approx(50.0)


def test_reindex_bfill():
    old_t = np.array([0.0, 0.2, 0.4])
    old_v = np.array([10.0, 30.0, 50.0])
    new_t = np.array([0.0, 0.1, 0.2])
    result = _reindex_by_method(
        old_t, old_v, new_t, "bfill", np.nan, None, None, u.s, np.dtype(float)
    )
    # At t=0.1, bfill from t=0.2 → 30.0
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(30.0)
    assert result[2] == pytest.approx(30.0)


def test_reindex_nearest():
    old_t = np.array([0.0, 0.2, 0.4])
    old_v = np.array([10.0, 30.0, 50.0])
    new_t = np.array([0.05, 0.15, 0.3])
    result = _reindex_by_method(
        old_t, old_v, new_t, "nearest", np.nan, None, None, u.s, np.dtype(float)
    )
    # 0.05 → nearest 0.0 → 10.0; 0.15 → equidistant, take L (0.0→10.0) or R (0.2→30.0)
    assert result[0] == pytest.approx(10.0)
    # 0.3 is between 0.2 and 0.4, equidistant — nearest picks one of them
    assert result[2] in (pytest.approx(30.0), pytest.approx(50.0))


def test_reindex_unknown_method_raises():
    old_t, old_v, new_t = _make_reindex_inputs()
    with pytest.raises(ValueError, match="Unknown asfreq method"):
        _reindex_by_method(
            old_t, old_v, new_t, "bad_method", np.nan, None, None, u.s, np.dtype(float)
        )


def test_reindex_interpolate_raises():
    old_t, old_v, new_t = _make_reindex_inputs()
    with pytest.raises(ValueError, match="asfreq does not interpolate"):
        _reindex_by_method(
            old_t, old_v, new_t, "interpolate", np.nan, None, None, u.s, np.dtype(float)
        )


def test_reindex_ffill_with_tolerance():
    old_t = np.array([0.0, 1.0])
    old_v = np.array([10.0, 20.0])
    new_t = np.array([0.5])
    # tolerance = 0.3 s, gap=0.5 → out of tolerance → stays fill_value=NaN
    result = _reindex_by_method(
        old_t, old_v, new_t, "ffill", np.nan, 0.3 * u.s, None, u.s, np.dtype(float)
    )
    assert np.isnan(result[0])


def test_reindex_bfill_with_tolerance():
    old_t = np.array([0.5])
    old_v = np.array([20.0])
    new_t = np.array([0.0])
    # bfill from 0.5 with tolerance 0.3 → gap 0.5 > tolerance → NaN
    result = _reindex_by_method(
        old_t, old_v, new_t, "bfill", np.nan, 0.3 * u.s, None, u.s, np.dtype(float)
    )
    assert np.isnan(result[0])


def test_reindex_nearest_with_tolerance():
    old_t = np.array([0.0])
    old_v = np.array([10.0])
    new_t = np.array([1.0])
    # nearest: distance=1.0, tolerance=0.5 → NaN
    result = _reindex_by_method(
        old_t, old_v, new_t, "nearest", np.nan, 0.5 * u.s, None, u.s, np.dtype(float)
    )
    assert np.isnan(result[0])


# ---------------------------------------------------------------------------
# TimeSeries.asfreq (integration via mixin)
# ---------------------------------------------------------------------------


def test_asfreq_basic():
    ts = _make_ts(np.arange(10, dtype=float), dt=0.1)
    result = ts.asfreq(0.1 * u.s)
    assert isinstance(result, TimeSeries)
    assert result.dt.to("s").value == pytest.approx(0.1)
    assert len(result) == len(ts)


def test_asfreq_nonsecond_axis_preserves_x0_and_values():
    source = TimeSeries([10.0, 20.0, 30.0], x0=1 * u.min, dt=1 * u.min)

    result = source.asfreq(1 * u.min)

    assert result.x0 == source.x0
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(result.value, source.value)


def test_asfreq_empty_nonsecond_axis_preserves_grid_x0():
    source = TimeSeries([10.0, 20.0, 30.0], x0=1 * u.min, dt=1 * u.min)

    result = source.asfreq(5 * u.min, origin="gps0")

    assert result.x0 == 5 * u.min
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(result.value, [])


def test_asfreq_upsampling_ffill():
    ts = _make_ts([0.0, 1.0, 2.0], dt=1.0)
    result = ts.asfreq(0.5 * u.s, method="ffill")
    assert isinstance(result, TimeSeries)
    assert len(result) > len(ts)


def test_asfreq_bfill():
    ts = _make_ts([0.0, 1.0, 2.0], dt=1.0)
    result = ts.asfreq(0.5 * u.s, method="bfill")
    assert isinstance(result, TimeSeries)


def test_asfreq_nearest():
    ts = _make_ts([0.0, 1.0, 2.0], dt=1.0)
    result = ts.asfreq(0.5 * u.s, method="nearest")
    assert isinstance(result, TimeSeries)


def test_asfreq_string_rule():
    ts = _make_ts(np.arange(10, dtype=float), dt=0.1)
    result = ts.asfreq("0.1s")
    assert isinstance(result, TimeSeries)


def test_asfreq_floor_align():
    ts = _make_ts(np.arange(10, dtype=float), dt=0.1)
    result = ts.asfreq(0.1 * u.s, align="floor")
    assert isinstance(result, TimeSeries)


def test_asfreq_origin_gps0():
    ts = _make_ts(np.arange(10, dtype=float), dt=0.1, t0=0.0)
    result = ts.asfreq(0.1 * u.s, origin="gps0")
    assert isinstance(result, TimeSeries)


def test_asfreq_with_fill_value():
    ts = _make_ts([0.0, 1.0, 2.0], dt=1.0)
    result = ts.asfreq(0.5 * u.s, method=None, fill_value=0.0)
    assert isinstance(result, TimeSeries)
    # Non-matched points should be 0.0
    assert np.all(np.isfinite(result.value))


def test_asfreq_int16_fractional_fill_promotes_and_preserves_values():
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=0.5)

    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result.value, [10.0, 0.5, 20.0, 0.5, 30.0, 0.5])


@pytest.mark.parametrize(
    ("fill_value", "expected_dtype", "expected_values"),
    [
        (
            np.array(0.5),
            np.dtype("float64"),
            [10.0, 0.5, 20.0, 0.5, 30.0, 0.5],
        ),
        (
            np.array(1 + 2j),
            np.dtype("complex128"),
            [10 + 0j, 1 + 2j, 20 + 0j, 1 + 2j, 30 + 0j, 1 + 2j],
        ),
    ],
)
def test_asfreq_int16_zero_dimensional_fill_preserves_values(
    fill_value, expected_dtype, expected_values
):
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=fill_value)

    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result.value, expected_values)


@pytest.mark.parametrize("fill_value", [np.inf, -np.inf])
def test_asfreq_int16_nonfinite_fill_promotes_and_preserves_values(fill_value):
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=fill_value)

    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result.value[::2], [10.0, 20.0, 30.0])
    assert np.all(np.isinf(result.value[1::2]))
    assert np.all(np.signbit(result.value[1::2]) == np.signbit(fill_value))


def test_asfreq_int16_float32_fill_allocates_float32_output():
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=np.float32(0.5))

    assert result.dtype == np.dtype("float32")
    np.testing.assert_array_equal(result.value, [10.0, 0.5, 20.0, 0.5, 30.0, 0.5])


def test_asfreq_int16_complex_fill_preserves_both_components():
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=1 + 2j)

    assert result.dtype == np.dtype("complex128")
    np.testing.assert_array_equal(
        result.value, [10 + 0j, 1 + 2j, 20 + 0j, 1 + 2j, 30 + 0j, 1 + 2j]
    )


def test_asfreq_int16_integer_fill_keeps_integer_dtype_kind():
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=0)

    assert result.dtype.kind == "i"
    np.testing.assert_array_equal(result.value, [10, 0, 20, 0, 30, 0])


@pytest.mark.parametrize("fill_value", [0, np.int64(0)])
def test_asfreq_uint64_representable_integer_fill_preserves_exact_values(fill_value):
    values = np.array([2**63 + 1, 2**63 + 3], dtype=np.uint64)
    ts = TimeSeries(values, dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=fill_value)

    assert result.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(result.value[::2], values)
    np.testing.assert_array_equal(result.value[1::2], [0, 0])


def test_asfreq_uint64_rejects_negative_integer_fill():
    ts = TimeSeries(np.array([2**63 + 1, 2**63 + 3], dtype=np.uint64), dt=1.0 * u.s)

    with pytest.raises(ValueError, match="cannot be represented"):
        ts.asfreq(0.5 * u.s, method=None, fill_value=-1)


@pytest.mark.parametrize("fill_value", [None, np.nan])
def test_asfreq_int16_none_and_nan_fill_remain_float64_nan(fill_value):
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.int16), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, fill_value=fill_value)

    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result.value[::2], [10.0, 20.0, 30.0])
    assert np.all(np.isnan(result.value[1::2]))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"fill_value": 0.5}, [10.0, 0.5, 20.0, 0.5, 30.0, 0.5]),
        ({}, [10.0, np.nan, 20.0, np.nan, 30.0, np.nan]),
    ],
)
def test_asfreq_float32_finite_and_default_nan_fill_remain_float32(kwargs, expected):
    ts = TimeSeries(np.array([10, 20, 30], dtype=np.float32), dt=1.0 * u.s)

    result = ts.asfreq(0.5 * u.s, method=None, **kwargs)

    assert result.dtype == np.dtype("float32")
    np.testing.assert_equal(result.value, expected)


def test_asfreq_complex64_default_nan_fill_remains_complex128():
    ts = TimeSeries(
        np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64), dt=1.0 * u.s
    )

    result = ts.asfreq(0.5 * u.s, method=None)

    assert result.dtype == np.dtype("complex128")
    np.testing.assert_array_equal(result.value[::2], ts.value)
    assert np.all(np.isnan(result.value[1::2].real))
    np.testing.assert_array_equal(result.value[1::2].imag, [0.0, 0.0, 0.0])


def test_asfreq_preserves_unit():
    # Check that output unit is preserved from input
    ts = _make_ts(np.arange(5, dtype=float), dt=0.1, unit=u.m)
    result = ts.asfreq(0.1 * u.s)
    assert result.unit == u.m


# ---------------------------------------------------------------------------
# TimeSeries.resample — time-bin path (string rule)
# ---------------------------------------------------------------------------


def test_resample_string_rule_uses_timebin():
    data = np.arange(100, dtype=float)
    ts = TimeSeries(data, dt=0.01 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts.resample("0.1s")
    assert isinstance(result, TimeSeries)
    # 100 samples @ 0.01s → 1.0s total; bins of 0.1s → ~10 bins
    assert len(result) <= 10


def test_resample_quantity_time_rule():
    data = np.arange(100, dtype=float)
    ts = TimeSeries(data, dt=0.01 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts.resample(0.1 * u.s)
    assert isinstance(result, TimeSeries)


def test_resample_time_bin_preserves_nonsecond_axis_authority():
    source = TimeSeries(
        np.array([1.0, 2.0, 3.0, 4.0]),
        x0=1 * u.min,
        dt=1 * u.min,
        unit=u.V,
    )

    result = source.resample(2 * u.min)

    np.testing.assert_allclose(result.value, [1.5, 3.5])
    assert result.x0 == 1 * u.min
    assert result.t0 == 1 * u.min
    assert result.dt == 2 * u.min
    np.testing.assert_allclose(result.times.to_value(u.min), [1.0, 3.0])


def test_resample_time_bin_empty_preserves_nonsecond_axis_authority():
    source = TimeSeries(
        np.array([1.0]),
        x0=1 * u.min,
        dt=1 * u.min,
        unit=u.V,
    )

    result = source.resample(2 * u.min, align="ceil", offset=1.5 * u.min)

    assert result.size == 0
    assert result.x0 == 2.5 * u.min
    assert result.dt == 2 * u.min
    assert result.unit == u.V


def test_resample_time_bin_closed_right_assigns_boundary_to_previous_bin():
    ts = TimeSeries(
        np.array([1.0, 3.0, 5.0, 7.0]),
        dt=1.0 * u.s,
        t0=0.0 * u.s,
        unit=u.m,
        name="boundary-test",
        channel="H1:TEST",
    )

    result = ts.resample("2s", closed="right")

    # Right-closed bins are [0, 2], then (2, 4].
    np.testing.assert_allclose(result.value, [3.0, 7.0])
    assert result.t0.to_value(u.s) == pytest.approx(0.0)
    assert result.dt.to_value(u.s) == pytest.approx(2.0)
    assert result.unit == u.m
    assert result.name == "boundary-test"
    assert str(result.channel) == "H1:TEST"


def test_resample_time_bin_closed_left_preserves_current_public_behavior():
    ts = TimeSeries(
        np.array([1.0, 3.0, 5.0, 7.0]),
        dt=1.0 * u.s,
        t0=0.0 * u.s,
        unit=u.m,
    )

    result = ts.resample("2s", closed="left")

    np.testing.assert_allclose(result.value, [2.0, 6.0])


@pytest.mark.parametrize(
    ("label", "expected_t0"),
    [("left", 0.0), ("right", 2.0), ("center", 1.0)],
)
def test_resample_time_bin_label_changes_coordinate_not_closed_right_membership(
    label, expected_t0
):
    ts = TimeSeries(
        np.array([1.0, 3.0, 5.0, 7.0]),
        dt=1.0 * u.s,
        t0=0.0 * u.s,
        unit=u.m,
    )

    result = ts.resample("2s", closed="right", label=label)

    np.testing.assert_allclose(result.value, [3.0, 7.0])
    assert result.t0.to_value(u.s) == pytest.approx(expected_t0)


def test_resample_time_bin_closed_right_excludes_samples_before_ceil_grid_start():
    ts = TimeSeries(
        np.array([10.0, 20.0, 30.0]),
        dt=1.0 * u.s,
        t0=1.0 * u.s,
        unit=u.m,
    )

    result = ts.resample("2s", closed="right", origin="gps0", align="ceil")

    # The 1 s sample is before the generated first edge at 2 s.
    np.testing.assert_allclose(result.value, [25.0])
    assert result.t0.to_value(u.s) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "offset",
    [1.0 * u.m, np.nan * u.s, np.inf * u.s, np.array([1.0, 2.0]) * u.s],
)
def test_resample_time_bin_rejects_invalid_offsets(offset):
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    with pytest.raises(ValueError):
        ts.resample("2s", offset=offset)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("closed", "invalid"),
        ("label", "invalid"),
        ("origin", "invalid"),
        ("align", "invalid"),
        ("nan_policy", "invalid"),
    ],
)
def test_resample_time_bin_rejects_invalid_enum_values(keyword, value):
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    with pytest.raises(ValueError):
        ts.resample("2s", **{keyword: value})


def test_resample_time_bin_rejects_invalid_nan_policy_before_ignore_nan_alias():
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    with pytest.raises(ValueError):
        ts.resample("2s", nan_policy="invalid", ignore_nan=True)


@pytest.mark.parametrize(
    "rule",
    ["0s", "-1s", np.nan * u.s, np.inf * u.s, -np.inf * u.s],
)
def test_resample_time_bin_rejects_nonpositive_or_nonfinite_width(rule):
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    with pytest.raises(ValueError):
        ts.resample(rule)


@pytest.mark.parametrize("rule", [np.array([1.0, 2.0]) * u.s, 1j * u.s])
def test_resample_time_bin_rejects_nonreal_or_nonscalar_width(rule):
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    with pytest.raises(ValueError):
        ts.resample(rule)


def test_resample_time_bin_accepts_positive_convertible_width():
    ts = _make_ts([1.0, 3.0, 5.0, 7.0], dt=1.0)

    result = ts.resample(2000 * u.ms)

    np.testing.assert_allclose(result.value, [2.0, 6.0])


def test_resample_time_bin_accepts_dimensionless_string_rule():
    ts = TimeSeries(
        np.array([1.0, 3.0, 5.0, 7.0]),
        times=np.arange(4) * u.dimensionless_unscaled,
        unit=u.m,
    )

    result = ts.resample("2")

    np.testing.assert_allclose(result.value, [2.0, 6.0])


# ---------------------------------------------------------------------------
# _resample_time_bin — aggregation variants
# ---------------------------------------------------------------------------


def test_resample_time_bin_mean():
    data = np.array([1.0, 3.0, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s")
    assert isinstance(result, TimeSeries)
    # Bins: [0,2) → mean(1,3)=2, [2,4) → mean(5,7)=6
    np.testing.assert_allclose(result.value, [2.0, 6.0])


def test_resample_time_bin_sum():
    data = np.array([1.0, 3.0, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", agg="sum")
    assert isinstance(result, TimeSeries)
    np.testing.assert_allclose(result.value, [4.0, 12.0])


def test_resample_time_bin_median():
    pytest.importorskip("scipy")
    data = np.array([1.0, 3.0, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", agg="median")
    assert isinstance(result, TimeSeries)
    np.testing.assert_allclose(result.value, [2.0, 6.0])


def test_resample_time_bin_callable_agg():
    data = np.array([1.0, 3.0, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", agg=np.sum)
    assert isinstance(result, TimeSeries)
    np.testing.assert_allclose(result.value, [4.0, 12.0])


def test_resample_time_bin_unknown_agg_raises():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    with pytest.raises(ValueError, match="Unknown aggregation"):
        ts._resample_time_bin("2s", agg="bad_agg")


def test_resample_time_bin_nan_policy_omit():
    data = np.array([1.0, np.nan, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", agg="mean", nan_policy="omit")
    # Bin 0: mean(1.0) = 1.0 (nan omitted); Bin 1: mean(5,7) = 6.0
    assert np.isfinite(result.value[0])
    assert result.value[0] == pytest.approx(1.0)


def test_resample_time_bin_nan_policy_propagate():
    data = np.array([1.0, np.nan, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", agg="mean", nan_policy="propagate")
    # Bin 0 contains NaN → NaN propagates
    assert np.isnan(result.value[0])


def test_resample_time_bin_ignore_nan_flag():
    data = np.array([1.0, np.nan, 5.0, 7.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", ignore_nan=True)
    assert np.isfinite(result.value[0])


def test_resample_time_bin_label_right():
    data = np.array([1.0, 2.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", label="right")
    assert isinstance(result, TimeSeries)
    assert result.t0.to("s").value == pytest.approx(2.0)


def test_resample_time_bin_label_center():
    data = np.array([1.0, 2.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", label="center")
    assert isinstance(result, TimeSeries)
    assert result.t0.to("s").value == pytest.approx(1.0)


def test_resample_time_bin_origin_gps0():
    data = np.arange(4, dtype=float)
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", origin="gps0")
    assert isinstance(result, TimeSeries)


def test_resample_time_bin_align_ceil():
    data = np.arange(4, dtype=float)
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("2s", align="ceil")
    assert isinstance(result, TimeSeries)


def test_resample_time_bin_min_count():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    ts = TimeSeries(data, dt=1.0 * u.s, t0=0.0 * u.s, unit=u.m)
    result = ts._resample_time_bin("4s", agg="mean", min_count=5)
    # Only 4 samples in the bin, min_count=5 → NaN
    assert np.isnan(result.value[0])
