import numpy as np
import pytest
from astropy import units as u

from gwexpy.interop._time import LeapSecondConversionError, gps_to_datetime_utc
from gwexpy.timeseries import TimeSeries

# --- P0-1 Tests (TimeSeries Imputation) ---


def test_impute_max_gap_regular():
    # Regular series: 0, 1, NaN, NaN, NaN, 5, 6 ... dt=1
    data = np.array([0, 1, np.nan, np.nan, np.nan, 5, 6], dtype=float)
    ts = TimeSeries(data, dt=1 * u.s, t0=0 * u.s)

    # Gap between t=1 and t=5 is 4 seconds.

    # 1. No max_gap -> fills
    imputed = ts.impute(method="interpolate")
    assert not np.isnan(imputed.value).any()
    assert np.isclose(imputed.value[2], 2.0)

    # 2. max_gap=5s (larger than gap 4s) -> fills
    imputed_large = ts.impute(method="interpolate", max_gap=5 * u.s)
    assert not np.isnan(imputed_large.value).any()

    # 3. max_gap=3s (Actual gap is 4s) -> NaN
    imputed_small = ts.impute(method="interpolate", max_gap=3 * u.s)
    # Mid-points should be NaN
    assert np.isnan(imputed_small.value[2:5]).all()
    # End-points preserved
    assert imputed_small.value[1] == 1.0
    assert imputed_small.value[5] == 5.0


def test_impute_max_gap_irregular():
    # Irregular
    times = np.array([0, 1, 2, 3, 4, 5])
    val = np.array([0, 1, np.nan, np.nan, np.nan, 5], dtype=float)
    ts = TimeSeries(val, times=times * u.s)

    # 1. max_gap=2s (Gap 4s) -> NaN
    imputed = ts.impute(method="interpolate", max_gap=2 * u.s)
    assert np.isnan(imputed.value[2:5]).all()

    # 2. max_gap=5s -> Fill
    imputed2 = ts.impute(method="interpolate", max_gap=5 * u.s)
    assert not np.isnan(imputed2.value).any()


def test_impute_max_gap_edges_safe():
    # Requirement: "Endpoint missing data (start/end) -> Ensure safe side (no interpolation) even when max_gap is specified"
    data = np.array([np.nan, np.nan, 1, 2, np.nan], dtype=float)
    ts = TimeSeries(data, dt=1 * u.s)

    # Without max_gap, interp fills edges (extrapolates flat usually)
    imp_default = ts.impute(method="interpolate")
    assert not np.isnan(imp_default.value).any()

    # With max_gap, we mandate NO extrapolation at edges
    imp_safe = ts.impute(method="interpolate", max_gap=10 * u.s)
    assert np.isnan(imp_safe.value[0])
    assert np.isnan(imp_safe.value[1])
    assert imp_safe.value[2] == 1.0
    assert np.isnan(imp_safe.value[4])


# --- P0-2 Tests (Leap Seconds) ---


def test_leap_seconds():
    # 2016-12-31 23:59:60 UTC = GPS 1167264017
    leap_gps = 1167264017

    # 1. Raise
    with pytest.raises(LeapSecondConversionError):
        gps_to_datetime_utc(leap_gps, leap="raise")

    # 2. Floor
    # Expect 2016-12-31 23:59:59.999999
    dt_floor = gps_to_datetime_utc(leap_gps, leap="floor")
    assert dt_floor.year == 2016
    assert dt_floor.month == 12
    assert dt_floor.day == 31
    assert dt_floor.hour == 23
    assert dt_floor.minute == 59
    assert dt_floor.second == 59
    assert dt_floor.microsecond == 999999

    # 3. Ceil
    # Expect 2017-01-01 00:00:00
    dt_ceil = gps_to_datetime_utc(leap_gps, leap="ceil")
    assert dt_ceil.year == 2017
    assert dt_ceil.month == 1
    assert dt_ceil.day == 1
    assert dt_ceil.second == 0


# --- P0-3 Tests (I/O Stubs) ---
