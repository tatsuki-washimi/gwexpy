import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.preprocess import align_timeseries_collection


def test_align_intersection_empty_semi_open():
    ts1 = TimeSeries([1, 2, 3], t0=0, dt=1)
    ts2 = TimeSeries([4, 5], t0=3, dt=1)

    with pytest.raises(ValueError, match="No overlap"):
        align_timeseries_collection([ts1, ts2], how="intersection")


def test_align_intersection_end_exclusive_len():
    ts1 = TimeSeries([1, 2, 3, 4, 5], t0=0, dt=1)
    ts2 = TimeSeries([10, 20, 30], t0=1, dt=1)

    _, times, _ = align_timeseries_collection([ts1, ts2], how="intersection")

    np.testing.assert_array_equal(times.value, [1, 2, 3])
    assert len(times) == 3


def test_align_intersection_range():
    ts1 = TimeSeries([1, 2, 3, 4], t0=0, dt=1)
    ts2 = TimeSeries([10, 20, 30], t0=2, dt=1)

    mat, times, _ = align_timeseries_collection([ts1, ts2], how="intersection")

    np.testing.assert_array_equal(times.value, [2, 3])
    np.testing.assert_array_equal(mat[:, 0], [3, 4])
    np.testing.assert_array_equal(mat[:, 1], [10, 20])


def test_align_union_range_fill_value():
    ts1 = TimeSeries([1, 2, 3], t0=0, dt=1)
    ts2 = TimeSeries([10, 20, 30], t0=2, dt=1)

    mat, times, _ = align_timeseries_collection([ts1, ts2], how="union", fill_value=-1)

    np.testing.assert_array_equal(times.value, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(mat[:, 0], [1, 2, 3, -1, -1])
    np.testing.assert_array_equal(mat[:, 1], [-1, -1, 10, 20, 30])


def test_align_dt_selects_max():
    dt_fast = (1 / 1024) * u.s
    dt_slow = (1 / 256) * u.s
    ts_fast = TimeSeries([1, 2, 3, 4], t0=0 * u.s, dt=dt_fast)
    ts_slow = TimeSeries([10, 20, 30], t0=0 * u.s, dt=dt_slow)

    _, times, meta = align_timeseries_collection([ts_fast, ts_slow], how="union")

    assert meta["dt"].to(u.s).value == pytest.approx(dt_slow.to(u.s).value)
    diffs = np.diff(times.to(u.s).value)
    assert np.allclose(diffs, dt_slow.to(u.s).value)


def test_align_mixed_units_warn_and_normalize():
    ts_ms = TimeSeries([1, 2, 3], t0=0 * u.ms, dt=1 * u.ms)
    ts_s = TimeSeries([4, 5], t0=0 * u.s, dt=0.002 * u.s)

    with pytest.warns(UserWarning, match="Converting time units"):
        _, times, meta = align_timeseries_collection([ts_ms, ts_s], how="union")

    assert times.unit == u.s
    assert meta["dt"].unit == u.s


def test_align_dimensionless_defaults_to_time():
    ts1 = TimeSeries([1, 2, 3], t0=0, dt=1)
    ts2 = TimeSeries([4, 5, 6], t0=1, dt=1)

    _, times, meta = align_timeseries_collection([ts1, ts2], how="intersection")

    assert times.unit == u.s
    assert meta["dt"].unit.physical_type == "time"


def test_align_asfreq_offset_floor_origin():
    ts1 = TimeSeries([1, 2, 3], t0=0.5, dt=0.5)
    ts2 = TimeSeries([10, 20], t0=1.0, dt=0.5)

    mat, times, _ = align_timeseries_collection([ts1, ts2], how="intersection")

    np.testing.assert_array_equal(times.value, [1.0, 1.5])
    np.testing.assert_array_equal(mat[:, 0], [2, 3])
    np.testing.assert_array_equal(mat[:, 1], [10, 20])
