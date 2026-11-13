"""HDF5 round-trip of irregularly sampled ``TimeSeries``/``StateVector`` (#882)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
)

TIMES = np.array([1_400_000_000.0, 1_400_000_001.0, 1_400_000_003.0, 1_400_000_004.5])
VALUES = np.array([1.0, -2.0, 3.5, 4.25]) * 1e-21
BITS = ["locked", "science", "injection"]


def _ts(name="K1:STRAIN", dtype=np.float64):
    ts = TimeSeries(VALUES.astype(dtype), times=TIMES, unit="m", name=name)
    assert not ts.xindex.regular
    return ts


def _sv(name="K1:STATE"):
    sv = StateVector(
        np.array([0, 1, 5, 7], dtype=np.uint32), times=TIMES, bits=BITS, name=name
    )
    assert not sv.xindex.regular
    return sv


def _assert_ts_equal(back, original):
    assert type(back) is type(original)
    assert back.dtype == original.dtype
    np.testing.assert_array_equal(back.value, original.value)
    np.testing.assert_array_equal(back.times.value, original.times.value)
    assert back.times.unit == original.times.unit
    assert back.unit == original.unit
    assert back.name == original.name
    assert not back.xindex.regular
    # The reader's internal bookkeeping flag must not reach the caller.
    assert not [key for key in vars(back) if "irregular_hdf5" in key]


def test_irregular_csv_read_keeps_gwpy_failure(tmp_path):
    """Only native HDF5 skips GWpy's padding step; other formats are GWpy's."""
    target = tmp_path / "irregular.csv"
    target.write_text("0,10\n1,20\n2.5,30\n4,40\n", encoding="utf-8")

    with pytest.raises(AttributeError):
        TimeSeries.read(target, format="csv")
    back = TimeSeries.read(target, format="csv", gap="ignore")
    np.testing.assert_array_equal(back.times.value, [0.0, 1.0, 2.5, 4.0])


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128])
def test_irregular_timeseries_hdf5_roundtrip(tmp_path, dtype):
    ts = _ts(dtype=dtype)
    target = tmp_path / "irregular_ts.h5"

    ts.write(target, format="hdf5")

    _assert_ts_equal(TimeSeries.read(target, format="hdf5"), ts)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_nearly_regular_timeseries_hdf5_preserves_explicit_axis(tmp_path, dtype):
    ticks = np.arange(100)
    times = ticks * 1e-6 + np.where(ticks % 2, 1e-9, 0.0)
    values = np.linspace(-1.0, 1.0, ticks.size).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * np.linspace(1.0, -1.0, ticks.size)
    ts = TimeSeries(values, times=times, unit="m", name="K1:NEARLY_REGULAR")
    assert ts.xindex.regular  # GWpy's absolute tolerance misclassifies this axis.
    target = tmp_path / "nearly_regular_ts.h5"

    ts.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f[ts.name].attrs)
    assert "xindex" in attrs
    assert "dx" not in attrs
    assert attrs["xunit"] == "s"

    back = TimeSeries.read(target, format="hdf5")
    assert back.dtype == ts.dtype
    assert back.unit == ts.unit
    assert back.name == ts.name
    np.testing.assert_array_equal(back.times.value, times)
    np.testing.assert_array_equal(back.value, values)


def test_nearly_regular_timeseries_hdf5_preserves_caller_attrs(tmp_path):
    ticks = np.arange(100)
    times = ticks * 1e-6 + np.where(ticks % 2, 1e-9, 0.0)
    ts = TimeSeries(np.ones(ticks.size), times=times, unit="m", name="near")
    target = tmp_path / "nearly_regular_attrs.h5"

    ts.write(target, format="hdf5", attrs={"note": "caller"})
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["near"].attrs)
    assert attrs["note"] == "caller"
    assert "xindex" in attrs
    assert "dx" not in attrs
    np.testing.assert_array_equal(attrs["xindex"], times)


@pytest.mark.parametrize("start", [0.0, 2e-6])
def test_nearly_regular_timeseries_read_bounds_use_explicit_coordinates(
    tmp_path, start
):
    ticks = np.arange(100)
    times = ticks * 1e-6 + ticks**2 * 1e-9
    ts = TimeSeries(np.arange(ticks.size), times=times, name="near")
    assert ts.xindex.regular
    target = tmp_path / "bounded_nearly_regular.h5"
    ts.write(target, format="hdf5")

    end = 105e-6
    back = TimeSeries.read(target, format="hdf5", start=start, end=end, gap="raise")
    expected = (times >= start) & (times < end)

    np.testing.assert_array_equal(back.times.value, times[expected])
    np.testing.assert_array_equal(back.value, ts.value[expected])


@pytest.mark.parametrize("start", [0.0, 2e-6])
def test_nearly_regular_timeseriesdict_read_bounds_use_explicit_coordinates(
    tmp_path, start
):
    ticks = np.arange(100)
    times = ticks * 1e-6 + ticks**2 * 1e-9
    ts = TimeSeries(np.arange(ticks.size), times=times, name="near")
    target = tmp_path / "bounded_nearly_regular_dict.h5"
    TimeSeriesDict({"near": ts}).write(target, format="hdf5")

    back = TimeSeriesDict.read(
        target, format="hdf5", start=start, end=105e-6, gap="raise"
    )["near"]
    expected = (times >= start) & (times < 105e-6)

    np.testing.assert_array_equal(back.times.value, times[expected])
    np.testing.assert_array_equal(back.value, ts.value[expected])


def test_mixed_timeseriesdict_keeps_gwpy_coverage_for_regular_members(tmp_path):
    ticks = np.arange(100)
    irregular_times = ticks * 1e-6 + ticks**2 * 1e-9
    explicit = TimeSeries(np.arange(ticks.size), times=irregular_times, name="explicit")
    regular = TimeSeries(np.ones(ticks.size), t0=0.0, dt=1e-6, name="regular")
    target = tmp_path / "mixed_bounded.h5"
    TimeSeriesDict({"explicit": explicit, "regular": regular}).write(
        target, format="hdf5"
    )

    with pytest.raises(ValueError):
        TimeSeriesDict.read(target, format="hdf5", start=0.0, end=105e-6, gap="raise")


def test_nearly_regular_timeserieslist_read_bounds_use_explicit_coordinates(
    tmp_path,
):
    ticks = np.arange(100)
    times = ticks * 1e-6 + ticks**2 * 1e-9
    ts = TimeSeries(np.arange(ticks.size), times=times, name="near")
    target = tmp_path / "bounded_nearly_regular_list.h5"
    TimeSeriesList(ts).write(target, format="hdf5")

    back = TimeSeriesList.read(target, format="hdf5", start=2e-6, end=105e-6)[0]
    expected = (times >= 2e-6) & (times < 105e-6)

    np.testing.assert_array_equal(back.times.value, times[expected])
    np.testing.assert_array_equal(back.value, ts.value[expected])


def test_single_sample_explicit_timeseries_hdf5_roundtrip(tmp_path):
    ts = TimeSeries([2.0], times=[7.0], name="single")
    target = tmp_path / "single.h5"

    ts.write(target, format="hdf5")
    back = TimeSeries.read(target, format="hdf5", gap="ignore")

    np.testing.assert_array_equal(back.times.value, [7.0])
    np.testing.assert_array_equal(back.value, [2.0])


def test_irregular_timeseries_hdf5_layout_has_no_dx(tmp_path):
    ts = _ts()
    target = tmp_path / "irregular_ts.h5"

    ts.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f[ts.name].attrs)
    assert "dx" not in attrs
    assert attrs["xunit"] == "s"
    np.testing.assert_array_equal(attrs["xindex"], TIMES)


def test_regular_timeseries_hdf5_layout_unchanged(tmp_path):
    ts = TimeSeries(VALUES, t0=1_400_000_000, dt=0.5, unit="m", name="K1:REG")
    target = tmp_path / "regular_ts.h5"

    ts.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["K1:REG"].attrs)
        assert h5f.libver[0] == "earliest"
    assert attrs["x0"] == 1_400_000_000.0
    assert attrs["dx"] == 0.5
    assert attrs["xunit"] == "s"
    assert "xindex" not in attrs
    back = TimeSeries.read(target, format="hdf5")
    assert back.xindex.regular
    np.testing.assert_array_equal(back.times.value, ts.times.value)


def test_irregular_timeseries_hdf5_overwrite_and_append(tmp_path):
    target = tmp_path / "irregular_ts.h5"
    first = _ts(name="first")
    second = _ts(name="second")
    first.write(target, format="hdf5")

    with pytest.raises(OSError):
        first.write(target, format="hdf5")

    second.write(target, format="hdf5", append=True)
    _assert_ts_equal(TimeSeries.read(target, format="hdf5", path="first"), first)
    _assert_ts_equal(TimeSeries.read(target, format="hdf5", path="second"), second)

    first.write(target, format="hdf5", overwrite=True)
    _assert_ts_equal(TimeSeries.read(target, format="hdf5"), first)


def test_irregular_timeseries_hdf5_into_open_file(tmp_path):
    ts = _ts()
    target = tmp_path / "open.h5"

    with h5py.File(target, "w") as h5f:
        ts.write(h5f, format="hdf5")

    _assert_ts_equal(TimeSeries.read(target, format="hdf5"), ts)


def test_irregular_timeseries_hdf5_long_axis_roundtrip(tmp_path):
    n = 20000
    rng = np.random.default_rng(882)
    times = 1_400_000_000.0 + np.cumsum(rng.uniform(0.5, 1.5, n))
    ts = TimeSeries(np.arange(n, dtype=np.float64), times=times, name="long")
    assert not ts.xindex.regular
    target = tmp_path / "long.h5"

    ts.write(target, format="hdf5")
    back = TimeSeries.read(target, format="hdf5")

    np.testing.assert_array_equal(back.times.value, times)
    np.testing.assert_array_equal(back.value, ts.value)


@pytest.mark.parametrize("gap", ["raise", "pad"])
@pytest.mark.parametrize(
    "bounds",
    [{"start": TIMES[0] - 10}, {"end": TIMES[-1] + 10}],
    ids=["start-before-data", "end-after-data"],
)
def test_irregular_timeseries_hdf5_read_rejects_uncovered_interval(
    tmp_path, gap, bounds
):
    ts = _ts()
    target = tmp_path / "irregular_ts.h5"
    ts.write(target, format="hdf5")

    with pytest.raises(ValueError, match="does not cover requested interval"):
        TimeSeries.read(target, format="hdf5", gap=gap, **bounds)


def test_irregular_timeseries_hdf5_read_pad_explains_no_padding(tmp_path):
    ts = _ts()
    target = tmp_path / "irregular_ts.h5"
    ts.write(target, format="hdf5")

    with pytest.raises(ValueError, match="cannot be padded"):
        TimeSeries.read(target, format="hdf5", pad=0.0, end=TIMES[-1] + 10)


def test_irregular_timeseries_hdf5_read_ignore_skips_coverage_check(tmp_path):
    ts = _ts()
    target = tmp_path / "irregular_ts.h5"
    ts.write(target, format="hdf5")

    back = TimeSeries.read(target, format="hdf5", gap="ignore", end=TIMES[-1] + 10)

    _assert_ts_equal(back, ts)


def test_irregular_timeseries_hdf5_read_covered_interval(tmp_path):
    ts = _ts()
    target = tmp_path / "irregular_ts.h5"
    ts.write(target, format="hdf5")

    back = TimeSeries.read(target, format="hdf5", start=TIMES[1], end=TIMES[3])

    np.testing.assert_array_equal(back.times.value, TIMES[1:3])
    np.testing.assert_array_equal(back.value, ts.value[1:3])


def test_irregular_timeseriesdict_hdf5_read_rejects_uncovered_interval(tmp_path):
    tsd = TimeSeriesDict({"A": _ts(name="A"), "B": _ts(name="B")})
    target = tmp_path / "dict.h5"
    tsd.write(target, format="hdf5")

    with pytest.raises(ValueError, match="does not cover requested interval"):
        TimeSeriesDict.read(target, format="hdf5", start=TIMES[0] - 10)


def test_mixed_regular_and_irregular_timeseriesdict_hdf5_roundtrip(tmp_path):
    regular = TimeSeries(VALUES, t0=TIMES[0], dt=0.5, unit="m", name="R")
    tsd = TimeSeriesDict({"I": _ts(name="I"), "R": regular})
    target = tmp_path / "mixed.h5"

    tsd.write(target, format="hdf5")
    back = TimeSeriesDict.read(target, format="hdf5")

    assert list(back) == ["I", "R"]
    _assert_ts_equal(back["I"], tsd["I"])
    assert back["R"].xindex.regular
    np.testing.assert_array_equal(back["R"].times.value, regular.times.value)
    np.testing.assert_array_equal(back["R"].value, regular.value)


def test_irregular_statevector_hdf5_roundtrip(tmp_path):
    sv = _sv()
    target = tmp_path / "irregular_sv.h5"

    sv.write(target, format="hdf5")
    # ``StateVector`` is GWpy's class: its default read post-processing needs a
    # sample rate, so an irregular vector is read with ``gap="ignore"``.
    back = StateVector.read(target, format="hdf5", gap="ignore")

    assert back.dtype == sv.dtype
    np.testing.assert_array_equal(back.value, sv.value)
    np.testing.assert_array_equal(back.times.value, TIMES)
    assert back.times.unit == u.s
    assert back.name == sv.name
    assert list(back.bits) == list(sv.bits)
    assert not back.xindex.regular


def test_irregular_timeseriesdict_hdf5_roundtrip(tmp_path):
    tsd = TimeSeriesDict({"A": _ts(name="A"), "B": _ts(name="B")})
    target = tmp_path / "dict.h5"

    tsd.write(target, format="hdf5")
    back = TimeSeriesDict.read(target, format="hdf5")

    assert list(back) == ["A", "B"]
    for key in tsd:
        _assert_ts_equal(back[key], tsd[key])


def test_irregular_timeserieslist_hdf5_roundtrip(tmp_path):
    tsl = TimeSeriesList(_ts(name="A"), _ts(name="B"))
    target = tmp_path / "list.h5"

    tsl.write(target, format="hdf5")
    back = TimeSeriesList.read(target, format="hdf5")

    assert len(back) == 2
    for restored, original in zip(back, tsl):
        _assert_ts_equal(restored, original)


def test_irregular_statevectordict_hdf5_roundtrip(tmp_path):
    svd = StateVectorDict({"A": _sv(name="A"), "B": _sv(name="B")})
    target = tmp_path / "svdict.h5"

    svd.write(target, format="hdf5")
    back = StateVectorDict.read(target, format="hdf5", gap="ignore")

    assert list(back) == ["A", "B"]
    for key in svd:
        np.testing.assert_array_equal(back[key].value, svd[key].value)
        np.testing.assert_array_equal(back[key].times.value, TIMES)
        assert list(back[key].bits) == BITS
