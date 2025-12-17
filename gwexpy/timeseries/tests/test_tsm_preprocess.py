import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gwexpy.timeseries import TimeSeriesMatrix
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwexpy.timeseries.timeseries import (
    _extract_axis_info,
    _validate_common_axis,
)


def _axis_array(axis):
    return axis.value if hasattr(axis, "value") else np.asarray(axis)


def _method_available(name):
    if not hasattr(GWpyTimeSeries, name):
        print(f"Skipping {name}: gwpy TimeSeries has no method '{name}'")
        return False
    return True


def _build_matrix():
    n_samples = 1024
    base = np.linspace(0, 1, n_samples)
    data = np.empty((2, 2, n_samples))
    for i in range(2):
        for j in range(2):
            data[i, j] = base + (i + j)

    units = np.array([["m", "m"], ["m", "m"]], dtype=object)
    names = np.array([["s00", "s01"], ["s10", "s11"]], dtype=object)
    channels = np.array([["C0", "C1"], ["C2", "C3"]], dtype=object)
    return TimeSeriesMatrix(
        data,
        dt=1 / 1024,
        units=units,
        names=names,
        channels=channels,
    )


def test_detrend_and_taper():
    if not _method_available("detrend"):
        return
    if not _method_available("taper"):
        return

    tsm = _build_matrix()
    detrended = tsm.detrend()
    assert isinstance(detrended, TimeSeriesMatrix)
    assert detrended.shape == tsm.shape
    assert np.array_equal(_axis_array(detrended.times), _axis_array(tsm.times))

    tapered = tsm.taper(nsamples=16)
    assert isinstance(tapered, TimeSeriesMatrix)
    assert tapered.shape == tsm.shape
    assert np.array_equal(_axis_array(tapered.times), _axis_array(tsm.times))


def test_resample_updates_axis():
    if not _method_available("resample"):
        return

    new_rate = 512
    tsm = _build_matrix()
    resampled = tsm.resample(new_rate)

    assert isinstance(resampled, TimeSeriesMatrix)
    assert resampled.shape[:2] == tsm.shape[:2]
    assert resampled.shape[-1] == len(resampled.times)

    res_times = _axis_array(resampled.times)
    for i in range(resampled.shape[0]):
        for j in range(resampled.shape[1]):
            assert np.array_equal(_axis_array(resampled[i, j].times), res_times)

    if hasattr(resampled.sample_rate, "to_value"):
        sr_val = resampled.sample_rate.to_value("Hz")
    else:
        sr_val = resampled.sample_rate
    assert np.isclose(sr_val, new_rate)
    assert np.isclose(resampled.dt.to_value("s"), 1 / new_rate)


def test_inplace_detrend_modifies():
    if not _method_available("detrend"):
        return

    tsm = _build_matrix()
    before = np.array(tsm.value, copy=True)
    result = tsm.detrend(inplace=True)

    assert result is tsm
    assert not np.allclose(before, tsm.value)
    assert np.array_equal(_axis_array(tsm.times), _axis_array(result.times))


def test_metadata_preserved():
    if not _method_available("taper"):
        return

    tsm = _build_matrix()
    tapered = tsm.taper(nsamples=16)

    assert tapered.meta[0, 0].unit == tsm.meta[0, 0].unit
    assert tapered.meta[0, 0].name == tsm.meta[0, 0].name
    assert str(tapered.meta[0, 0].channel) == str(tsm.meta[0, 0].channel)


def test_axis_mismatch_raises():
    ts1 = GWpyTimeSeries(np.zeros(4), sample_rate=4, t0=0)
    ts2 = GWpyTimeSeries(np.zeros(4), sample_rate=4, t0=1e-9)
    infos = [_extract_axis_info(ts1), _extract_axis_info(ts2)]
    try:
        _validate_common_axis(infos, "detrend")
    except ValueError as e:
        msg = str(e)
        assert "t0" in msg or "dt" in msg or "length" in msg or "times" in msg
    else:
        raise AssertionError("Expected ValueError for mismatched axes")


if __name__ == "__main__":
    test_detrend_and_taper()
    test_resample_updates_axis()
    test_inplace_detrend_modifies()
    test_metadata_preserved()
    print("ALL PREPROCESS TESTS PASSED")
