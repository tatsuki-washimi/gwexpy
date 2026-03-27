"""Tests for gwexpy/timeseries/utils.py"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries.utils import (
    _coerce_t0_gps,
    _extract_axis_info,
    _extract_freq_axis_info,
    _validate_common_axis,
    _validate_common_epoch,
    _validate_common_frequency_axis,
)


# ---------------------------------------------------------------------------
# _coerce_t0_gps
# ---------------------------------------------------------------------------

def test_coerce_t0_gps_none():
    assert _coerce_t0_gps(None) is None


def test_coerce_t0_gps_quantity_time():
    q = 10.0 * u.s
    result = _coerce_t0_gps(q)
    assert result.to(u.s).value == pytest.approx(10.0)


def test_coerce_t0_gps_quantity_dimensionless():
    q = u.Quantity(5.0)  # dimensionless
    result = _coerce_t0_gps(q)
    assert result.unit == u.s
    assert result.value == pytest.approx(5.0)


def test_coerce_t0_gps_quantity_other_unit():
    # Non-time, non-dimensionless Quantity — returned as-is
    q = 3.0 * u.m
    result = _coerce_t0_gps(q)
    assert result is q


def test_coerce_t0_gps_int():
    result = _coerce_t0_gps(42)
    assert result.unit == u.s
    assert result.value == pytest.approx(42.0)


def test_coerce_t0_gps_float():
    result = _coerce_t0_gps(1234567890.0)
    assert result.unit == u.s
    assert result.value == pytest.approx(1234567890.0)


def test_coerce_t0_gps_invalid_raises():
    with pytest.raises(ValueError, match="Could not convert"):
        _coerce_t0_gps(object())


# ---------------------------------------------------------------------------
# _extract_axis_info
# ---------------------------------------------------------------------------

def _fake_ts(times, dt=None, t0=None):
    """Build a minimal TimeSeries-like namespace."""
    ns = SimpleNamespace(times=times)
    if dt is not None:
        ns.dt = dt
    if t0 is not None:
        ns.t0 = t0
    return ns


def test_extract_axis_info_no_times_raises():
    ns = SimpleNamespace(times=None)
    with pytest.raises(ValueError, match="times axis is required"):
        _extract_axis_info(ns)


def test_extract_axis_info_regular():
    times = np.arange(10) * 0.1 * u.s
    ts = _fake_ts(times, dt=0.1 * u.s, t0=0.0 * u.s)
    info = _extract_axis_info(ts)
    assert info["regular"] is True
    assert info["n"] == 10


def test_extract_axis_info_no_dt_attribute():
    """Object without 'dt' attribute — regular=False."""
    times = np.arange(5) * u.s
    ts = SimpleNamespace(times=times)  # no dt attribute
    info = _extract_axis_info(ts)
    assert info["regular"] is False
    assert info["dt"] is None


def test_extract_axis_info_non_quantity_dt():
    """dt given as plain float, not Quantity."""
    times = np.arange(10) * 0.1 * u.s
    ts = _fake_ts(times, dt=0.1, t0=0.0 * u.s)
    info = _extract_axis_info(ts)
    assert info["regular"] is True


def test_extract_axis_info_zero_dt():
    """dt=0 → irregular."""
    times = np.arange(5) * u.s
    ts = _fake_ts(times, dt=0.0 * u.s, t0=0.0 * u.s)
    info = _extract_axis_info(ts)
    assert info["regular"] is False


def test_extract_axis_info_nan_dt():
    """dt=NaN → irregular."""
    times = np.arange(5) * u.s
    ts = _fake_ts(times, dt=float("nan") * u.s, t0=0.0 * u.s)
    info = _extract_axis_info(ts)
    assert info["regular"] is False


def test_extract_axis_info_no_t0_attribute():
    """Has dt but no t0 attribute — falls back to axis[0]."""
    times = np.array([0.0, 0.1, 0.2]) * u.s
    ts = SimpleNamespace(times=times, dt=0.1 * u.s)  # no t0 attribute
    info = _extract_axis_info(ts)
    assert info["t0"] is not None


def test_extract_axis_info_t0_none_makes_irregular():
    """t0=None on object means regular=False."""
    times = np.arange(5) * 0.1 * u.s

    class FakeTS:
        dt = 0.1 * u.s
        t0 = None  # explicitly None

    FakeTS.times = times
    info = _extract_axis_info(FakeTS())
    assert info["regular"] is False


# ---------------------------------------------------------------------------
# _validate_common_axis
# ---------------------------------------------------------------------------

def _make_axis_info(n, dt=0.1 * u.s, t0=0.0 * u.s):
    times = np.arange(n) * dt
    return {
        "regular": True,
        "dt": dt,
        "t0": t0,
        "n": n,
        "times": times,
    }


def test_validate_common_axis_empty():
    result, n = _validate_common_axis([], "test")
    assert result is None
    assert n == 0


def test_validate_common_axis_single():
    info = _make_axis_info(10)
    times, n = _validate_common_axis([info], "test")
    assert n == 10


def test_validate_common_axis_consistent():
    info1 = _make_axis_info(10)
    info2 = _make_axis_info(10)
    times, n = _validate_common_axis([info1, info2], "test")
    assert n == 10


def test_validate_common_axis_length_mismatch_raises():
    info1 = _make_axis_info(10)
    info2 = _make_axis_info(20)
    with pytest.raises(ValueError, match="length"):
        _validate_common_axis([info1, info2], "test")


def test_validate_common_axis_dt_mismatch_raises():
    info1 = _make_axis_info(10, dt=0.1 * u.s)
    info2 = _make_axis_info(10, dt=0.2 * u.s)
    with pytest.raises(ValueError, match="dt"):
        _validate_common_axis([info1, info2], "test")


def test_validate_common_axis_t0_mismatch_raises():
    info1 = _make_axis_info(10, t0=0.0 * u.s)
    info2 = _make_axis_info(10, t0=1.0 * u.s)
    with pytest.raises(ValueError, match="t0"):
        _validate_common_axis([info1, info2], "test")


def test_validate_common_axis_irregular_identical():
    """Irregular axes (regular=False) with identical times arrays."""
    times = np.array([0.0, 0.1, 0.3]) * u.s
    info1 = {"regular": False, "dt": None, "t0": None, "n": 3, "times": times}
    info2 = {"regular": False, "dt": None, "t0": None, "n": 3, "times": times}
    result, n = _validate_common_axis([info1, info2], "test")
    assert n == 3


def test_validate_common_axis_irregular_unit_mismatch_raises():
    """Irregular axes with different units."""
    times1 = np.array([0.0, 0.1]) * u.s
    times2 = np.array([0.0, 0.1]) * u.ms
    info1 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times1}
    info2 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times2}
    with pytest.raises(ValueError, match="unit"):
        _validate_common_axis([info1, info2], "test")


def test_validate_common_axis_irregular_values_mismatch_raises():
    """Irregular axes with same unit but different values."""
    times1 = np.array([0.0, 0.1]) * u.s
    times2 = np.array([0.0, 0.2]) * u.s
    info1 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times1}
    info2 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times2}
    with pytest.raises(ValueError, match="identical"):
        _validate_common_axis([info1, info2], "test")


def test_validate_common_axis_irregular_no_unit():
    """Irregular axes as plain numpy arrays (no unit)."""
    times = np.array([0.0, 0.1, 0.3])
    info1 = {"regular": False, "dt": None, "t0": None, "n": 3, "times": times}
    info2 = {"regular": False, "dt": None, "t0": None, "n": 3, "times": times}
    result, n = _validate_common_axis([info1, info2], "test")
    assert n == 3


def test_validate_common_axis_irregular_no_unit_mismatch_raises():
    """Irregular plain arrays that differ."""
    times1 = np.array([0.0, 0.1])
    times2 = np.array([0.0, 0.2])
    info1 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times1}
    info2 = {"regular": False, "dt": None, "t0": None, "n": 2, "times": times2}
    with pytest.raises(ValueError, match="identical"):
        _validate_common_axis([info1, info2], "test")


# ---------------------------------------------------------------------------
# _extract_freq_axis_info
# ---------------------------------------------------------------------------

def _fake_fs(freqs, df=None, f0=None):
    ns = SimpleNamespace(frequencies=freqs)
    if df is not None:
        ns.df = df
    if f0 is not None:
        ns.f0 = f0
    return ns


def test_extract_freq_axis_info_no_freqs_raises():
    ns = SimpleNamespace(frequencies=None)
    with pytest.raises(ValueError, match="frequencies axis is required"):
        _extract_freq_axis_info(ns)


def test_extract_freq_axis_info_regular():
    freqs = np.arange(10) * 1.0 * u.Hz
    fs = _fake_fs(freqs, df=1.0 * u.Hz, f0=0.0 * u.Hz)
    info = _extract_freq_axis_info(fs)
    assert info["regular"] is True
    assert info["n"] == 10


def test_extract_freq_axis_info_no_df_attribute():
    freqs = np.arange(5) * u.Hz
    ns = SimpleNamespace(frequencies=freqs)  # no df attribute
    info = _extract_freq_axis_info(ns)
    assert info["regular"] is False
    assert info["df"] is None


def test_extract_freq_axis_info_non_quantity_df():
    freqs = np.arange(10) * 1.0 * u.Hz
    fs = _fake_fs(freqs, df=1.0, f0=0.0)
    info = _extract_freq_axis_info(fs)
    assert info["regular"] is True


def test_extract_freq_axis_info_zero_df():
    freqs = np.arange(5) * u.Hz
    fs = _fake_fs(freqs, df=0.0 * u.Hz, f0=0.0 * u.Hz)
    info = _extract_freq_axis_info(fs)
    assert info["regular"] is False


def test_extract_freq_axis_info_no_f0_attribute():
    """Has df but no f0 attribute — falls back to freqs[0]."""
    freqs = np.array([0.0, 1.0, 2.0]) * u.Hz
    ns = SimpleNamespace(frequencies=freqs, df=1.0 * u.Hz)  # no f0
    info = _extract_freq_axis_info(ns)
    assert info["f0"] is not None


def test_extract_freq_axis_info_f0_none_makes_irregular():
    """f0=None means irregular."""
    freqs = np.arange(5) * 1.0 * u.Hz

    class FakeFS:
        df = 1.0 * u.Hz
        f0 = None

    FakeFS.frequencies = freqs
    info = _extract_freq_axis_info(FakeFS())
    assert info["regular"] is False


# ---------------------------------------------------------------------------
# _validate_common_frequency_axis
# ---------------------------------------------------------------------------

def _make_freq_info(n, df=1.0 * u.Hz, f0=0.0 * u.Hz):
    freqs = np.arange(n) * 1.0 * u.Hz
    return {"regular": True, "df": df, "f0": f0, "n": n, "freqs": freqs}


def test_validate_common_frequency_axis_empty():
    result = _validate_common_frequency_axis([], "test")
    assert result == (None, None, None, 0)


def test_validate_common_frequency_axis_single():
    info = _make_freq_info(10)
    freqs, df, f0, n = _validate_common_frequency_axis([info], "test")
    assert n == 10


def test_validate_common_frequency_axis_consistent():
    info1 = _make_freq_info(10)
    info2 = _make_freq_info(10)
    _, _, _, n = _validate_common_frequency_axis([info1, info2], "test")
    assert n == 10


def test_validate_common_frequency_axis_length_mismatch_raises():
    info1 = _make_freq_info(10)
    info2 = _make_freq_info(20)
    with pytest.raises(ValueError, match="length"):
        _validate_common_frequency_axis([info1, info2], "test")


def test_validate_common_frequency_axis_df_mismatch_raises():
    info1 = _make_freq_info(10, df=1.0 * u.Hz)
    info2 = _make_freq_info(10, df=2.0 * u.Hz)
    with pytest.raises(ValueError, match="df"):
        _validate_common_frequency_axis([info1, info2], "test")


def test_validate_common_frequency_axis_f0_mismatch_raises():
    info1 = _make_freq_info(10, f0=0.0 * u.Hz)
    info2 = _make_freq_info(10, f0=1.0 * u.Hz)
    with pytest.raises(ValueError, match="f0"):
        _validate_common_frequency_axis([info1, info2], "test")


def test_validate_common_frequency_axis_irregular_identical():
    freqs = np.array([0.0, 1.0, 3.0]) * u.Hz
    info1 = {"regular": False, "df": None, "f0": None, "n": 3, "freqs": freqs}
    info2 = {"regular": False, "df": None, "f0": None, "n": 3, "freqs": freqs}
    result_freqs, _, _, n = _validate_common_frequency_axis([info1, info2], "test")
    assert n == 3


def test_validate_common_frequency_axis_irregular_unit_mismatch_raises():
    freqs1 = np.array([0.0, 1.0]) * u.Hz
    freqs2 = np.array([0.0, 1.0]) * u.kHz
    info1 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs1}
    info2 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs2}
    with pytest.raises(ValueError, match="unit"):
        _validate_common_frequency_axis([info1, info2], "test")


def test_validate_common_frequency_axis_irregular_values_mismatch_raises():
    freqs1 = np.array([0.0, 1.0]) * u.Hz
    freqs2 = np.array([0.0, 2.0]) * u.Hz
    info1 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs1}
    info2 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs2}
    with pytest.raises(ValueError, match="identical"):
        _validate_common_frequency_axis([info1, info2], "test")


def test_validate_common_frequency_axis_irregular_no_unit():
    freqs = np.array([0.0, 1.0, 3.0])
    info1 = {"regular": False, "df": None, "f0": None, "n": 3, "freqs": freqs}
    info2 = {"regular": False, "df": None, "f0": None, "n": 3, "freqs": freqs}
    _, _, _, n = _validate_common_frequency_axis([info1, info2], "test")
    assert n == 3


def test_validate_common_frequency_axis_irregular_no_unit_mismatch_raises():
    freqs1 = np.array([0.0, 1.0])
    freqs2 = np.array([0.0, 2.0])
    info1 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs1}
    info2 = {"regular": False, "df": None, "f0": None, "n": 2, "freqs": freqs2}
    with pytest.raises(ValueError, match="identical"):
        _validate_common_frequency_axis([info1, info2], "test")


# ---------------------------------------------------------------------------
# _validate_common_epoch
# ---------------------------------------------------------------------------

def test_validate_common_epoch_empty():
    assert _validate_common_epoch([], "test") is None


def test_validate_common_epoch_single():
    assert _validate_common_epoch([42.0], "test") == 42.0


def test_validate_common_epoch_identical():
    assert _validate_common_epoch([1.0, 1.0, 1.0], "test") == 1.0


def test_validate_common_epoch_mismatch_raises():
    with pytest.raises(ValueError, match="epoch"):
        _validate_common_epoch([1.0, 2.0], "test")


def test_validate_common_epoch_none_values():
    assert _validate_common_epoch([None, None], "test") is None
