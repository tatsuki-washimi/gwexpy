"""Tests for gwexpy/types/_stats.py — StatisticalMethodsMixin."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.timeseries import TimeSeries


def _make_sm(data=None, unit=None):
    if data is None:
        data = np.arange(12.0).reshape(2, 2, 3)
    return SeriesMatrix(data, xindex=np.arange(data.shape[-1]), unit=unit)


def _make_ts(data=None, unit=None):
    """Create a TimeSeries which has a real .unit attribute."""
    if data is None:
        data = np.arange(6.0)
    return TimeSeries(data, sample_rate=100, unit=unit)


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

def test_mean_default():
    sm = _make_sm()
    result = sm.mean()
    assert result == pytest.approx(5.5)


def test_mean_ignore_nan_false():
    sm = _make_sm()
    result = sm.mean(ignore_nan=False)
    assert result == pytest.approx(5.5)


def test_mean_with_nan_ignored():
    data = np.array([[[1.0, np.nan, 3.0]]])
    sm = _make_sm(data)
    result = sm.mean()
    assert result == pytest.approx(2.0)


def test_mean_with_unit():
    ts = _make_ts(unit=u.m)
    result = ts.mean()
    assert hasattr(result, "unit")


def test_mean_axis():
    sm = _make_sm()
    result = sm.mean(axis=0)
    assert result.shape == (2, 3)


def test_mean_keepdims():
    sm = _make_sm()
    result = sm.mean(axis=0, keepdims=True)
    assert result.shape == (1, 2, 3)


# ---------------------------------------------------------------------------
# std
# ---------------------------------------------------------------------------

def test_std_default():
    sm = _make_sm()
    assert sm.std() > 0


def test_std_ignore_nan_false():
    # Line 64-65 — func_raw branch
    sm = _make_sm()
    result = sm.std(ignore_nan=False)
    assert result > 0


def test_std_with_unit():
    ts = _make_ts(unit=u.s)
    result = ts.std()
    assert hasattr(result, "unit")


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------

def test_var_default():
    sm = _make_sm()
    assert sm.var() > 0


def test_var_ignore_nan_false():
    # Line 88-89 — func_raw branch
    sm = _make_sm()
    result = sm.var(ignore_nan=False)
    assert result > 0


def test_var_with_unit():
    ts = _make_ts(unit=u.kg)
    result = ts.var()
    assert hasattr(result, "unit")


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------

def test_min_default():
    sm = _make_sm()
    assert sm.min() == pytest.approx(0.0)


def test_min_ignore_nan_false():
    sm = _make_sm()
    result = sm.min(ignore_nan=False)
    assert result == pytest.approx(0.0)


def test_max_default():
    sm = _make_sm()
    assert sm.max() == pytest.approx(11.0)


def test_max_ignore_nan_false():
    sm = _make_sm()
    result = sm.max(ignore_nan=False)
    assert result == pytest.approx(11.0)


def test_min_with_unit():
    ts = _make_ts(unit=u.V)
    result = ts.min()
    assert hasattr(result, "unit")


def test_max_with_unit():
    ts = _make_ts(unit=u.V)
    result = ts.max()
    assert hasattr(result, "unit")


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------

def test_median_default():
    sm = _make_sm()
    assert sm.median() == pytest.approx(5.5)


def test_median_ignore_nan_false():
    # Lines 179-185 — overwrite_input / func_raw branch
    sm = _make_sm()
    result = sm.median(ignore_nan=False)
    assert result == pytest.approx(5.5)


def test_median_overwrite_input():
    sm = _make_sm()
    result = sm.median(overwrite_input=True)
    assert result == pytest.approx(5.5)


def test_median_with_unit():
    ts = _make_ts(unit=u.Hz)
    result = ts.median()
    assert hasattr(result, "unit")


def test_median_axis():
    sm = _make_sm()
    result = sm.median(axis=0)
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

def test_rms_basic():
    data = np.array([[[3.0, 4.0]]])
    sm = _make_sm(data)
    result = sm.rms()
    assert result == pytest.approx(np.sqrt((9 + 16) / 2))


def test_rms_with_unit():
    # Lines 205-213 — unit branch
    ts = _make_ts(np.array([1.0, 2.0, 3.0]), unit=u.m)
    result = ts.rms()
    assert hasattr(result, "unit")
    assert result.unit == u.m


def test_rms_ignore_nan_false():
    sm = _make_sm()
    result = sm.rms(ignore_nan=False)
    assert result > 0


def test_rms_with_nan_ignored():
    data = np.array([[[1.0, np.nan, 3.0]]])
    sm = _make_sm(data)
    result = sm.rms()
    # nanmean([1, 9]) = 5 → sqrt(5)
    assert result == pytest.approx(np.sqrt(5.0))


def test_rms_keepdims():
    sm = _make_sm()
    result = sm.rms(axis=0, keepdims=True)
    assert result.shape == (1, 2, 3)


# ---------------------------------------------------------------------------
# skewness / kurtosis
# ---------------------------------------------------------------------------

def test_skewness_basic():
    data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    sm = _make_sm(data)
    result = sm.skewness()
    assert result == pytest.approx(0.0, abs=1e-6)


def test_skewness_axis():
    data = np.random.default_rng(42).normal(size=(2, 2, 20))
    sm = _make_sm(data)
    result = sm.skewness(axis=0)
    assert result.shape == (2, 20)


def test_kurtosis_basic():
    data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    sm = _make_sm(data)
    result = sm.kurtosis()
    # Fisher kurtosis of uniform-like data
    assert isinstance(float(result), float)


def test_kurtosis_pearson():
    data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    sm = _make_sm(data)
    result = sm.kurtosis(fisher=False)
    # Pearson kurtosis = Fisher + 3
    fisher_val = sm.kurtosis(fisher=True)
    assert result == pytest.approx(fisher_val + 3.0, rel=1e-5)


def test_kurtosis_axis():
    data = np.random.default_rng(0).normal(size=(2, 2, 20))
    sm = _make_sm(data)
    result = sm.kurtosis(axis=0)
    assert result.shape == (2, 20)
