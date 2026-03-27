"""Tests for gwexpy/timeseries/_core.py — TimeSeriesCore."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


def _make_ts(n=100, sample_rate=100.0, t0=0.0):
    data = np.sin(2 * np.pi * 5.0 * np.arange(n) / sample_rate)
    return TimeSeries(data, sample_rate=sample_rate, t0=t0)


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

class TestTail:
    def test_tail_default(self):
        ts = _make_ts()
        result = ts.tail()
        assert len(result) == 5

    def test_tail_n_int(self):
        ts = _make_ts()
        result = ts.tail(10)
        assert len(result) == 10

    def test_tail_n_none(self):
        # Line 52 — n=None returns self
        ts = _make_ts()
        result = ts.tail(None)
        assert result is ts

    def test_tail_n_zero(self):
        # Line 55 — n<=0 returns empty slice
        ts = _make_ts()
        result = ts.tail(0)
        assert len(result) == 0

    def test_tail_n_negative(self):
        ts = _make_ts()
        result = ts.tail(-5)
        assert len(result) == 0

    def test_tail_larger_than_ts(self):
        ts = _make_ts(n=10)
        result = ts.tail(100)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------

class TestCrop:
    def test_crop_basic(self):
        ts = _make_ts(n=100, sample_rate=100.0, t0=0.0)
        result = ts.crop(start=0.0, end=0.5)
        assert len(result) <= 50

    def test_crop_start_none(self):
        ts = _make_ts(n=100, sample_rate=100.0, t0=0.0)
        result = ts.crop(end=0.5)
        assert len(result) <= 100

    def test_crop_end_none(self):
        ts = _make_ts(n=100, sample_rate=100.0, t0=0.0)
        result = ts.crop(start=0.2)
        assert len(result) <= 100

    def test_crop_array_start(self):
        # Lines 70-72 — start is array → extract first element
        ts = _make_ts(n=100, sample_rate=100.0, t0=0.0)
        result = ts.crop(start=np.array([0.1, 0.2]))
        assert result is not None

    def test_crop_array_end(self):
        # Lines 74-77 — end is array → extract first element
        ts = _make_ts(n=100, sample_rate=100.0, t0=0.0)
        result = ts.crop(end=np.array([0.8, 0.9]))
        assert result is not None


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

class TestAppend:
    def test_append_inplace_true(self):
        # Line 94-95 — inplace=True returns self
        ts1 = _make_ts(n=50, t0=0.0)
        ts2 = _make_ts(n=50, t0=0.5)
        result = ts1.append(ts2, inplace=True)
        assert result is ts1

    def test_append_inplace_false(self):
        # Lines 96-104 — inplace=False
        ts1 = _make_ts(n=50, t0=0.0)
        ts2 = _make_ts(n=50, t0=0.5)
        result = ts1.append(ts2, inplace=False)
        assert result is not None


# ---------------------------------------------------------------------------
# find_peaks
# ---------------------------------------------------------------------------

class TestFindPeaks:
    def test_find_peaks_basic(self):
        # Lines 106-204 — basic peak finding
        ts = _make_ts(n=200)
        peaks, props = ts.find_peaks()
        assert isinstance(props, dict)
        assert len(peaks) > 0

    def test_find_peaks_no_peaks(self):
        # Lines 188-192 — no peaks found → empty result
        ts = TimeSeries(np.ones(100), sample_rate=100.0, t0=0.0)
        peaks, props = ts.find_peaks()
        assert len(peaks) == 0

    def test_find_peaks_with_height(self):
        # Lines 142 — height parameter
        ts = _make_ts(n=200)
        peaks, props = ts.find_peaks(height=0.5)
        assert isinstance(props, dict)

    def test_find_peaks_with_quantity_height(self):
        # Line 135-138 — height as Quantity with .value
        ts = TimeSeries(np.sin(2*np.pi*5*np.arange(200)/100.0),
                       sample_rate=100.0, t0=0.0, unit=u.V)
        peaks, props = ts.find_peaks(height=0.5 * u.V)
        assert isinstance(peaks, TimeSeries)

    def test_find_peaks_with_distance_quantity(self):
        # Lines 154-155 — distance as time Quantity
        ts = _make_ts(n=200)
        peaks, props = ts.find_peaks(distance=0.1 * u.s)
        assert isinstance(props, dict)

    def test_find_peaks_with_prominence(self):
        # Line 144 — prominence
        ts = _make_ts(n=200)
        peaks, props = ts.find_peaks(prominence=0.1)
        assert isinstance(props, dict)

    def test_find_peaks_width_list(self):
        # Lines 158-171 — width as iterable
        ts = _make_ts(n=400)
        peaks, props = ts.find_peaks(width=[0.05 * u.s, 0.2 * u.s])
        assert isinstance(props, dict)

    def test_find_peaks_width_single_quantity(self):
        # Lines 172-173 — width as single Quantity
        ts = _make_ts(n=400)
        peaks, props = ts.find_peaks(width=0.05 * u.s)
        assert isinstance(props, dict)

    def test_find_peaks_width_plain_float(self):
        # Lines 164, 168-170 — width items without .to
        ts = _make_ts(n=400)
        peaks, props = ts.find_peaks(width=[3, 20])  # samples
        assert isinstance(props, dict)

    def test_find_peaks_width_tuple(self):
        # Width as tuple (also wraps as tuple in output)
        ts = _make_ts(n=400)
        peaks, props = ts.find_peaks(width=(3, 20))
        assert isinstance(props, dict)

    def test_find_peaks_name_set(self):
        # Line 201 — peaks have name set
        ts = TimeSeries(np.sin(2*np.pi*5*np.arange(200)/100.0),
                       sample_rate=100.0, t0=0.0, name="myts")
        peaks, props = ts.find_peaks(height=0.5)
        if len(peaks) > 0:
            assert "peaks" in peaks.name

    def test_find_peaks_returns_timeseries(self):
        ts = _make_ts(n=200)
        peaks, props = ts.find_peaks()
        assert isinstance(peaks, TimeSeries)
