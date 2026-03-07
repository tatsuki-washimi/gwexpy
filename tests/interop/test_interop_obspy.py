"""Tests for ObsPy interop adapter."""

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from gwexpy.interop.obspy_ import from_obspy, to_obspy
from gwexpy.timeseries import TimeSeries


def _make_ts(n=100, t0=1000000000.0):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=t0, dt=0.01, unit="m", name="test",
    )


class TestToObspy:
    def test_ts_to_trace(self):
        ts = _make_ts()
        tr = to_obspy(ts)
        assert isinstance(tr, obspy.Trace)
        assert len(tr.data) == 100
        np.testing.assert_allclose(tr.data, ts.value)

    def test_sampling_rate_preserved(self):
        ts = _make_ts()
        tr = to_obspy(ts)
        assert np.isclose(tr.stats.sampling_rate, ts.sample_rate.value)


class TestFromObspy:
    def test_trace_to_ts(self):
        tr = obspy.Trace(data=np.ones(50))
        tr.stats.delta = 0.01
        tr.stats.starttime = obspy.UTCDateTime(0)

        ts = from_obspy(TimeSeries, tr, unit="m")
        assert len(ts) == 50
        np.testing.assert_array_equal(ts.value, np.ones(50))

    def test_roundtrip(self):
        ts = _make_ts()
        tr = to_obspy(ts)
        ts2 = from_obspy(TimeSeries, tr, unit="m")
        np.testing.assert_allclose(ts2.value, ts.value)
        assert np.isclose(ts2.sample_rate.value, ts.sample_rate.value)

    def test_dict_to_stream(self):
        from gwexpy.timeseries import TimeSeriesDict

        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=1000000000.0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=1000000000.0, dt=0.01, name="ch2"),
        })
        st = to_obspy(tsd)
        assert isinstance(st, obspy.Stream)
        assert len(st) == 2
