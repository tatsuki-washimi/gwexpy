"""Tests for ObsPy interop adapter."""

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from gwexpy.interop.obspy_ import from_obspy, to_obspy
from gwexpy.timeseries import TimeSeries


def _make_ts(n=100, t0=1000000000.0):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=t0,
        dt=0.01,
        unit="m",
        name="test",
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

        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=1000000000.0, dt=0.01, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=1000000000.0, dt=0.01, name="ch2"),
            }
        )
        st = to_obspy(tsd)
        assert isinstance(st, obspy.Stream)
        assert len(st) == 2


def _make_trace(data, *, channel, t0=0):
    tr = obspy.Trace(data=np.asarray(data, dtype=float))
    tr.stats.delta = 0.01
    tr.stats.starttime = obspy.UTCDateTime(t0)
    tr.stats.channel = channel
    return tr


class TestStreamToDict:
    def test_stream_to_timeseriesdict(self):
        from gwexpy.timeseries import TimeSeriesDict

        st = obspy.Stream(
            [
                _make_trace(np.ones(50), channel="HHZ"),
                _make_trace(np.zeros(50), channel="HHN"),
            ]
        )
        tsd = TimeSeriesDict.from_obspy(st)
        assert isinstance(tsd, TimeSeriesDict)
        assert len(tsd) == 2
        keys = list(tsd.keys())
        np.testing.assert_array_equal(tsd[keys[0]].value, np.ones(50))
        np.testing.assert_array_equal(tsd[keys[1]].value, np.zeros(50))

    def test_duplicate_keys_are_unique(self):
        from gwexpy.timeseries import TimeSeriesDict

        st = obspy.Stream(
            [
                _make_trace(np.ones(50), channel="HHZ"),
                _make_trace(np.full(50, 2.0), channel="HHZ"),
            ]
        )
        tsd = TimeSeriesDict.from_obspy(st)
        # Both traces preserved despite identical ids.
        assert len(tsd) == 2

    def test_roundtrip_dict(self):
        from gwexpy.timeseries import TimeSeriesDict

        tsd = TimeSeriesDict(
            {
                "ch1": TimeSeries(
                    np.arange(50.0), t0=1000000000.0, dt=0.01, name="ch1"
                ),
                "ch2": TimeSeries(np.ones(50), t0=1000000000.0, dt=0.01, name="ch2"),
            }
        )
        st = to_obspy(tsd)
        tsd2 = TimeSeriesDict.from_obspy(st)
        assert len(tsd2) == 2
        values = sorted((ts.value.tolist() for ts in tsd2.values()))
        expected = sorted((ts.value.tolist() for ts in tsd.values()))
        assert values == expected


class TestStreamToTimeSeries:
    def test_single_trace_stream(self):
        st = obspy.Stream([_make_trace(np.ones(50), channel="HHZ")])
        ts = TimeSeries.from_obspy(st, unit="m")
        assert isinstance(ts, TimeSeries)
        np.testing.assert_array_equal(ts.value, np.ones(50))

    def test_multi_trace_stream_raises(self):
        st = obspy.Stream(
            [
                _make_trace(np.ones(50), channel="HHZ"),
                _make_trace(np.zeros(50), channel="HHN"),
            ]
        )
        with pytest.raises(TypeError, match="TimeSeriesDict"):
            TimeSeries.from_obspy(st)


def test_spectrogram_integer_precision_survives_obspy_roundtrip():
    from gwexpy.spectrogram import Spectrogram

    values = np.array([[2**53 + 1, 2**53 + 3], [2**53 + 5, 2**53 + 7]], dtype=np.int64)
    source = Spectrogram(values, t0=1234567890, dt=1, frequencies=[10.0, 20.0])
    result = from_obspy(Spectrogram, to_obspy(source))
    assert result.dtype == values.dtype
    assert int(result.value[0, 0]) == 2**53 + 1
    np.testing.assert_array_equal(result.value, values)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_spectrogram_obspy_stream_preserves_dtype_and_is_independent(dtype):
    from gwexpy.spectrogram import Spectrogram

    values = np.arange(6).reshape(3, 2).astype(dtype)
    source = Spectrogram(values, t0=1234567890, dt=1, frequencies=[10.0, 20.0])
    stream = to_obspy(source)
    assert stream[0].data.dtype == values.dtype
    np.testing.assert_array_equal(stream[0].data, values[:, 0])
    stream[0].data[0] = 99
    assert source.value[0, 0] == 0
