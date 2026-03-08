"""Tests for MNE interop adapter."""

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from gwexpy.interop.mne_ import from_mne_raw, to_mne_rawarray
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _make_ts(n=100, name="test"):
    return TimeSeries(
        np.random.default_rng(42).standard_normal(n),
        t0=0, dt=0.01, name=name,
    )


class TestToMneRawArray:
    def test_single_ts(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        assert isinstance(raw, mne.io.RawArray)
        data = raw.get_data()
        assert data.shape == (1, 100)
        np.testing.assert_allclose(data[0], ts.value)

    def test_sampling_rate(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        assert np.isclose(raw.info["sfreq"], 100.0)

    def test_multi_channel(self):
        tsd = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.01, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.01, name="ch2"),
        })
        raw = to_mne_rawarray(tsd)
        assert raw.info["nchan"] == 2
        assert set(raw.ch_names) == {"ch1", "ch2"}

    def test_channel_names_preserved(self):
        tsd = TimeSeriesDict({
            "X1": TimeSeries(np.ones(20), t0=0, dt=0.1, name="X1"),
            "Y2": TimeSeries(np.ones(20), t0=0, dt=0.1, name="Y2"),
        })
        raw = to_mne_rawarray(tsd)
        assert "X1" in raw.ch_names
        assert "Y2" in raw.ch_names


class TestFromMneRaw:
    def test_roundtrip(self):
        tsd = TimeSeriesDict({
            "ch0": TimeSeries(np.arange(30, dtype=float), t0=0, dt=0.01, name="ch0"),
        })
        raw = to_mne_rawarray(tsd)
        tsd2 = from_mne_raw(TimeSeriesDict, raw)
        assert "ch0" in tsd2
        np.testing.assert_allclose(tsd2["ch0"].value, tsd["ch0"].value)

    def test_sfreq_preserved(self):
        ts = _make_ts()
        raw = to_mne_rawarray(ts)
        tsd = from_mne_raw(TimeSeriesDict, raw)
        key = next(iter(tsd))
        assert np.isclose(tsd[key].sample_rate.value, 100.0)
