"""Edge-case tests for I/O readers: empty data, special floats, multi-channel."""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.frequencyseries import FrequencySeries


class TestCsvEdgeCases:
    def test_multi_sample_roundtrip(self, tmp_path):
        ts = TimeSeries(np.arange(10, dtype=float), dt=0.5, t0=100, unit="strain", name="multi")
        path = tmp_path / "multi.csv"
        ts.write(str(path), format="csv")
        ts2 = TimeSeries.read(str(path), format="csv")
        np.testing.assert_allclose(ts2.value, ts.value)

    def test_frequencyseries_roundtrip(self, tmp_path):
        fs = FrequencySeries(
            np.array([1.0, 2.0, 3.0, 4.0]),
            f0=0, df=10, unit="1/Hz", name="psd_test",
        )
        path = tmp_path / "fs.csv"
        fs.write(str(path), format="csv")
        fs2 = FrequencySeries.read(str(path), format="csv")
        np.testing.assert_allclose(fs2.value, fs.value)

    def test_large_values(self, tmp_path):
        ts = TimeSeries([1e30, -1e30, 1e-30], dt=1.0, t0=0, name="large")
        path = tmp_path / "large.csv"
        ts.write(str(path), format="csv")
        ts2 = TimeSeries.read(str(path), format="csv")
        np.testing.assert_allclose(ts2.value, ts.value, rtol=1e-6)


class TestHdf5EdgeCases:
    def test_empty_timeseries(self, tmp_path):
        ts = TimeSeries([], dt=1.0, t0=0, name="empty")
        path = tmp_path / "empty.hdf5"
        ts.write(str(path), format="hdf5")
        ts2 = TimeSeries.read(str(path), format="hdf5")
        assert len(ts2) == 0

    def test_dict_roundtrip_channel_names(self, tmp_path):
        d = TimeSeriesDict({
            "X1:CHANNEL_A": TimeSeries(np.ones(10), dt=1, t0=0, name="X1:CHANNEL_A"),
            "X1:CHANNEL_B": TimeSeries(np.zeros(10), dt=1, t0=0, name="X1:CHANNEL_B"),
        })
        path = tmp_path / "dict.hdf5"
        d.write(str(path), format="hdf5")
        d2 = TimeSeriesDict.read(str(path), format="hdf5")
        assert set(d2.keys()) == {"X1:CHANNEL_A", "X1:CHANNEL_B"}
        np.testing.assert_array_equal(d2["X1:CHANNEL_A"].value, np.ones(10))


class TestReadErrors:
    def test_nonexistent_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            TimeSeries.read(str(tmp_path / "does_not_exist.csv"), format="csv")
