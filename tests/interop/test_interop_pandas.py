"""Tests for pandas interop adapter roundtrips."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from gwexpy.timeseries import TimeSeries


@pytest.fixture
def ts():
    return TimeSeries(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        dt=0.1,
        t0=1000000000,
        unit="m/s",
        name="test_channel",
    )


class TestToPandasSeries:
    def test_datetime_index(self, ts):
        s = ts.to_pandas(index="datetime")
        assert isinstance(s, pd.Series)
        assert len(s) == 5
        np.testing.assert_array_equal(s.values, ts.value)
        assert s.index.tz is not None  # Should be UTC-aware

    def test_gps_index(self, ts):
        s = ts.to_pandas(index="gps")
        assert s.index.name == "gps_time"
        np.testing.assert_allclose(s.index[0], float(ts.t0.value), rtol=1e-6)

    def test_seconds_index(self, ts):
        s = ts.to_pandas(index="seconds")
        assert s.index.name == "time_unix"
        assert len(s) == 5

    def test_invalid_index_raises(self, ts):
        with pytest.raises(ValueError, match="Unknown index type"):
            ts.to_pandas(index="invalid")

    def test_name_preserved(self, ts):
        s = ts.to_pandas()
        assert s.name == "test_channel"

    def test_values_preserved(self, ts):
        s = ts.to_pandas()
        np.testing.assert_array_equal(s.values, ts.value)


class TestFromPandasSeries:
    def test_roundtrip_gps(self, ts):
        s = ts.to_pandas(index="gps")
        ts2 = TimeSeries.from_pandas(s, unit="m/s")
        np.testing.assert_array_equal(ts2.value, ts.value)
        assert str(ts2.unit) == str(ts.unit)

    def test_roundtrip_datetime(self, ts):
        s = ts.to_pandas(index="datetime")
        ts2 = TimeSeries.from_pandas(s, unit="m/s")
        np.testing.assert_array_equal(ts2.value, ts.value)

    def test_roundtrip_preserves_length(self, ts):
        s = ts.to_pandas()
        ts2 = TimeSeries.from_pandas(s, unit="m/s")
        assert len(ts2) == len(ts)


class TestToPandasDataFrame:
    def test_dict_to_dataframe(self):
        from gwexpy.timeseries import TimeSeriesDict

        d = TimeSeriesDict({
            "ch1": TimeSeries(np.arange(5, dtype=float), dt=0.1, t0=0, name="ch1"),
            "ch2": TimeSeries(np.arange(5, dtype=float) * 2, dt=0.1, t0=0, name="ch2"),
        })
        df = d.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"ch1", "ch2"}
        assert len(df) == 5
