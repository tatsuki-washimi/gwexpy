"""
Extended tests for gwexpy/interop/polars_.py using mock Polars.
"""
from __future__ import annotations

import sys
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.frequencyseries import FrequencySeries


def _fake_pl():
    """Return a more capable fake polars module."""
    class FakeSeries:
        def __init__(self, values, name=None):
            self.name = name
            self._values = np.asarray(values)

        def to_numpy(self):
            return self._values

        def to_list(self):
            return list(self._values)

        def __len__(self):
            return len(self._values)

        def __getitem__(self, idx):
            return self._values[idx]

    class FakeDataFrame:
        def __init__(self, data=None):
            self._data = data or {}
            self.columns = list(self._data.keys())

        def __getitem__(self, key):
            if key not in self._data:
                raise KeyError(key)
            return FakeSeries(self._data[key], name=key)
        
        def __len__(self):
            if not self._data: return 0
            return len(next(iter(self._data.values())))

    return SimpleNamespace(
        Series=FakeSeries,
        DataFrame=FakeDataFrame,
    )


class TestToPolarsExtended:
    @pytest.fixture(autouse=True)
    def mock_polars(self):
        pl = _fake_pl()
        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            self.polars_mod = polars_mod
            yield

    def test_to_polars_dataframe_time_units(self):
        ts = TimeSeries([1, 2, 3], t0=1000000000 * u.s, dt=1 * u.s, name="val")
        
        # GPS
        df_gps = self.polars_mod.to_polars_dataframe(ts, time_unit="gps")
        assert_allclose = np.testing.assert_allclose
        assert_allclose(df_gps["time"].to_numpy(), [1000000000, 1000000001, 1000000002])
        
        # Unix
        df_unix = self.polars_mod.to_polars_dataframe(ts, time_unit="unix")
        # GPS 1000000000 is 2011-09-14 01:46:25 UTC -> Unix 1315964785
        assert df_unix["time"].to_numpy()[0] == pytest.approx(1315964785.0)

        # Invalid
        with pytest.raises(ValueError, match="Unknown time_unit"):
            self.polars_mod.to_polars_dataframe(ts, time_unit="invalid")

    def test_to_polars_frequencyseries_index_unit(self):
        fs = FrequencySeries([10, 20], frequencies=[1000, 2000] * u.Hz, name="psd")
        
        # Default Hz
        df_hz = self.polars_mod.to_polars_frequencyseries(fs, index_unit="Hz")
        np.testing.assert_array_equal(df_hz["frequency"].to_numpy(), [1000, 2000])
        
        # Convert to kHz
        df_khz = self.polars_mod.to_polars_frequencyseries(fs, index_unit="kHz")
        np.testing.assert_array_equal(df_khz["frequency"].to_numpy(), [1, 2])
        
        # Invalid unit
        with pytest.raises(u.UnitConversionError):
            self.polars_mod.to_polars_frequencyseries(fs, index_unit="m")

    def test_to_polars_dict(self):
        tsd = TimeSeriesDict()
        tsd["A"] = TimeSeries([1, 2], t0=0, dt=1)
        tsd["B"] = TimeSeries([3, 4], t0=0, dt=1)
        
        df = self.polars_mod.to_polars_dict(tsd, time_unit="gps")
        assert set(df.columns) == {"time", "A", "B"}
        np.testing.assert_array_equal(df["A"].to_numpy(), [1, 2])
        np.testing.assert_array_equal(df["B"].to_numpy(), [3, 4])


class TestFromPolarsExtended:
    @pytest.fixture(autouse=True)
    def mock_polars(self):
        pl = _fake_pl()
        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            self.polars_mod = polars_mod
            self.pl = pl
            yield

    def test_from_polars_dataframe_datetime(self):
        # Mocking a DF with datetime values
        # GPS 1000000000
        t0_dt = datetime.datetime(2011, 9, 14, 1, 46, 25, tzinfo=datetime.timezone.utc)
        t1_dt = t0_dt + datetime.timedelta(seconds=1)
        
        data = {"time": [t0_dt, t1_dt], "val": [10.0, 20.0]}
        df = self.pl.DataFrame(data)
        
        ts = self.polars_mod.from_polars_dataframe(TimeSeries, df)
        assert ts.t0.value == pytest.approx(1000000000.0)
        assert ts.dt.value == pytest.approx(1.0)

    def test_from_polars_dataframe_datetime64(self):
        # Need to fix np.datetime64 to be relative to Unix 1970
        # GPS 1000000000 is 2011-09-14 01:46:25 UTC
        t0_64 = np.datetime64("2011-09-14T01:46:25")
        t1_64 = t0_64 + np.timedelta64(1, "s")
        
        data = {"time": [t0_64, t1_64], "val": [10.0, 20.0]}
        df = self.pl.DataFrame(data)
        
        ts = self.polars_mod.from_polars_dataframe(TimeSeries, df)
        assert ts.t0.value == pytest.approx(1000000000.0)
        assert ts.dt.value == pytest.approx(1.0)

    def test_from_polars_dict(self):
        data = {"time": [0, 1], "A": [10, 20], "B": [30, 40]}
        df = self.pl.DataFrame(data)
        
        tsd = self.polars_mod.from_polars_dict(TimeSeriesDict, df, unit_map={"A": "m", "B": "s"})
        assert isinstance(tsd, TimeSeriesDict)
        assert tsd["A"].unit == u.m
        assert tsd["B"].unit == u.s
        assert tsd["A"].t0.value == 0
        assert tsd["A"].dt.value == 1
