"""Tests for gwexpy/interop/polars_.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


def _fake_pl():
    """Return a minimal fake polars module."""
    class FakeSeries:
        def __init__(self, *, name="", values=None):
            self.name = name
            self._values = np.asarray(values) if values is not None else np.array([])

        def to_numpy(self):
            return self._values

    class FakeDataFrame:
        def __init__(self, data=None):
            self._data = data or {}
            self.columns = list(self._data.keys())

        def __getitem__(self, key):
            vals = self._data[key]
            col = _FakeColumn(vals)
            return col

    class _FakeColumn:
        def __init__(self, values):
            self._values = list(values) if not isinstance(values, np.ndarray) else values

        def __len__(self):
            return len(self._values)

        def __getitem__(self, idx):
            return self._values[idx]

        @property
        def columns(self):
            return []

        def to_numpy(self):
            return np.asarray(self._values)

        def to_list(self):
            return list(self._values)

    return SimpleNamespace(
        Series=FakeSeries,
        DataFrame=FakeDataFrame,
    )


def _make_ts(n=8, t0=0.0, dt=1.0, unit="m", name="ch"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit, name=name)


class TestToPolarsSeriesRaisesWithoutPolars:
    def test_raises_import_error(self):
        with patch.dict(sys.modules, {"polars": None}):
            from gwexpy.interop.polars_ import to_polars_series
            with pytest.raises(ImportError):
                to_polars_series(_make_ts())


class TestToPolarsSeriesWithMock:
    def test_basic_conversion(self):
        pl = _fake_pl()
        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            ts = _make_ts(n=5, name="test")
            s = polars_mod.to_polars_series(ts)
        assert s.name == "test"
        np.testing.assert_array_equal(s.to_numpy(), ts.value)

    def test_custom_name(self):
        pl = _fake_pl()
        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            ts = _make_ts(n=3)
            s = polars_mod.to_polars_series(ts, name="custom")
        assert s.name == "custom"


class TestFromPolarsSeriesNoImport:
    def test_basic_conversion(self):
        from gwexpy.interop.polars_ import from_polars_series
        fake_series = SimpleNamespace(to_numpy=lambda: np.arange(5.0))
        ts = from_polars_series(TimeSeries, fake_series, unit="m", t0=0.0, dt=1.0)
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 5

    def test_t0_and_dt_passed(self):
        from gwexpy.interop.polars_ import from_polars_series
        fake_series = SimpleNamespace(to_numpy=lambda: np.ones(4))
        ts = from_polars_series(TimeSeries, fake_series, t0=100.0, dt=0.5)
        assert ts.t0.value == pytest.approx(100.0)
        assert ts.dt.value == pytest.approx(0.5)

    def test_unit_preserved(self):
        from gwexpy.interop.polars_ import from_polars_series
        fake_series = SimpleNamespace(to_numpy=lambda: np.ones(3))
        ts = from_polars_series(TimeSeries, fake_series, unit="km")
        assert str(ts.unit) == "km"


class TestToPolarsFrequencyseriesRaisesWithoutPolars:
    def test_raises_import_error(self):
        from gwexpy.frequencyseries import FrequencySeries
        fs = FrequencySeries(np.ones(5), df=1.0)
        with patch.dict(sys.modules, {"polars": None}):
            from gwexpy.interop.polars_ import to_polars_frequencyseries
            with pytest.raises(ImportError):
                to_polars_frequencyseries(fs)


class TestToPolarsDataframeRaisesWithoutPolars:
    def test_raises_import_error(self):
        with patch.dict(sys.modules, {"polars": None}):
            from gwexpy.interop.polars_ import to_polars_dataframe
            with pytest.raises(ImportError):
                to_polars_dataframe(_make_ts())


class TestFromPolarsDataframeRaisesWithoutPolars:
    def test_raises_import_error(self):
        fake_df = SimpleNamespace(columns=["time", "ch"], **{"__getitem__": lambda s, k: None})
        with patch.dict(sys.modules, {"polars": None}):
            from gwexpy.interop.polars_ import from_polars_dataframe
            with pytest.raises(ImportError):
                from_polars_dataframe(TimeSeries, fake_df)


class TestFromPolarsDataframeWithMock:
    def test_basic_regular_grid(self):
        pl = _fake_pl()

        class MockDF:
            columns = ["time", "ch"]

            def __getitem__(self, key):
                if key == "time":
                    return _FakePolarsList([0.0, 1.0, 2.0, 3.0])
                return _FakePolarsList([1.0, 2.0, 3.0, 4.0])

        class _FakePolarsList:
            def __init__(self, vals):
                self._vals = list(vals)

            def __len__(self):
                return len(self._vals)

            def __getitem__(self, idx):
                return self._vals[idx]

            def to_numpy(self):
                return np.array(self._vals)

            def to_list(self):
                return self._vals

        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            ts = polars_mod.from_polars_dataframe(TimeSeries, MockDF())
        assert isinstance(ts, TimeSeries)
        assert len(ts) == 4

    def test_no_data_column_raises(self):
        pl = _fake_pl()

        class MockDF:
            columns = ["time"]

            def __getitem__(self, key):
                return []

        with patch.dict(sys.modules, {"polars": pl}):
            import importlib
            import gwexpy.interop.polars_ as polars_mod
            importlib.reload(polars_mod)
            with pytest.raises(ValueError, match="data column"):
                polars_mod.from_polars_dataframe(TimeSeries, MockDF())
