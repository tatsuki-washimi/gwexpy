"""Tests for gwexpy/interop/sqlite_.py."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from gwexpy.interop.sqlite_ import _ensure_schema, from_sqlite, to_sqlite
from gwexpy.timeseries import TimeSeries


def _make_ts(n=5, t0=0.0, dt=1.0, unit="m", name="ch1"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit, name=name)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


class TestEnsureSchema:
    def test_creates_series_table(self, conn):
        _ensure_schema(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t[0] for t in tables]
        assert "series" in names

    def test_creates_samples_table(self, conn):
        _ensure_schema(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [t[0] for t in tables]
        assert "samples" in names

    def test_idempotent(self, conn):
        _ensure_schema(conn)
        _ensure_schema(conn)  # should not raise


class TestToSqlite:
    def test_basic_write(self, conn):
        ts = _make_ts()
        sid = to_sqlite(ts, conn)
        assert sid == "ch1"

    def test_returns_series_id(self, conn):
        ts = _make_ts(name="test_ch")
        sid = to_sqlite(ts, conn)
        assert sid == "test_ch"

    def test_custom_series_id(self, conn):
        ts = _make_ts()
        sid = to_sqlite(ts, conn, series_id="custom_id")
        assert sid == "custom_id"

    def test_duplicate_raises(self, conn):
        ts = _make_ts()
        to_sqlite(ts, conn)
        with pytest.raises(ValueError, match="exists"):
            to_sqlite(ts, conn)

    def test_overwrite_replaces(self, conn):
        ts = _make_ts()
        to_sqlite(ts, conn)
        to_sqlite(ts, conn, overwrite=True)  # should not raise

    def test_no_name_defaults_to_unknown(self, conn):
        ts = TimeSeries(np.arange(3.0), t0=0, dt=1.0)
        sid = to_sqlite(ts, conn)
        assert sid == "unknown"

    def test_data_rows_inserted(self, conn):
        ts = _make_ts(n=4)
        to_sqlite(ts, conn)
        rows = conn.execute("SELECT * FROM samples WHERE series_id='ch1' ORDER BY i").fetchall()
        assert len(rows) == 4


class TestFromSqlite:
    def test_basic_roundtrip(self, conn):
        ts = _make_ts(n=5, t0=10.0, dt=0.5, unit="m", name="ch")
        to_sqlite(ts, conn)
        ts2 = from_sqlite(TimeSeries, conn, "ch")
        np.testing.assert_array_almost_equal(ts2.value, ts.value)

    def test_t0_restored(self, conn):
        ts = _make_ts(t0=100.0)
        to_sqlite(ts, conn)
        ts2 = from_sqlite(TimeSeries, conn, "ch1")
        assert ts2.t0.value == pytest.approx(100.0)

    def test_dt_restored(self, conn):
        ts = _make_ts(dt=0.25)
        to_sqlite(ts, conn)
        ts2 = from_sqlite(TimeSeries, conn, "ch1")
        assert ts2.dt.value == pytest.approx(0.25)

    def test_unit_restored(self, conn):
        ts = _make_ts(unit="s")
        to_sqlite(ts, conn)
        ts2 = from_sqlite(TimeSeries, conn, "ch1")
        assert str(ts2.unit) == "s"

    def test_missing_series_raises(self, conn):
        _ensure_schema(conn)
        with pytest.raises(KeyError, match="not found"):
            from_sqlite(TimeSeries, conn, "nonexistent")
