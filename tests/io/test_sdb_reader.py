"""Tests for SDB (SQLite weather DB) reader."""

import sqlite3

import numpy as np
import pytest

from gwexpy.timeseries.io.sdb import read_timeseries_sdb, read_timeseriesdict_sdb


def _create_weather_db(path, n_records=10, start_unix=1700000000, interval=300):
    """Create a minimal weather SQLite database for testing."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE archive ("
        "  dateTime INTEGER,"
        "  outTemp REAL,"
        "  outHumidity REAL,"
        "  barometer REAL"
        ")"
    )
    for i in range(n_records):
        conn.execute(
            "INSERT INTO archive VALUES (?, ?, ?, ?)",
            (start_unix + i * interval, 70.0 + i, 50.0 + i * 0.5, 29.92),
        )
    conn.commit()
    conn.close()


class TestSdbReader:
    def test_basic_read(self, tmp_path):
        db_path = tmp_path / "test.sdb"
        _create_weather_db(db_path)

        tsd = read_timeseriesdict_sdb(str(db_path))
        assert len(tsd) > 0
        assert "outTemp" in tsd
        assert "outHumidity" in tsd
        assert "barometer" in tsd

    def test_unit_conversion_temperature(self, tmp_path):
        db_path = tmp_path / "temp.sdb"
        _create_weather_db(db_path, n_records=1)

        tsd = read_timeseriesdict_sdb(str(db_path))
        # 70F = 21.111C
        np.testing.assert_allclose(tsd["outTemp"].value[0], (70.0 - 32) / 1.8, rtol=1e-6)

    def test_unit_conversion_pressure(self, tmp_path):
        db_path = tmp_path / "pres.sdb"
        _create_weather_db(db_path, n_records=1)

        tsd = read_timeseriesdict_sdb(str(db_path))
        # 29.92 inHg * 33.8639 = ~1013.25 hPa
        np.testing.assert_allclose(
            tsd["barometer"].value[0], 29.92 * 33.8639, rtol=1e-4
        )

    def test_sample_rate(self, tmp_path):
        db_path = tmp_path / "rate.sdb"
        interval = 300  # 5 minutes
        _create_weather_db(db_path, interval=interval)

        tsd = read_timeseriesdict_sdb(str(db_path))
        ts = tsd["outTemp"]
        assert np.isclose(ts.sample_rate.value, 1.0 / interval)

    def test_empty_table(self, tmp_path):
        db_path = tmp_path / "empty.sdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE archive (dateTime INTEGER, outTemp REAL)"
        )
        conn.commit()
        conn.close()

        tsd = read_timeseriesdict_sdb(str(db_path))
        assert len(tsd) == 0

    def test_single_timeseries(self, tmp_path):
        db_path = tmp_path / "single.sdb"
        _create_weather_db(db_path)

        ts = read_timeseries_sdb(str(db_path))
        assert len(ts) == 10

    def test_column_selection(self, tmp_path):
        db_path = tmp_path / "cols.sdb"
        _create_weather_db(db_path)

        tsd = read_timeseriesdict_sdb(str(db_path), columns=["outTemp"])
        assert "outTemp" in tsd
        assert "barometer" not in tsd
