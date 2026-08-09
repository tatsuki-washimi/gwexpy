"""Public contract tests for SDB direct I/O."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError

from gwexpy.gui.loaders.loaders import load_products
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _create_weather_db(path, *, start_unix=1700000000, interval=300):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE archive ("
        "  dateTime INTEGER,"
        "  outTemp REAL,"
        "  outHumidity REAL,"
        "  barometer REAL"
        ")"
    )
    conn.executemany(
        "INSERT INTO archive VALUES (?, ?, ?, ?)",
        [
            (start_unix + 0 * interval, 70.0, 50.0, 29.92),
            (start_unix + 1 * interval, 71.0, 51.0, 29.92),
            (start_unix + 2 * interval, 72.0, 52.0, 29.92),
        ],
    )
    conn.commit()
    conn.close()


def test_sdb_public_read_entrypoints_work_for_canonical_format(tmp_path):
    path = tmp_path / "weather.sdb"
    _create_weather_db(path)

    tsd = TimeSeriesDict.read(path, format="sdb")
    assert sorted(tsd.keys()) == ["barometer", "outHumidity", "outTemp"]
    np.testing.assert_allclose(tsd["outTemp"].value[0], (70.0 - 32.0) / 1.8)

    ts = TimeSeries.read(path, format="sdb")
    assert ts.name == "outTemp"
    assert len(ts) == 3


@pytest.mark.parametrize("extension", (".sqlite", ".sqlite3"))
def test_removed_sdb_aliases_and_extensions_are_not_supported(tmp_path, extension):
    path = tmp_path / f"weather{extension}"
    _create_weather_db(path)

    for fmt in ("sqlite", "sqlite3"):
        with pytest.raises(IORegistryError):
            TimeSeriesDict.read(path, format=fmt)

    with pytest.raises(IORegistryError):
        TimeSeriesDict.read(path)

    with pytest.raises(RuntimeError, match="Unsupported file format"):
        load_products(str(path))


def test_sdb_matrix_reader_remains_available_as_implementation_surface(tmp_path):
    path = tmp_path / "weather.sdb"
    _create_weather_db(path)

    matrix = TimeSeriesMatrix.read(path, format="sdb")
    assert matrix.shape == (3, 1, 3)
