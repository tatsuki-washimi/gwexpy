
import os
import sqlite3
import tempfile

import numpy as np
import pytest
from gwpy.time import LIGOTimeGPS

from gwexpy.timeseries import TimeSeries


def test_sqlite_interop():
    t0 = LIGOTimeGPS(1234567890, 0)
    dt = 0.01
    data = np.arange(100, dtype=float)
    ts = TimeSeries(data, t0=t0, dt=dt, name="sql_ts", unit="m")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = sqlite3.connect(db_path)

        # 1. to_sqlite
        ts.to_sqlite(conn, series_id="my_series")

        # 2. from_sqlite
        ts_restored = TimeSeries.from_sqlite(conn, "my_series")

        assert ts_restored.name == "my_series"
        assert ts_restored.unit.to_string() == "m"
        assert np.isclose(ts_restored.t0.value, float(t0))
        np.testing.assert_allclose(ts_restored.dt.value, dt)
        np.testing.assert_array_equal(ts_restored.value, data)

        # Test overwrite
        ts2 = TimeSeries(data * 2, t0=t0, dt=dt, name="sql_ts2", unit="m")
        with pytest.raises(ValueError):
             ts2.to_sqlite(conn, series_id="my_series", overwrite=False)

        ts2.to_sqlite(conn, series_id="my_series", overwrite=True)
        ts_restored2 = TimeSeries.from_sqlite(conn, "my_series")
        np.testing.assert_array_equal(ts_restored2.value, data * 2)

        conn.close()

try:
    import h5py  # noqa: F401 - availability check
except ImportError:
    h5py = None

@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_hdf5_interop():
    t0 = LIGOTimeGPS(100, 0)
    dt = 1.0
    data = np.ones(10)
    ts = TimeSeries(data, t0=t0, dt=dt, name="h5_ts", unit="V")

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test.h5")
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("data")

            # 1. to_hdf5
            ts.to_hdf5_dataset(grp, "ts1")

            # 2. from_hdf5
            ts_restored = TimeSeries.from_hdf5_dataset(grp, "ts1")

            assert ts_restored.name == "h5_ts"
            assert np.isclose(ts_restored.t0.value, float(t0))
            np.testing.assert_array_equal(ts_restored.value, data)

try:
    import obspy
except ImportError:
    obspy = None

@pytest.mark.skipif(obspy is None, reason="obspy not installed")
def test_obspy_interop():
    t0 = LIGOTimeGPS(1234567890, 0)
    dt = 0.01
    data = np.random.randn(50)
    ts = TimeSeries(data, t0=t0, dt=dt, name="obs_ts", channel="H1:STRAIN")

    # 1. to_obspy
    tr = ts.to_obspy()
    assert isinstance(tr, obspy.Trace)
    assert tr.stats.sampling_rate == 100.0
    # Obspy starttime is UTCDateTime

    # 2. from_obspy
    ts_restored = TimeSeries.from_obspy(tr)
    # Precision check
    assert np.isclose(ts_restored.t0.value, float(t0), atol=1e-6)
    np.testing.assert_allclose(ts_restored.dt.value, dt)
