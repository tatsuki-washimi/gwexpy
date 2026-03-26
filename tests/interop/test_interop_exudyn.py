"""Tests for Exudyn sensor interoperability.

Uses temporary text files and numpy arrays to simulate Exudyn output.
Does NOT require Exudyn to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.interop.exudyn_ import from_exudyn_sensor

N_SAMPLES = 200
DT = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sensor_array(n_cols=3):
    """Simulate mbs.GetSensorStoredData() output: col0=time, rest=values."""
    rng = np.random.default_rng(5)
    time = np.arange(N_SAMPLES) * DT
    values = rng.random((N_SAMPLES, n_cols))
    return np.column_stack([time, values])


def _write_sensor_file(tmp_path, n_cols=3):
    """Write sensor data to a text file."""
    arr = _make_sensor_array(n_cols)
    fpath = tmp_path / "sensor.txt"
    np.savetxt(fpath, arr)
    return fpath


# ---------------------------------------------------------------------------
# from_exudyn_sensor — ndarray input
# ---------------------------------------------------------------------------


class TestFromExudynArray:
    def test_single_col_returns_ts(self):
        arr = _make_sensor_array(n_cols=1)
        result = from_exudyn_sensor(TimeSeries, arr)
        assert isinstance(result, TimeSeries)
        assert len(result) == N_SAMPLES

    def test_multi_col_returns_tsm(self):
        arr = _make_sensor_array(n_cols=3)
        result = from_exudyn_sensor(TimeSeriesMatrix, arr)
        assert isinstance(result, TimeSeriesMatrix)
        assert result.shape[0] == 3  # 3 channels

    def test_unit_displacement(self):
        arr = _make_sensor_array(n_cols=2)
        result = from_exudyn_sensor(
            TimeSeriesMatrix, arr, output_variable="Displacement"
        )
        assert result.units is not None

    def test_unit_velocity(self):
        arr = _make_sensor_array(n_cols=1)
        result = from_exudyn_sensor(
            TimeSeries, arr, output_variable="Velocity"
        )
        assert result.unit is not None  # TimeSeries has .unit

    def test_column_names(self):
        arr = _make_sensor_array(n_cols=2)
        names = ["ux", "uy"]
        result = from_exudyn_sensor(
            TimeSeriesMatrix, arr, column_names=names
        )
        assert isinstance(result, TimeSeriesMatrix)


# ---------------------------------------------------------------------------
# from_exudyn_sensor — file input
# ---------------------------------------------------------------------------


class TestFromExudynFile:
    def test_file_returns_tsm(self, tmp_path):
        fpath = _write_sensor_file(tmp_path, n_cols=3)
        result = from_exudyn_sensor(TimeSeriesMatrix, fpath)
        assert isinstance(result, TimeSeriesMatrix)
        assert result.shape[0] == 3

    def test_file_single_col(self, tmp_path):
        fpath = _write_sensor_file(tmp_path, n_cols=1)
        result = from_exudyn_sensor(TimeSeries, fpath)
        assert isinstance(result, TimeSeries)

    def test_dt_correct(self, tmp_path):
        fpath = _write_sensor_file(tmp_path, n_cols=1)
        result = from_exudyn_sensor(TimeSeries, fpath)
        dt_val = result.dt.value if hasattr(result.dt, "value") else float(result.dt)
        assert abs(dt_val - DT) < 1e-10
