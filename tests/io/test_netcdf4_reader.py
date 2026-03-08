from datetime import UTC, datetime

import numpy as np

from gwpy.time import to_gps

from gwexpy.timeseries.io.netcdf4_ import _infer_time_axis


def test_infer_time_axis_numeric_since_units():
    vals = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    t0, dt = _infer_time_axis(vals, {"units": "seconds since 2024-01-01 00:00:00"})

    assert np.isclose(t0, float(to_gps(datetime(2024, 1, 1, tzinfo=UTC))))
    assert np.isclose(dt, 0.5)


def test_infer_time_axis_numeric_absolute_fallback():
    vals = np.array([1234.0, 1234.25, 1234.5], dtype=np.float64)
    t0, dt = _infer_time_axis(vals, {})

    assert np.isclose(t0, 1234.0)
    assert np.isclose(dt, 0.25)
