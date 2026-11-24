"""HDF5 persistence of predictive uncertainty."""

import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.timeseries import TimeSeries


def test_forecast_uncertainty_hdf5_roundtrip(tmp_path):
    series = TimeSeries([1.0, 2.0, 3.0, 4.0], t0=100, dt=1, unit="V", name="signal")
    quantiles = {
        level: TimeSeries(
            np.array(series.value) + offset, t0=100, dt=1, unit="V", name="signal"
        )
        for level, offset in [(0.1, -1), (0.5, 0), (0.9, 1)]
    }
    series._set_uncertainty(
        quantiles=quantiles,
        center_kind="median",
        uncertainty_kind="predictive",
        interval_levels=(0.1, 0.9),
    )
    path = tmp_path / "forecast.h5"
    series.write(path, format="hdf5", path="data")
    native = GwpyTimeSeries.read(path, format="hdf5", path="data")
    np.testing.assert_array_equal(native.value, series.value)
    restored = TimeSeries.read(path, format="hdf5", path="data", start=101, end=103)
    np.testing.assert_array_equal(restored.value, [2.0, 3.0])
    assert restored.interval_levels == (0.1, 0.9)
    np.testing.assert_array_equal(restored.quantiles[0.1].value, [1.0, 2.0])
    np.testing.assert_array_equal(restored.quantiles[0.9].value, [3.0, 4.0])
