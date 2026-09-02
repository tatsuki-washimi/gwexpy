"""GWpy constructor compatibility for :class:`gwexpy.plot.SkyMap`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from gwpy.plot import Plot as GwpyPlot
from gwpy.timeseries import TimeSeries

from gwexpy.plot import SkyMap


def test_skymap_accepts_parent_timeseries_constructor_surface():
    """Data-bearing calls must not receive the empty-sky default projection."""
    series = TimeSeries(
        np.arange(4.0),
        t0=1_000_000_000,
        dt=0.25,
        unit="V",
        name="signal",
    )
    expected = GwpyPlot(series)
    actual = SkyMap(series)
    try:
        assert len(actual.axes) == len(expected.axes) == 1
        assert actual.axes[0].name == expected.axes[0].name
        assert len(actual.axes[0].lines) == len(expected.axes[0].lines) == 1
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_xdata(),
            expected.axes[0].lines[0].get_xdata(),
        )
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_ydata(),
            expected.axes[0].lines[0].get_ydata(),
        )
    finally:
        plt.close(actual)
        plt.close(expected)
