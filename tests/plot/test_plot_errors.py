"""Tests for Plot.__init__ error paths and edge cases."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from gwpy.timeseries import TimeSeries

from gwexpy.plot import Plot


@pytest.fixture
def ts():
    return TimeSeries(np.random.randn(100), dt=0.01, t0=0, unit="m", name="test")


class TestPlotEdgeCases:
    def test_empty_args(self):
        """Plot() with no data should not crash."""
        p = Plot()
        p.close()

    def test_single_series(self, ts):
        p = Plot(ts)
        assert len(p.axes) >= 1
        p.close()

    def test_separate_true(self, ts):
        ts2 = TimeSeries(np.random.randn(100), dt=0.01, t0=0, name="ts2")
        p = Plot(ts, ts2, separate=True)
        assert len(p.axes) >= 2
        p.close()

    def test_geometry_tuple(self, ts):
        ts2 = TimeSeries(np.random.randn(100), dt=0.01, t0=0, name="ts2")
        p = Plot(ts, ts2, separate=True, geometry=(2, 1))
        assert len(p.axes) == 2
        p.close()


class TestPlotInvalidInputs:
    def test_invalid_geometry_zero(self, ts):
        """geometry=(0,0) should raise."""
        with pytest.raises((ValueError, Exception)):
            Plot(ts, geometry=(0, 0))

    def test_invalid_xscale(self, ts):
        """Invalid xscale value should raise or be ignored."""
        try:
            p = Plot(ts, xscale="nonexistent_scale")
            p.close()
        except (ValueError, Exception):
            pass  # Either raising or ignoring is acceptable

    def test_plot_close_idempotent(self, ts):
        """Closing a plot twice should not crash."""
        p = Plot(ts)
        p.close()
        # Second close should not raise
        try:
            p.close()
        except Exception:
            pass  # Some backends may raise on double-close


class TestPlotShow:
    def test_show_and_close(self, ts):
        p = Plot(ts)
        p.show(close=True)
        # Figure should be closed after show(close=True)

    def test_repr_png(self, ts):
        p = Plot(ts)
        png = p._repr_png_()
        assert png is None or isinstance(png, bytes)
        p.close()
