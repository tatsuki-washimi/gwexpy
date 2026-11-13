"""End-to-end error-bar units for ROOT graphs and histograms (#773)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.root_ import to_tgraph, to_th1d
from gwexpy.timeseries import TimeSeries

ROOT = pytest.importorskip("ROOT")


def _graph_errors(graph, n):
    return np.array([graph.GetErrorY(i) for i in range(n)])


def _data():
    return TimeSeries(np.array([1.0, 2.0, 3.0]), t0=0, dt=1, unit="m", name="x")


def test_tgraph_incompatible_quantity_error_raises():
    with pytest.raises(u.UnitConversionError):
        to_tgraph(_data(), error=np.full(3, 10.0) * u.s)


def test_tgraph_incompatible_series_error_raises():
    error = TimeSeries(np.full(3, 10.0), t0=0, dt=1, unit="s")

    with pytest.raises(u.UnitConversionError):
        to_tgraph(_data(), error=error)


@pytest.mark.parametrize(
    "error",
    [
        np.full(3, 50.0) * u.cm,
        TimeSeries(np.full(3, 50.0), t0=0, dt=1, unit="cm"),
        np.full(3, 0.5) * u.m,
        np.full(3, 0.5),
        [0.5, 0.5, 0.5],
    ],
    ids=["quantity-cm", "series-cm", "quantity-m", "ndarray", "list"],
)
def test_tgraph_error_bars_are_in_the_data_unit(error):
    graph = to_tgraph(_data(), error=error)

    assert isinstance(graph, ROOT.TGraphErrors)
    np.testing.assert_allclose(_graph_errors(graph, 3), 0.5, rtol=1e-12)
    np.testing.assert_allclose(
        [graph.GetPointY(i) for i in range(3)], [1.0, 2.0, 3.0], rtol=1e-12
    )


def test_th1d_incompatible_quantity_error_raises():
    spectrum = FrequencySeries(np.array([1.0, 2.0, 3.0]), f0=0, df=1, unit="m")

    with pytest.raises(u.UnitConversionError):
        to_th1d(spectrum, error=np.full(3, 10.0) * u.s)


def test_th1d_compatible_quantity_error_is_scaled():
    spectrum = FrequencySeries(np.array([1.0, 2.0, 3.0]), f0=0, df=1, unit="m")

    hist = to_th1d(spectrum, error=np.full(3, 50.0) * u.cm)

    np.testing.assert_allclose(
        [hist.GetBinError(i) for i in range(1, 4)], 0.5, rtol=1e-12
    )
