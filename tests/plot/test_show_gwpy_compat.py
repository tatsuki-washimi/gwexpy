"""GWpy differential contracts for :meth:`gwexpy.plot.Plot.show`."""

from __future__ import annotations

import inspect

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from gwpy.plot import Plot as GwpyPlot

from gwexpy.plot import FieldPlot, Plot, SkyMap


def _managed_plot() -> Plot:
    """Create a Plot registered with pyplot so close state is observable."""
    return plt.figure(FigureClass=Plot)


@pytest.mark.parametrize("plot_type", [Plot, FieldPlot, SkyMap])
def test_show_preserves_gwpy_positional_surface(plot_type):
    """The GWexpy-only ``close`` control must not shift GWpy arguments."""
    gwpy = inspect.signature(GwpyPlot.show).parameters
    actual = inspect.signature(plot_type.show).parameters

    for name in ("self", "warn", "block"):
        assert actual[name].kind == gwpy[name].kind
        assert actual[name].default == gwpy[name].default
    assert list(actual)[:3] == ["self", "warn", "block"]
    assert actual["close"].kind is inspect.Parameter.KEYWORD_ONLY
    assert actual["close"].default is False


def test_show_default_keeps_figure_open_like_gwpy():
    plot = _managed_plot()
    number = plot.number
    try:
        plot.show(warn=False, block=False)
        assert plt.fignum_exists(number)
    finally:
        plt.close(plot)


def test_show_close_is_explicit_extension():
    plot = _managed_plot()
    number = plot.number
    plot.show(False, False, close=True)
    assert not plt.fignum_exists(number)


def test_show_rejects_legacy_positional_close_slot():
    plot = _managed_plot()
    try:
        with pytest.raises(TypeError):
            plot.show(False, False, True)
    finally:
        plt.close(plot)
