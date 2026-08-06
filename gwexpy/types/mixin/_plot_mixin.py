"""PlotMixin — centralised deferred import for plot()."""

from __future__ import annotations

from typing import Any


class PlotMixin:
    """Mixin providing ``plot()`` via a single deferred import of Plot.

    Classes that mix this in no longer need their own
    ``from gwexpy.plot import Plot`` at call-time.
    """

    def plot(self, **kwargs: Any) -> Any:
        """Plot this object using :class:`gwexpy.plot.Plot`."""
        from gwexpy.interop._registry import ConverterRegistry

        Plot = ConverterRegistry.get_constructor("Plot")
        return Plot(self, **kwargs)
