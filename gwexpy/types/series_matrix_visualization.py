from __future__ import annotations

from typing import Any

from gwexpy.types.mixin._plot_mixin import PlotMixin


class SeriesMatrixVisualizationMixin(PlotMixin):
    """Mixin for SeriesMatrix plotting and visualization."""

    def step(self, where: str = "post", **kwargs: Any) -> Any:
        """Plot the matrix as a step function."""
        kwargs.setdefault("drawstyle", f"steps-{where}")
        return self.plot(**kwargs)
