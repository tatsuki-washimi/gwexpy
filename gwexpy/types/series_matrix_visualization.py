from __future__ import annotations

from typing import Any


class SeriesMatrixVisualizationMixin:
    """Mixin for SeriesMatrix plotting and visualization."""

    def plot(self, **kwargs: Any) -> Any:
        """Plot this SeriesMatrix using gwexpy.plot.Plot."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)

    def step(self, where: str = "post", **kwargs: Any) -> Any:
        """Plot the matrix as a step function."""
        kwargs.setdefault("drawstyle", f"steps-{where}")
        return self.plot(**kwargs)
