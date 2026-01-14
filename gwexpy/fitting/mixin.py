"""
Mixin for fitting functionality.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class FittingMixin:
    """
    Mixin class that adds .fit() method to Series objects.
    """

    def fit(
        self,
        model: Any,
        x_range: tuple[float, float] | None = None,
        sigma: Any | None = None,
        p0: dict[str, float] | None = None,
        limits: dict[str, tuple[float, float]] | None = None,
        fixed: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        "Fit the series data to a model."
        from gwexpy.fitting import fit_series

        return fit_series(
            self,
            model,
            x_range=x_range,
            sigma=sigma,
            p0=p0,
            limits=limits,
            fixed=fixed,
            **kwargs,
        )
