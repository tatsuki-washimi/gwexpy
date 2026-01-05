"""
Mixin for fitting functionality.
"""

from __future__ import annotations
from typing import Any, Iterable, Optional

class FittingMixin:
    """
    Mixin class that adds .fit() method to Series objects.
    """

    def fit(self, model: Any, x_range: Optional[tuple[float, float]] = None,
            sigma: Optional[Any] = None, p0: Optional[dict[str, float]] = None,
            limits: Optional[dict[str, tuple[float, float]]] = None,
            fixed: Optional[Iterable[str]] = None, **kwargs: Any) -> Any:
        "Fit the series data to a model."
        from gwexpy.fitting import fit_series
        return fit_series(self, model, x_range=x_range, sigma=sigma,
                          p0=p0, limits=limits, fixed=fixed, **kwargs)
