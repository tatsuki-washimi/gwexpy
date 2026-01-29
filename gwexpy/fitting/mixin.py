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
        """
        Fit the data to a model using iminuit.

        Parameters
        ----------
        model : callable or str
            The model function to fit. Can be a callable with signature
            ``f(x, p1, p2, ...)`` or a string name of a pre-defined model.
        x_range : tuple of float, optional
            The (min, max) range of the x-axis to include in the fit.
        sigma : array-like or scalar, optional
            The errors or weights for the data points.
        p0 : dict, optional
            Initial guesses for the parameter values.
        limits : dict, optional
            Lower and upper bounds for parameters.
        fixed : iterable of str, optional
            Names of parameters to keep fixed during the fit.
        **kwargs
            Additional arguments passed to the fitting engine.

        Returns
        -------
        FitResult
            An object containing the fit results, including best-fit parameters,
            errors, and plotting methods.
        """
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
