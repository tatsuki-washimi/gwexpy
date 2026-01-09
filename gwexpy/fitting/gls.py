"""
Generalized Least Squares (GLS) cost function for iminuit.

This module provides a cost function class for fitting with a full
covariance matrix, enabling χ² minimization that properly accounts
for correlations between data points.
"""
from __future__ import annotations

import numpy as np
from iminuit import Minuit
from iminuit.util import describe

__all__ = ["GeneralizedLeastSquares"]


class GeneralizedLeastSquares:
    """
    Generalized Least Squares (GLS) cost function.

    Minimizes χ² = r.T @ cov_inv @ r where r = y - model(x, **params).

    This cost function accounts for correlations between data points
    through the inverse covariance matrix.

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., frequency array).
    y : array-like
        Observed data (real-valued).
    cov_inv : ndarray
        Inverse covariance matrix, shape (n, n) where n = len(y).
        Can be obtained from `BifrequencyMap.inverse().value`.
    model : callable
        Model function with signature `model(x, *params) -> y`.
        The first argument must be `x`, followed by fit parameters.

    Attributes
    ----------
    errordef : float
        Error definition for Minuit. Set to `Minuit.LEAST_SQUARES` (= 1.0).

    Examples
    --------
    >>> def linear(x, a, b):
    ...     return a * x + b
    >>> gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
    >>> m = Minuit(gls, a=1, b=0)
    >>> m.migrad()
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cov_inv: np.ndarray,
        model: callable,
    ) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cov_inv = np.asarray(cov_inv)
        self.model = model

        # Validate dimensions
        n = len(self.y)
        if self.cov_inv.shape != (n, n):
            raise ValueError(
                f"cov_inv shape {self.cov_inv.shape} does not match "
                f"data length {n}. Expected ({n}, {n})."
            )

        # Extract parameter names from model (skip first arg 'x')
        params = describe(model)[1:]
        self._parameters = {name: None for name in params}

    def __call__(self, *args) -> float:
        """
        Compute χ² for given parameter values.

        Parameters
        ----------
        *args : float
            Parameter values in the order defined by the model signature.

        Returns
        -------
        float
            χ² value: r.T @ cov_inv @ r
        """
        # Model prediction
        ym = self.model(self.x, *args)
        # Residual vector
        r = self.y - ym
        # χ² = r.T @ cov_inv @ r
        chi2 = float(r @ self.cov_inv @ r)
        return chi2

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return len(self.y)
