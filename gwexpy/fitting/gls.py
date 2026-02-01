"""
Generalized Least Squares (GLS) cost function for iminuit.

This module provides a cost function class for fitting with a full
covariance matrix, enabling χ² minimization that properly accounts
for correlations between data points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from iminuit import Minuit
from iminuit.util import describe
from scipy.linalg import solve_triangular

__all__ = ["GeneralizedLeastSquares", "GLS"]


class GLS:
    """
    Direct solver for Generalized Least Squares problems (Linear).

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_params).
    y : array-like
        Observation vector (n_samples,).
    cov : array-like, optional
        Covariance matrix (n_samples, n_samples).
    cov_inv : array-like, optional
        Inverse covariance matrix (n_samples, n_samples).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cov: np.ndarray | None = None,
        cov_inv: np.ndarray | None = None,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if cov_inv is not None:
            self.cov_inv = np.asarray(cov_inv)
        elif cov is not None:
            self.cov_inv = np.linalg.inv(np.asarray(cov))
        else:
            # Ordinary Least Squares (identity weight)
            self.cov_inv = np.eye(len(y))

    def solve(self) -> np.ndarray:
        """
        Solve the linear GLS problem: beta = (X.T @ W @ X)^-1 @ X.T @ W @ y
        where W = cov_inv.
        """
        W = self.cov_inv
        XTW = self.X.T @ W
        # Use np.linalg.solve for better stability than explicit inverse
        beta = np.linalg.solve(XTW @ self.X, XTW @ self.y)
        return beta


class GeneralizedLeastSquares:
    """
    Generalized Least Squares (GLS) cost function.

    Minimizes χ² = r.T @ cov_inv @ r where r = y - ``model(x, **params)``.

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
    cov : ndarray, optional
        Original covariance matrix. If provided, Cholesky decomposition
        is used for better numerical stability.

    Notes
    -----
    `errordef` is set to `Minuit.LEAST_SQUARES` (= 1.0) for iminuit.

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
        model: Callable[..., Any],
        cov: np.ndarray | None = None,
    ) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.cov_inv = np.asarray(cov_inv)
        self.cov = np.asarray(cov) if cov is not None else None
        self.model = model

        # Validate dimensions
        n = len(self.y)
        if self.cov_inv.shape != (n, n):
            raise ValueError(
                f"cov_inv shape {self.cov_inv.shape} does not match "
                f"data length {n}. Expected ({n}, {n})."
            )

        # Precompute Cholesky factor if covariance is available for better stability
        self.cov_cho = None
        if self.cov is not None:
            try:
                self.cov_cho = np.linalg.cholesky(self.cov)
            except np.linalg.LinAlgError:
                # Fallback to cov_inv if not positive definite
                self.cov_cho = None

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

        # Compute χ² based on available covariance information
        if self.cov_cho is not None:
            # Use Cholesky factor for better numerical stability
            # r.T @ inv(cov) @ r == ||inv(L) @ r||^2

            # solve L @ w = r
            w = solve_triangular(self.cov_cho, r, lower=True)
            chi2 = float(np.sum(np.abs(w) ** 2))
        else:
            # χ² = r.T @ cov_inv @ r
            chi2 = float(r @ self.cov_inv @ r)

        return chi2

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return len(self.y)
