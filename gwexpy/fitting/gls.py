"""Generalized Least Squares (GLS) cost function for iminuit.

This module provides a cost function class for fitting with a full
covariance matrix, enabling χ² minimization that properly accounts
for correlations between data points.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from iminuit import Minuit
from iminuit.util import describe

try:
    from scipy.linalg import solve_triangular
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.fitting. Install with: pip install scipy"
    ) from _exc

__all__ = ["GeneralizedLeastSquares", "GLS"]


class GLS:
    """Direct solver for Generalized Least Squares problems (Linear).

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
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        n_samples, n_params = self.X.shape
        if len(self.y) != n_samples:
            raise ValueError(
                f"X and y length mismatch: X has {n_samples} rows but y has "
                f"{len(self.y)} elements"
            )
        if n_samples < n_params:
            raise ValueError(
                f"Underdetermined system: n_samples={n_samples} < "
                f"n_params={n_params}; GLS requires n_samples >= n_params"
            )
        if cov_inv is not None:
            self.cov_inv = np.asarray(cov_inv)
        elif cov is not None:
            cov_arr = np.asarray(cov, dtype=float)
            cond = np.linalg.cond(cov_arr)
            if cond > 1.0 / np.finfo(float).eps:
                warnings.warn(
                    f"GLS: covariance matrix is ill-conditioned (cond={cond:.3e}); "
                    "the computed inverse may be inaccurate. "
                    "Consider supplying cov_inv directly.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.cov_inv = np.linalg.inv(cov_arr)
        else:
            # Ordinary Least Squares (identity weight)
            self.cov_inv = np.eye(n_samples)

    def solve(self) -> np.ndarray:
        """Solve the linear GLS problem.

        beta = (X.T @ W @ X)^-1 @ X.T @ W @ y
        where W = cov_inv.
        """
        W = self.cov_inv
        XTW = self.X.T @ W
        A = XTW @ self.X
        b = XTW @ self.y
        cond = np.linalg.cond(A)
        if cond > 1.0 / np.finfo(float).eps:
            warnings.warn(
                f"GLS.solve: normal equations matrix is ill-conditioned "
                f"(cond={cond:.3e}); solution may be inaccurate.",
                RuntimeWarning,
                stacklevel=2,
            )
        beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return beta


class GeneralizedLeastSquares:
    """Generalized Least Squares (GLS) cost function.

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
        cov_inv_arr = np.asarray(cov_inv, dtype=float)
        self.cov = np.asarray(cov) if cov is not None else None
        self.model = model

        # Validate dimensions
        n = len(self.y)
        if cov_inv_arr.shape != (n, n):
            raise ValueError(
                f"cov_inv shape {cov_inv_arr.shape} does not match "
                f"data length {n}. Expected ({n}, {n})."
            )

        if not np.all(np.isfinite(cov_inv_arr)):
            raise ValueError("cov_inv must contain only finite values (no NaN/Inf)")

        # Warn if cov_inv is indefinite (negative eigenvalue in the symmetric
        # part): chi2 = r^T W r depends only on (W+W^T)/2, so negative
        # eigenvalues produce chi2 < 0 making the Minuit objective non-convex.
        # Emit RuntimeWarning rather than raise so that pipeline callers (e.g.
        # fit_bootstrap_spectrum) that compute cov_inv from empirical data can
        # degrade gracefully (issue #457 / G6-F2).
        W_sym = (cov_inv_arr + cov_inv_arr.T) / 2.0
        eigvals = np.linalg.eigvalsh(W_sym)
        min_eig = float(eigvals.min())
        scale = max(float(eigvals.max()), 1.0)
        if min_eig < -1e-10 * scale:
            warnings.warn(
                f"GeneralizedLeastSquares: cov_inv is indefinite "
                f"(min eigenvalue of symmetric part={min_eig:.3e}); "
                "chi2 may be negative, making the fit non-convex. "
                "Consider supplying a positive (semi-)definite cov_inv.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.cov_inv = cov_inv_arr

        # Precompute Cholesky factor if covariance is available for better stability
        self.cov_cho = None
        if self.cov is not None:
            try:
                self.cov_cho = np.linalg.cholesky(self.cov)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "GeneralizedLeastSquares: cov is not positive definite; "
                    "Cholesky decomposition failed. Falling back to direct cov_inv. "
                    "Numerical stability may be reduced.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.cov_cho = None

        # Extract parameter names from model (skip first arg 'x')
        params = describe(model)[1:]
        self._parameters = {name: None for name in params}

    def __call__(self, *args) -> float:
        """Compute χ² for given parameter values.

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
