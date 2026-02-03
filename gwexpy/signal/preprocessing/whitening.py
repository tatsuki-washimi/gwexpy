"""
gwexpy.signal.preprocessing.whitening
--------------------------------------

Whitening algorithms for signal processing.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from gwexpy.numerics import SAFE_FLOOR_STRAIN, get_safe_epsilon
except ImportError:
    # Fallbacks matching the Phase 1 numerics design (variance-scaled epsilon).
    SAFE_FLOOR_STRAIN = 1e-50  # floor below GW strain power scale (~1e-42)
    REL_EPS = 1e-6  # relative variance tolerance for auto epsilon

    def get_safe_epsilon(data, rel_tol=REL_EPS, abs_tol=SAFE_FLOOR_STRAIN):
        """Return an epsilon relative to data variance (temporary local logic)."""
        arr = np.asarray(data.value if hasattr(data, "value") else data)
        if arr.size == 0:
            return abs_tol
        var = np.nanvar(arr)
        if not np.isfinite(var) or var <= 0:
            return abs_tol
        return max(abs_tol, var * rel_tol)


def _resolve_eps(eps, data):
    if eps is None or (isinstance(eps, str) and eps == "auto"):
        return get_safe_epsilon(data)
    if isinstance(eps, (int, float, np.floating)):
        eps_val = float(eps)
        if not np.isfinite(eps_val) or eps_val < 0:
            raise ValueError("eps must be a non-negative finite float")
        return eps_val
    raise TypeError("eps must be a float, None, or 'auto'")


class WhiteningModel:
    """Model resulting from whitening transformation.

    Parameters
    ----------
    mean : ndarray
        Mean of the original data.
    W : ndarray
        Whitening matrix.
    """

    def __init__(self, mean, W):
        self.mean = mean
        self.W = W
        self.W_inv = np.linalg.pinv(W)

    def inverse_transform(self, X_w):
        """
        Project whitened data back to original space.

        Parameters
        ----------
        X_w : ndarray or array-like
            Whitened data with shape (n_samples, n_components).

        Returns
        -------
        X_rec : ndarray
            Reconstructed data.
        """
        if hasattr(X_w, "value"):
            val = X_w.value
        else:
            val = X_w

        X_rec = (val @ self.W_inv.T) + self.mean
        return X_rec


def whiten(X, *, method="pca", eps=None, n_components=None, return_model=True):
    """
    Whiten an array using PCA or ZCA whitening.

    Parameters
    ----------
    X : ndarray
        Input data with shape (n_samples, n_features).
    method : str, optional
        Whitening method: 'pca' or 'zca'. Default is 'pca'.

        - ``'pca'``: Principal Component Analysis whitening. Projects data onto
          principal components and scales to unit variance. The output may have
          a different orientation relative to the original feature space.
        - ``'zca'``: Zero-phase Component Analysis whitening (also known as
          Mahalanobis whitening). The whitened data maintains maximum correlation
          with the original data while achieving decorrelation. This preserves
          the original axes alignment better than PCA.

    eps : float or str or None, optional
        Small constant added to eigenvalues to avoid division by zero.
        If None or 'auto' (default), the value is determined from data variance.
    n_components : int, optional
        Number of components to keep. If None, keep all components.
        For PCA, reduces dimensionality. For ZCA, reduces dimensionality but
        loses the channel-preserving property, and a warning is issued.
    return_model : bool, optional
        If True, return (X_whitened, model). If False, return only X_whitened.

    Returns
    -------
    X_whitened : ndarray
        Whitened data with shape (n_samples, n_components) or (n_samples, n_features)
        if n_components is None.
    model : WhiteningModel, optional
        Model for inverse transformation (if return_model=True).

    Notes
    -----
    Both PCA and ZCA whitening produce data with approximately identity covariance
    matrix (assuming sufficient samples). The difference is in the rotation:

    - PCA: ``W = S^(-1/2) @ U^T`` where U and S are from SVD of covariance matrix.
    - ZCA: ``W = U @ S^(-1/2) @ U^T``, which applies the inverse rotation.

    The inverse_transform method uses the pseudo-inverse of the whitening matrix
    to project back to the original space.
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Resolve eps
    if eps is None or (isinstance(eps, str) and eps == "auto"):
        # For whitening, we want eps to be relative to the eigenvalues of the covariance matrix.
        # Eigenvalues have units of variance. std(X)**2 is a good proxy for the scale of eigenvalues.
        var = np.nanvar(X_centered)
        eps_val = max(SAFE_FLOOR_STRAIN, var * 1e-6)
    elif isinstance(eps, (int, float, np.floating)):
        eps_val = float(eps)
    else:
        raise TypeError("eps must be a float, None, or 'auto'")

    cov = np.cov(X_centered, rowvar=False)

    # Handle 1D case
    if cov.ndim == 0:
        cov = np.array([[cov]])

    U, S, Vt = np.linalg.svd(cov)

    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps_val))

    if method == "pca":
        W = S_inv_sqrt @ U.T
    elif method == "zca":
        W = U @ S_inv_sqrt @ U.T
    else:
        raise ValueError(f"method must be 'pca' or 'zca', got '{method}'")

    if n_components is not None:
        if method == "zca":
            warnings.warn("n_components ignores channel mapping for ZCA if reduced.")
        W = W[:n_components, :]

    X_whitened = X_centered @ W.T

    if return_model:
        model = WhiteningModel(mean, W)
        return X_whitened, model
    return X_whitened


__all__ = ["WhiteningModel", "whiten"]
