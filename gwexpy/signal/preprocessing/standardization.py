"""
gwexpy.signal.preprocessing.standardization
--------------------------------------------

Standardization algorithms for signal processing.
"""

import numpy as np


class StandardizationModel:
    """Model resulting from standardization transformation.

    Parameters
    ----------
    mean : ndarray
        Mean of the original data.
    scale : ndarray
        Scale (std or MAD) of the original data.
    axis : int or str
        Axis along which standardization was performed.
    """

    def __init__(self, mean, scale, axis):
        self.mean = mean
        self.scale = scale
        self.axis = axis

    def inverse_transform(self, X_std):
        """
        Undo standardization: X = X_std * scale + mean

        Parameters
        ----------
        X_std : ndarray or array-like
            Standardized data.

        Returns
        -------
        X : ndarray
            Original-scale data.
        """
        if hasattr(X_std, "value"):
            val = X_std.value
        else:
            val = X_std

        return val * self.scale + self.mean


def standardize(X, *, method="zscore", ddof=0, axis=-1, return_model=True):
    """
    Standardize an array using z-score or robust standardization.

    Parameters
    ----------
    X : ndarray
        Input data.
    method : str, optional
        Standardization method: 'zscore' or 'robust' (alias: 'mad').
        Default is 'zscore'.

        - ``'zscore'``: Uses mean and standard deviation: ``(X - mean) / std``.
        - ``'robust'`` or ``'mad'``: Uses median and MAD (median absolute deviation):
          ``(X - median) / (1.4826 * MAD)``. The constant 1.4826 is the reciprocal of
          the MAD of a standard normal distribution, which ensures that the resulting
          scale is equivalent to the standard deviation for Gaussian data.

    ddof : int, optional
        Delta degrees of freedom for std calculation. Default is 0.
    axis : int, optional
        Axis along which to standardize. Default is -1.
    return_model : bool, optional
        If True, return (X_standardized, model). If False, return only X_standardized.

    Returns
    -------
    X_standardized : ndarray
        Standardized data.
    model : StandardizationModel, optional
        Model for inverse transformation (if return_model=True).

    Notes
    -----
    NaN values are handled using ``nanmean``/``nanstd`` for zscore and
    ``nanmedian`` for robust methods, so NaN values in the input are ignored
    during computation but preserved in the output.

    If the scale (std or MAD) is zero, a value of 1.0 is used to avoid
    division by zero.
    """
    if method in ("robust", "mad"):
        center = np.nanmedian(X, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(X - center), axis=axis, keepdims=True)
        scale = 1.4826 * mad
        scale = np.where(scale == 0, 1.0, scale)
    elif method == "zscore":
        center = np.nanmean(X, axis=axis, keepdims=True)
        scale = np.nanstd(X, axis=axis, ddof=ddof, keepdims=True)
        scale = np.where(scale == 0, 1.0, scale)
    else:
        raise ValueError(
            f"Unknown standardization method '{method}'. "
            f"Supported methods are 'zscore', 'robust'."
        )

    X_standardized = (X - center) / scale

    if return_model:
        model = StandardizationModel(
            mean=np.squeeze(center), scale=np.squeeze(scale), axis=axis
        )
        return X_standardized, model
    return X_standardized


__all__ = ["StandardizationModel", "standardize"]
