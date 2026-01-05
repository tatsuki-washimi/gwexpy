
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

try:
    from sklearn.decomposition import PCA, FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .preprocess import (
    whiten_matrix
)

@dataclass
class PCAResult:
    sklearn_model: Any
    channel_labels: List[str]
    preprocessing: Dict[str, Any]
    input_meta: Dict[str, Any] = None

    @property
    def components(self):
        return self.sklearn_model.components_

    @property
    def components_(self):
        """Alias to match scikit-learn API."""
        return self.sklearn_model.components_

    @property
    def explained_variance_ratio(self):
        return self.sklearn_model.explained_variance_ratio_

    @property
    def explained_variance_ratio_(self):
        """Alias to match scikit-learn API."""
        return self.sklearn_model.explained_variance_ratio_

    def summary_dict(self):
        return {
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "n_components": self.sklearn_model.n_components_,
        }

@dataclass
class ICAResult:
    sklearn_model: Any
    channel_labels: List[str]
    preprocessing: Dict[str, Any]
    input_meta: Dict[str, Any] = None


def _check_sklearn(name="scikit-learn"):
    if not SKLEARN_AVAILABLE:
        raise ImportError(f"{name} is required. pip install scikit-learn")

def _handle_nan_policy(matrix, policy, impute_kwargs=None):
    if impute_kwargs is None:
        impute_kwargs = {}

    if np.any(np.isnan(matrix.value)):
        if policy == "raise":
            raise ValueError("Input contains NaNs and nan_policy is 'raise'.")
        elif policy == "impute":
            mat_copy = matrix.copy()
            val = mat_copy.value

            method = impute_kwargs.get("method", "interpolate")
            impute_kwargs.get("limit", None)

            for i in range(val.shape[1]):
                col = val[:, i]
                nans = np.isnan(col)
                if np.any(nans):
                    if method == "interpolate":
                        valid = ~nans
                        if not np.any(valid):
                            continue
                        x = np.arange(len(col))
                        col[nans] = np.interp(x[nans], x[valid], col[valid])
                    elif method == "mean":
                        col[nans] = np.nanmean(col)
                    # For simplicity, minimal implementation of others

            matrix = mat_copy

    return matrix

def _fit_scaler(matrix, method, ddof=0):
    val = matrix.value
    if method == "robust":
        med = np.nanmedian(val, axis=0)
        diff = np.abs(val - med)
        mad = np.nanmedian(diff, axis=0)
        scale = 1.4826 * mad
        scale[scale==0] = 1.0
        return {"mean": med, "scale": scale, "method": method}
    elif method == "zscore":
        mean = np.nanmean(val, axis=0)
        std = np.nanstd(val, axis=0, ddof=ddof)
        std[std==0] = 1.0
        return {"mean": mean, "scale": std, "method": method}
    return None

def _apply_scaler(matrix, preprocessing):
    scaler = preprocessing.get("scaler_stats")
    if not scaler:
        return matrix

    val = matrix.value.copy()
    val = (val - scaler["mean"]) / scaler["scale"]

    new_mat = matrix.__class__(val, t0=matrix.t0, dt=matrix.dt)
    if hasattr(new_mat, 'channel_names'):
        new_mat.channel_names = getattr(matrix, 'channel_names', None)
    return new_mat

def _inverse_scaler(val, preprocessing):
    scaler = preprocessing.get("scaler_stats")
    if not scaler:
        return val

    return val * scaler["scale"] + scaler["mean"]


def pca_fit(
    matrix,
    *,
    n_components=None,
    svd_solver="auto",
    whiten=False,
    center=True,
    scale=None,
    nan_policy="raise",
    impute_kwargs=None,
    random_state=None,
):
    """
    Fit a PCA model to the TimeSeriesMatrix.

    Parameters
    ----------
    matrix : TimeSeriesMatrix
        Input data matrix (channels, cols, time).
    n_components : int, optional
        Number of components to keep.
    svd_solver : str, default='auto'
        SVD solver to use.
    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
    center : bool, default=True
        Whether to center the data before fitting.
    scale : str, optional
        Scaling method ('zscore' or 'robust').
    nan_policy : str, default='raise'
        'raise', 'impute'.
    impute_kwargs : dict, optional
        Arguments for imputation if nan_policy='impute'.
    random_state : int, optional
        Random seed.

    Returns
    -------
    PCAResult
        Object containing the fitted PCA model and metadata.
    """
    _check_sklearn()

    matrix = _handle_nan_policy(matrix, nan_policy, impute_kwargs)

    preprocessing = {"center": center, "scale": scale, "whiten_param": whiten}

    X_proc = matrix

    if scale in ["zscore", "robust"]:
        scaler_stats = _fit_scaler(matrix, scale)
        preprocessing["scaler_stats"] = scaler_stats
        X_proc = _apply_scaler(matrix, preprocessing)

    # PCA
    # Input to sklearn PCA: (n_samples, n_features)
    # Our data: (channels, cols, time) -> 3D.
    # Flatten features: (features, time).
    # Then Transpose: (time, features).
    X_features = X_proc.value.reshape(-1, X_proc.shape[-1])
    X_sklearn = X_features.T

    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        whiten=whiten,
        random_state=random_state
    )

    pca.fit(X_sklearn)

    return PCAResult(
        sklearn_model=pca,
        channel_labels=getattr(matrix, "channel_names", []),
        preprocessing=preprocessing,
        input_meta={"t0": matrix.t0, "dt": matrix.dt}
    )

def pca_transform(pca_res, matrix, n_components=None):
    """
    Apply PCA dimensionality reduction to the matrix.

    Parameters
    ----------
    pca_res : PCAResult
        Fitted PCA result object.
    matrix : TimeSeriesMatrix
        Data to transform.
    n_components : int, optional
        Number of components to use (can be less than fitted).

    Returns
    -------
    TimeSeriesMatrix
        Transformed matrix (components, 1, time).
    """
    _check_sklearn()

    # Preprocessing
    matrix = _apply_scaler(matrix, pca_res.preprocessing)

    X_features = matrix.value.reshape(-1, matrix.shape[-1])
    X_sklearn = X_features.T

    scores = pca_res.sklearn_model.transform(X_sklearn)

    if n_components is not None:
        scores = scores[:, :n_components]

    n_pcs = scores.shape[1]
    labels = [f"PC{i+1}" for i in range(n_pcs)]

    # Output: (samples, components).
    # We want TimeSeriesMatrix (components, 1, samples).
    # components becomes channels.
    scores_new = scores.T[:, None, :] # (components, 1, samples)

    new_mat = matrix.__class__(
        scores_new,
        t0=matrix.t0,
        dt=matrix.dt,
    )
    if hasattr(new_mat, 'channel_names'):
        new_mat.channel_names = labels

    return new_mat

def pca_inverse_transform(pca_res, scores_matrix):
    """
    Transform data back to its original space.

    Parameters
    ----------
    pca_res : PCAResult
        Fitted PCA result object.
    scores_matrix : TimeSeriesMatrix
        PCA scores (transformed data).

    Returns
    -------
    TimeSeriesMatrix
        Reconstructed matrix in original space (channels, cols, time).
    """
    _check_sklearn()

    if hasattr(scores_matrix, 'value'):
        val = scores_matrix.value
    else:
        val = scores_matrix

    # Input scores: (components, 1, samples) -> (samples, components)
    # Assuming standard shape from our transform.
    # Flatten if needed?
    # val is (components, 1, samples).
    # reshape to (components, samples).T -> (samples, components)
    scores_features = val.reshape(-1, val.shape[-1])
    scores_sklearn = scores_features.T

    X_rec_val = pca_res.sklearn_model.inverse_transform(scores_sklearn)

    # X_rec_val is (samples, features) -> (features, samples)
    X_rec_val = X_rec_val.T

    # Reshape back to original 3D structure?
    # We don't know original structure explicitly unless we stored shape in preprocessing?
    # But usually we return flattened channels if structure unknown, or try to infer.
    # For now, return (features, 1, samples) as default flat structure.
    # Or try to look at input_meta?
    # preprocess 'standardize' maintained shape.
    # 'whiten' maintained shape if same dim.
    # 'pca' inverse: features count matches original.
    # If we knew original shape, we could reshape.
    # Let's assume (features, 1, samples) for now.

    X_rec_3d = X_rec_val[:, None, :]

    # Undo scaler (scaler handles 3D if generic axis logic used in inverse)
    # _inverse_scaler simply mult/add.
    X_rec_3d = _inverse_scaler(X_rec_3d, pca_res.preprocessing)

    # We reconstruct original channels
    # We assume we can get original t0/dt from input stats or scores matrix
    new_mat = scores_matrix.__class__(
        X_rec_3d,
        t0=scores_matrix.t0,
        dt=scores_matrix.dt
    )
    if pca_res.channel_labels:
         if hasattr(new_mat, 'channel_names'):
             new_mat.channel_names = pca_res.channel_labels

    return new_mat

def ica_fit(
    matrix,
    *,
    n_components=None,
    algorithm="parallel",
    fun="logcosh",
    whiten="unit-variance",
    max_iter=200,
    tol=1e-4,
    random_state=None,
    center=True,
    scale=None,
    prewhiten=True,
    nan_policy="raise",
    impute_kwargs=None,
):
    """
    Fit an ICA model to the TimeSeriesMatrix.

    Parameters
    ----------
    matrix : TimeSeriesMatrix
        Input data matrix.
    n_components : int, optional
        Number of components to use.
    algorithm : str, default='parallel'
        Applied algorithm for FastICA.
    fun : str, default='logcosh'
        Functional form of the G function used in the approximation to neg-entropy.
    whiten : str or bool, default='unit-variance'
        Whitening strategy.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance.
    random_state : int, optional
        Random seed.
    center : bool, default=True
        Whether to center the data.
    scale : str, optional
        Scaling method ('zscore' or 'robust').
    prewhiten : bool, default=True
        Whether to perform pre-whitening using PCA/ZCA before ICA.
    nan_policy : str, default='raise'
        NaN handling policy.
    impute_kwargs : dict, optional
        Imputation arguments.

    Returns
    -------
    ICAResult
        Fitted ICA result object.
    """
    _check_sklearn()

    matrix = _handle_nan_policy(matrix, nan_policy, impute_kwargs)

    preprocessing = {"center": center, "scale": scale, "prewhiten": prewhiten}

    X_proc = matrix

    if scale in ["zscore", "robust"]:
        scaler_stats = _fit_scaler(matrix, scale)
        preprocessing["scaler_stats"] = scaler_stats
        X_proc = _apply_scaler(matrix, preprocessing)

    white_model = None
    if prewhiten:
        # Use our whitening
        X_proc, white_model = whiten_matrix(X_proc, method="pca", n_components=n_components)
        preprocessing["whitening_model"] = white_model
        whiten_arg = False # sklearn should not whiten again usually, or we adjust
    else:
        whiten_arg = whiten

    ica = FastICA(
        n_components=n_components if whiten_arg else None,
        algorithm=algorithm,
        fun=fun,
        whiten=whiten_arg,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    # Input to sklearn: (samples, features)
    X_features = X_proc.value.reshape(-1, X_proc.shape[-1])
    start_val = X_features.T
    ica.fit(start_val)

    return ICAResult(
        sklearn_model=ica,
        channel_labels=getattr(matrix, "channel_names", []),
        preprocessing=preprocessing,
        input_meta={"t0": matrix.t0, "dt": matrix.dt}
    )

def ica_transform(ica_res, matrix):
    """
    Recover the sources from the matrix (apply ICA).

    Parameters
    ----------
    ica_res : ICAResult
        Fitted ICA result object.
    matrix : TimeSeriesMatrix
        Data to transform.

    Returns
    -------
    TimeSeriesMatrix
        Recovered sources (components, 1, time).
    """
    _check_sklearn()

    X_proc = _apply_scaler(matrix, ica_res.preprocessing)

    if ica_res.preprocessing.get("prewhiten"):
         wm = ica_res.preprocessing["whitening_model"]
         # wm model applied in 'whiten_matrix'
         # We need to reuse that logic or whiten_matrix(..., return_model=False) but whiten_matrix is strict.
         # But wm.W and wm.mean are available.
         # Matrix value (channels, time).
         # Whitening: (val.T - mean) @ W.T -> (time, components)
         val = X_proc.value.reshape(-1, X_proc.shape[-1]) #(features, time)
         val_centered = (val.T) - wm.mean #(time, features)
         val_w = val_centered @ wm.W.T #(time, components)

         X_input = val_w # (samples, features)
    else:
         X_input = X_proc.value.reshape(-1, X_proc.shape[-1]).T # (samples, channels)

    sources = ica_res.sklearn_model.transform(X_input)
    # sources: (samples, components)

    n_ics = sources.shape[1]
    labels = [f"IC{i+1}" for i in range(n_ics)]

    # Output (components, 1, samples)
    sources_new = sources.T[:, None, :]

    new_mat = matrix.__class__(sources_new, t0=matrix.t0, dt=matrix.dt)
    if hasattr(new_mat, 'channel_names'):
        new_mat.channel_names = labels
    return new_mat

def ica_inverse_transform(ica_res, sources):
    """
    Transform sources back to the mixed signal space.

    Parameters
    ----------
    ica_res : ICAResult
        Fitted ICA result object.
    sources : TimeSeriesMatrix
        Independent components (sources).

    Returns
    -------
    TimeSeriesMatrix
        Reconstructed matrix in mixed space.
    """
    _check_sklearn()

    # sources: (components, 1, samples)
    if hasattr(sources, 'value'):
        val = sources.value
    else:
        val = sources

    # Reshape to (samples, components) for sklearn
    val_flat = val.reshape(-1, val.shape[-1]).T # (components, time).T -> (time, components)

    rec = ica_res.sklearn_model.inverse_transform(val_flat)

    # rec is (samples, features) if not prewhitened, or (samples, whitened_components) if prewhitened?
    # inverse_transform returns to the input space of fit().
    # fit() input was X_proc (whitened if prewhiten=True).
    # So rec is in X_proc space.

    # Undo prewhiten
    if ica_res.preprocessing.get("prewhiten"):
         wm = ica_res.preprocessing["whitening_model"]
         # wm model: inverse_transform(X_w).
         # X_w input to inverse_transform should be (n_samples, n_components)?
         # Or it might need to match whitening logic.
         # Our whitening maps (time, features) -> (time, components).
         # So rec is (time, components).
         rec = wm.inverse_transform(rec)
         # Now rec is (time, features) i.e. (samples, channels)

    # Undo scaler
    # _inverse_scaler expects (channels, ..., time) or generic broadcasting?
    # rec is (samples, channels).
    # transpose back to (channels, 1, samples)
    rec_3d = rec.T[:, None, :]

    rec_final = _inverse_scaler(rec_3d, ica_res.preprocessing)

    new_mat = sources.__class__(rec_final, t0=sources.t0, dt=sources.dt)
    if ica_res.channel_labels:
         if hasattr(new_mat, 'channel_names'):
             new_mat.channel_names = ica_res.channel_labels

    return new_mat
