import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy import units as u

from .collections import TimeSeriesDict, TimeSeriesList
from .decomposition import (
    ica_fit,
    ica_inverse_transform,
    ica_transform,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
)
from .matrix import TimeSeriesMatrix
from .preprocess import WhiteningModel, impute_timeseries, whiten_matrix
from .timeseries import TimeSeries


def _is_collection(x):
    return isinstance(x, (TimeSeriesDict, TimeSeriesList))


def _to_matrix_from_collection(
    obj,
    *,
    align: str = "intersection",
):
    if isinstance(obj, TimeSeriesDict):
        return obj.to_matrix(align=align), obj
    if isinstance(obj, TimeSeriesList):
        return obj.to_matrix(align=align), obj
    return obj, None


def _restore_collection(result, original):
    if original is None:
        return result
    if isinstance(original, TimeSeriesDict) and isinstance(result, TimeSeriesMatrix):
        return result.to_dict()
    if isinstance(original, TimeSeriesList) and isinstance(result, TimeSeriesMatrix):
        return result.to_list()
    return result


class Transform:
    """
    Minimal transform interface for TimeSeries-like objects.
    """

    supports_inverse = False

    def fit(self, x):
        """Fit the transform to the data. Returns self."""
        return self

    def transform(self, x):
        """Apply the transform to data. Must be implemented by subclasses."""
        raise NotImplementedError

    def fit_transform(self, x):
        """Fit and transform in one step."""
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        """Reverse the transform. Not all transforms support this."""
        raise NotImplementedError(
            "inverse_transform is not implemented for this transform"
        )


class Pipeline:
    """
    Sequentially apply a list of transforms.
    """

    def __init__(self, steps: Sequence[tuple[str, Transform]]):
        """Initialize pipeline with named transform steps.

        Parameters
        ----------
        steps : list of (name, Transform) tuples
            Sequence of transforms to apply.
        """
        self.steps: list[tuple[str, Transform]] = []
        for name, step in steps:
            if not isinstance(step, Transform):
                raise TypeError(f"Step '{name}' must be a Transform")
            self.steps.append((name, step))
        self._is_fitted = False

    def fit(self, x):
        """Fit all transforms in sequence."""
        data = x
        for _, step in self.steps:
            data = step.fit_transform(data)
        self._is_fitted = True
        return self

    def transform(self, x):
        """Apply all transforms in sequence."""
        data = x
        for _, step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, x):
        """Fit and transform in one step."""
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y, *, strict: bool = True):
        """Apply inverse transforms in reverse order.

        Parameters
        ----------
        y : data
            Transformed data.
        strict : bool, optional
            If True, raise error if any step doesn't support inverse.
        """
        data = y
        for name, step in reversed(self.steps):
            if getattr(step, "supports_inverse", False):
                data = step.inverse_transform(data)
            else:
                if strict:
                    raise ValueError(
                        f"Step '{name}' does not support inverse_transform"
                    )
        return data


class ImputeTransform(Transform):
    """
    Impute missing values using existing lower-level helpers.
    """

    def __init__(self, method: str = "interpolate", **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    def transform(self, x):
        """Apply imputation to TimeSeries, Matrix, or Collections."""
        if isinstance(x, TimeSeries):
            return impute_timeseries(x, method=self.method, **self.kwargs)
        if isinstance(x, TimeSeriesMatrix):
            return x.impute(method=self.method, **self.kwargs)
        if isinstance(x, TimeSeriesDict):
            return x.__class__({k: self.transform(v) for k, v in x.items()})
        if isinstance(x, TimeSeriesList):
            return x.__class__([self.transform(v) for v in x])
        raise TypeError(f"Unsupported type for ImputeTransform: {type(x)}")


class StandardizeTransform(Transform):
    """
    Standardize TimeSeries/Matrix objects with optional robust scaling.
    """

    supports_inverse = True

    def __init__(
        self,
        method: str = "zscore",
        ddof: int = 0,
        robust: bool = False,
        axis: str = "time",
        *,
        multivariate: bool = False,
        align: str = "intersection",
    ):
        super().__init__()
        self.method = "robust" if robust else method
        self.ddof = ddof
        self.axis = axis
        self.multivariate = multivariate
        self.align = align
        self.params: dict[str, Any] | None = None

    def _compute_stats_ts(self, ts: TimeSeries):
        if self.method == "robust":
            center = np.nanmedian(ts.value)
            scale = 1.4826 * np.nanmedian(np.abs(ts.value - center))
        else:
            center = np.nanmean(ts.value)
            scale = np.nanstd(ts.value, ddof=self.ddof)
        if scale == 0:
            scale = 1.0
        return float(center), float(scale)

    def _compute_stats_matrix(self, mat: TimeSeriesMatrix):
        val = np.asarray(mat.value)
        np_axis = -1 if self.axis == "time" else (0, 1)
        if self.method == "robust":
            center = np.nanmedian(val, axis=np_axis, keepdims=True)
            scale = 1.4826 * np.nanmedian(
                np.abs(val - center), axis=np_axis, keepdims=True
            )
        else:
            center = np.nanmean(val, axis=np_axis, keepdims=True)
            scale = np.nanstd(val, axis=np_axis, ddof=self.ddof, keepdims=True)
        scale = np.where(scale == 0, 1.0, scale)
        return center, scale

    def fit(self, x):
        """Fit standardization parameters (center, scale) for the input data."""
        data, original = (
            _to_matrix_from_collection(x, align=self.align)
            if self.multivariate
            else (x, None)
        )
        if isinstance(data, TimeSeries):
            center, scale = self._compute_stats_ts(data)
            self.params = {"type": "ts", "center": center, "scale": scale, "meta": data}
        elif isinstance(data, TimeSeriesMatrix):
            center, scale = self._compute_stats_matrix(data)
            self.params = {
                "type": "matrix",
                "center": center,
                "scale": scale,
                "axis": self.axis,
                "meta": data,
                "collection": original,
            }
        elif isinstance(data, TimeSeriesDict):
            stats = {k: self._compute_stats_ts(v) for k, v in data.items()}
            self.params = {"type": "dict", "stats": stats}
        elif isinstance(data, TimeSeriesList):
            stats = [self._compute_stats_ts(v) for v in data]
            self.params = {"type": "list", "stats": stats}
        else:
            raise TypeError(f"Unsupported type for StandardizeTransform: {type(data)}")
        return self

    def _apply_ts(self, ts: TimeSeries, center: float, scale: float):
        new_ts = ts.copy()
        new_ts.value[:] = (ts.value - center) / scale
        try:
            new_ts.unit = u.dimensionless_unscaled
        except AttributeError:
            pass
        return new_ts

    def _apply_matrix(self, mat: TimeSeriesMatrix, center, scale):
        val = (np.asarray(mat.value) - center) / scale
        new_mat = mat.copy()
        new_mat.value[:] = val
        return new_mat

    def transform(self, x):
        """Apply standardization using fitted parameters."""
        if self.params is None:
            self.fit(x)

        params = copy.deepcopy(self.params)
        data, original = (
            _to_matrix_from_collection(x, align=self.align)
            if self.multivariate
            else (x, None)
        )

        if params["type"] == "ts" and isinstance(data, TimeSeries):
            return self._apply_ts(data, params["center"], params["scale"])
        if params["type"] == "matrix" and isinstance(data, TimeSeriesMatrix):
            result = self._apply_matrix(data, params["center"], params["scale"])
            return (
                _restore_collection(result, params.get("collection"))
                if self.multivariate
                else result
            )
        if params["type"] == "dict" and isinstance(data, TimeSeriesDict):
            out = data.__class__()
            for k, ts in data.items():
                if k not in params["stats"]:
                    params["stats"][k] = self._compute_stats_ts(ts)
                center, scale = params["stats"][k]
                out[k] = self._apply_ts(ts, center, scale)
            return out
        if params["type"] == "list" and isinstance(data, TimeSeriesList):
            out_list = data.__class__()
            for idx, ts in enumerate(data):
                if idx >= len(params["stats"]):
                    params["stats"].append(self._compute_stats_ts(ts))
                center, scale = params["stats"][idx]
                out_list.append(self._apply_ts(ts, center, scale))
            return out_list

        raise TypeError(
            f"Incompatible input for StandardizeTransform.transform: {type(data)}"
        )

    def inverse_transform(self, y):
        """Reverse standardization transformation."""
        if self.params is None:
            raise ValueError("StandardizeTransform has not been fitted.")

        params = self.params
        data = y
        if params["type"] == "ts" and isinstance(data, TimeSeries):
            center, scale = params["center"], params["scale"]
            new_ts = data.copy()
            new_ts.value[:] = data.value * scale + center
            return new_ts
        if params["type"] == "matrix" and isinstance(data, TimeSeriesMatrix):
            center, scale = params["center"], params["scale"]
            val = data.value * scale + center
            new_mat = data.copy()
            new_mat.value[:] = val
            return (
                _restore_collection(new_mat, params.get("collection"))
                if self.multivariate
                else new_mat
            )
        if params["type"] == "dict" and isinstance(data, TimeSeriesDict):
            out = data.__class__()
            for k, ts in data.items():
                if k not in params["stats"]:
                    raise KeyError(
                        f"Key '{k}' not present in fitted StandardizeTransform stats."
                    )
                center, scale = params["stats"][k]
                new_ts = ts.copy()
                new_ts.value[:] = ts.value * scale + center
                out[k] = new_ts
            return out
        if params["type"] == "list" and isinstance(data, TimeSeriesList):
            out_list = data.__class__()
            for idx, ts in enumerate(data):
                if idx >= len(params["stats"]):
                    raise IndexError(f"No fitted stats for index {idx}")
                center, scale = params["stats"][idx]
                new_ts = ts.copy()
                new_ts.value[:] = ts.value * scale + center
                out_list.append(new_ts)
            return out_list

        raise TypeError(
            f"Incompatible input for StandardizeTransform.inverse_transform: {type(data)}"
        )


class WhitenTransform(Transform):
    """
    Whitening using PCA or ZCA on TimeSeriesMatrix-like data.
    """

    supports_inverse = True

    def __init__(
        self,
        method: str = "pca",
        eps: float = 1e-12,
        n_components: int | None = None,
        *,
        multivariate: bool = True,
        align: str = "intersection",
    ):
        super().__init__()
        self.method = method
        self.eps = eps
        self.n_components = n_components
        self.multivariate = multivariate
        self.align = align
        self.model: WhiteningModel | None = None
        self._channel_names: list[str] | None = None

    def _to_matrix(self, x):
        if isinstance(x, TimeSeriesMatrix):
            return x
        if _is_collection(x):
            mat, original = _to_matrix_from_collection(x, align=self.align)
            return mat, original
        if isinstance(x, TimeSeries):
            mat = TimeSeriesMatrix(x.value[None, None, :], t0=x.t0, dt=x.dt)
            return mat, None
        raise TypeError(f"Unsupported type for whitening: {type(x)}")

    def fit(self, x):
        """Fit whitening parameters for the input data."""
        mat_data = x
        original = None
        if self.multivariate:
            mat_data, original = _to_matrix_from_collection(x, align=self.align)
        mat_data, orig2 = (
            (mat_data, None)
            if isinstance(mat_data, TimeSeriesMatrix)
            else self._to_matrix(mat_data)
        )
        original = original or orig2

        whitened, model = whiten_matrix(
            mat_data, method=self.method, eps=self.eps, n_components=self.n_components
        )
        self.model = model
        self._channel_names = getattr(whitened, "channel_names", None)
        self._original_collection = original
        return self

    def transform(self, x):
        """Apply whitening transform."""
        if self.model is None:
            self.fit(x)

        mat_data = x
        original = None
        if self.multivariate:
            mat_data, original = _to_matrix_from_collection(x, align=self.align)
        mat_data, orig2 = (
            (mat_data, None)
            if isinstance(mat_data, TimeSeriesMatrix)
            else self._to_matrix(mat_data)
        )
        original = original or orig2

        X = mat_data.value.reshape(-1, mat_data.shape[-1]).T  # (time, features)
        X_centered = X - self.model.mean
        X_w = X_centered @ self.model.W.T
        new_val = X_w.T[:, None, :]
        new_mat = TimeSeriesMatrix(new_val, t0=mat_data.t0, dt=mat_data.dt)
        if self._channel_names:
            new_mat.channel_names = self._channel_names
        return _restore_collection(new_mat, original if self.multivariate else None)

    def inverse_transform(self, y):
        """Reverse whitening transformation."""
        if self.model is None:
            raise ValueError("WhitenTransform has not been fitted.")
        mat_data, original = (
            _to_matrix_from_collection(y, align=self.align)
            if self.multivariate
            else (y, None)
        )
        if not isinstance(mat_data, TimeSeriesMatrix):
            raise TypeError(
                "inverse_transform expects TimeSeriesMatrix-compatible input."
            )
        X_w = mat_data.value.reshape(-1, mat_data.shape[-1]).T
        X_rec = self.model.inverse_transform(X_w)
        X_rec = X_rec.T[:, None, :]
        new_mat = TimeSeriesMatrix(X_rec, t0=mat_data.t0, dt=mat_data.dt)
        return _restore_collection(new_mat, original)


class PCATransform(Transform):
    """
    PCA wrapper using existing decomposition helpers.
    """

    supports_inverse = True

    def __init__(
        self,
        n_components: int | None = None,
        *,
        multivariate: bool = True,
        align: str = "intersection",
        **kwargs,
    ):
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.multivariate = multivariate
        self.align = align
        self.model = None
        self._collection = None

    def _ensure_matrix(self, x):
        if isinstance(x, TimeSeriesMatrix):
            return x, None
        if _is_collection(x):
            mat, original = _to_matrix_from_collection(x, align=self.align)
            return mat, original
        raise TypeError(
            "PCATransform expects TimeSeriesMatrix or collection convertible to matrix."
        )

    def fit(self, x):
        """Fit PCA model for the input data."""
        mat, collection = self._ensure_matrix(x) if self.multivariate else (x, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError(
                "PCATransform requires TimeSeriesMatrix when multivariate=False."
            )
        self.model = pca_fit(mat, n_components=self.n_components, **self.kwargs)
        self._collection = collection
        return self

    def transform(self, x):
        """Apply PCA transformation (project to scores)."""
        if self.model is None:
            self.fit(x)
        mat, collection = self._ensure_matrix(x) if self.multivariate else (x, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError("PCATransform requires TimeSeriesMatrix input.")
        scores = pca_transform(self.model, mat, n_components=self.n_components)
        return _restore_collection(scores, collection if self.multivariate else None)

    def inverse_transform(self, y):
        """Reverse PCA transformation (reconstruct from scores)."""
        if self.model is None:
            raise ValueError("PCATransform has not been fitted.")
        mat, collection = self._ensure_matrix(y) if self.multivariate else (y, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError(
                "PCATransform inverse_transform expects TimeSeriesMatrix input."
            )
        rec = pca_inverse_transform(self.model, mat)
        return _restore_collection(rec, collection if self.multivariate else None)


class ICATransform(Transform):
    """
    ICA wrapper using existing decomposition helpers.
    """

    supports_inverse = True

    def __init__(
        self,
        n_components: int | None = None,
        *,
        multivariate: bool = True,
        align: str = "intersection",
        **kwargs,
    ):
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.multivariate = multivariate
        self.align = align
        self.model = None
        self._collection = None

    def _ensure_matrix(self, x):
        if isinstance(x, TimeSeriesMatrix):
            return x, None
        if _is_collection(x):
            mat, original = _to_matrix_from_collection(x, align=self.align)
            return mat, original
        raise TypeError(
            "ICATransform expects TimeSeriesMatrix or collection convertible to matrix."
        )

    def fit(self, x):
        """Fit ICA model for the input data."""
        mat, collection = self._ensure_matrix(x) if self.multivariate else (x, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError(
                "ICATransform requires TimeSeriesMatrix when multivariate=False."
            )
        self.model = ica_fit(mat, n_components=self.n_components, **self.kwargs)
        self._collection = collection
        return self

    def transform(self, x):
        """Apply ICA transformation (project to sources)."""
        if self.model is None:
            self.fit(x)
        mat, collection = self._ensure_matrix(x) if self.multivariate else (x, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError("ICATransform requires TimeSeriesMatrix input.")
        sources = ica_transform(self.model, mat)
        return _restore_collection(sources, collection if self.multivariate else None)

    def inverse_transform(self, y):
        """Reverse ICA transformation (reconstruct from sources)."""
        if self.model is None:
            raise ValueError("ICATransform has not been fitted.")
        mat, collection = self._ensure_matrix(y) if self.multivariate else (y, None)
        if not isinstance(mat, TimeSeriesMatrix):
            raise TypeError(
                "ICATransform inverse_transform expects TimeSeriesMatrix input."
            )
        rec = ica_inverse_transform(self.model, mat)
        return _restore_collection(rec, collection if self.multivariate else None)
