from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.numerics.scaling import _resolve_epsilon

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
    """Base interface for reusable preprocessing transforms.

    Subclasses implement :meth:`transform` and optionally :meth:`fit` and
    :meth:`inverse_transform`. The interface accepts individual
    `TimeSeries` objects as well as GWexpy collections such as
    `TimeSeriesMatrix`, `TimeSeriesDict`, and `TimeSeriesList`.

    Examples
    --------
    >>> transform = ImputeTransform(method="interpolate")
    >>> filled = transform.fit_transform(ts)
    >>> filled.shape == ts.shape
    True

    """

    supports_inverse = False

    def fit(self, x):
        """Fit the transform to the input data and return `self`."""
        return self

    def transform(self, x):
        """Apply the transform to input data.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def fit_transform(self, x):
        """Fit the transform and immediately apply it to `x`."""
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        """Reverse the transform when supported by the subclass."""
        raise NotImplementedError(
            "inverse_transform is not implemented for this transform"
        )


class Pipeline:
    """Compose multiple transforms into a deterministic preprocessing chain.

    The pipeline applies each step in order during :meth:`fit` and
    :meth:`transform`, mirroring common ML preprocessing workflows while
    preserving GWexpy metadata and collection types.

    Parameters
    ----------
    steps : sequence of (str, Transform)
        Ordered `(name, transform)` pairs to execute.

    Examples
    --------
    >>> pipeline = Pipeline(
    ...     [
    ...         ("impute", ImputeTransform(method="interpolate")),
    ...         ("standardize", StandardizeTransform()),
    ...     ]
    ... )
    >>> standardized = pipeline.fit_transform(ts_matrix)
    >>> standardized.shape == ts_matrix.shape
    True

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
        """Fit each step in order using the output of the previous step."""
        data = x
        for _, step in self.steps:
            data = step.fit_transform(data)
        self._is_fitted = True
        return self

    def transform(self, x):
        """Apply every fitted transform in sequence."""
        data = x
        for _, step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, x):
        """Fit the pipeline and return the transformed output."""
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y, *, strict: bool = True):
        """Apply supported inverse transforms in reverse order.

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
    """Impute missing values using existing lower-level helpers."""

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
            out = x.__class__()
            for v in x:
                out.append(self.transform(v))
            return out
        raise TypeError(f"Unsupported type for ImputeTransform: {type(x)}")


class StandardizeTransform(Transform):
    """Standardize TimeSeries or matrix objects with optional robust scaling."""

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
        if self.method not in {"zscore", "robust"}:
            raise ValueError(f"Unknown standardization method: {self.method}")
        if axis not in {"time", "channel"}:
            raise ValueError(f"Unknown standardization axis: {axis}")
        self.ddof = ddof
        self.axis = axis
        self.multivariate = multivariate
        self.align = align
        self.params: dict[str, Any] | None = None

    def _compute_stats_ts(self, ts: TimeSeries):
        dtype = np.complex128 if np.iscomplexobj(ts.value) else np.float64
        values = np.asarray(ts.value, dtype=dtype)
        if self.method == "robust":
            center = self._stable_median(values)
            scale = 1.4826 * np.nanmedian(np.abs(values - center))
        else:
            center = self._stable_mean(values)
            scale = self._stable_std(values)
        center, scale = self._checked_stats(values, center, scale)
        return center.item(), float(scale)

    def _stable_mean(self, values, *, axis=None):
        """Compute a finite mean without overflowing the raw sum."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            magnitude = np.nanmax(
                self._safe_magnitude(values), axis=axis, keepdims=axis is not None
            )
            divisor = np.where(magnitude == 0, 1.0, magnitude)
            mean = np.nanmean(values / divisor, axis=axis, keepdims=axis is not None)
        with np.errstate(invalid="ignore"):
            return mean * divisor

    def _stable_median(self, values, *, axis=None):
        """Find a median without overflowing the average of middle samples."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            magnitude = np.nanmax(
                self._safe_magnitude(values), axis=axis, keepdims=axis is not None
            )
            divisor = np.where(magnitude == 0, 1.0, magnitude)
            normalized = values / divisor
            median = np.nanmedian(normalized, axis=axis, keepdims=axis is not None)
            finite = np.where(np.isfinite(values), values, np.nan)
            minimum = np.nanmin(finite, axis=axis, keepdims=axis is not None)
            maximum = np.nanmax(finite, axis=axis, keepdims=axis is not None)
        constant = np.isfinite(minimum) & (minimum == maximum)
        with np.errstate(invalid="ignore"):
            return np.where(constant, minimum, median * divisor)

    def _stable_std(self, values, *, axis=None):
        """Compute standard deviation without squaring tiny raw magnitudes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            magnitude = np.nanmax(
                self._safe_magnitude(values), axis=axis, keepdims=axis is not None
            )
            divisor = np.where(magnitude == 0, 1.0, magnitude)
            normalized = values / divisor
            scaled_std = np.nanstd(
                normalized,
                axis=axis,
                ddof=self.ddof,
                keepdims=axis is not None,
            )
        with np.errstate(invalid="ignore"):
            return scaled_std * divisor

    @staticmethod
    def _safe_magnitude(values):
        """Bound a complex value without overflowing its Euclidean magnitude."""
        if np.iscomplexobj(values):
            return np.maximum(np.abs(values.real), np.abs(values.imag))
        return np.abs(values)

    def _checked_stats(self, values, center, scale, *, axis=None):
        """Center exact constants precisely and reject invalid scales."""
        if not np.all(np.isfinite(scale)):
            raise ValueError("Standardization scale must be finite")
        finite = np.where(np.isfinite(values), values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            minimum = np.nanmin(finite, axis=axis, keepdims=axis is not None)
            maximum = np.nanmax(finite, axis=axis, keepdims=axis is not None)
        constant = np.isfinite(minimum) & np.isfinite(maximum) & (minimum == maximum)
        if np.any(constant):
            warnings.warn(
                "Finite constant channel standardized to zero",
                UserWarning,
                stacklevel=3,
            )
        if np.any((scale == 0) & ~constant):
            raise ValueError("Nonconstant channel has zero standardization scale (MAD)")
        return np.where(constant, minimum, center), np.where(constant, 1.0, scale)

    def _matrix_values(self, mat: TimeSeriesMatrix, target_unit=None):
        dtype = np.complex128 if np.iscomplexobj(mat.value) else np.float64
        values = np.asarray(mat.value, dtype=dtype).copy()
        if self.axis == "channel":
            if target_unit is None:
                target_unit = u.Unit(mat.units.flat[0])
            for index in np.ndindex(mat.units.shape):
                try:
                    values[index] *= u.Unit(mat.units[index]).to(target_unit)
                except u.UnitConversionError as exc:
                    raise ValueError(
                        "Incompatible channel units for pooled standardization"
                    ) from exc
        return values

    def _compute_stats_matrix(self, mat: TimeSeriesMatrix, target_unit=None):
        val = self._matrix_values(mat, target_unit)
        np_axis = -1 if self.axis == "time" else (0, 1)
        if self.method == "robust":
            center = self._stable_median(val, axis=np_axis)
            scale = 1.4826 * np.nanmedian(
                np.abs(val - center), axis=np_axis, keepdims=True
            )
        else:
            center = self._stable_mean(val, axis=np_axis)
            scale = self._stable_std(val, axis=np_axis)
        center, scale = self._checked_stats(val, center, scale, axis=np_axis)
        return center, scale

    def _collection_matrix(self, collection):
        """Align a collection while retaining each channel's fit-time unit."""
        originals = (
            list(collection.values())
            if isinstance(collection, TimeSeriesDict)
            else list(collection)
        )
        if self.axis != "channel":
            aligned = collection.to_matrix(align=self.align)
            aligned.meta.units = np.array([[ts.unit] for ts in originals], dtype=object)
            return aligned
        target_unit = originals[0].unit
        converted = []
        for ts in originals:
            try:
                converted.append(ts.to(target_unit))
            except u.UnitConversionError as exc:
                raise ValueError(
                    "Incompatible channel units for pooled standardization"
                ) from exc
        aligned = TimeSeriesList(converted).to_matrix(align=self.align)
        aligned.meta.units = np.full((len(originals), 1), target_unit, dtype=object)
        if isinstance(collection, TimeSeriesDict):
            aligned.channel_names = list(collection)
        return aligned

    def fit(self, x):
        """Fit standardization parameters (center, scale) for the input data."""
        original = None
        if self.multivariate and isinstance(x, TimeSeriesDict):
            original = x.__class__({k: copy.deepcopy(v) for k, v in x.items()})
        elif self.multivariate and isinstance(x, TimeSeriesList):
            original = x.__class__([copy.deepcopy(v) for v in x])
        data = self._collection_matrix(x) if original is not None else x
        if isinstance(data, TimeSeries):
            center, scale = self._compute_stats_ts(data)
            self.params = {"type": "ts", "center": center, "scale": scale, "meta": data}
        elif isinstance(data, TimeSeriesMatrix):
            pooled_unit = data.units.flat[0] if self.axis == "channel" else None
            center, scale = self._compute_stats_matrix(data, pooled_unit)
            self.params = {
                "type": "matrix",
                "center": center,
                "scale": scale,
                "axis": self.axis,
                "meta": data,
                "collection": original,
                "pooled_unit": pooled_unit,
                "units": data.units.copy(),
            }
        elif isinstance(data, TimeSeriesDict):
            stats = {k: self._compute_stats_ts(v) for k, v in data.items()}
            self.params = {"type": "dict", "stats": stats, "meta": copy.deepcopy(data)}
        elif isinstance(data, TimeSeriesList):
            stats_list = [self._compute_stats_ts(v) for v in data]
            self.params = {
                "type": "list",
                "stats": stats_list,
                "meta": copy.deepcopy(data),
            }
        else:
            raise TypeError(f"Unsupported type for StandardizeTransform: {type(data)}")
        return self

    def _apply_ts(self, ts: TimeSeries, center: float, scale: float):
        dtype = np.complex128 if np.iscomplexobj(ts.value) else np.float64
        new_ts = ts.astype(dtype) / ts.unit
        new_ts.value[:] = (ts.value - center) / scale
        return new_ts

    def _apply_matrix(self, mat: TimeSeriesMatrix, center, scale, pooled_unit=None):
        val = (self._matrix_values(mat, pooled_unit) - center) / scale
        new_mat = mat.astype(val.dtype)
        new_mat.value[:] = val
        new_mat.meta.units = np.full(
            mat.units.shape, u.dimensionless_unscaled, dtype=object
        )
        return new_mat

    def transform(self, x):
        """Apply standardization using fitted parameters."""
        if self.params is None:
            self.fit(x)

        assert self.params is not None
        params = copy.deepcopy(self.params)
        data = (
            self._collection_matrix(x) if self.multivariate and _is_collection(x) else x
        )

        if params["type"] == "ts" and isinstance(data, TimeSeries):
            return self._apply_ts(data, params["center"], params["scale"])
        if params["type"] == "matrix" and isinstance(data, TimeSeriesMatrix):
            return self._apply_matrix(
                data, params["center"], params["scale"], params["pooled_unit"]
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
            dtype = np.complex128 if np.iscomplexobj(data.value) else np.float64
            new_ts = data.astype(dtype) * params["meta"].unit
            new_ts.value[:] = data.value * scale + center
            return new_ts
        if params["type"] == "matrix" and isinstance(data, TimeSeriesMatrix):
            center, scale = params["center"], params["scale"]
            original = params.get("collection")
            row_by_key = None
            if isinstance(original, TimeSeriesDict):
                labels = list(data.channel_names)
                keys = list(original)
                if len(labels) != len(keys) or set(labels) != set(keys):
                    unknown = set(labels) - set(keys)
                    raise KeyError(
                        f"Key '{next(iter(unknown), '<missing>')}' not present in fitted StandardizeTransform stats."
                    )
                row_by_key = {key: index for index, key in enumerate(labels)}
                if center.shape[0] == len(keys):
                    fitted_row = {key: index for index, key in enumerate(keys)}
                    order = [fitted_row[key] for key in labels]
                    center = center[order]
                    scale = scale[order]
            val = (
                np.asarray(
                    data.value,
                    dtype=np.complex128 if np.iscomplexobj(data.value) else np.float64,
                )
                * scale
                + center
            )
            new_mat = data.astype(val.dtype)
            new_mat.value[:] = val
            if original is not None:
                sources = (
                    list(original.values())
                    if isinstance(original, TimeSeriesDict)
                    else list(original)
                )
                if data.shape[0] > len(sources):
                    if isinstance(original, TimeSeriesDict):
                        raise KeyError(
                            "Unknown key in inverse StandardizeTransform matrix"
                        )
                    raise IndexError("No fitted stats for extra matrix channel")
                out = original.__class__()
                for index, source in enumerate(sources):
                    row = (
                        row_by_key[list(original)[index]]
                        if row_by_key is not None
                        else index
                    )
                    values = val[row, 0]
                    if params["pooled_unit"] is not None:
                        values = u.Quantity(values, params["pooled_unit"]).to_value(
                            source.unit
                        )
                    restored = source.copy()
                    source_times = source.times.to_value(data.times.unit)
                    aligned_times = data.times.value
                    tolerance = max(1e-9, float(np.max(np.spacing(source_times))) * 4)
                    if len(source_times) == len(aligned_times):
                        if not np.allclose(
                            source_times, aligned_times, rtol=0, atol=tolerance
                        ):
                            restored.times = data.times
                        restored.value[:] = values
                    else:
                        positions = np.searchsorted(source_times, aligned_times)
                        valid = positions < len(source_times)
                        matched = np.zeros(len(positions), dtype=bool)
                        matched[valid] = np.isclose(
                            source_times[positions[valid]],
                            aligned_times[valid],
                            rtol=0,
                            atol=tolerance,
                        )
                        if np.all(matched):
                            restored.value[positions] = values
                        else:
                            if len(source) >= len(values):
                                restored = source[: len(values)].copy()
                            else:
                                restored = TimeSeries(
                                    np.empty(len(values), dtype=val.dtype),
                                    dt=source.dt,
                                    unit=source.unit,
                                    name=source.name,
                                    channel=source.channel,
                                )
                            restored.times = data.times
                            restored.value[:] = values
                    if isinstance(out, TimeSeriesDict):
                        out[list(original)[index]] = restored
                    else:
                        out.append(restored)
                return out
            if params["pooled_unit"] is not None:
                for cell in np.ndindex(new_mat.units.shape):
                    val[cell] = u.Quantity(val[cell], params["pooled_unit"]).to_value(
                        params["units"][cell]
                    )
                new_mat.value[:] = val
            new_mat.meta.units = params["units"].copy()
            return new_mat
        if params["type"] == "dict" and isinstance(data, TimeSeriesDict):
            out = data.__class__()
            for k, ts in data.items():
                if k not in params["stats"]:
                    raise KeyError(
                        f"Key '{k}' not present in fitted StandardizeTransform stats."
                    )
                center, scale = params["stats"][k]
                new_ts = ts * params["meta"][k].unit if "meta" in params else ts.copy()
                new_ts.value[:] = ts.value * scale + center
                out[k] = new_ts
            return out
        if params["type"] == "list" and isinstance(data, TimeSeriesList):
            out_list = data.__class__()
            for idx, ts in enumerate(data):
                if idx >= len(params["stats"]):
                    raise IndexError(f"No fitted stats for index {idx}")
                center, scale = params["stats"][idx]
                new_ts = (
                    ts * params["meta"][idx].unit if "meta" in params else ts.copy()
                )
                new_ts.value[:] = ts.value * scale + center
                out_list.append(new_ts)
            return out_list

        raise TypeError(
            f"Incompatible input for StandardizeTransform.inverse_transform: {type(data)}"
        )


class WhitenTransform(Transform):
    """Whiten TimeSeriesMatrix-like data with PCA or ZCA.

    Whitening operates on the input's numerical values without converting
    feature units. For mixed-unit features, the whitening result therefore
    depends on the numerical unit representation. Inverse transformation
    restores the units recorded at fit time but does not resolve this
    dimensional ambiguity.
    """

    supports_inverse = True

    def __init__(
        self,
        method: Literal["pca", "zca"] = "pca",
        eps: float | Literal["auto"] | None = "auto",
        n_components: int | None = None,
        *,
        multivariate: bool = True,
        align: str = "intersection",
    ):
        super().__init__()
        self.method = method
        _resolve_epsilon(eps)
        self.eps = eps
        self.n_components = n_components
        self.multivariate = multivariate
        self.align = align
        self.model: WhiteningModel | None = None
        self._channel_names: list[str] | None = None
        self._original_collection: TimeSeriesDict | TimeSeriesList | None = None
        self._fit_units: np.ndarray | None = None

    @staticmethod
    def _units_for_input(x) -> np.ndarray | None:
        """Return input feature units in the order used by whitening."""
        if isinstance(x, TimeSeriesMatrix):
            return x.units.copy().reshape(-1)
        if isinstance(x, TimeSeriesDict):
            return np.asarray([series.unit for series in x.values()], dtype=object)
        if isinstance(x, TimeSeriesList):
            return np.asarray([series.unit for series in x], dtype=object)
        if isinstance(x, TimeSeries):
            return np.asarray([x.unit], dtype=object)
        return None

    def _to_matrix(self, x):
        if isinstance(x, TimeSeriesMatrix):
            return x
        if _is_collection(x):
            mat, original = _to_matrix_from_collection(x, align=self.align)
            return mat, original
        if isinstance(x, TimeSeries):
            mat = TimeSeriesMatrix(
                x.value[None, None, :],
                x0=x.x0,
                dt=x.dt,
                xunit=x.xunit,
            )
            return mat, None
        raise TypeError(f"Unsupported type for whitening: {type(x)}")

    def fit(self, x):
        """Fit whitening parameters for the input data."""
        fit_units = self._units_for_input(x)
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
        self._fit_units = fit_units
        return self

    def transform(self, x):
        """Apply whitening transform."""
        if self.model is None:
            self.fit(x)
        assert self.model is not None

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
        if not np.all(np.isfinite(X)):
            raise ValueError("whitening requires finite input data")
        model = self.model
        X_centered = X - model.mean
        X_w = X_centered @ model.W.T
        new_val = X_w.T[:, None, :]
        new_mat = TimeSeriesMatrix(
            new_val, x0=mat_data.x0, dt=mat_data.dt, xunit=mat_data.xunit
        )
        new_mat.meta.units = np.full(
            new_mat.meta.shape, u.dimensionless_unscaled, dtype=object
        )
        new_mat.unit = u.dimensionless_unscaled
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
        new_mat = TimeSeriesMatrix(
            X_rec, x0=mat_data.x0, dt=mat_data.dt, xunit=mat_data.xunit
        )
        if self._fit_units is not None and self._fit_units.size == new_mat.meta.size:
            new_mat.meta.units = self._fit_units.reshape(new_mat.meta.shape)
        return _restore_collection(new_mat, original)


class PCATransform(Transform):
    """Wrap PCA using existing decomposition helpers."""

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
    """Wrap ICA using existing decomposition helpers."""

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
