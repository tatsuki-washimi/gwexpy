from __future__ import annotations

import logging
import pickle
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Self, SupportsIndex, cast

import numpy as np
from astropy import units as u

from gwexpy.types.mixin import InteropMixin, RegularityMixin

from ._stats import StatisticalMethodsMixin
from .metadata import MetaData, MetaDataDict, MetaDataMatrix

if TYPE_CHECKING:
    from gwexpy.types.typing import IndexLike, MetaDataCollectionType, UnitLike
from .series_matrix_analysis import SeriesMatrixAnalysisMixin
from .series_matrix_core import SeriesMatrixCoreMixin
from .series_matrix_indexing import SeriesMatrixIndexingMixin
from .series_matrix_io import SeriesMatrixIOMixin
from .series_matrix_math import SeriesMatrixMathMixin
from .series_matrix_structure import SeriesMatrixStructureMixin
from .series_matrix_validation_mixin import SeriesMatrixValidationMixin
from .series_matrix_visualization import SeriesMatrixVisualizationMixin
from .seriesmatrix_validation import (
    _check_attribute_consistency,
    _check_shape_consistency,
    _fill_missing_attributes,
    _make_meta_matrix,
    _normalize_input,
    build_index_if_needed,
    check_add_sub_compatibility,
    check_shape_xindex_compatibility,
    convert_add_sub_values,
)

logger = logging.getLogger(__name__)

_SERIESMATRIX_PICKLE_METADATA_FIELDS = (
    "_xindex",
    "meta",
    "rows",
    "cols",
    "name",
    "epoch",
    "attrs",
)

# Marker meaning "keep each cell's own unit" for :func:`_copy_meta_cells`.
_UNSET_UNIT_SENTINEL = object()

_COMPARISON_UFUNCS = {
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
    np.greater,
    np.greater_equal,
}

_ADD_SUB_COMPARISON_UFUNCS = {np.add, np.subtract} | _COMPARISON_UFUNCS

# Ufuncs whose second operand is a divisor: a zero anywhere in it is reported
# as ``ZeroDivisionError`` instead of silently producing ``inf``/``nan``.
_DIVISION_UFUNCS = {
    np.divide,
    np.true_divide,
    np.floor_divide,
    np.mod,
    np.remainder,
    np.fmod,
    np.divmod,
}

_LOGICAL_UFUNCS = {
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.logical_not,
    np.isfinite,
    np.isinf,
    np.isnan,
}

# Ufuncs that carry the first operand's per-cell unit through unchanged.
_META_PASSTHROUGH_UFUNCS = _LOGICAL_UFUNCS | {
    np.sign,
    np.floor,
    np.ceil,
    np.trunc,
    np.rint,
    np.mod,
    np.remainder,
}


def _copy_meta_cells(
    meta_matrix: MetaDataMatrix, unit: Any = _UNSET_UNIT_SENTINEL
) -> MetaDataMatrix:
    """Rebuild a metadata matrix with independent per-cell ``MetaData``.

    Every cell keeps its own ``name``/``channel``.  When *unit* is given it
    replaces each cell's unit (used for comparison results, which are
    dimensionless); otherwise the original per-cell unit is preserved.

    Rebuilding rather than calling ``MetaDataMatrix.copy()`` matters because
    ``copy()`` on an object array duplicates only the references, so the copy
    would keep sharing ``MetaData`` instances with the source (issue #577a).
    """
    arr = np.empty(meta_matrix.shape, dtype=object)
    for idx in np.ndindex(meta_matrix.shape):
        src = meta_matrix[idx]
        if not isinstance(src, MetaData):
            # MetaDataMatrix.__new__ guarantees every cell is a MetaData
            # instance; numpy's object-dtype __getitem__ stubs cannot express
            # that invariant, so this check both narrows the type for mypy
            # and guards the invariant at runtime.
            raise TypeError(
                f"MetaDataMatrix cell at {idx} is {type(src).__name__}, "
                "expected MetaData"
            )
        payload = deepcopy(dict(src))
        payload["unit"] = src.unit if unit is _UNSET_UNIT_SENTINEL else unit
        arr[idx] = MetaData(**payload)
    return MetaDataMatrix(
        arr,
        row_keys=deepcopy(getattr(meta_matrix, "row_keys", None)),
        col_keys=deepcopy(getattr(meta_matrix, "col_keys", None)),
    )


def _scalar_power_exponent(operand: Any) -> Any:
    """Return *operand* as a scalar exponent, or ``None`` if it is not one.

    A dimensionless scalar ``Quantity`` is unwrapped to its numeric value; a
    ``Quantity`` carrying a real unit raises, since ``unit ** (1 m)`` has no
    meaning.
    """
    if isinstance(operand, u.Quantity):
        value = np.asarray(operand.value)
        if value.ndim != 0:
            return None
        if not operand.unit.is_equivalent(u.dimensionless_unscaled):
            raise u.UnitConversionError(
                f"exponent must be dimensionless, got {operand.unit}"
            )
        return operand.to_value(u.dimensionless_unscaled).item()
    if isinstance(operand, (bool, np.bool_)):
        # Python/NumPy semantics: True == 1, False == 0.
        return int(operand)
    if isinstance(operand, np.number):
        # Unwrap to a plain Python scalar: MetaDataMatrix.__array_ufunc__'s
        # ``_to_array`` helper does not accept ``np.number`` (only
        # ``int``/``float``/``complex``), so passing a bare ``np.int64``
        # through unwrapped made every ``matrix ** np.int64(n)`` take the
        # vectorized-metadata path's exception+fallback branch -- correct
        # result, but a full traceback log and a ``PerformanceWarning`` on
        # every call.
        return operand.item()
    if isinstance(operand, (int, float, complex)):
        return operand
    if isinstance(operand, np.ndarray) and operand.ndim == 0:
        return operand.item()
    return None


class _UnsetType:
    """Sentinel type for omitted constructor arguments."""


_UNSET = _UnsetType()


def _copy_xindex(xindex: Any) -> Any:
    """Copy an x-axis index without assuming a specific index implementation."""
    if xindex is None:
        return None
    try:
        return xindex.copy()
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        return deepcopy(xindex)


def _copy_metadata_matrix(meta: MetaDataMatrix) -> MetaDataMatrix:
    """Deep-copy a metadata matrix and its per-cell metadata entries."""
    return _copy_meta_cells(meta)


def _copy_metadata_dict(meta: MetaDataDict, key_prefix: str) -> MetaDataDict:
    """Deep-copy row or column metadata while preserving keys."""
    items = OrderedDict()
    for key, value in meta.items():
        items[key] = MetaData(**deepcopy(dict(value)))
    return MetaDataDict(items, expected_size=len(items), key_prefix=key_prefix)


def _pickle_safe_attrs(attrs: Any, protocol: int) -> dict[str, Any]:
    """Return attrs entries that can be serialized by pickle."""
    if not isinstance(attrs, dict):
        return {}
    safe_attrs = {}
    dropped = []
    for key, value in attrs.items():
        try:
            pickle.dumps((key, value), protocol=protocol)
        except (
            pickle.PicklingError,
            TypeError,
            AttributeError,
            ValueError,
            RuntimeError,
            RecursionError,
        ):
            dropped.append(key)
            continue
        safe_attrs[key] = value
    if dropped:
        warnings.warn(
            f"Dropping attrs entries that cannot be pickled: {dropped}",
            stacklevel=2,
        )
    return safe_attrs


class PerformanceWarning(RuntimeWarning):
    """Warning raised when an operation falls back to a slower implementation."""

    pass


class SeriesMatrix(  # type: ignore[misc]
    RegularityMixin,
    InteropMixin,
    SeriesMatrixCoreMixin,
    SeriesMatrixIndexingMixin,
    SeriesMatrixIOMixin,
    SeriesMatrixMathMixin,
    SeriesMatrixAnalysisMixin,
    SeriesMatrixStructureMixin,
    SeriesMatrixVisualizationMixin,
    SeriesMatrixValidationMixin,
    StatisticalMethodsMixin,
    np.ndarray,
):
    """N-dimensional matrix of aligned series values with per-cell metadata.

    Arithmetic contract
    -------------------
    ``SeriesMatrix`` sets ``__array_ufunc__ = None``, so NumPy ufuncs never
    operate on it directly.  Everything users need is exposed through explicit
    operators (see
    :class:`~gwexpy.types.series_matrix_math.SeriesMatrixMathMixin`):
    ``+ - * / // % ** @``, their reflected and in-place forms, the six
    comparisons and unary ``+ - abs()``.  In exchange, an expression such as
    ``(2 * u.s) * matrix`` keeps the matrix type, its per-cell units and all
    axis metadata instead of collapsing to a bare ``Quantity`` (issue #575).

    Applying a ufunc directly -- ``np.sqrt(matrix)``, ``np.add.reduce(matrix)``
    -- raises ``TypeError`` rather than silently discarding metadata.  Operate
    on ``matrix.value`` when a raw NumPy result is what you want; for example
    ``np.isfinite(matrix.value)`` is the supported way to get a finiteness
    mask (as a plain boolean `numpy.ndarray`, without axis metadata).

    ``%``, ``//`` and ``divmod()`` are supported only between dimensionless
    operands: NumPy's remainder/floor-divide do not convert units, so
    applying them to a unit-bearing matrix would silently ignore a unit
    mismatch rather than convert or fail. ``divmod()`` is not implemented at
    all and always raises ``TypeError``, even for dimensionless operands.
    Full unit-aware floor-division and remainder are deferred to the v0.2.0
    semantic-contract redesign (issue #637).
    """

    def __new__(
        cls,
        data: Any = None,
        *,
        meta: MetaDataMatrix | np.ndarray | list | None = None,
        unit: object | None = None,
        units: np.ndarray | object | None = None,
        names: np.ndarray | None = None,
        channels: np.ndarray | None = None,
        rows: MetaDataCollectionType = None,
        cols: MetaDataCollectionType = None,
        shape: tuple[int, ...] | None = None,
        xindex: IndexLike | None = None,
        dx: u.Quantity | None = None,
        x0: u.Quantity | None = None,
        xunit: UnitLike = None,
        name: str | _UnsetType = _UNSET,
        epoch: float | _UnsetType = _UNSET,
        attrs: dict[str, Any] | None | _UnsetType = _UNSET,
    ) -> SeriesMatrix:
        """Create a SeriesMatrix with normalized inputs and metadata."""
        if unit is not None:
            if units is not None:
                raise ValueError("give only one of unit or units")
            units = unit

        source_matrix = data if isinstance(data, SeriesMatrix) else None
        explicit_units = units is not None
        explicit_names = names is not None
        explicit_channels = channels is not None

        if xindex is not None and xunit is not None:
            try:
                xindex = u.Quantity(xindex, xunit)
            except (TypeError, ValueError, AttributeError) as e:
                # xunit implies the caller intended a unit conversion; warn and continue
                # so that downstream code still has the raw xindex to work with.
                logger.warning("SeriesMatrix xindex conversion failed: %s", e)

        value_array, data_attrs, detected_xindex = _normalize_input(
            data=data,
            units=units,
            names=names,
            channels=channels,
            shape=shape,
            xindex=xindex,
            dx=dx,
            x0=x0,
            xunit=xunit,
        )

        if meta is None and source_matrix is not None:
            meta_matrix = _copy_metadata_matrix(source_matrix.meta)
            if explicit_units:
                meta_matrix.units = data_attrs["unit"]
            if explicit_names:
                meta_matrix.names = data_attrs["name"]
            if explicit_channels:
                meta_matrix.channels = data_attrs["channel"]
        else:
            if meta is not None:
                if not isinstance(meta, MetaDataMatrix):
                    try:
                        meta = MetaDataMatrix(meta)
                    except (TypeError, ValueError) as e:
                        raise TypeError(
                            "meta must be a MetaDataMatrix or a 2D array-like of MetaData/dict"
                        ) from e
                if units is None:
                    data_attrs["unit"] = None
                if names is None:
                    data_attrs["name"] = None
                if channels is None:
                    data_attrs["channel"] = None
                _check_attribute_consistency(data_attrs=data_attrs, meta=meta)
                units_arr, names_arr, channels_arr = _fill_missing_attributes(
                    data_attrs=data_attrs, meta=meta
                )
            else:
                units_arr = data_attrs.get("unit", None)
                names_arr = data_attrs.get("name", None)
                channels_arr = data_attrs.get("channel", None)

            if meta is not None:
                meta_matrix = _copy_metadata_matrix(meta)
                meta_matrix.units = units_arr
                meta_matrix.names = names_arr
                meta_matrix.channels = channels_arr
            else:
                meta_matrix = _make_meta_matrix(
                    shape=value_array.shape[:2],
                    units=units_arr,
                    names=names_arr,
                    channels=channels_arr,
                )

        if xindex is None:
            if detected_xindex is not None:
                xindex = (
                    _copy_xindex(detected_xindex)
                    if source_matrix is not None
                    else detected_xindex
                )
            else:
                if value_array.shape[2] == 0 and dx is None and x0 is None:
                    xindex = np.asarray([])
                else:
                    xindex = build_index_if_needed(
                        xindex=None,
                        dx=dx,
                        x0=x0,
                        xunit=xunit,
                        length=value_array.shape[2],
                    )

        _check_shape_consistency(
            value_array=value_array,
            meta_matrix=meta_matrix,
            xindex=cast(Optional[np.ndarray], xindex),
        )

        obj = np.asarray(value_array).view(cls)

        obj._value = obj.view(np.ndarray)

        obj.meta = meta_matrix
        N, M = value_array.shape[:2]
        if source_matrix is not None:
            if rows is None and getattr(source_matrix, "rows", None) is not None:
                rows = _copy_metadata_dict(source_matrix.rows, "row")
            if cols is None and getattr(source_matrix, "cols", None) is not None:
                cols = _copy_metadata_dict(source_matrix.cols, "col")
        if isinstance(rows, dict) and not isinstance(rows, OrderedDict):
            rows = OrderedDict(rows)
        if isinstance(cols, dict) and not isinstance(cols, OrderedDict):
            cols = OrderedDict(cols)
        if name is _UNSET:
            name = (
                getattr(source_matrix, "name", "") if source_matrix is not None else ""
            )
        if epoch is _UNSET:
            epoch = (
                getattr(source_matrix, "epoch", 0.0)
                if source_matrix is not None
                else 0.0
            )
        if attrs is _UNSET:
            attrs = (
                deepcopy(getattr(source_matrix, "attrs", {}))
                if source_matrix is not None
                else {}
            )
        elif attrs is None:
            attrs = {}
        obj.rows = MetaDataDict(cast(Any, rows), expected_size=N, key_prefix="row")
        obj.cols = MetaDataDict(cast(Any, cols), expected_size=M, key_prefix="col")
        obj.xindex = xindex
        obj.name = cast(str, name)
        obj.epoch = cast(float, epoch)
        obj.attrs = cast(dict[str, Any], attrs) or {}

        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._value = self.view(np.ndarray)
        self._suppress_xindex_check = True
        self.xindex = getattr(obj, "xindex", None)
        if hasattr(self, "_suppress_xindex_check"):
            delattr(self, "_suppress_xindex_check")
        from typing import cast

        from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix

        self.meta = cast(MetaDataMatrix, getattr(obj, "meta", None))
        self.rows = cast(MetaDataDict, getattr(obj, "rows", None))
        self.cols = cast(MetaDataDict, getattr(obj, "cols", None))
        self.name = getattr(obj, "name", "")
        self.epoch = getattr(obj, "epoch", 0.0)
        # NOTE: attrs is shared by reference here on purpose -- a plain
        # ``.view()`` is a lightweight view and the metadata contract
        # (test_typed_view_preserves_metadata_by_reference) requires
        # ``viewed.attrs is matrix.attrs``.  Operations that produce a
        # genuinely new logical object (math ops, crop/append/diff/pad via
        # _get_meta_for_constructor, and __array_ufunc__) deep-copy attrs at
        # their own construction site instead.
        self.attrs = getattr(obj, "attrs", getattr(self, "attrs", {}))

        # Propagate custom _gwex_ attributes
        for key, val in getattr(obj, "__dict__", {}).items():
            if key.startswith("_gwex_") and key not in self.__dict__:
                self.__dict__[key] = val

    def _component_result(self, values: np.ndarray, *, name: str | None) -> Any:
        """Rebuild a unary component with fully independent public metadata."""
        matrix_cls = cast(Any, self.__class__)
        return matrix_cls(
            values,
            xindex=_copy_xindex(self.xindex),
            meta=_copy_metadata_matrix(self.meta),
            rows=_copy_metadata_dict(self.rows, "row") if self.rows else self.rows,
            cols=_copy_metadata_dict(self.cols, "col") if self.cols else self.cols,
            name=name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    @property
    def real(self) -> Self:
        """Return the real component without aliasing public metadata."""
        return cast(
            Self,
            self._component_result(
                self.view(np.ndarray).real,
                name=f"{self.name}.real" if self.name else "",
            ),
        )

    @real.setter
    def real(self, value: Any) -> None:
        self.view(np.ndarray).real = value

    @property
    def imag(self) -> Self:
        """Return the imaginary component without aliasing public metadata."""
        return cast(
            Self,
            self._component_result(
                self.view(np.ndarray).imag,
                name=f"{self.name}.imag" if self.name else "",
            ),
        )

    @imag.setter
    def imag(self, value: Any) -> None:
        self.view(np.ndarray).imag = value

    def conj(self) -> Self:
        """Return the conjugate with fully independent public metadata."""
        return cast(
            Self,
            self._component_result(
                np.conjugate(self.view(np.ndarray)),
                name=self.name,
            ),
        )

    def copy(self, order: Any = "C") -> Any:
        """Copy values and every public metadata field independently.

        This narrow override retains explicit ``MetaDataMatrix`` row/column
        keys, which the legacy structural mixin predates.
        """
        matrix_cls = cast(Any, self.__class__)
        return matrix_cls(
            np.array(self.view(np.ndarray), copy=True, order=order),
            meta=_copy_metadata_matrix(self.meta),
            rows=_copy_metadata_dict(self.rows, "row"),
            cols=_copy_metadata_dict(self.cols, "col"),
            xindex=_copy_xindex(self.xindex),
            name=self.name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    def astype(
        self,
        dtype: Any,
        order: Any = "K",
        casting: Any = "unsafe",
        subok: Any = True,
        copy: Any = True,
    ) -> Any:
        """Cast values while retaining independent public metadata."""
        values = self.value.astype(
            dtype, order=order, casting=casting, subok=subok, copy=copy
        )
        if not copy and values is self.value:
            return self
        matrix_cls = cast(Any, self.__class__)
        return matrix_cls(
            values,
            meta=_copy_metadata_matrix(self.meta),
            rows=_copy_metadata_dict(self.rows, "row"),
            cols=_copy_metadata_dict(self.cols, "col"),
            xindex=_copy_xindex(self.xindex),
            name=self.name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    def transpose(self, *axes: Any) -> Any:
        """Transpose rows/columns with independent, correctly keyed metadata."""
        if axes:
            return np.transpose(self.value, axes)
        matrix_cls = cast(Any, self.__class__)
        return matrix_cls(
            np.transpose(self.value, (1, 0, 2)),
            meta=MetaDataMatrix(
                deepcopy(np.asarray(self.meta, dtype=object).T),
                row_keys=deepcopy(self.meta.col_keys),
                col_keys=deepcopy(self.meta.row_keys),
            ),
            rows=_copy_metadata_dict(self.cols, "row"),
            cols=_copy_metadata_dict(self.rows, "col"),
            xindex=_copy_xindex(self.xindex),
            name=f"{self.name}.T" if self.name else "",
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    @property
    def T(self) -> SeriesMatrix:
        """Return the B0 row/column transpose."""
        return cast(SeriesMatrix, self.transpose())

    def reshape(
        self,
        *shape: Any,
        order: Any = "C",
        copy: Any = None,
    ) -> Any:
        """Reshape without sharing B0 public state with the source."""
        requested = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else tuple(shape)
        )
        nsamp = self._value.shape[2]
        if len(requested) == 2:
            target_shape = (*requested, nsamp)
        elif len(requested) == 3:
            if requested[2] != nsamp:
                raise ValueError(
                    f"Cannot reshape sample axis: expected {nsamp}, got {requested[2]}"
                )
            target_shape = requested
        else:
            raise ValueError("Reshape target must be 2D or 3D")
        target_shape = cast(
            tuple[int, int, int], tuple(int(size) for size in target_shape)
        )
        values = self._value.reshape(target_shape, order=order)
        if copy is not None:
            values = np.array(values, copy=copy)
        meta_values = deepcopy(
            np.asarray(self.meta, dtype=object).reshape(target_shape[:2], order=order)
        )
        same_layout = target_shape[:2] == self.meta.shape
        matrix_cls = cast(Any, self.__class__)
        return matrix_cls(
            values,
            meta=MetaDataMatrix(
                meta_values,
                row_keys=deepcopy(self.meta.row_keys) if same_layout else None,
                col_keys=deepcopy(self.meta.col_keys) if same_layout else None,
            ),
            rows=_copy_metadata_dict(self.rows, "row") if same_layout else None,
            cols=_copy_metadata_dict(self.cols, "col") if same_layout else None,
            xindex=_copy_xindex(self.xindex),
            name=self.name,
            epoch=self.epoch,
            attrs=deepcopy(self.attrs),
        )

    def _reduce_with_protocol(self, protocol: int) -> tuple[Any, ...]:
        """Include SeriesMatrix metadata in ndarray-subclass pickle state."""
        picked = list(np.ndarray.__reduce__(self))
        matrix_state = {
            key: _pickle_safe_attrs(value, protocol) if key == "attrs" else value
            for key, value in self.__dict__.items()
            if key in _SERIESMATRIX_PICKLE_METADATA_FIELDS
        }
        base_state = picked[2] if isinstance(picked[2], tuple) else ()
        picked[2] = base_state + (matrix_state,)
        return tuple(picked)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Include SeriesMatrix metadata using the active pickle protocol."""
        reducer = type(self).__reduce__
        if reducer is not SeriesMatrix.__reduce__:
            return cast(Any, reducer)(self)
        return self._reduce_with_protocol(protocol.__index__())

    def __reduce__(self) -> tuple[Any, ...]:
        """Include SeriesMatrix metadata for legacy pickle callers."""
        return self._reduce_with_protocol(pickle.DEFAULT_PROTOCOL)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        """Restore SeriesMatrix metadata from pickle state."""
        if state and isinstance(state[-1], dict):
            matrix_state = state[-1]
            ndarray_state = state[:-1]
        else:
            matrix_state = {}
            ndarray_state = state
        super().__setstate__(ndarray_state)
        self.__dict__.update(matrix_state)
        self._value = self.view(np.ndarray)

    # NumPy ufunc opt-out.  ``None`` makes every ufunc refuse SeriesMatrix
    # operands, which is what forces ``Quantity * matrix`` (and any other
    # left operand with its own ``__array_ufunc__``) back through Python's
    # reflected-operator protocol and into the explicit operator suite in
    # :class:`~gwexpy.types.series_matrix_math.SeriesMatrixMathMixin`.
    # Without it, ``Quantity.__array_ufunc__`` wins the NEP 13 dispatch,
    # unwraps the matrix via its public ``.value`` and silently returns a
    # bare ``Quantity`` with the wrong unit (issue #575).
    __array_ufunc__ = None  # type: ignore[assignment]

    def _cast_ufunc_operand(self, operand: Any) -> Any:
        """Broadcast *operand* into a matrix aligned with ``self``.

        Scalars fill every cell; 1-D input broadcasts along the sample axis;
        2-D input broadcasts along the sample axis after being read as one
        value per cell; 3-D input must match the full shape.  Returns
        ``NotImplemented`` for operand types this class cannot interpret.
        """
        if isinstance(operand, SeriesMatrix):
            return operand
        if isinstance(operand, u.Quantity):
            values, unit = np.asarray(operand.value), operand.unit
        elif isinstance(operand, np.ndarray):
            values, unit = np.asarray(operand), None
        elif isinstance(operand, (bool, np.bool_, int, float, complex, np.number)):
            values, unit = np.asarray(operand), u.dimensionless_unscaled
        else:
            return NotImplemented

        shape = self._value.shape
        N, M, K = shape
        if values.ndim == 0:
            broadcast = np.full(shape, values)
        elif values.ndim == 1:
            if values.shape != (K,):
                raise ValueError(
                    f"1D operand must have length N_samples={K}, got {values.shape}"
                )
            broadcast = np.broadcast_to(values.reshape(1, 1, K), shape)
        elif values.ndim == 2:
            if values.shape != (N, M):
                raise ValueError(
                    f"2D operand must have shape (Nrow,Ncol)={(N, M)}, got {values.shape}"
                )
            broadcast = np.broadcast_to(values.reshape(N, M, 1), shape)
        elif values.ndim == 3:
            if values.shape != shape:
                raise ValueError(
                    f"3D operand must have shape {shape}, got {values.shape}"
                )
            broadcast = values
        else:
            raise ValueError(f"operand with ndim={values.ndim} is not supported")

        # Inherit this matrix's per-cell name/channel.  MetaData's ufunc rules
        # take the *first* operand's name, so a synthetic placeholder here
        # would surface as the result's name in every reflected operation
        # (``quantity * matrix``).
        if unit is None:
            unit = u.dimensionless_unscaled
        meta_array = np.empty((N, M), dtype=object)
        for i in range(N):
            for j in range(M):
                source = self.meta[i, j]
                payload = deepcopy(dict(source))
                payload["unit"] = unit
                meta_array[i, j] = MetaData(**payload)
        return self.__class__(
            broadcast,
            xindex=_copy_xindex(self.xindex),
            meta=MetaDataMatrix(
                meta_array,
                row_keys=deepcopy(getattr(self.meta, "row_keys", None)),
                col_keys=deepcopy(getattr(self.meta, "col_keys", None)),
            ),
            shape=shape,
        )

    def _ufunc_meta_operands(
        self, ufunc: Any, casted_inputs: list[Any], inputs: tuple[Any, ...]
    ) -> list[Any]:
        """Return the operands used to derive the result's per-cell metadata.

        Normally this is just each operand's ``MetaDataMatrix``.  ``power`` is
        the exception: the exponent is a *number*, not a unit-carrying
        quantity, so it is passed through unwrapped.  Broadcasting it into a
        dimensionless matrix instead -- as every other operand is -- is what
        made ``matrix ** 2`` fail with ``UnitConversionError`` while
        ``np.square(matrix)`` succeeded (issue #577e).
        """
        meta_matrices = [inp.meta for inp in casted_inputs]
        if ufunc is not np.power or len(inputs) != 2:
            return cast(list[Any], meta_matrices)

        exponent = _scalar_power_exponent(inputs[1])
        if exponent is not None:
            return [meta_matrices[0], exponent]

        raise u.UnitConversionError(
            "power with a non-scalar exponent is not supported for SeriesMatrix"
        )

    def _ufunc_result_meta(
        self,
        ufunc: Any,
        meta_operands: list[Any],
        *,
        bool_result: bool,
        meta_passthrough: bool,
        ufunc_kwargs: dict[str, Any],
    ) -> MetaDataMatrix:
        """Derive the result's metadata matrix for a single ufunc call."""
        base_meta = meta_operands[0]
        if ufunc in _COMPARISON_UFUNCS:
            return _copy_meta_cells(base_meta, u.dimensionless_unscaled)
        if bool_result or meta_passthrough:
            return _copy_meta_cells(base_meta)
        try:
            return MetaDataMatrix(ufunc(*meta_operands, **ufunc_kwargs))
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            if isinstance(e, u.UnitConversionError):
                raise
            logger.exception("MetaData vectorized ufunc failed; falling back to loop.")
            warnings.warn(
                f"MetaData vectorized ufunc failed; falling back to loop. Error: {e}",
                PerformanceWarning,
                stacklevel=2,
            )
            result_meta = np.empty(base_meta.shape, dtype=object)
            for idx in np.ndindex(base_meta.shape):
                meta_args = [
                    operand[idx] if isinstance(operand, MetaDataMatrix) else operand
                    for operand in meta_operands
                ]
                result_meta[idx] = ufunc(*meta_args, **ufunc_kwargs)
            return MetaDataMatrix(result_meta)

    def _ufunc_dispatch(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Apply *ufunc* to matrix operands, propagating per-cell metadata.

        This is the implementation that used to be exposed as
        ``__array_ufunc__``.  Since ``__array_ufunc__`` is now ``None`` NumPy
        never calls it; the explicit operators in
        :class:`~gwexpy.types.series_matrix_math.SeriesMatrixMathMixin` do.

        Only ``method="__call__"`` is meaningful here.  ``reduce``,
        ``accumulate``, ``reduceat``, ``outer`` and ``at`` used to be delegated
        to the bare ndarray, which silently discarded all metadata (issue
        #577d); they are now rejected outright, and the reduction-shaped public
        methods (``sum``/``prod``/``cumsum``/``any``/``all``) are provided as
        documented metadata-free overrides instead.

        ``out=`` and ``where=`` are rejected rather than silently dropped
        (issue #577c); use an in-place operator such as ``matrix *= other`` to
        write into an existing matrix.
        """
        if method != "__call__":
            raise TypeError(
                f"SeriesMatrix does not support ufunc method {method!r}; "
                "only '__call__' propagates per-cell metadata"
            )
        if kwargs.get("out") is not None:
            raise TypeError(
                "SeriesMatrix does not support the 'out' argument; "
                "use an in-place operator (e.g. 'a *= b') instead"
            )
        if kwargs.get("where", True) is not True:
            raise TypeError("SeriesMatrix does not support the 'where' argument")
        ufunc_kwargs = {k: v for k, v in kwargs.items() if k not in ("out", "where")}

        # Reject every non-scalar exponent before operand broadcasting or
        # dtype coercion.  Lists and tuples otherwise take a generic
        # ``NotImplemented`` path and leak a TypeError instead of the unit
        # contract's atomic UnitConversionError.
        if ufunc is np.power and len(inputs) == 2:
            # A bare Unit has no exponent value and retains the established
            # explicit TypeError path below.  Every value-bearing non-scalar
            # exponent is rejected here before casting or broadcasting.
            if (
                not isinstance(inputs[1], u.UnitBase)
                and _scalar_power_exponent(inputs[1]) is None
            ):
                raise u.UnitConversionError(
                    "power with a non-scalar exponent is not supported for SeriesMatrix"
                )

        casted_inputs = []
        for inp in inputs:
            casted = self._cast_ufunc_operand(inp)
            if casted is NotImplemented:
                return NotImplemented
            casted_inputs.append(casted)

        check_shape_xindex_compatibility(*casted_inputs)

        if ufunc in _ADD_SUB_COMPARISON_UFUNCS:
            check_add_sub_compatibility(*casted_inputs)
            value_arrays = convert_add_sub_values(casted_inputs)
        else:
            value_arrays = [inp.view(np.ndarray) for inp in casted_inputs]

        if ufunc in _DIVISION_UFUNCS and len(value_arrays) == 2:
            if np.any(value_arrays[1] == 0):
                raise ZeroDivisionError(
                    f"{ufunc.__name__} by zero in SeriesMatrix operation"
                )

        result_values = self._ufunc_values(ufunc, value_arrays, ufunc_kwargs)
        bool_result = bool(np.issubdtype(result_values.dtype, np.bool_))
        meta_passthrough = ufunc in _META_PASSTHROUGH_UFUNCS or getattr(
            ufunc, "__name__", None
        ) in {"clip"}

        result_meta_matrix = self._ufunc_result_meta(
            ufunc,
            self._ufunc_meta_operands(ufunc, casted_inputs, inputs),
            bool_result=bool_result,
            meta_passthrough=meta_passthrough,
            ufunc_kwargs=ufunc_kwargs,
        )
        rows = self.rows
        cols = self.cols
        return self.__class__(
            result_values,
            xindex=_copy_xindex(self.xindex),
            meta=result_meta_matrix,
            units=result_meta_matrix.units,
            rows=_copy_metadata_dict(rows, "row") if rows else rows,
            cols=_copy_metadata_dict(cols, "col") if cols else cols,
            name=getattr(self, "name", ""),
            epoch=getattr(self, "epoch", 0.0),
            # Deep-copy so the result does not alias the source's attrs dict
            # (the #442 metadata-sharing defect).
            attrs=deepcopy(getattr(self, "attrs", {})),
        )

    def _ufunc_values(
        self, ufunc: Any, value_arrays: list[np.ndarray], ufunc_kwargs: dict[str, Any]
    ) -> np.ndarray:
        """Evaluate *ufunc* on raw value arrays, looping only if forced to."""
        try:
            return np.asarray(ufunc(*value_arrays, **ufunc_kwargs))
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            warnings.warn(
                f"ufunc {ufunc.__name__} failed vectorized execution; "
                f"falling back to loop. Error: {e}",
                PerformanceWarning,
                stacklevel=2,
            )
        N, M = self._value.shape[:2]
        # Collect first, then let NumPy infer the dtype: pre-allocating with the
        # input dtype would quietly coerce a boolean comparison result back to
        # float.
        cells = [
            [ufunc(*[v[i, j] for v in value_arrays], **ufunc_kwargs) for j in range(M)]
            for i in range(N)
        ]
        return np.asarray(cells)
