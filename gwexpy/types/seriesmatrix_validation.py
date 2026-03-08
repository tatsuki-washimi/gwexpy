from __future__ import annotations

from collections import OrderedDict
from typing import cast

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from astropy import units as u
from gwpy.types.array import Array
from gwpy.types.index import Index
from gwpy.types.series import Series

from .metadata import MetaData, MetaDataDict, MetaDataMatrix


# --- Common utilities ---
def to_series(val, xindex, name="s", epoch=0.0):
    if isinstance(val, Series):
        return val
    elif isinstance(val, Array):
        return Series(
            val.value,
            xindex=xindex,
            unit=val.unit,
            name=val.name,
            channel=val.channel,
            epoch=val.epoch,
        )
    elif isinstance(val, u.Quantity):
        if np.isscalar(val.value):
            if xindex is None:
                raise ValueError(
                    "Cannot create Series from scalar Quantity without xindex"
                )
            return Series(
                np.full(len(xindex), val.value), xindex=xindex, unit=val.unit, name=name
            )
        else:
            return Series(val.value, xindex=xindex, unit=val.unit, name=name)
    elif isinstance(val, np.ndarray) and val.ndim == 1:
        return Series(val, xindex=xindex, name=name)
    elif np.isscalar(val):
        if xindex is None:
            raise ValueError("Cannot create Series from scalar without xindex")
        return Series(np.full(len(xindex), val), xindex=xindex, name=name)
    else:
        raise TypeError(f"Unsupported element type: {type(val)}")


def infer_xindex_from_items(items):
    """Try to extract a usable xindex from a list of Series-like objects."""
    for item in items:
        if isinstance(item, Series) and item.xindex is not None:
            return item.xindex
    return None


def build_index_if_needed(xindex, dx, x0, xunit, length):
    """Create a gwpy Index if not explicitly provided."""
    if xindex is not None:
        return xindex
    if dx is not None and x0 is not None:
        _xunit = (
            u.Unit(xunit)
            if xunit
            else (dx.unit if isinstance(dx, u.Quantity) else u.dimensionless_unscaled)
        )
        _dx = dx.to_value(_xunit) if isinstance(dx, u.Quantity) else dx
        _x0 = x0.to_value(_xunit) if isinstance(x0, u.Quantity) else x0

        # Create Index directly with scalar values to minimize Quantity object lifetime
        try:
            return Index.define(_x0, _dx, length, unit=_xunit)
        except TypeError:
            # Fallback: older versions may not support unit parameter
            start = u.Quantity(_x0, _xunit)
            step = u.Quantity(_dx, _xunit)
            return Index.define(start, step, length)
    raise ValueError("xindex or (x0, dx) must be specified")


def check_add_sub_compatibility(*seriesmatrices):
    """Validate unit equality across SeriesMatrix (or MetaDataMatrix) operands."""
    n_matrices = len(seriesmatrices)
    shape = seriesmatrices[0].shape
    for sm in seriesmatrices:
        if sm.shape != shape:
            raise ValueError(f"Shape mismatch: {shape} vs {sm.shape}")
    for i in range(shape[0]):
        for j in range(shape[1]):
            u0 = seriesmatrices[0].meta[i, j].unit
            for k in range(1, n_matrices):
                uk = seriesmatrices[k].meta[i, j].unit
                if u0 != uk:
                    raise u.UnitConversionError(
                        f"Unit mismatch at cell ({i},{j}): {u0} vs {uk}"
                    )
    return True


def check_shape_xindex_compatibility(*seriesmatrices):
    """Ensure shape and xindex match across SeriesMatrix inputs."""
    shape = seriesmatrices[0].shape
    xindex = seriesmatrices[0].xindex
    xunit = getattr(xindex, "unit", None)

    def _to_first_unit(arr):
        if xunit is None:
            return arr
        try:
            return arr.to_value(xunit)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return arr

    for sm in seriesmatrices:
        if sm.shape != shape:
            raise ValueError(f"Shape mismatch: {shape} vs {sm.shape}")
        if hasattr(sm, "xindex"):
            if isinstance(xindex, (np.ndarray, list)) or hasattr(xindex, "__array__"):
                lhs = _to_first_unit(sm.xindex)
                rhs = _to_first_unit(xindex)
                try:
                    equal = np.array_equal(lhs, rhs)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    equal = False
                if not equal:
                    raise ValueError("xindex mismatch (array content not equal)")
            else:
                if sm.xindex != xindex:
                    raise ValueError("xindex mismatch")
    return True


def check_unit_dimension_compatibility(*seriesmatrices, expected_dim=None):
    """Check physical dimension consistency across all cells."""
    shape = seriesmatrices[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            dims = [sm.meta[i, j].unit.physical_type for sm in seriesmatrices]
            if expected_dim is not None and not all(d == expected_dim for d in dims):
                raise u.UnitConversionError(f"Dimension mismatch at ({i},{j}): {dims}")
            if len(set(dims)) > 1:
                raise u.UnitConversionError(f"Dimension mismatch at ({i},{j}): {dims}")
    return True


def check_xindex_monotonic(seriesmatrix):
    """Validate that xindex is strictly monotonic (increasing or decreasing)."""
    xindex = seriesmatrix.xindex
    arr = np.array(xindex)
    if not (np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)):
        raise ValueError("xindex is not monotonic")
    return True


def check_labels_unique(seriesmatrix):
    """Validate that row/col/channel labels are unique."""
    if len(set(seriesmatrix.row_keys())) != len(seriesmatrix.row_keys()):
        raise ValueError("Duplicate row labels found.")
    if len(set(seriesmatrix.col_keys())) != len(seriesmatrix.col_keys()):
        raise ValueError("Duplicate col labels found.")
    chans = [meta.channel for meta in seriesmatrix.meta.flatten()]
    if None not in chans and len(set(chans)) != len(chans):
        raise ValueError("Duplicate channel labels found.")
    return True


def check_no_nan_inf(seriesmatrix):
    """Raise if the value array contains NaN or Inf."""
    if np.isnan(seriesmatrix.value).any():
        raise ValueError("SeriesMatrix contains NaN values")
    if np.isinf(seriesmatrix.value).any():
        raise ValueError("SeriesMatrix contains Inf values")
    return True


def check_epoch_and_sampling(seriesmatrix1, seriesmatrix2):
    if hasattr(seriesmatrix1, "epoch") and hasattr(seriesmatrix2, "epoch"):
        if seriesmatrix1.epoch != seriesmatrix2.epoch:
            raise ValueError("Epoch mismatch")
    if hasattr(seriesmatrix1, "dx") and hasattr(seriesmatrix2, "dx"):
        if seriesmatrix1.dx != seriesmatrix2.dx:
            raise ValueError("Sampling step dx mismatch")
    return True


def _broadcast_attr(attr, shape2d, name):
    """
    Broadcast user-supplied units/names/channels to the expected 2D shape.
    """
    if attr is None:
        return None
    arr = np.asarray(attr, dtype=object)
    try:
        return np.broadcast_to(arr, shape2d).copy()
    except ValueError as e:
        raise ValueError(
            f"{name} shape mismatch: expected broadcastable to {shape2d}, got {np.shape(attr)}"
        ) from e


def _expand_key(key, ndim):
    """
    Expand indexing keys with ellipsis to full ndim elements and pad with slice(None).
    """
    if not isinstance(key, tuple):
        key = (key,)
    key_list = list(key)
    if Ellipsis in key_list:
        ell_idx = key_list.index(Ellipsis)
        n_missing = ndim - (len(key_list) - 1)
        key_list = (
            key_list[:ell_idx] + [slice(None)] * n_missing + key_list[ell_idx + 1 :]
        )
    while len(key_list) < ndim:
        key_list.append(slice(None))
    return tuple(key_list[:ndim])


def _slice_metadata_dict(meta_dict, key, prefix):
    """
    Slice MetaDataDict to align with ndarray slicing on rows/cols.
    """
    items = list(meta_dict.items())
    if isinstance(key, slice):
        subset = items[key]
    elif isinstance(key, (list, np.ndarray, tuple)):
        subset = [items[int(i)] for i in key]
    elif isinstance(key, (int, np.integer)):
        subset = [items[int(key)]]
    else:
        # Fallback: return original (no slicing)
        return meta_dict
    return MetaDataDict(
        OrderedDict(subset), expected_size=len(subset), key_prefix=prefix
    )


# ---------------------------------------------------------------------------
# Shared utilities for _normalize_input handlers
# ---------------------------------------------------------------------------


def _make_attr_arrays(units, names, channels, shape2d, unit_default=None,
                      name_default=None, channel_default=None):
    """Build unit/name/channel 2-D arrays by broadcasting or filling defaults.

    Parameters
    ----------
    units, names, channels : user-supplied overrides (or None)
    shape2d : (N, M) tuple
    unit_default, name_default, channel_default : default value when the
        corresponding user override is None.  Can be a scalar (broadcast)
        or a pre-built ndarray.
    """
    def _resolve(user_val, attr_name, default):
        if user_val is not None:
            return _broadcast_attr(user_val, shape2d, attr_name)
        if isinstance(default, np.ndarray):
            return default
        return np.full(shape2d, default, dtype=object)

    unit_arr = _resolve(units, "units", unit_default)
    name_arr = _resolve(names, "names", name_default)
    channel_arr = _resolve(channels, "channels", channel_default)
    return unit_arr, name_arr, channel_arr


def _pack_attrs(unit_arr, name_arr, channel_arr):
    """Wrap the three attribute arrays into the dict expected by callers."""
    return {"unit": unit_arr, "name": name_arr, "channel": channel_arr}


def _convert_units_loop(arr, unit_arr, base_unit, N, M, err_prefix=""):
    """Convert *arr* in-place so that ``arr[i,j]`` is expressed in
    ``unit_arr[i,j]``, assuming the original data is in *base_unit*.

    If *err_prefix* is given it is prepended to the error message.
    """
    for i in range(N):
        for j in range(M):
            tgt = unit_arr[i, j]
            try:
                arr[i, j] = u.Quantity(arr[i, j], base_unit).to_value(tgt)
            except (u.UnitConversionError, TypeError, ValueError) as e:
                msg = f"Unit conversion failed at ({i},{j}): {e}"
                if err_prefix:
                    msg = f"{err_prefix}{e}"
                raise u.UnitConversionError(msg)


def _convert_units_with_explicit(arr, unit_arr, intrinsic_units, explicit_unit,
                                 N, M):
    """Convert units only for cells where *explicit_unit* is True.

    *intrinsic_units* is the per-cell unit array extracted from the input
    objects.  *unit_arr* is the user-requested target unit array
    (already broadcast).  The function updates *unit_arr* in-place for
    cells where the target was ``None``.
    """
    for i in range(N):
        for j in range(M):
            tgt = unit_arr[i, j]
            if tgt is None:
                unit_arr[i, j] = intrinsic_units[i, j]
                continue
            if explicit_unit[i, j]:
                try:
                    arr[i, j] = u.Quantity(
                        arr[i, j], intrinsic_units[i, j]
                    ).to_value(tgt)
                except (u.UnitConversionError, TypeError, ValueError) as e:
                    raise u.UnitConversionError(
                        f"Unit conversion failed at ({i},{j}): {e}"
                    )


def _infer_length_from_items(items, shape):
    """Determine the time-axis length from a flat list of cell values."""
    for v in items:
        if isinstance(v, Series):
            return len(v)
        if isinstance(v, Array):
            try:
                return len(v)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
        if isinstance(v, u.Quantity) and not np.isscalar(v.value):
            try:
                return len(v.value)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
        if isinstance(v, np.ndarray) and v.ndim == 1:
            return len(v)
    if shape is not None and len(shape) == 3:
        return shape[2]
    return 1


def _extract_series_attrs(series_list, N, M):
    """Extract per-cell unit/name/channel arrays from a 2-D list of Series."""
    unit_arr = np.empty((N, M), dtype=object)
    name_arr = np.full((N, M), None, dtype=object)
    channel_arr = np.full((N, M), None, dtype=object)
    for i, row in enumerate(series_list):
        for j, s in enumerate(row):
            unit_arr[i, j] = (
                s.unit if hasattr(s, "unit") else u.dimensionless_unscaled
            )
            name_arr[i, j] = s.name if hasattr(s, "name") else None
            channel_arr[i, j] = s.channel if hasattr(s, "channel") else None
    return unit_arr, name_arr, channel_arr


def _detect_xindex_from_series(series_list):
    """Return common xindex from a flat iterable of Series, or None."""
    all_xindex = [s.xindex for s in series_list if s.xindex is not None]
    if all_xindex and all(np.array_equal(ix, all_xindex[0]) for ix in all_xindex):
        return all_xindex[0]
    return None


def _resolve_scalar_shape(shape, xindex):
    """Resolve 3-D shape for a scalar input."""
    if shape is None:
        if xindex is not None:
            return (1, 1, len(xindex))
        return (1, 1, 1)
    if len(shape) == 2:
        return (*shape, 1)
    if len(shape) != 3:
        raise ValueError(f"shape must be 2D or 3D, got {len(shape)}D")
    return shape


# ---------------------------------------------------------------------------
# Per-type handler functions
# ---------------------------------------------------------------------------


def _handle_none(data, units, names, channels, shape, xindex, dx, x0, xunit):
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, (0, 0),
        unit_default=np.empty((0, 0), dtype=object),
        name_default=np.empty((0, 0), dtype=object),
        channel_default=np.empty((0, 0), dtype=object),
    )
    return np.empty((0, 0, 0)), _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_scalar(data, units, names, channels, shape, xindex, dx, x0, xunit):
    shape = _resolve_scalar_shape(shape, xindex)
    arr = np.full(shape, data)
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, shape[:2],
        unit_default=u.dimensionless_unscaled,
        name_default=None,
        channel_default=None,
    )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_scalar_quantity(data, units, names, channels, shape, xindex, dx,
                            x0, xunit):
    shape = _resolve_scalar_shape(shape, xindex)
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, shape[:2],
        unit_default=data.unit,
        name_default=None,
        channel_default=None,
    )
    arr = np.empty(shape, dtype=np.result_type(data.value, float))
    for i in range(shape[0]):
        for j in range(shape[1]):
            tgt = unit_arr[i, j]
            if tgt is None:
                tgt = data.unit
            try:
                val = data.to_value(tgt)
            except (u.UnitConversionError, TypeError, ValueError) as e:
                raise u.UnitConversionError(
                    f"Unit conversion failed for scalar Quantity: {e}"
                )
            arr[i, j, :] = val
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_series(data, units, names, channels, shape, xindex, dx, x0, xunit):
    arr = np.asarray(data.value).reshape(1, 1, -1)
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, (1, 1),
        unit_default=np.array([[data.unit]], dtype=object),
        name_default=np.array([[getattr(data, "name", None)]], dtype=object),
        channel_default=np.array([[getattr(data, "channel", None)]], dtype=object),
    )
    if units is not None:
        tgt = unit_arr[0, 0]
        try:
            arr[0, 0] = u.Quantity(arr[0, 0], data.unit).to_value(tgt)
        except (u.UnitConversionError, TypeError, ValueError) as e:
            raise u.UnitConversionError(
                f"Unit conversion failed for Series input: {e}"
            )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), data.xindex


def _handle_array(data, units, names, channels, shape, xindex, dx, x0, xunit):
    arr = np.asarray(data.value).reshape(1, 1, -1)
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, (1, 1),
        unit_default=np.array([[data.unit]], dtype=object),
        name_default=np.array([[getattr(data, "name", None)]], dtype=object),
        channel_default=np.array([[getattr(data, "channel", None)]], dtype=object),
    )
    if units is not None:
        tgt = unit_arr[0, 0]
        try:
            arr[0, 0] = u.Quantity(arr[0, 0], data.unit).to_value(tgt)
        except (u.UnitConversionError, TypeError, ValueError) as e:
            raise u.UnitConversionError(
                f"Unit conversion failed for Array input: {e}"
            )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_ndarray_1d_2d(data, units, names, channels, shape, xindex, dx, x0,
                          xunit):
    is_quantity = isinstance(data, u.Quantity)
    if is_quantity:
        quantity_data = cast(u.Quantity, data)
        base_unit = quantity_data.unit
        arr_raw = quantity_data.value
    else:
        base_unit = u.dimensionless_unscaled
        arr_raw = data

    if data.ndim == 1:
        arr = np.asarray(arr_raw).reshape(1, 1, -1)
        N, M = (1, 1)
    else:
        N, M = arr_raw.shape
        arr = np.asarray(arr_raw).reshape(N, 1, M)

    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, (N, M),
        unit_default=base_unit,
        name_default=None,
        channel_default=None,
    )

    if is_quantity and units is not None:
        _convert_units_loop(arr, unit_arr, base_unit, N, M)
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_ndarray_3d(data, units, names, channels, shape, xindex, dx, x0,
                       xunit):
    arr = data.value if isinstance(data, u.Quantity) else data
    _unit = data.unit if isinstance(data, u.Quantity) else u.dimensionless_unscaled
    N, M, _ = arr.shape

    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, (N, M),
        unit_default=_unit,
        name_default=None,
        channel_default=None,
    )

    if isinstance(data, u.Quantity) and units is not None:
        out = np.empty_like(arr, dtype=np.result_type(arr, float))
        _convert_units_loop(out, unit_arr, _unit, N, M)
        # We need to copy original data into out first, then convert.
        # Actually the loop reads arr[i,j] via Quantity -- replicate original:
        for i in range(N):
            for j in range(M):
                tgt = unit_arr[i, j]
                try:
                    out[i, j] = u.Quantity(arr[i, j], _unit).to_value(tgt)
                except (u.UnitConversionError, TypeError, ValueError) as e:
                    raise u.UnitConversionError(
                        f"Unit conversion failed at ({i},{j}): {e}"
                    )
        arr = out
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


def _handle_dict(data, units, names, channels, shape, xindex, dx, x0, xunit):
    data_list = [list(row) for row in data.values()]
    row_names = list(data.keys())
    N = len(row_names)
    M = len(data_list[0]) if data_list else 0

    # Infer time-axis length from first element for xindex construction
    inferred_len = None
    if data_list and data_list[0]:
        first_elem = data_list[0][0]
        if hasattr(first_elem, "__len__") and not np.isscalar(first_elem):
            try:
                inferred_len = len(first_elem)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                inferred_len = None
    if (
        xindex is None
        and dx is not None
        and x0 is not None
        and inferred_len is not None
    ):
        xindex = build_index_if_needed(None, dx, x0, xunit, inferred_len)

    # Detect common xindex from Series-like elements
    all_series = [v for row in data_list for v in row if hasattr(v, "xindex")]
    detected_xindex = None
    if all_series:
        all_xindex = [s.xindex for s in all_series]
        if all(np.array_equal(ix, all_xindex[0]) for ix in all_xindex):
            detected_xindex = all_xindex[0]

    # Convert each element to Series and collect values/attrs
    series_list = []
    unit_arr = np.empty((N, M), dtype=object)
    name_arr = np.empty((N, M), dtype=object)
    channel_arr = np.empty((N, M), dtype=object)
    explicit_unit = np.zeros((N, M), dtype=bool)

    for i, row in enumerate(data_list):
        row_series = []
        for j, v in enumerate(row):
            explicit_unit[i, j] = isinstance(v, (Series, Array, u.Quantity))
            if isinstance(v, Series):
                s = v
            else:
                series_xindex = xindex if xindex is not None else detected_xindex
                s = to_series(v, xindex=series_xindex, name=f"elem_{i}_{j}")

            row_series.append(s.value)
            unit_arr[i, j] = (
                s.unit if hasattr(s, "unit") else u.dimensionless_unscaled
            )
            name_arr[i, j] = s.name if hasattr(s, "name") else None
            channel_arr[i, j] = s.channel if hasattr(s, "channel") else None

        series_list.append(row_series)

    if N == 0 or M == 0:
        arr = np.empty((N, M, 0))
    else:
        arr = np.stack([np.stack(r, axis=0) for r in series_list], axis=0)

    if units is not None:
        target_units = _broadcast_attr(units, (N, M), "units")
        _convert_units_with_explicit(
            arr, target_units, unit_arr, explicit_unit, N, M
        )
        unit_arr = target_units

    name_arr = (
        _broadcast_attr(names, (N, M), "names") if names is not None else name_arr
    )
    channel_arr = (
        _broadcast_attr(channels, (N, M), "channels")
        if channels is not None
        else channel_arr
    )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), detected_xindex


def _handle_list(data, units, names, channels, shape, xindex, dx, x0, xunit):
    # Empty list
    if len(data) == 0:
        unit_arr, name_arr, channel_arr = _make_attr_arrays(
            units, names, channels, (0, 0),
            unit_default=np.empty((0, 0), dtype=object),
            name_default=np.empty((0, 0), dtype=object),
            channel_default=np.empty((0, 0), dtype=object),
        )
        return (
            np.empty((0, 0, 0)),
            _pack_attrs(unit_arr, name_arr, channel_arr),
            None,
        )

    # Promote 1-D list to column vector
    if not isinstance(data[0], (list, tuple)):
        data = [[v] for v in data]

    N = len(data)
    M = len(data[0]) if N > 0 else 0
    for r in data:
        if len(r) != M:
            raise ValueError("Ragged 2D list is not supported for SeriesMatrix")

    if xindex is None:
        xindex = infer_xindex_from_items([v for row in data for v in row])

    items = [v for row in data for v in row]
    inferred_len = _infer_length_from_items(items, shape)

    if xindex is None and dx is not None and x0 is not None:
        xindex = build_index_if_needed(None, dx, x0, xunit, inferred_len)

    # Build Series for each cell
    series_list = []
    explicit_unit = np.zeros((N, M), dtype=bool)
    for row in data:
        series_row = []
        i = len(series_list)
        for j, v in enumerate(row):
            explicit_unit[i, j] = isinstance(v, (Series, Array, u.Quantity))
            series_xindex = xindex if xindex is not None else None
            s = to_series(v, xindex=series_xindex, name=f"elem_{i}_{j}")
            series_row.append(s)
        series_list.append(series_row)

    all_series = [s for row in series_list for s in row]
    detected_xindex = _detect_xindex_from_series(all_series)

    value_list = [s.value for row in series_list for s in row]
    if N == 0 or M == 0:
        unit_arr, name_arr, channel_arr = _make_attr_arrays(
            units, names, channels, (N, M),
            unit_default=np.empty((N, M), dtype=object),
            name_default=np.empty((N, M), dtype=object),
            channel_default=np.empty((N, M), dtype=object),
        )
        return (
            np.empty((N, M, 0)),
            _pack_attrs(unit_arr, name_arr, channel_arr),
            detected_xindex,
        )

    arr = np.stack(value_list).reshape(N, M, -1)

    unit_arr, name_arr, channel_arr = _extract_series_attrs(series_list, N, M)

    if units is not None:
        target_units = _broadcast_attr(units, (N, M), "units")
        _convert_units_with_explicit(
            arr, target_units, unit_arr, explicit_unit, N, M
        )
        unit_arr = target_units

    name_arr = (
        _broadcast_attr(names, (N, M), "names") if names is not None else name_arr
    )
    channel_arr = (
        _broadcast_attr(channels, (N, M), "channels")
        if channels is not None
        else channel_arr
    )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), detected_xindex


def _handle_seriesmatrix(data, units, names, channels, shape, xindex, dx, x0,
                         xunit):
    arr = np.array(data)
    unit_arr, name_arr, channel_arr = _make_attr_arrays(
        units, names, channels, data.units.shape,
        unit_default=data.units.copy(),
        name_default=data.names.copy(),
        channel_default=data.channels.copy(),
    )
    return arr, _pack_attrs(unit_arr, name_arr, channel_arr), None


# ---------------------------------------------------------------------------
# Type detection and dispatch
# ---------------------------------------------------------------------------


def _detect_input_type(data):
    """Return a string key identifying the input type for dispatch.

    The ordering of checks mirrors the original ``_normalize_input`` logic
    so that, e.g., a scalar ``Quantity`` is detected before the generic
    ``ndarray`` branch.
    """
    if data is None:
        return "none"
    if np.isscalar(data):
        return "scalar"
    if isinstance(data, u.Quantity) and np.isscalar(data.value):
        return "scalar_quantity"
    if isinstance(data, Series):
        return "series"
    if isinstance(data, Array):
        return "array"
    if isinstance(data, (np.ndarray, u.Quantity)) and getattr(data, "ndim", 0) in (
        1, 2,
    ):
        return "ndarray_1d_2d"
    if isinstance(data, (np.ndarray, u.Quantity)) and getattr(data, "ndim", 0) == 3:
        return "ndarray_3d"
    if isinstance(data, dict):
        return "dict"
    if isinstance(data, list):
        return "list"
    # Defer SeriesMatrix check to the dispatcher (requires lazy import)
    return "unknown"


_NORMALIZE_HANDLERS = {
    "none": _handle_none,
    "scalar": _handle_scalar,
    "scalar_quantity": _handle_scalar_quantity,
    "series": _handle_series,
    "array": _handle_array,
    "ndarray_1d_2d": _handle_ndarray_1d_2d,
    "ndarray_3d": _handle_ndarray_3d,
    "dict": _handle_dict,
    "list": _handle_list,
    "seriesmatrix": _handle_seriesmatrix,
}


def _normalize_input(
    data,
    units=None,
    names=None,
    channels=None,
    shape=None,
    xindex=None,
    dx=None,
    x0=None,
    xunit=None,
) -> tuple:
    """
    Normalize heterogeneous inputs into a 3D value array and attribute arrays.

    Returns
    -------
    tuple
        (value_array, attr_dict, detected_xindex)
        where attr_dict contains per-cell unit/name/channel arrays.
    """
    type_key = _detect_input_type(data)

    # SeriesMatrix requires a lazy import, so handle the "unknown" fallback
    # and the explicit SeriesMatrix check here.
    if type_key == "unknown":
        from .seriesmatrix import SeriesMatrix

        if isinstance(data, SeriesMatrix):
            type_key = "seriesmatrix"
        else:
            raise TypeError(
                f"Unsupported data type for SeriesMatrix: {type(data)}"
            )

    handler = _NORMALIZE_HANDLERS[type_key]
    return handler(data, units, names, channels, shape, xindex, dx, x0, xunit)


def _check_attribute_consistency(data_attrs: dict, meta: MetaDataMatrix) -> None:
    """
    Validate that overlapping attributes between data_attrs and meta match.
    """
    for attr in ["unit", "name", "channel"]:
        data_arr = data_attrs.get(attr, None)
        if data_arr is not None:
            meta_arr = getattr(meta, attr + "s", None)
            if meta_arr is not None:
                if attr == "unit":
                    mask = np.vectorize(
                        lambda x, y: (
                            x.is_equivalent(y)
                            if x is not None and y is not None
                            else True
                        )
                    )(data_arr, meta_arr)
                elif attr == "channel":

                    def _ch_equal(x, y):
                        if x is None or y is None:
                            return True
                        try:
                            return str(getattr(x, "name", x)) == str(
                                getattr(y, "name", y)
                            )
                        except (
                            IndexError,
                            KeyError,
                            TypeError,
                            ValueError,
                            AttributeError,
                        ):
                            return False

                    mask = np.vectorize(_ch_equal)(data_arr, meta_arr)
                else:
                    mask = (
                        (data_arr == meta_arr) | (meta_arr is None) | (data_arr is None)
                    )
                if not np.all(mask):
                    idxs = np.argwhere(~mask)
                    raise ValueError(f"Inconsistent {attr}: mismatch at indices {idxs}")
    return


def _fill_missing_attributes(
    data_attrs: dict, meta: MetaDataMatrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill missing unit/name/channel attributes from a MetaDataMatrix.
    """
    N, M = meta.shape
    # units
    data_units = data_attrs.get("unit", None)
    if data_units is not None:
        units = data_units
    else:
        units = meta.units
    # names
    data_names = data_attrs.get("name", None)
    if data_names is not None:
        names = data_names
    else:
        names = meta.names
    # channels
    data_channels = data_attrs.get("channel", None)
    if data_channels is not None:
        channels = data_channels
    else:
        channels = meta.channels
    return units, names, channels


def _make_meta_matrix(
    shape: tuple[int, int],
    units: np.ndarray | None,
    names: np.ndarray | None,
    channels: np.ndarray | None,
) -> MetaDataMatrix:
    """Build a MetaDataMatrix from per-cell unit/name/channel arrays."""
    N, M = shape
    meta_array = np.empty((N, M), dtype=object)
    for i in range(N):
        for j in range(M):
            meta_array[i, j] = MetaData(
                unit=units[i, j] if units is not None else None,
                name=names[i, j] if names is not None else None,
                channel=channels[i, j] if channels is not None else None,
            )
    return MetaDataMatrix(meta_array)


def _check_shape_consistency(
    value_array: np.ndarray, meta_matrix: MetaDataMatrix, xindex: np.ndarray | None
) -> None:
    """Validate shape consistency among value array, metadata, and xindex."""
    N, M = value_array.shape[:2]
    if meta_matrix.shape != (N, M):
        raise ValueError(
            f"MetaDataMatrix shape mismatch: {meta_matrix.shape} vs {(N, M)}"
        )
    if xindex is not None:
        if value_array.shape[-1] != len(xindex):
            raise ValueError(
                f"xindex length mismatch: {value_array.shape[-1]} vs {len(xindex)}"
            )
    return
