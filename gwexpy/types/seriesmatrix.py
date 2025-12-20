import warnings
import itertools
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import matplotlib.pyplot as plt
from html import escape
import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, Union, Mapping, Any
from copy import deepcopy
from datetime import datetime
from astropy import units as u
from gwpy.time import LIGOTimeGPS, to_gps
from gwpy.types.array import Array
from gwpy.types.index import Index
from gwpy.types.series import Series
from gwexpy.interop._optional import require_optional

from .metadata import MetaData, MetaDataDict, MetaDataMatrix

# --- Common utilities ---
def to_series(val, xindex, name="s", epoch=0.0):
    if isinstance(val, Series):
        return val
    elif isinstance(val, Array):
        return Series(val.value, xindex=xindex, unit=val.unit,
                      name=val.name, channel=val.channel, epoch=val.epoch)
    elif isinstance(val, u.Quantity):
        if np.isscalar(val.value):
            if xindex is None:
                raise ValueError("Cannot create Series from scalar Quantity without xindex")
            return Series(np.full(len(xindex), val.value), xindex=xindex, unit=val.unit, name=name)
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
        _xunit = u.Unit(xunit) if xunit else (
            dx.unit if isinstance(dx, u.Quantity) else u.dimensionless_unscaled
        )
        _dx = dx.to_value(_xunit) if isinstance(dx, u.Quantity) else dx
        _x0 = x0.to_value(_xunit) if isinstance(x0, u.Quantity) else x0

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
                    raise u.UnitConversionError(f"Unit mismatch at cell ({i},{j}): {u0} vs {uk}")
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
        except Exception:
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
                except Exception:
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
        raise ValueError(f"{name} shape mismatch: expected broadcastable to {shape2d}, got {np.shape(attr)}") from e

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
        key_list = key_list[:ell_idx] + [slice(None)] * n_missing + key_list[ell_idx + 1:]
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
    return MetaDataDict(OrderedDict(subset), expected_size=len(subset), key_prefix=prefix)

def _normalize_input(
    data,
    units=None,
    names=None,
    channels=None,
    shape=None,
    xindex=None,
    dx=None,
    x0=None,
    xunit=None
) -> tuple:
    """
    Normalize heterogeneous inputs into a 3D value array and attribute arrays.

    Returns
    -------
    tuple
        (value_array, attr_dict, detected_xindex)
        where attr_dict contains per-cell unit/name/channel arrays.
    """
    # 0. None -> empty matrix
    if data is None:
        arr = np.empty((0, 0, 0))
        unit_arr = _broadcast_attr(units, (0, 0), "units") if units is not None else np.empty((0, 0), dtype=object)
        name_arr = _broadcast_attr(names, (0, 0), "names") if names is not None else np.empty((0, 0), dtype=object)
        channel_arr = _broadcast_attr(channels, (0, 0), "channels") if channels is not None else np.empty((0, 0), dtype=object)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    # 1. scalar input -> broadcast to 3D
    if np.isscalar(data):
        if shape is None:
            if xindex is not None:
                shape = (1, 1, len(xindex))
            else:
                shape = (1, 1, 1)
        if len(shape) == 2:
            shape = (*shape, 1)
        elif len(shape) != 3:
            raise ValueError(f"shape must be 2D or 3D, got {len(shape)}D")
        arr = np.full(shape, data)
        unit_arr = _broadcast_attr(units, shape[:2], "units") if units is not None else np.full(shape[:2], u.dimensionless_unscaled)
        name_arr = _broadcast_attr(names, shape[:2], "names") if names is not None else np.full(shape[:2], None)
        channel_arr = _broadcast_attr(channels, shape[:2], "channels") if channels is not None else np.full(shape[:2], None)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    if isinstance(data, u.Quantity) and np.isscalar(data.value):
        if shape is None:
            if xindex is not None:
                shape = (1, 1, len(xindex))
            else:
                shape = (1, 1, 1)
        if len(shape) == 2:
            shape = (*shape, 1)
        elif len(shape) != 3:
            raise ValueError(f"shape must be 2D or 3D, got {len(shape)}D")
        unit_arr = _broadcast_attr(units, shape[:2], "units") if units is not None else np.full(shape[:2], data.unit)
        arr = np.empty(shape, dtype=np.result_type(data.value, float))
        for i in range(shape[0]):
            for j in range(shape[1]):
                tgt = unit_arr[i, j]
                if tgt is None:
                    tgt = data.unit
                try:
                    val = data.to_value(tgt)
                except Exception as e:
                    raise u.UnitConversionError(f"Unit conversion failed for scalar Quantity: {e}")
                arr[i, j, :] = val
        name_arr = _broadcast_attr(names, shape[:2], "names") if names is not None else np.full(shape[:2], None)
        channel_arr = _broadcast_attr(channels, shape[:2], "channels") if channels is not None else np.full(shape[:2], None)
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    # 2. Series/Array -> (1,1,K)
    if isinstance(data, Series):
        arr = np.asarray(data.value).reshape(1, 1, -1)
        unit_arr = _broadcast_attr(units, (1, 1), "units") if units is not None else np.array([[data.unit]], dtype=object)
        name_arr = _broadcast_attr(names, (1, 1), "names") if names is not None else np.array([[getattr(data, "name", None)]], dtype=object)
        channel_arr = _broadcast_attr(channels, (1, 1), "channels") if channels is not None else np.array([[getattr(data, "channel", None)]], dtype=object)
        if units is not None:
            # Series has authoritative unit: require convertibility and convert values
            tgt = unit_arr[0, 0]
            try:
                arr[0, 0] = u.Quantity(arr[0, 0], data.unit).to_value(tgt)
            except Exception as e:
                raise u.UnitConversionError(f"Unit conversion failed for Series input: {e}")
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, data.xindex

    if isinstance(data, Array):
        arr = np.asarray(data.value).reshape(1, 1, -1)
        unit_arr = _broadcast_attr(units, (1, 1), "units") if units is not None else np.array([[data.unit]], dtype=object)
        name_arr = _broadcast_attr(names, (1, 1), "names") if names is not None else np.array([[getattr(data, "name", None)]], dtype=object)
        channel_arr = _broadcast_attr(channels, (1, 1), "channels") if channels is not None else np.array([[getattr(data, "channel", None)]], dtype=object)
        if units is not None:
            tgt = unit_arr[0, 0]
            try:
                arr[0, 0] = u.Quantity(arr[0, 0], data.unit).to_value(tgt)
            except Exception as e:
                raise u.UnitConversionError(f"Unit conversion failed for Array input: {e}")
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    # 3. 1D/2D ndarray/Quantity
    if isinstance(data, (np.ndarray, u.Quantity)) and getattr(data, "ndim", 0) in (1, 2):
        is_quantity = isinstance(data, u.Quantity)
        base_unit = data.unit if is_quantity else u.dimensionless_unscaled
        arr_raw = data.value if is_quantity else data
        if data.ndim == 1:
            arr = np.asarray(arr_raw).reshape(1, 1, -1)
            N, M = (1, 1)
        else:
            N, M = arr_raw.shape
            arr = np.asarray(arr_raw).reshape(N, M, 1)

        unit_arr = _broadcast_attr(units, (N, M), "units") if units is not None else np.full((N, M), base_unit)
        name_arr = _broadcast_attr(names, (N, M), "names") if names is not None else np.full((N, M), None)
        channel_arr = _broadcast_attr(channels, (N, M), "channels") if channels is not None else np.full((N, M), None)

        if is_quantity and units is not None:
            # Quantity has authoritative unit: convert each cell to target unit
            for i in range(N):
                for j in range(M):
                    tgt = unit_arr[i, j]
                    try:
                        arr[i, j] = u.Quantity(arr[i, j], base_unit).to_value(tgt)
                    except Exception as e:
                        raise u.UnitConversionError(f"Unit conversion failed at ({i},{j}): {e}")
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    # 4. 3D ndarray/Quantity
    if isinstance(data, (np.ndarray, u.Quantity)) and getattr(data, "ndim", 0) == 3:
        arr = data.value if isinstance(data, u.Quantity) else data
        _unit = data.unit if isinstance(data, u.Quantity) else u.dimensionless_unscaled
        N, M, _ = arr.shape
        unit_arr = _broadcast_attr(units, (N, M), "units") if units is not None else np.full((N, M), _unit)
        name_arr = _broadcast_attr(names, (N, M), "names") if names is not None else np.full((N, M), None)
        channel_arr = _broadcast_attr(channels, (N, M), "channels") if channels is not None else np.full((N, M), None)
        if isinstance(data, u.Quantity) and units is not None:
            # Quantity has authoritative unit: convert each cell to target unit
            out = np.empty_like(arr, dtype=np.result_type(arr, float))
            for i in range(N):
                for j in range(M):
                    tgt = unit_arr[i, j]
                    try:
                        out[i, j] = u.Quantity(arr[i, j], _unit).to_value(tgt)
                    except Exception as e:
                        raise u.UnitConversionError(f"Unit conversion failed at ({i},{j}): {e}")
            arr = out
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    # 3. dict input
    if isinstance(data, dict):
        data_list = [list(row) for row in data.values()]
        row_names = list(data.keys())
        N = len(row_names)
        M = len(data_list[0]) if data_list else 0
        
        inferred_len = None
        if data_list and data_list[0]:
            first_elem = data_list[0][0]
            if hasattr(first_elem, "__len__") and not np.isscalar(first_elem):
                try:
                    inferred_len = len(first_elem)
                except Exception:
                    inferred_len = None
        if xindex is None and dx is not None and x0 is not None and inferred_len is not None:
            xindex = build_index_if_needed(None, dx, x0, xunit, inferred_len)

        all_series = [v for row in data_list for v in row if hasattr(v, "xindex")]
        detected_xindex = None
        if all_series:
            all_xindex = [s.xindex for s in all_series]
            if all(np.array_equal(ix, all_xindex[0]) for ix in all_xindex):
                detected_xindex = all_xindex[0]
        
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
                unit_arr[i, j] = s.unit if hasattr(s, 'unit') else u.dimensionless_unscaled
                name_arr[i, j] = s.name if hasattr(s, 'name') else None
                channel_arr[i, j] = s.channel if hasattr(s, 'channel') else None
            
            series_list.append(row_series)
        
        if N == 0 or M == 0:
            arr = np.empty((N, M, 0))
        else:
            arr = np.stack([np.stack(r, axis=0) for r in series_list], axis=0)

        if units is not None:
            target_units = _broadcast_attr(units, (N, M), "units")
            for i in range(N):
                for j in range(M):
                    tgt = target_units[i, j]
                    if tgt is None:
                        target_units[i, j] = unit_arr[i, j]
                        continue
                    if explicit_unit[i, j]:
                        try:
                            arr[i, j] = u.Quantity(arr[i, j], unit_arr[i, j]).to_value(tgt)
                        except Exception as e:
                            raise u.UnitConversionError(f"Unit conversion failed at ({i},{j}): {e}")
            unit_arr = target_units

        name_arr = _broadcast_attr(names, (N, M), "names") if names is not None else name_arr
        channel_arr = _broadcast_attr(channels, (N, M), "channels") if channels is not None else channel_arr
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, detected_xindex

    # 4. list input (1D -> column vector, 2D -> matrix)
    if isinstance(data, list):
        if len(data) == 0:
            arr = np.empty((0, 0, 0))
            unit_arr = _broadcast_attr(units, (0, 0), "units") if units is not None else np.empty((0, 0), dtype=object)
            name_arr = _broadcast_attr(names, (0, 0), "names") if names is not None else np.empty((0, 0), dtype=object)
            channel_arr = _broadcast_attr(channels, (0, 0), "channels") if channels is not None else np.empty((0, 0), dtype=object)
            return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

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
        inferred_len = None
        for v in items:
            if isinstance(v, Series):
                inferred_len = len(v)
                break
            if isinstance(v, Array):
                try:
                    inferred_len = len(v)
                    break
                except Exception:
                    pass
            if isinstance(v, u.Quantity) and not np.isscalar(v.value):
                try:
                    inferred_len = len(v.value)
                    break
                except Exception:
                    pass
            if isinstance(v, np.ndarray) and v.ndim == 1:
                inferred_len = len(v)
                break
        if inferred_len is None:
            if shape is not None and len(shape) == 3:
                inferred_len = shape[2]
            else:
                inferred_len = 1

        if xindex is None and dx is not None and x0 is not None:
            xindex = build_index_if_needed(None, dx, x0, xunit, inferred_len)

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
        detected_xindex = None
        if all_series:
            all_xindex = [s.xindex for s in all_series if s.xindex is not None]
            if all_xindex and all(np.array_equal(ix, all_xindex[0]) for ix in all_xindex):
                detected_xindex = all_xindex[0]
        
        value_list = [s.value for row in series_list for s in row]
        if N == 0 or M == 0:
            arr = np.empty((N, M, 0))
            unit_arr = _broadcast_attr(units, (N, M), "units") if units is not None else np.empty((N, M), dtype=object)
            name_arr = _broadcast_attr(names, (N, M), "names") if names is not None else np.empty((N, M), dtype=object)
            channel_arr = _broadcast_attr(channels, (N, M), "channels") if channels is not None else np.empty((N, M), dtype=object)
            return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, detected_xindex
        arr = np.stack(value_list).reshape(N, M, -1)
        
        unit_arr = np.empty((N, M), dtype=object)
        name_arr = np.full((N, M), None, dtype=object)
        channel_arr = np.full((N, M), None, dtype=object)
        
        for i, row in enumerate(series_list):
            for j, s in enumerate(row):
                unit_arr[i, j] = s.unit if hasattr(s, "unit") else u.dimensionless_unscaled
                name_arr[i, j] = s.name if hasattr(s, "name") else None
                channel_arr[i, j] = s.channel if hasattr(s, "channel") else None

        if units is not None:
            target_units = _broadcast_attr(units, (N, M), "units")
            for i in range(N):
                for j in range(M):
                    tgt = target_units[i, j]
                    if tgt is None:
                        target_units[i, j] = unit_arr[i, j]
                        continue
                    if explicit_unit[i, j]:
                        try:
                            arr[i, j] = u.Quantity(arr[i, j], unit_arr[i, j]).to_value(tgt)
                        except Exception as e:
                            raise u.UnitConversionError(f"Unit conversion failed at ({i},{j}): {e}")
            unit_arr = target_units

        name_arr = _broadcast_attr(names, (N, M), "names") if names is not None else name_arr
        channel_arr = _broadcast_attr(channels, (N, M), "channels") if channels is not None else channel_arr
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, detected_xindex

    # 5. SeriesMatrix input
    if isinstance(data, SeriesMatrix):
        arr = np.array(data)
        unit_arr = _broadcast_attr(units, data.units.shape, "units") if units is not None else data.units.copy()
        name_arr = _broadcast_attr(names, data.names.shape, "names") if names is not None else data.names.copy()
        channel_arr = _broadcast_attr(channels, data.channels.shape, "channels") if channels is not None else data.channels.copy()
        return arr, {"unit": unit_arr, "name": name_arr, "channel": channel_arr}, None

    raise TypeError(f"Unsupported data type for SeriesMatrix: {type(data)}")



def _check_attribute_consistency(
    data_attrs: dict,
    meta: "MetaDataMatrix"
) -> None:
    """
    Validate that overlapping attributes between data_attrs and meta match.
    """
    for attr in ["unit", "name", "channel"]:
        data_arr = data_attrs.get(attr, None)
        if data_arr is not None:
            meta_arr = getattr(meta, attr + "s", None)
            if meta_arr is not None:
                if attr == "unit":
                    mask = np.vectorize(lambda x, y: x.is_equivalent(y) if x is not None and y is not None else True)(data_arr, meta_arr)
                elif attr == "channel":
                    def _ch_equal(x, y):
                        if x is None or y is None:
                            return True
                        try:
                            return str(getattr(x, "name", x)) == str(getattr(y, "name", y))
                        except Exception:
                            return False
                    mask = np.vectorize(_ch_equal)(data_arr, meta_arr)
                else:
                    mask = (data_arr == meta_arr) | (meta_arr == None) | (data_arr == None)
                if not np.all(mask):
                    idxs = np.argwhere(~mask)
                    raise ValueError(f"Inconsistent {attr}: mismatch at indices {idxs}")
    return


def _fill_missing_attributes(
    data_attrs: dict,
    meta: "MetaDataMatrix"
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
    units: Optional[np.ndarray],
    names: Optional[np.ndarray],
    channels: Optional[np.ndarray]
) -> "MetaDataMatrix":
    """Build a MetaDataMatrix from per-cell unit/name/channel arrays."""
    N, M = shape
    meta_array = np.empty((N, M), dtype=object)
    for i in range(N):
        for j in range(M):
            meta_array[i, j] = MetaData(
                unit=units[i, j] if units is not None else None,
                name=names[i, j] if names is not None else None,
                channel=channels[i, j] if channels is not None else None
            )
    return MetaDataMatrix(meta_array)

def _check_shape_consistency(
    value_array: np.ndarray,
    meta_matrix: "MetaDataMatrix",
    xindex: Optional[np.ndarray]
) -> None:
    """Validate shape consistency among value array, metadata, and xindex."""
    N, M = value_array.shape[:2]
    if meta_matrix.shape != (N, M):
        raise ValueError(f"MetaDataMatrix shape mismatch: {meta_matrix.shape} vs {(N, M)}")
    if xindex is not None:
        if value_array.shape[-1] != len(xindex):
            raise ValueError(f"xindex length mismatch: {value_array.shape[-1]} vs {len(xindex)}")
    return


########################################
### SeriesMatrix
#######################################
class SeriesMatrix(np.ndarray):
    def __new__(cls, 
                data=None,
                *,
                meta: Optional[Union["MetaDataMatrix", np.ndarray, list]] = None,
                unit: Optional[object] = None,
                units: Optional[np.ndarray] = None,
                names: Optional[np.ndarray] = None,
                channels: Optional[np.ndarray] = None,
                rows=None,
                cols=None,
                shape=None,
                xindex=None,
                dx=None,
                x0=None,
                xunit=None,
                name="",
                epoch=0.0,
                attrs=None):
        """
        Create a SeriesMatrix with normalized inputs and metadata.
        """

        if unit is not None:
            if units is not None:
                raise ValueError("give only one of unit or units")
            units = unit

        value_array, data_attrs, detected_xindex = _normalize_input(
            data=data,
            units=units,
            names=names,
            channels=channels,
            shape=shape,
            xindex=xindex,
            dx=dx,
            x0=x0,
            xunit=xunit
        )

        if meta is not None:
            if not isinstance(meta, MetaDataMatrix):
                try:
                    meta = MetaDataMatrix(meta)
                except Exception as e:
                    raise TypeError(
                        "meta must be a MetaDataMatrix or a 2D array-like of MetaData/dict"
                    ) from e
            if units is None:
                data_attrs["unit"] = None
            if names is None:
                data_attrs["name"] = None
            if channels is None:
                data_attrs["channel"] = None
            _check_attribute_consistency(
                data_attrs=data_attrs,
                meta=meta
            )
            units_arr, names_arr, channels_arr = _fill_missing_attributes(
                data_attrs=data_attrs,
                meta=meta
            )
        else:
            units_arr = data_attrs.get("unit", None)
            names_arr = data_attrs.get("name", None)
            channels_arr = data_attrs.get("channel", None)

        meta_matrix = _make_meta_matrix(
            shape=value_array.shape[:2],
            units=units_arr,
            names=names_arr,
            channels=channels_arr
        )

        if xindex is None:
            if detected_xindex is not None:
                xindex = detected_xindex
            else:
                if value_array.shape[2] == 0 and dx is None and x0 is None:
                    xindex = np.asarray([])
                else:
                    xindex = build_index_if_needed(
                        xindex=None,
                        dx=dx,
                        x0=x0,
                        xunit=xunit,
                        length=value_array.shape[2]
                    )
        
        _check_shape_consistency(
            value_array=value_array,
            meta_matrix=meta_matrix,
            xindex=xindex
        )

        obj = np.asarray(value_array).view(cls)

        obj._value = obj.view(np.ndarray)

        obj.meta = meta_matrix
        N, M = value_array.shape[:2]
        if isinstance(rows, dict) and not isinstance(rows, OrderedDict):
            rows = OrderedDict(rows)
        if isinstance(cols, dict) and not isinstance(cols, OrderedDict):
            cols = OrderedDict(cols)
        obj.rows = MetaDataDict(rows, expected_size=N, key_prefix="row")
        obj.cols = MetaDataDict(cols, expected_size=M, key_prefix="col")
        obj.xindex = xindex
        obj.name = name
        obj.epoch = epoch
        obj.attrs = attrs or {}

        return obj



    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self._value = self.view(np.ndarray)
        self._suppress_xindex_check = True
        self.xindex = getattr(obj, 'xindex', None)
        if hasattr(self, "_suppress_xindex_check"):
            delattr(self, "_suppress_xindex_check")
        self.meta   = getattr(obj, 'meta', None)
        self.rows   = getattr(obj, 'rows', None)
        self.cols   = getattr(obj, 'cols', None)
        self.name   = getattr(obj, 'name', "")
        self.epoch  = getattr(obj, 'epoch', 0.0)
        self.attrs  = getattr(obj, 'attrs', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            base_inputs = [inp.view(np.ndarray) if isinstance(inp, SeriesMatrix) else inp for inp in inputs]
            try:
                return np.ndarray.__array_ufunc__(self.view(np.ndarray), ufunc, method, *base_inputs, **kwargs)
            except Exception:
                return NotImplemented
    
        casted_inputs = []
        xindex = self.xindex
        shape  = self._value.shape
        meta   = self.meta
        rows   = self.rows
        cols   = self.cols
        epoch  = getattr(self, "epoch", 0.0)
        name   = getattr(self, "name", "")
        attrs  = getattr(self, "attrs", {})
    
        for inp in inputs:
            if isinstance(inp, SeriesMatrix):
                casted_inputs.append(inp)
            elif isinstance(inp, u.Quantity):
                val = np.asarray(inp.value)
                N, M, K = shape
                if val.ndim == 0:
                    arr = np.full(shape, val)
                elif val.ndim == 1:
                    if val.shape != (K,):
                        raise ValueError(f"1D Quantity must have length N_samples={K}, got {val.shape}")
                    arr = np.broadcast_to(val.reshape(1, 1, K), shape)
                elif val.ndim == 2:
                    if val.shape != (N, M):
                        raise ValueError(f"2D Quantity must have shape (Nrow,Ncol)={(N, M)}, got {val.shape}")
                    arr = np.broadcast_to(val.reshape(N, M, 1), shape)
                elif val.ndim == 3:
                    if val.shape != shape:
                        raise ValueError(f"3D Quantity must have shape {shape}, got {val.shape}")
                    arr = val
                else:
                    raise ValueError(f"Quantity with ndim={val.ndim} is not supported in __array_ufunc__")
                unit = inp.unit
                meta_array = np.empty(self._value.shape[:2], dtype=object)
                for i in range(self._value.shape[0]):
                    for j in range(self._value.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
            elif isinstance(inp, (float, int, complex)):
                arr = np.full(self._value.shape, inp)
                unit = u.dimensionless_unscaled
                meta_array = np.empty(self._value.shape[:2], dtype=object)
                for i in range(self._value.shape[0]):
                    for j in range(self._value.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
            elif isinstance(inp, np.ndarray):
                val = np.asarray(inp)
                N, M, K = shape
                if val.ndim == 0:
                    arr = np.full(shape, val)
                elif val.ndim == 1:
                    if val.shape != (K,):
                        raise ValueError(
                            f"1D ndarray must have length N_samples={K}, got {val.shape}"
                        )
                    arr = np.broadcast_to(val.reshape(1, 1, K), shape)
                elif val.ndim == 2:
                    if val.shape != (N, M):
                        raise ValueError(
                            f"2D ndarray must have shape (Nrow,Ncol)={(N, M)}, got {val.shape}"
                        )
                    arr = np.broadcast_to(val.reshape(N, M, 1), shape)
                elif val.ndim == 3:
                    if val.shape != shape:
                        raise ValueError(f"3D ndarray must have shape {shape}, got {val.shape}")
                    arr = val
                else:
                    raise ValueError(
                        f"ndarray with ndim={val.ndim} is not supported in __array_ufunc__"
                    )

                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, shape=self._value.shape))
            else:
                return NotImplemented


    
        check_shape_xindex_compatibility(*casted_inputs)
    
        if ufunc in [np.add, np.subtract, np.less, np.less_equal, np.equal, np.not_equal, np.greater, np.greater_equal]:
            check_add_sub_compatibility(*casted_inputs)
    
        value_arrays = [inp.view(np.ndarray) for inp in casted_inputs]
    
        meta_matrices = [inp.meta for inp in casted_inputs]
    
        ufunc_kwargs = {k: v for k, v in kwargs.items() if k not in ('out', 'where')}
        
        N, M = self._value.shape[:2]
        logical_ufuncs = {
            np.logical_and, np.logical_or, np.logical_xor, np.logical_not,
            np.isfinite, np.isinf, np.isnan, np.isclose
        }
        meta_passthrough_ufuncs = logical_ufuncs | {
            np.sign, np.floor, np.ceil, np.trunc, np.rint,
            np.mod, np.remainder, np.clip
        }
        try:
            probe_val_args = [v[0, 0] for v in value_arrays]
            probe_result = ufunc(*probe_val_args, **ufunc_kwargs)
            boolean_ufuncs = {np.less, np.less_equal, np.equal, np.not_equal,
                              np.greater, np.greater_equal,
                              np.logical_and, np.logical_or, np.logical_xor,
                              np.logical_not, np.isfinite, np.isinf, np.isnan,
                              np.isclose}
            if ufunc in boolean_ufuncs:
                result_dtype = np.bool_
            else:
                result_dtype = np.asarray(probe_result).dtype
        except Exception:
            result_dtype = self._value.dtype
        
        result_values = np.empty(self._value.shape, dtype=result_dtype)
        result_meta   = np.empty(self._value.shape[:2], dtype=object)
        bool_result = np.issubdtype(result_dtype, np.bool_)

        for i in range(N):
            for j in range(M):
                val_args = [v[i, j] for v in value_arrays]
                meta_args = [m[i, j] for m in meta_matrices]
                try:
                    result_values[i, j] = ufunc(*val_args, **ufunc_kwargs)
                except Exception as e:
                    raise type(e)(f"Error at cell ({i},{j}): {e}")
                try:
                    if bool_result or ufunc in meta_passthrough_ufuncs or ufunc.__name__ == "clip":
                        result_meta[i, j] = meta_args[0]
                    else:
                        result_meta[i, j] = ufunc(*meta_args, **ufunc_kwargs)
                except Exception as e:
                    raise type(e)(f"MetaData ufunc error at ({i},{j}): {e}")
    
        result_meta_matrix = MetaDataMatrix(result_meta)
        result_units = result_meta_matrix.units
        return self.__class__(
            result_values,
            xindex=self.xindex,
            meta=result_meta_matrix,
            units=result_units,
            rows=rows,
            cols=cols,
            name=name,
            epoch=epoch,
            attrs=attrs
        )

    


    ##### xindex Information #####
    @property
    def xindex(self):
        return getattr(self, "_xindex", None)

    @xindex.setter
    def xindex(self, value):
        if value is None:
            self._xindex = None
        else:
            if isinstance(value, Index):
                xi = value
            elif isinstance(value, u.Quantity):
                xi = value
            else:
                xi = np.asarray(value)
            try:
                n_samples = self._value.shape[2]
            except Exception:
                n_samples = None
            suppress = getattr(self, "_suppress_xindex_check", False)
            if (not suppress) and n_samples is not None and hasattr(xi, "__len__") and len(xi) != n_samples:
                raise ValueError(f"xindex length mismatch: expected {n_samples}, got {len(xi)}")
            self._xindex = xi
        for attr in ("_dx", "_x0"):
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def x0(self):
        try:
            return self._x0
        except AttributeError:
            try:
                self._x0 = self.xindex[0]
            except (AttributeError, IndexError):
                self._x0 = u.Quantity(0, self.xunit)
            return self._x0
        
    @property
    def dx(self):
        try:
            return self._dx
        except AttributeError:
            if not hasattr(self, "xindex") or self.xindex is None:
                raise AttributeError("dx is undefined because xindex is not set")
            if hasattr(self.xindex, "regular") and not self.xindex.regular:
                raise AttributeError("This SeriesMatrix has an irregular x-axis index, so 'dx' is not well defined")
            dx = self.xindex[1] - self.xindex[0]
            if not isinstance(dx, u.Quantity):
                xunit = getattr(self.xindex, 'unit', u.dimensionless_unscaled)
                dx = u.Quantity(dx, xunit)
            self._dx = dx
            return self._dx

    
    @property
    def xspan(self):
        xindex = self.xindex
        try:
            if hasattr(xindex, "regular") and xindex.regular:
                return (xindex[0], xindex[-1] + self.dx)
            if len(xindex) > 1:
                step = xindex[-1] - xindex[-2]
                return (xindex[0], xindex[-1] + step)
            return (xindex[0], xindex[0])
        except Exception:
            return (xindex[0], xindex[-1])
    
    @property
    def xunit(self):
        try:
            return self._dx.unit
        except AttributeError:
            try:
                return self._x0.unit
            except AttributeError:
                return u.dimensionless_unscaled
    
    @property
    def N_samples(self):
        return len(self.xindex) if self.xindex is not None else 0

    @property
    def xarray(self):
        """
        Return the sample axis values. Do NOT multiply by `self.xunit` here because
        `self.xindex` may already be a Quantity with units. Returning `self.xindex`
        avoids double-applying units.
        """
        return self.xindex

    @property
    def duration(self):
        """
        Duration covered by the samples. Computed directly from `xindex` to avoid
        unit duplication (i.e. `xindex[-1] - xindex[0]`).
        """
        if self.N_samples == 0:
            try:
                return 0 * self.xunit
            except Exception:
                return 0
        return self.xindex[-1] - self.xindex[0]

    def is_contiguous(self, other, tol=1/2.**18):
        """
        Check contiguity using xspan endpoints (gwpy-like semantics).
        """
        if not isinstance(other, SeriesMatrix):
            raise TypeError("is_contiguous expects SeriesMatrix")
        self.is_compatible(other)

        base_unit = getattr(self.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        span_self = self.xspan
        span_other = other.xspan
        diff1 = _to_base(span_self[1]) - _to_base(span_other[0])
        diff2 = _to_base(span_other[1]) - _to_base(span_self[0])
        if abs(diff1) < tol:
            return 1
        elif abs(diff2) < tol:
            return -1
        else:
            return 0

    def is_contiguous_exact(self, other, tol=1/2.**18):
        """
        Strict contiguity check (requires identical shape and matching endpoints).
        """
        if self._value.shape != other._value.shape:
            raise ValueError(f"shape does not match: {self._value.shape} vs {other._value.shape}")

        base_unit = getattr(self.xindex, "unit", None)
        if base_unit is None:
            base_unit = getattr(other.xindex, "unit", None)

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        xspan_self = self.xspan
        xspan_other = other.xspan
        diff1 = _to_base(xspan_self[1]) - _to_base(xspan_other[0])
        diff2 = _to_base(xspan_other[1]) - _to_base(xspan_self[0])
        if abs(diff1) < tol:
            return 1
        elif abs(diff2) < tol:
            return -1
        else:
            return 0
    
    def is_compatible(self, other):
        """
        Compatibility check similar to gwpy: compares shape, xindex/dx, and units.
        """
        if not isinstance(other, SeriesMatrix):
            arr = np.asarray(other)
            if arr.shape != self._value.shape:
                raise ValueError(f"shape does not match: {self._value.shape} vs {arr.shape}")
            return True

        if self._value.shape[:2] != other._value.shape[:2]:
            raise ValueError(f"matrix shape does not match: {self._value.shape[:2]} vs {other._value.shape[:2]}")

        if list(self.rows.keys()) != list(other.rows.keys()):
            raise ValueError("row keys do not match")
        if list(self.cols.keys()) != list(other.cols.keys()):
            raise ValueError("col keys do not match")

        xunit_self = getattr(self.xindex, "unit", None)
        xunit_other = getattr(other.xindex, "unit", None)
        if xunit_self is not None and xunit_other is not None:
            try:
                if not u.Unit(xunit_self).is_equivalent(u.Unit(xunit_other)):
                    raise ValueError(f"xindex unit does not match: {xunit_self} vs {xunit_other}")
            except Exception:
                raise ValueError(f"xindex unit does not match: {xunit_self} vs {xunit_other}")

        try:
            dx_self = self.dx
            dx_other = other.dx
            if dx_self != dx_other:
                raise ValueError(f"dx does not match: {dx_self} vs {dx_other}")
        except Exception:
            lhs = np.asarray(self.xindex)
            rhs = np.asarray(other.xindex)
            if not np.array_equal(lhs, rhs):
                raise ValueError("xindex does not match")

        for i in range(self._value.shape[0]):
            for j in range(self._value.shape[1]):
                u1 = self.meta[i, j].unit
                u2 = other.meta[i, j].unit
                if u1 != u2:
                    raise ValueError(f"unit does not match at ({i},{j}): {u1} vs {u2}")
        return True

    def is_compatible_exact(self, other):
        """
        Strict compatibility: requires identical shape, xindex, and metadata units.
        """
        base_unit = getattr(self.xindex, "unit", None)
        lhs = self.xindex
        rhs = other.xindex
        if base_unit is not None:
            try:
                lhs = lhs.to_value(base_unit)
                rhs = rhs.to_value(base_unit)
            except Exception:
                pass
        try:
            equal = np.array_equal(lhs, rhs)
        except Exception:
            equal = lhs == rhs
        if not equal:
            raise ValueError(f"xindex does not match: {self.xindex} vs {other.xindex}")

        if self._value.shape != other._value.shape:
            raise ValueError(f"shape does not match: {self._value.shape} vs {other._value.shape}")

        if list(self.rows.keys()) != list(other.rows.keys()):
            raise ValueError("row keys do not match")
        if list(self.cols.keys()) != list(other.cols.keys()):
            raise ValueError("col keys do not match")

        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        return True

    
    ##### rows/cols Information #####
    def row_keys(self):
        return tuple(self.rows.keys())

    def col_keys(self):
        return tuple(self.cols.keys())
        
    def keys(self):
        return (self.row_keys(), self.col_keys())

    def row_index(self, key): #Return the index of the given row key.
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key): #Return the index of the given column key.
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def get_index(self, key_row, key_col): #Return the (i,j) index for given row and column keys.
        return self.row_index(key_row), self.col_index(key_col)


    ##### Elements Metadata #####    
    @property
    def MetaDataMatrix(self):
        return self.meta
        
    @property
    def units(self):
        return self.meta.units
    
    @property
    def names(self):
        return self.meta.names

    @property
    def channels(self):
        return self.meta.channels

    ##### Elements (Series object) acess #####
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, (int, str)) and isinstance(col_key, (int, str)):
                i = self.row_index(row_key) if not isinstance(row_key, int) else row_key
                j = self.col_index(col_key) if not isinstance(col_key, int) else col_key
                meta = self.meta[i, j]
                values = np.ndarray.__getitem__(self._value, (i, j)) 
                return Series(values, xindex=self.xindex.copy(),
                              unit=meta.unit, name=meta.name, channel=meta.channel)

        result = np.ndarray.__getitem__(self, key)
        if isinstance(result, SeriesMatrix):
            row_sel, col_sel, sample_sel = _expand_key(key, self.ndim)
            try:
                xi = self.xindex[sample_sel]
                if np.isscalar(xi):
                    xi = u.Quantity([xi]) if isinstance(xi, u.Quantity) else np.asarray([xi])
                result.xindex = xi
            except Exception:
                result.xindex = self.xindex

            try:
                meta_slice = self.meta[row_sel, col_sel]
                if meta_slice.ndim == 1:
                    if isinstance(row_sel, (int, np.integer)):
                        meta_slice = meta_slice.reshape(1, -1)
                    elif isinstance(col_sel, (int, np.integer)):
                        meta_slice = meta_slice.reshape(-1, 1)
                result.meta = MetaDataMatrix(meta_slice)
            except Exception:
                result.meta = self.meta

            try:
                result.rows = _slice_metadata_dict(self.rows, row_sel, "row")
                result.cols = _slice_metadata_dict(self.cols, col_sel, "col")
            except Exception:
                result.rows = self.rows
                result.cols = self.cols
            if result.ndim == 2:
                # NOTE:
                # - If the sample axis is reduced to a scalar (e.g. [:, :, k]),
                #   the remaining 2D shape is (nrow, ncol) and we re-add a
                #   length-1 sample axis.
                # - If a row/col axis is reduced to a scalar but the sample axis
                #   remains (e.g. [:, j] or [i, :]), the 2D shape is
                #   (nrow, nsample) or (ncol, nsample). In this case we re-add
                #   the missing row/col axis (size 1) and keep the sample axis.
                sample_is_scalar = isinstance(sample_sel, (int, np.integer))
                row_is_scalar = isinstance(row_sel, (int, np.integer))
                col_is_scalar = isinstance(col_sel, (int, np.integer))

                if sample_is_scalar:
                    result = result.reshape(result.shape[0], result.shape[1], 1).view(
                        SeriesMatrix
                    )
                    result.xindex = (
                        u.Quantity(result.xindex).reshape(1)
                        if isinstance(result.xindex, u.Quantity)
                        else np.asarray(result.xindex).reshape(1)
                    )
                else:
                    if col_is_scalar and not row_is_scalar:
                        # (nrow, nsample) -> (nrow, 1, nsample)
                        result = result.reshape(
                            result.shape[0], 1, result.shape[1]
                        ).view(SeriesMatrix)
                    elif row_is_scalar and not col_is_scalar:
                        # (ncol, nsample) -> (1, ncol, nsample)
                        result = result.reshape(
                            1, result.shape[0], result.shape[1]
                        ).view(SeriesMatrix)
                    else:
                        # Fallback: treat as single-sample matrix
                        result = result.reshape(result.shape[0], result.shape[1], 1).view(
                            SeriesMatrix
                        )
                        result.xindex = (
                            u.Quantity(result.xindex).reshape(1)
                            if isinstance(result.xindex, u.Quantity)
                            else np.asarray(result.xindex).reshape(1)
                        )

                result.meta = MetaDataMatrix(result.meta)
            result._value = result.view(np.ndarray)
        return result
    
    def __setitem__(self, key, value):
        if not (isinstance(key, tuple) and len(key) == 2):
            return np.ndarray.__setitem__(self, key, value)
        
        row_key, col_key = key
        i = self.row_index(row_key) if not isinstance(row_key, int) else row_key
        j = self.col_index(col_key) if not isinstance(col_key, int) else col_key
        if isinstance(value, Series):
            if len(value) != self._value.shape[2]:
                raise ValueError("xindex length mismatch")
            if hasattr(value, "xindex") and self.xindex is not None:
                base_unit = getattr(self.xindex, "unit", None)

                def _to_base(x):
                    if base_unit is None:
                        return np.asarray(x)
                    return u.Quantity(x).to_value(base_unit)

                try:
                    lhs = _to_base(value.xindex)
                    rhs = _to_base(self.xindex)
                    if not np.array_equal(lhs, rhs):
                        raise ValueError("Assigned Series has incompatible xindex")
                except Exception:
                    raise ValueError("Assigned Series has incompatible xindex")
            self._value[i, j] = value.value
            self.meta[i, j] = MetaData(unit=value.unit, name=value.name, channel=value.channel)
        else:
            raise TypeError("Only Series objects can be assigned to SeriesMatrix elements.")
       
    def plot(self, **kwargs):
        """
        Plot this SeriesMatrix using gwexpy.plot.Plot.
        """
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)

    ##### as a Matrix #####

    def _all_element_units_equivalent(self):
        """
        Check whether all element units are mutually equivalent.

        Returns
        -------
        tuple(bool, UnitBase)
            (is_equivalent, reference_unit)
        """
        ref_unit = self.meta[0, 0].unit
        try:
            units = np.array(self.meta.units, dtype=object)
            # If all cells are None, consider them equivalent
            if units.size == 0:
                return True, ref_unit
            # Vectorized check for equivalence to ref_unit
            def _eq(u_):
                try:
                    return u_.is_equivalent(ref_unit)
                except Exception:
                    return False
            mask = np.vectorize(_eq)(units)
            if np.all(mask):
                return True, ref_unit
            return False, ref_unit
        except Exception:
            # Fallback: sequential check
            for meta in self.meta.flat:
                try:
                    if not meta.unit.is_equivalent(ref_unit):
                        return False, ref_unit
                except Exception:
                    return False, ref_unit
            return True, ref_unit

    def _to_common_unit_values(self, ref_unit):
        """
        Convert all element values to a common reference unit.
        """
        N, M, K = self._value.shape
        units = np.array(self.meta.units, dtype=object)
        if units.size == 0:
            return np.array(self._value, copy=True)

        # Batch conversion for equivalent units
        try:
            def _eq(u_):
                try:
                    return u_.is_equivalent(ref_unit)
                except Exception:
                    return False
            mask = np.vectorize(_eq)(units)
            if np.all(mask):
                return u.Quantity(self._value, units[0, 0]).to_value(ref_unit)
        except Exception:
            pass

        # Mixed units: convert per cell
        result = np.empty((N, M, K), dtype=np.result_type(self._value, float))
        for i in range(N):
            for j in range(M):
                u_ij = self.meta[i, j].unit
                try:
                    vals = u.Quantity(self._value[i, j], u_ij).to_value(ref_unit)
                except Exception as e:
                    raise type(e)(f"Unit conversion failed at ({i},{j}): {e}")
                result[i, j] = vals
        return result

    @property
    def shape3D(self):
        return self._value.shape[0], self._value.shape[1], self.N_samples
        
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        np.copyto(self._value, new_value)

    @property
    def loc(self):
        class _LocAccessor:
            def __init__(self, parent):
                self._parent = parent

            def __getitem__(self, key):
                return self._parent._value[key]

            def __setitem__(self, key, value):
                self._parent._value[key] = value

        return _LocAccessor(self)

    def submatrix(self, row_keys, col_keys):
        row_indices = [self.row_index(k) for k in row_keys]
        col_indices = [self.col_index(k) for k in col_keys]
        new_data = self.value[np.ix_(row_indices, col_indices)]
        new_meta = MetaDataMatrix(self.meta[np.ix_(row_indices, col_indices)])
    
        new_rows = OrderedDict((k, self.rows[k]) for k in row_keys)
        new_cols = OrderedDict((k, self.cols[k]) for k in col_keys)
    
        return SeriesMatrix(new_data, xindex=self.xindex, name=self.name,
                            epoch=self.epoch, rows=new_rows, cols=new_cols,
                            meta=new_meta, attrs=self.attrs)

    def to_series_2Dlist(self):
        return [[self[row,col] for col in self.col_keys()] for row in self.row_keys()]

    def to_series_1Dlist(self):
        return [self[row,col] for col in self.col_keys() for row in self.row_keys()]


        
    ##### Mathematics #####
    def astype(self, dtype, copy=True):
        new_value = np.array(self.value, dtype=dtype, copy=copy)
        return self.__class__(new_value,
                              meta=self.meta.copy(),
                              xindex=self.xindex,
                              name=self.name,
                              epoch=self.epoch,
                              attrs=self.attrs,
                             )
    
    @property
    def real(self):
        new = (self + self.conj()) / 2
        return new.astype(float)
    
    @property
    def imag(self):
        new = (self - self.conj()) / (2j)
        return new.astype(float)

    def abs(self):
        new = abs(self)
        return new.astype(float)
        
    def angle(self, deg: bool = False):
        new = self.copy()
        new.meta = deepcopy(self.meta)
        new.value = np.angle(self.value, deg=deg)
        unit = u.deg if deg else u.rad
        for meta in new.meta.flat:
            meta["unit"] = unit
        return new.astype(float)

    @property
    def T(self):
        new_data = np.transpose(self._value, (1, 0, 2))
        new_meta = np.transpose(self.meta, (1, 0))
        return self.__class__(
            new_data,
            xindex=self.xindex,
            name=self.name,
            epoch=self.epoch,
            rows=self.cols,
            cols=self.rows,
            meta=MetaDataMatrix(new_meta),
            attrs=self.attrs,
        )

    
    def transpose(self):
        return self.T

    @property
    def dagger(self):
        """Hermitian transpose (complex conjugate then transpose)."""
        return self.conj().T


    def trace(self):
        """
        Matrix trace along the diagonal.

        Returns
        -------
        Series
            Sum of diagonal elements per sample.

        Raises
        ------
        ValueError
            If the matrix is not square.
        UnitConversionError
            If diagonal element units are not mutually equivalent.
        """
        nrow, ncol, _ = self._value.shape
        if nrow != ncol:
            raise ValueError("trace requires a square matrix")
        ref_unit = self.meta[0, 0].unit
        diag_values = []
        for i in range(nrow):
            u_ii = self.meta[i, i].unit
            if not u_ii.is_equivalent(ref_unit):
                raise u.UnitConversionError(f"Diagonal units not equivalent: {u_ii} vs {ref_unit}")
            diag_values.append(u.Quantity(self._value[i, i], u_ii).to_value(ref_unit))
        summed = np.sum(diag_values, axis=0)
        try:
            xi = self.xindex.copy()
        except Exception:
            xi = deepcopy(self.xindex)
        name = f"trace({self.name})" if getattr(self, "name", "") else "trace"
        return Series(summed, xindex=xi, unit=ref_unit, name=name)

    def diagonal(self, output: str = "list"):
        """
        Extract diagonal elements in various formats.

        Parameters
        ----------
        output : {'list', 'vector', 'matrix'}, optional
            Output form. See docstring for details.
        """
        nrow, ncol, nsamp = self._value.shape
        n = min(nrow, ncol)
        diag_series = []
        for i in range(n):
            meta = self.meta[i, i]
            try:
                xi = self.xindex.copy()
            except Exception:
                xi = deepcopy(self.xindex)
            diag_series.append(
                Series(
                    self._value[i, i],
                    xindex=xi,
                    unit=meta.unit,
                    name=meta.name,
                    channel=meta.channel,
                )
            )

        if output == "list":
            return diag_series

        if output == "vector":
            values = np.empty((n, 1, nsamp), dtype=self._value.dtype)
            meta_arr = np.empty((n, 1), dtype=object)
            for i in range(n):
                values[i, 0] = diag_series[i].value
                meta_arr[i, 0] = MetaData(**dict(self.meta[i, i]))
            rows_dict = _slice_metadata_dict(self.rows, list(range(n)), "row")
            cols_dict = MetaDataDict({"diag": MetaData()}, expected_size=1, key_prefix="col")
            return SeriesMatrix(
                values,
                xindex=self.xindex,
                rows=rows_dict,
                cols=cols_dict,
                meta=MetaDataMatrix(meta_arr),
                name=getattr(self, "name", ""),
                epoch=getattr(self, "epoch", 0.0),
                attrs=getattr(self, "attrs", {}),
            )

        if output == "matrix":
            values = np.zeros_like(self._value)
            for i in range(n):
                values[i, i] = self._value[i, i]
            # preserve metadata/labels as-is
            return SeriesMatrix(
                values,
                xindex=self.xindex,
                rows=self.rows,
                cols=self.cols,
                meta=self.meta,
                name=getattr(self, "name", ""),
                epoch=getattr(self, "epoch", 0.0),
                attrs=getattr(self, "attrs", {}),
            )

        raise ValueError("output must be one of {'list', 'vector', 'matrix'}")

    def det(self):
        """
        Determinant per sample.

        Returns
        -------
        Series
            Determinant values with appropriate unit.
        """
        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("det requires a square matrix")
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for det()")
        common = self._to_common_unit_values(ref_unit)
        # Batch computation: (N, N, K) -> (K, N, N)
        mats = np.moveaxis(common, 2, 0)
        det_vals = np.linalg.det(mats)
        result_unit = ref_unit ** nrow
        try:
            xi = self.xindex.copy()
        except Exception:
            xi = deepcopy(self.xindex)
        name = f"det({self.name})" if getattr(self, "name", "") else "det"
        return Series(det_vals, xindex=xi, unit=result_unit, name=name)

    def inv(self, swap_rowcol: bool = True):
        """
        Matrix inverse per sample.

        Parameters
        ----------
        swap_rowcol : bool, optional
            If True, swap row/col label dictionaries in the output.
        """
        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("inv requires a square matrix")
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for inv()")
        common = self._to_common_unit_values(ref_unit)
        mats = np.moveaxis(common, 2, 0)  # (K, N, N)
        inv_stack = np.linalg.inv(mats)
        inv_vals = np.moveaxis(inv_stack, 0, 2)  # (N, N, K)

        inv_unit = ref_unit ** -1
        meta_arr = np.empty((nrow, ncol), dtype=object)
        for i in range(nrow):
            for j in range(ncol):
                meta_arr[i, j] = MetaData(unit=inv_unit, name="", channel=None)
        meta_matrix = MetaDataMatrix(meta_arr)

        def _copy_meta_dict(md: MetaDataDict, prefix: str):
            items = OrderedDict()
            for k, v in md.items():
                items[k] = MetaData(**dict(v))
            return MetaDataDict(items, expected_size=len(md), key_prefix=prefix)

        rows_out = _copy_meta_dict(self.cols, "row") if swap_rowcol else _copy_meta_dict(self.rows, "row")
        cols_out = _copy_meta_dict(self.rows, "col") if swap_rowcol else _copy_meta_dict(self.cols, "col")

        return SeriesMatrix(
            inv_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=meta_matrix,
            name=f"inv({self.name})" if getattr(self, "name", "") else "inv",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    def schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None):
        """
        Schur complement of a selected block.

        Parameters
        ----------
        keep_rows : list
            Row keys/indices to keep.
        keep_cols : list, optional
            Column keys/indices to keep. Defaults to keep_rows.
        eliminate_rows/eliminate_cols : list, optional
            Indices to eliminate. Defaults to complement of keep sets.
        """
        nrow, ncol, nsamp = self._value.shape
        if keep_cols is None:
            keep_cols = keep_rows

        def _row_idx(k):
            return int(k) if isinstance(k, (int, np.integer)) else self.row_index(k)

        def _col_idx(k):
            return int(k) if isinstance(k, (int, np.integer)) else self.col_index(k)

        all_row_idx = list(range(nrow))
        all_col_idx = list(range(ncol))
        keep_rows_idx = [_row_idx(k) for k in keep_rows]
        keep_cols_idx = [_col_idx(k) for k in keep_cols]
        if eliminate_rows is None:
            eliminate_rows_idx = [i for i in all_row_idx if i not in keep_rows_idx]
        else:
            eliminate_rows_idx = [_row_idx(k) for k in eliminate_rows]
        if eliminate_cols is None:
            eliminate_cols_idx = [j for j in all_col_idx if j not in keep_cols_idx]
        else:
            eliminate_cols_idx = [_col_idx(k) for k in eliminate_cols]

        if len(eliminate_rows_idx) != len(eliminate_cols_idx):
            raise ValueError("Eliminated row/col sets must have the same size for Schur complement")
        if not keep_rows_idx or not keep_cols_idx:
            raise ValueError("Keep sets must be non-empty")

        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for schur()")
        common = self._to_common_unit_values(ref_unit)

        r_keep = len(keep_rows_idx)
        c_keep = len(keep_cols_idx)

        if len(eliminate_rows_idx) == 0:
            # If nothing to remove, extract as is
            result_vals = common[np.ix_(keep_rows_idx, keep_cols_idx)]
        else:
            # Batch linear algebra: (N, N, K) -> (K, N, N)
            stack = np.moveaxis(common, 2, 0)
            A = np.take(np.take(stack, keep_rows_idx, axis=1), keep_cols_idx, axis=2)
            B = np.take(np.take(stack, keep_rows_idx, axis=1), eliminate_cols_idx, axis=2)
            C = np.take(np.take(stack, eliminate_rows_idx, axis=1), keep_cols_idx, axis=2)
            D = np.take(np.take(stack, eliminate_rows_idx, axis=1), eliminate_cols_idx, axis=2)

            D_inv = np.linalg.inv(D)
            schur_block = A - np.matmul(np.matmul(B, D_inv), C)
            result_vals = np.moveaxis(schur_block, 0, 2)

        # build metadata for kept block (units set to ref_unit)
        meta_arr = np.empty((r_keep, c_keep), dtype=object)
        for ii, ri in enumerate(keep_rows_idx):
            for jj, cj in enumerate(keep_cols_idx):
                base_meta = self.meta[ri, cj]
                meta_arr[ii, jj] = MetaData(unit=ref_unit, name=base_meta.name, channel=base_meta.channel)

        def _subset_meta_dict(md: MetaDataDict, indices, prefix):
            items = OrderedDict()
            keys = list(md.keys())
            for idx in indices:
                key = keys[idx]
                items[key] = MetaData(**dict(md[key]))
            return MetaDataDict(items, expected_size=len(indices), key_prefix=prefix)

        rows_out = _subset_meta_dict(self.rows, keep_rows_idx, "row")
        cols_out = _subset_meta_dict(self.cols, keep_cols_idx, "col")

        return SeriesMatrix(
            result_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=MetaDataMatrix(meta_arr),
            name=f"schur({self.name})" if getattr(self, "name", "") else "schur",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

      
    ##### Edit forrwing the Sampling axis   ##### 
    def crop(self, start=None, end=None, copy=False):
        """
        Extract a subset along the sample axis based on xindex range.

        Parameters
        ----------
        start : float or Quantity or None
            Lower bound on xindex (inclusive). If None, start from the beginning.
        end : float or Quantity or None
            Upper bound on xindex (exclusive). If None, go to the end.
        copy : bool, optional
            If True, copy the underlying data; otherwise return a view.

        Returns
        -------
        SeriesMatrix
            New SeriesMatrix cropped to the requested xindex span.
        """
        xindex = self.xindex
        xunit = getattr(xindex, "unit", None)
        xvalues = getattr(xindex, "value", np.asarray(xindex))

        def _as_base(val):
            if val is None:
                return None
            if xunit is not None:
                return u.Quantity(val, xunit).to_value(xunit)
            return u.Quantity(val).value if isinstance(val, u.Quantity) else val

        start_val = _as_base(start)
        end_val = _as_base(end)

        idx0 = np.searchsorted(xvalues, start_val, side='left') if start_val is not None else 0
        idx1 = np.searchsorted(xvalues, end_val, side='left') if end_val is not None else len(xvalues)
        new_data = self.value[:, :, idx0:idx1] if self._value.ndim == 3 else self.value[:, idx0:idx1]
        if copy:
            new_data = np.array(new_data, copy=True)
        new_xindex = xindex[idx0:idx1]
        return SeriesMatrix(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )

    def append(self, other, inplace=True, pad=None, gap=None, resize=True):
        """
        Append another SeriesMatrix along the sample axis with gwpy-like gap handling.
        """
        if gap is None:
            gap = 'raise' if pad is None else 'pad'
        if pad is None and gap == 'pad':
            pad = 0.0

        # Use self directly here to avoid unnecessary deep copy even if not inplace
        # (append_exact/is_contiguous creates new array, so self is not destroyed)
        target = self

        base_unit = getattr(target.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return u.Quantity(val, base_unit).to_value(base_unit)

        def _is_regular(xidx):
            if hasattr(xidx, "regular"):
                return xidx.regular
            try:
                arr = np.asarray(xidx, dtype=float)
                if arr.size < 2:
                    return True
                diffs = np.diff(arr)
                return np.allclose(diffs, diffs[0])
            except Exception:
                return False

        def _concat_ignore(a: "SeriesMatrix", b: "SeriesMatrix") -> "SeriesMatrix":
            new_data = np.concatenate([a.value, b.value], axis=2)
            if base_unit is None:
                new_xindex = np.concatenate([np.asarray(a.xindex), np.asarray(b.xindex)])
            else:
                ax = u.Quantity(a.xindex).to_value(base_unit)
                bx = u.Quantity(b.xindex).to_value(base_unit)
                new_xindex = np.concatenate([ax, bx]) * base_unit
            return SeriesMatrix(
                new_data,
                xindex=new_xindex,
                rows=a.rows,
                cols=a.cols,
                meta=a.meta,
                name=a.name,
                epoch=a.epoch,
                attrs=a.attrs,
            )

        cont = target.is_contiguous(other)
        if cont != 1:
            _gap_is_numeric = isinstance(gap, (int, float, np.number, u.Quantity))
            _use_pad = (gap == 'pad') or (_gap_is_numeric and pad is not None)
            if _use_pad:
                if not _is_regular(target.xindex):
                    raise ValueError("Padding gap requires regular xindex")
                # Even if pad is allowed, reject reversed or duplicate order first
                s0, s1 = target.xspan
                o0, _ = other.xspan
                s0b = _to_base(s0)
                s1b = _to_base(s1)
                o0b = _to_base(o0)
                if s0b <= o0b < s1b:
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                if o0b < s0b:
                    raise ValueError("Cannot append that starts before this one")

                gap_base = np.inf if gap == "pad" else _to_base(gap)
                out_full = target.append_exact(other, inplace=False, pad=pad, gap=gap_base)
            elif gap == 'ignore':
                out_full = _concat_ignore(target, other)
            else:
                s0, s1 = target.xspan
                o0, _ = other.xspan
                if _to_base(s0) < _to_base(o0) < _to_base(s1):
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                raise ValueError("Cannot append discontiguous SeriesMatrix")
        else:
            out_full = target.append_exact(other, inplace=False, pad=None, gap=None)

        if not resize:
            keep = target._value.shape[2]
            if out_full._value.shape[2] > keep:
                out_full = out_full[:, :, -keep:]

        if not inplace:
            return out_full

        try:
            self.resize(out_full.shape, refcheck=False)
            self[:] = out_full[:]
            self.xindex = out_full.xindex
            self.meta = out_full.meta
            self.rows = out_full.rows
            self.cols = out_full.cols
            self.name = out_full.name
            self.epoch = out_full.epoch
            self.attrs = out_full.attrs
            self._value = self.view(np.ndarray)
            return self
        except Exception:
            # Fallback: return out_full non-destructively
            return out_full

    def append_exact(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        """
        Append another SeriesMatrix along the sample axis without resizing rows or columns.

        Parameters
        ----------
        other : SeriesMatrix
            Matrix to append.
        inplace : bool, optional
            If True, mutate this instance; otherwise return a new matrix.
        pad : number or None, optional
            Fill value used when ``gap`` is allowed; ``None`` or ``\"nan\"`` uses ``np.nan``.
        gap : float or None, optional
            Maximum additional spacing (in base xindex units) tolerated beyond one sample.
        tol : float, optional
            Absolute tolerance for detecting contiguity.

        Returns
        -------
        SeriesMatrix
            Concatenated matrix.
        """
        if self._value.shape[:2] != other._value.shape[:2]:
            raise ValueError(f"shape[:2] does not match: {self._value.shape[:2]} vs {other._value.shape[:2]}")
        if self._value.ndim != other._value.ndim:
            raise ValueError(f"ndim does not match: {self._value.ndim} vs {other._value.ndim}")
        if list(self.rows.keys()) != list(other.rows.keys()):
            raise ValueError("row keys do not match")
        if list(self.cols.keys()) != list(other.cols.keys()):
            raise ValueError("col keys do not match")
        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}")

        base_unit = getattr(self.xindex, "unit", None)
        if base_unit is None:
            base_unit = getattr(other.xindex, "unit", None)

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        if hasattr(self.xindex, "regular") and self.xindex.regular:
            dx_q = self.dx
        else:
            if len(self.xindex) < 2:
                raise ValueError("Cannot infer dx from xindex with length < 2")
            dx_q = self.xindex[-1] - self.xindex[-2]
        dx_self = _to_base(dx_q)
        self_last = _to_base(self.xindex[-1])
        other_first = _to_base(other.xindex[0])
        diff = other_first - self_last

        if np.isclose(diff, dx_self, atol=tol) or abs(diff) < tol:
            new_data = np.concatenate([self.value, other.value], axis=2)
            if hasattr(self.xindex, 'regular') and self.xindex.regular:
                new_xindex = Index.define(self.xindex[0], dx_q, new_data.shape[2])
            else:
                new_xindex = np.concatenate([self.xindex, other.xindex])
        elif gap is not None and abs(diff - dx_self) <= gap:
            n_gap = int(np.round((diff - dx_self) / dx_self))
            pad_value = np.nan if pad is None or pad == "nan" else pad
            pad_block = np.full((self._value.shape[0], self._value.shape[1], n_gap), pad_value, dtype=self._value.dtype)
            if base_unit is not None:
                pad_times = self_last + dx_self * np.arange(1, n_gap+1)
                pad_xindex = pad_times * base_unit
            else:
                pad_xindex = self_last + dx_self * np.arange(1, n_gap+1)
            new_data = np.concatenate([self.value, pad_block, other.value], axis=2)
            if hasattr(self.xindex, 'regular') and self.xindex.regular:
                new_xindex = Index.define(self.xindex[0], dx_q, new_data.shape[2])
            else:
                new_xindex = np.concatenate([self.xindex, pad_xindex, other.xindex])
        else:
            raise ValueError(
                f"gap detected: {diff} [{base_unit if base_unit is not None else 'dimensionless'}], but gap={gap} specified."
            )
        result = SeriesMatrix(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )
        if inplace:
            self.value = result.value
            self.xindex = result.xindex
            return self
        return result



    def prepend(self, other, inplace=True, pad=None, gap=None, resize=True):
        """
        Prepend another SeriesMatrix along the sample axis.

        Parameters
        ----------
        other : SeriesMatrix
            Matrix to place before this one.
        inplace : bool, optional
            If True, mutate this instance; otherwise return a new matrix.
        pad : number or None, optional
            Fill value used when ``gap`` is allowed.
        gap : float or None, optional
            Maximum tolerated gap (in base xindex units) beyond one sample.
        resize : bool, optional
            If False, behave like ``update`` (no resizing); otherwise allow resizing.

        Returns
        -------
            SeriesMatrix
                Prepend result.
        """
        out = other.append(self, inplace=False, pad=pad, gap=gap, resize=resize)
        if not inplace:
            return out
        try:
            self.resize(out.shape, refcheck=False)
            self[:] = out[:]
            self.xindex = out.xindex
            self.meta = out.meta
            self.rows = out.rows
            self.cols = out.cols
            self.name = out.name
            self.epoch = out.epoch
            self.attrs = out.attrs
            self._value = self.view(np.ndarray)
            return self
        except Exception:
            return out

    def prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        """Exact prepend counterpart of :meth:`append_exact`."""
        return other.append_exact(self, inplace=inplace, pad=pad, gap=gap, tol=tol)

    def update(self, other, inplace=True, pad=None, gap=None):
        """
        Append without resizing, mirroring ``Series.update`` in gwpy.

        Parameters
        ----------
        other : SeriesMatrix
            Matrix to append.
        inplace : bool, optional
            If True, mutate this instance.
        pad : number or None, optional
            Fill value used when ``gap`` is allowed.
        gap : float or None, optional
            Maximum tolerated gap (in base xindex units) beyond one sample.

        Returns
        -------
        SeriesMatrix
            Updated matrix.
        """
        return self.append(other, inplace=inplace, pad=pad, gap=gap, resize=False)

    def diff(self, n=1, axis=2):
        """
        Compute the n-th discrete difference along the sample axis.

        Parameters
        ----------
        n : int, optional
            Order of the difference.
        axis : int, optional
            Must be 2 (sample axis).

        Returns
        -------
        SeriesMatrix
            Differenced matrix with shortened sample axis.
        """
        if axis != 2:
            raise ValueError("SeriesMatrix.diff supports only axis=2 (sample axis)")
        if n < 0:
            raise ValueError("n must be non-negative")
        new_values = np.diff(self.value, n=n, axis=2)
        if hasattr(self.xindex, "__len__") and len(self.xindex) > n:
            try:
                dx = self.dx
                new_xindex = Index.define(self.xindex[n], dx, new_values.shape[2])
            except Exception:
                new_xindex = self.xindex[n:]
        else:
            new_xindex = self.xindex
        return self.__class__(
            new_values,
            xindex=new_xindex,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )

    def value_at(self, x):
        """
        Return the (Nrow, Ncol) values at the first matching xindex entry.

        Parameters
        ----------
        x : scalar or Quantity
            Target xindex value.

        Returns
        -------
        ndarray
            Matrix of values corresponding to ``x``.
        """
        xidx = self.xindex
        base_unit = getattr(xidx, "unit", None)
        if base_unit is not None:
            x_val = u.Quantity(x, base_unit).to_value(base_unit)
            x_array = u.Quantity(xidx).to_value(base_unit)
        else:
            x_val = float(u.Quantity(x).value) if isinstance(x, u.Quantity) else x
            x_array = np.asarray(xidx)
        matches = np.nonzero(x_array == x_val)[0]
        if len(matches) == 0:
            raise ValueError(f"Value {x} not found in xindex")
        idx = matches[0]
        return self.value[:, :, idx]

    def pad(self, pad_width, **kwargs):
        """
        Pad along the sample axis (axis=2). Assumes a regular xindex.

        Parameters
        ----------
        pad_width : int or tuple(int, int)
            Number of samples to add before and after. A single int applies symmetrically.
        **kwargs :
            Additional keyword arguments forwarded to ``np.pad``.

        Returns
        -------
        SeriesMatrix
            Padded matrix with updated xindex.
        """
        kwargs.setdefault("mode", "constant")
        if isinstance(pad_width, int):
            before = after = pad_width
        else:
            try:
                before, after = pad_width
            except Exception:
                raise ValueError("pad_width must be int or tuple of (before, after)")
        pad_spec = ((0, 0), (0, 0), (before, after))
        new_values = np.pad(self.value, pad_spec, **kwargs)

        dx = self.dx
        start = self.xindex[0] - dx * before
        new_len = new_values.shape[2]
        new_xindex = Index.define(start, dx, new_len)

        result = self.__class__(
            new_values,
            xindex=new_xindex,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )
        return result

    def shift(self, delta):
        """
        Shift the xindex by a constant offset.

        Parameters
        ----------
        delta : scalar or Quantity
            Offset added to each entry in ``xindex``.

        Returns
        -------
        SeriesMatrix
            The modified instance.
        """
        xidx = self.xindex
        xunit = getattr(xidx, "unit", None)
        if xunit is not None:
            delta_val = u.Quantity(delta, xunit)
            new_xindex = xidx + delta_val
        else:
            delta_val = float(u.Quantity(delta).value) if isinstance(delta, u.Quantity) else delta
            new_xindex = np.asarray(xidx) + delta_val
        self.xindex = new_xindex
        try:
            self._x0 = new_xindex[0]
        except Exception:
            pass
        return self

    def copy(self, order='C'):
        """
        Return a deep copy of values, xindex, and metadata.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Memory order passed to ``np.array`` when copying values.

        Returns
        -------
        SeriesMatrix
            A new instance with duplicated data and metadata.
        """
        new_values = np.array(self._value, copy=True, order=order)
        try:
            xindex_copy = self.xindex.copy()
        except Exception:
            xindex_copy = deepcopy(self.xindex)

        def _copy_meta_dict(md: MetaDataDict, prefix: str):
            items = OrderedDict()
            for k, v in md.items():
                items[k] = MetaData(**dict(v))
            return MetaDataDict(items, expected_size=len(md), key_prefix=prefix)

        rows_copy = _copy_meta_dict(self.rows, "row")
        cols_copy = _copy_meta_dict(self.cols, "col")

        meta_arr = np.empty(self.meta.shape, dtype=object)
        for idx, m in enumerate(self.meta.flat):
            meta_arr.flat[idx] = MetaData(**dict(m))
        meta_copy = MetaDataMatrix(meta_arr)

        attrs_copy = deepcopy(getattr(self, "attrs", {}))

        return self.__class__(
            new_values,
            xindex=xindex_copy,
            meta=meta_copy,
            rows=rows_copy,
            cols=cols_copy,
            name=self.name,
            epoch=self.epoch,
            attrs=attrs_copy,
        )

    def step(self, where="post", **kwargs):
        """
        Convenience wrapper for step-style plotting.

        Parameters
        ----------
        where : {'pre', 'post', 'mid'}, optional
            Location parameter forwarded to matplotlib step plotting.
        **kwargs :
            Additional plotting keyword arguments.

        Returns
        -------
        object
            Plot object returned by :meth:`plot`.
        """
        kwargs.setdefault("drawstyle", f"steps-{where}")
        return self.plot(method="plot", **kwargs)

    # -- I/O (HDF5) -------------------------------------------------
    def to_pandas(self, format="wide"):
        """
        Convert SeriesMatrix to pandas DataFrame.

        Parameters
        ----------
        format : {'wide', 'long'}, optional
            'wide': (Default) Index is xindex. Columns are flattened channel keys "row_col".
            'long': Columns are [index, row, col, value].

        Returns
        -------
        pandas.DataFrame
        """
        pd = require_optional("pandas")

        if format == "wide":
            N, M, K = self._value.shape
            
            # Move sample axis to front: (K, N, M)
            val_T = np.moveaxis(self._value, -1, 0)
            # Flatten N and M: (K, N*M)
            val_flat = val_T.reshape(K, -1)
            
            # Prepare columns
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            
            col_names = []
            for r in r_keys:
                for c in c_keys:
                    col_names.append(f"{r}_{c}")
            
            # Prepare index
            xidx = self.xindex
            idx_name = "index"
            if isinstance(xidx, u.Quantity):
                idx_name = f"index [{xidx.unit}]"
                xidx = xidx.value
                
            df = pd.DataFrame(val_flat, index=xidx, columns=col_names)
            df.index.name = idx_name
            return df

        elif format == "long":
            N, M, K = self._value.shape
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            
            xidx = self.xindex
            if isinstance(xidx, u.Quantity):
                xidx = xidx.value
            
            long_index = np.tile(xidx, N * M)
            
            val_list = []
            row_list = []
            col_list = []
            
            for r in r_keys:
                for c in c_keys:
                    i = self.row_index(r)
                    j = self.col_index(c)
                    val_list.append(self._value[i, j])
                    row_list.extend([r] * K)
                    col_list.extend([c] * K)
                    
            long_values = np.concatenate(val_list)
            
            df = pd.DataFrame({
                "index": long_index,
                "row": row_list,
                "col": col_list,
                "value": long_values
            })
            return df
            
        else:
            raise ValueError(f"Unknown format: {format}")

    def write(self, target, format=None, **kwargs):
        """
        Write SeriesMatrix to file.

        Parameters
        ----------
        target : str
            Target filename.
        format : str, optional
            'hdf5', 'csv', 'parquet'. If None, inferred from extension.
        **kwargs
            Passed to underlying writer.
        """
        from pathlib import Path
        
        if format is None:
            ext = Path(target).suffix.lower()
            if ext in [".h5", ".hdf5", ".hdf"]:
                format = "hdf5"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".parquet", ".pq"]:
                format = "parquet"
            else:
                format = "hdf5"
        
        if format == "hdf5":
            return self.to_hdf5(target, **kwargs)
        elif format == "csv":
            df = self.to_pandas(format="wide")
            return df.to_csv(target, **kwargs)
        elif format == "parquet":
            df = self.to_pandas(format="wide")
            return df.to_parquet(target, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_hdf5(self, filepath, **kwargs):
        """
        Write the SeriesMatrix to a simple HDF5 representation.


        Parameters
        ----------
        filepath : str or path-like
            Destination path.
        **kwargs :
            Extra keyword arguments passed to ``h5py.File``.
        """
        import h5py

        with h5py.File(filepath, "w", **kwargs) as f:
            f.attrs["name"] = str(getattr(self, "name", ""))
            try:
                f.attrs["epoch"] = float(getattr(self, "epoch", 0.0))
            except Exception:
                pass
            attrs_dict = getattr(self, "attrs", None)
            if attrs_dict is not None:
                try:
                    f.attrs["attrs_json"] = json.dumps(attrs_dict)
                except Exception:
                    pass

            # data
            f.create_dataset("data", data=self.value)

            # xindex
            grp_x = f.create_group("xindex")
            if isinstance(self.xindex, u.Quantity):
                grp_x.create_dataset("value", data=np.asarray(self.xindex.value))
                grp_x.attrs["unit"] = str(self.xindex.unit)
            else:
                grp_x.create_dataset("value", data=np.asarray(self.xindex))

            # meta (per element)
            meta_grp = f.create_group("meta")
            units = np.vectorize(lambda u_: "" if u_ is None else str(u_))(self.meta.units)
            names = np.vectorize(lambda n: "" if n is None else str(n))(self.meta.names)
            channels = np.vectorize(lambda c: "" if c is None else str(c))(self.meta.channels)
            meta_grp.create_dataset("units", data=units.astype("S"))
            meta_grp.create_dataset("names", data=names.astype("S"))
            meta_grp.create_dataset("channels", data=channels.astype("S"))

            # rows
            row_grp = f.create_group("rows")
            row_keys = np.array(list(self.rows.keys()), dtype="S")
            row_names = np.array([str(v.name) for v in self.rows.values()], dtype="S")
            row_units = np.array([str(v.unit) for v in self.rows.values()], dtype="S")
            row_channels = np.array([str(v.channel) for v in self.rows.values()], dtype="S")
            row_grp.create_dataset("keys", data=row_keys)
            row_grp.create_dataset("names", data=row_names)
            row_grp.create_dataset("units", data=row_units)
            row_grp.create_dataset("channels", data=row_channels)

            # cols
            col_grp = f.create_group("cols")
            col_keys = np.array(list(self.cols.keys()), dtype="S")
            col_names = np.array([str(v.name) for v in self.cols.values()], dtype="S")
            col_units = np.array([str(v.unit) for v in self.cols.values()], dtype="S")
            col_channels = np.array([str(v.channel) for v in self.cols.values()], dtype="S")
            col_grp.create_dataset("keys", data=col_keys)
            col_grp.create_dataset("names", data=col_names)
            col_grp.create_dataset("units", data=col_units)
            col_grp.create_dataset("channels", data=col_channels)

    @classmethod
    def read(cls, filepath, **kwargs):
        """
        Read a SeriesMatrix from the HDF5 format produced by :meth:`write`.

        Parameters
        ----------
        filepath : str or path-like
            Source path.
        **kwargs :
            Extra keyword arguments passed to ``h5py.File``.

        Returns
        -------
        SeriesMatrix
            Reconstructed instance.
        """
        import h5py

        with h5py.File(filepath, "r", **kwargs) as f:
            data = f["data"][...]

            # xindex
            grp_x = f["xindex"]
            xval = grp_x["value"][...]
            xunit = grp_x.attrs.get("unit", None)
            xindex = u.Quantity(xval, xunit) if xunit else xval

            # meta
            meta_grp = f["meta"]
            units = meta_grp["units"][...].astype(str)
            names = meta_grp["names"][...].astype(str)
            channels = meta_grp["channels"][...].astype(str)
            N, M = units.shape
            meta_arr = np.empty((N, M), dtype=object)
            for i in range(N):
                for j in range(M):
                    unit_str = units[i, j]
                    name_str = names[i, j]
                    chan_str = channels[i, j]
                    meta_arr[i, j] = MetaData(
                        unit=unit_str if unit_str else u.dimensionless_unscaled,
                        name=name_str,
                        channel=chan_str,
                    )
            meta_matrix = MetaDataMatrix(meta_arr)

            # rows
            row_grp = f["rows"]
            row_keys = row_grp["keys"][...].astype(str)
            row_names = row_grp["names"][...].astype(str)
            row_units = row_grp["units"][...].astype(str)
            row_channels = row_grp["channels"][...].astype(str)
            rows = OrderedDict()
            for k, n, ustr, c in zip(row_keys, row_names, row_units, row_channels):
                rows[k] = MetaData(name=n, unit=ustr if ustr else u.dimensionless_unscaled, channel=c)

            # cols
            col_grp = f["cols"]
            col_keys = col_grp["keys"][...].astype(str)
            col_names = col_grp["names"][...].astype(str)
            col_units = col_grp["units"][...].astype(str)
            col_channels = col_grp["channels"][...].astype(str)
            cols = OrderedDict()
            for k, n, ustr, c in zip(col_keys, col_names, col_units, col_channels):
                cols[k] = MetaData(name=n, unit=ustr if ustr else u.dimensionless_unscaled, channel=c)

            name = f.attrs.get("name", "")
            epoch = f.attrs.get("epoch", 0.0)
            attrs_json = f.attrs.get("attrs_json", None)
            try:
                attrs = json.loads(attrs_json) if attrs_json is not None else {}
            except Exception:
                attrs = {}

        return cls(
            data,
            xindex=xindex,
            meta=meta_matrix,
            rows=rows,
            cols=cols,
            name=name,
            epoch=epoch,
            attrs=attrs,
        )

        
    ##### Visualizations #####
    def __repr__(self): 
        try:
            return f"<SeriesMatrix shape={self.shape3D} rows={self.row_keys()} cols={self.col_keys()}>"
        except Exception:
            return "<SeriesMatrix (incomplete or empty)>"

    def __str__(self):
        info = (
            f"SeriesMatrix(shape={self._value.shape},  name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xunit   : {self.xunit}\n"
            f"  samples : {self.N_samples}\n"
        )
    
        info += "\n[ Row metadata ]\n" + str(self.rows)
        info += "\n\n[ Column metadata ]\n" + str(self.cols)
    
        if hasattr(self, 'meta'):
            info += "\n\n[ Elements metadata ]\n" + str(self.meta)
    
        return info
    
    def _repr_html_(self):
        html = f"<h3>SeriesMatrix: shape={self._value.shape}, name='{escape(str(self.name))}'</h3>"
        html += "<ul>"
        html += f"<li><b>epoch:</b> {escape(str(self.epoch))}</li>"
        html += f"<li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li>"
        html += f"<li><b>xunit:</b> {escape(str(self.xunit))}</li>"
        html += "</ul>"
    
        html += "<h4>Row Metadata</h4>" + self.rows._repr_html_()
        html += "<h4>Column Metadata</h4>" + self.cols._repr_html_()
    
        if hasattr(self, 'meta'):
            html += "<h4>Element Metadata</h4>" + self.meta._repr_html_()
    
        if self.attrs:
            html += "<h4>Attributes</h4><pre>" + escape(str(self.attrs)) + "</pre>"
    
        return html

