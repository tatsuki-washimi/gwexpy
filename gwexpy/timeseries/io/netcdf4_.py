"""NetCDF4 reader/writer for gwexpy (via xarray).

Reads variables that have a ``time`` dimension and converts them to
:class:`~gwexpy.timeseries.TimeSeries`.
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from collections import OrderedDict

import numpy as np

from gwexpy.io.time_selection import apply_time_selection, pop_time_selection
from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_dependency,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

logger = logging.getLogger(__name__)
_MATRIX_VAR_PREFIX = "__gwexpy_matrix__"
_NETCDF_SCHEMA_VERSION = 2
_NETCDF_AXIS_ENCODING = "t(i)=t0+i*dt"


def _to_json_native(val):
    """Convert a value to a JSON-serializable Python native type.

    numpy integer/float/bool scalars expose .item() which maps to the
    corresponding Python built-in.  Tuples and lists are recursively converted.
    Exotic types (datetime64, timedelta64, …) that remain non-serializable
    fall back to str().
    """
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, (list, tuple)):
        return [_to_json_native(item) for item in val]
    if not isinstance(val, (bool, int, float, str, type(None))):
        return str(val)
    return val


def _encode_netcdf_var_name(key) -> str:
    """Convert mapping keys to NetCDF-safe variable names.

    For tuple keys the variable name uses a SHA-256 hash of the repr so that
    the name is always unique, contains only hex characters, and contains no
    illegal NetCDF4 characters (parentheses, spaces, …).  The true row/col
    identity is always stored in ``gwexpy_row_key``/``gwexpy_col_key``
    attributes; the variable name itself only needs to be unique and valid.
    """
    if isinstance(key, tuple):
        import hashlib

        h = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
        return f"{_MATRIX_VAR_PREFIX}{h}"
    return str(key)


def _decode_netcdf_key(raw):
    """Deserialize a key stored as JSON; fall back to str for legacy files.

    Recursively converts nested lists to tuples (matching the Zarr decoder
    behavior) so that round-trips preserve tuple keys and maintain hashability.
    """
    if raw is None:
        return None

    def _normalize(decoded):
        """Recursively convert lists to tuples."""
        if isinstance(decoded, list):
            return tuple(_normalize(item) for item in decoded)
        return decoded

    try:
        result = json.loads(raw)
        return _normalize(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(raw)


def _import_xarray():
    try:
        xr = ensure_dependency("xarray", extra="netcdf4")
    except ImportError as exc:
        raise ImportError(
            "xarray is required for reading/writing NetCDF4 files. "
            "Install with `pip install 'gwexpy[netcdf4]'`."
        ) from exc
    return xr


def _normalize_channels(channels, available: list[str]) -> list[str]:
    """Validate a channel selector before any variable values are loaded."""
    if channels is None:
        return sorted(available)
    selected = [channels] if isinstance(channels, str) else list(channels)
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate NetCDF channels requested: {duplicates}")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"NetCDF channels not found: {missing}")
    return selected


def _v2_timing_attrs(t0: float, dt: float) -> dict[str, object]:
    """Encode exact float timing plus interoperable GPS seconds/nanoseconds."""
    gps_seconds = math.floor(t0)
    gps_nanoseconds = int(round((t0 - gps_seconds) * 1_000_000_000))
    if gps_nanoseconds >= 1_000_000_000:
        gps_seconds += 1
        gps_nanoseconds -= 1_000_000_000
    if gps_nanoseconds < 0:
        gps_seconds -= 1
        gps_nanoseconds += 1_000_000_000
    numerator, denominator = dt.as_integer_ratio()
    return {
        "gwexpy_netcdf_schema_version": _NETCDF_SCHEMA_VERSION,
        "gwexpy_t0_float_hex": t0.hex(),
        "gwexpy_t0_gps_seconds": np.int64(gps_seconds),
        "gwexpy_t0_gps_nanoseconds": np.int32(gps_nanoseconds),
        "gwexpy_dt_numerator": str(numerator),
        "gwexpy_dt_denominator": str(denominator),
        "gwexpy_axis_encoding": _NETCDF_AXIS_ENCODING,
    }


def _read_v2_timing(ds) -> tuple[float, float]:
    """Decode and validate the v2 timing metadata."""
    attrs = ds.attrs
    required = {
        "gwexpy_t0_float_hex",
        "gwexpy_t0_gps_seconds",
        "gwexpy_t0_gps_nanoseconds",
        "gwexpy_dt_numerator",
        "gwexpy_dt_denominator",
        "gwexpy_axis_encoding",
    }
    missing = sorted(required.difference(attrs))
    if missing:
        raise ValueError(f"NetCDF v2 file is missing timing metadata: {missing}")
    if attrs["gwexpy_axis_encoding"] != _NETCDF_AXIS_ENCODING:
        raise ValueError("unsupported NetCDF v2 axis encoding")
    t0 = float.fromhex(str(attrs["gwexpy_t0_float_hex"]))
    numerator = int(str(attrs["gwexpy_dt_numerator"]))
    denominator = int(str(attrs["gwexpy_dt_denominator"]))
    if denominator <= 0:
        raise ValueError("NetCDF v2 dt denominator must be positive")
    dt = numerator / denominator
    if not math.isfinite(t0) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("NetCDF v2 timing must be finite with positive dt")
    gps_seconds = int(attrs["gwexpy_t0_gps_seconds"])
    gps_nanoseconds = int(attrs["gwexpy_t0_gps_nanoseconds"])
    if not 0 <= gps_nanoseconds < 1_000_000_000:
        raise ValueError("NetCDF v2 GPS nanoseconds must be normalized")
    if abs((gps_seconds + gps_nanoseconds / 1_000_000_000) - t0) > 0.5e-9:
        raise ValueError("NetCDF v2 GPS timing disagrees with t0 float metadata")
    return t0, dt


def _validate_v2_series(tsd) -> tuple[int, float, float]:
    """Reject unsupported values and heterogeneous axes before opening target."""
    if not tsd:
        raise ValueError("Cannot write empty TimeSeriesDict to NetCDF4")
    first = next(iter(tsd.values()))
    n_samples = len(first)
    t0 = float(first.t0.value)
    dt = float(first.dt.value)
    if n_samples == 0:
        raise ValueError("NetCDF v2 cannot write empty time series")
    if not math.isfinite(t0) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("NetCDF v2 requires finite t0 and positive dt")
    for key, ts in tsd.items():
        values = np.asarray(ts.value)
        dtype = values.dtype
        native = (dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8)) or dtype in (
            np.dtype("float32"),
            np.dtype("float64"),
        )
        if not native:
            raise TypeError(f"unsupported dtype for NetCDF v2 channel {key!r}: {dtype}")
        if len(ts) == 0:
            raise ValueError("NetCDF v2 cannot write empty time series")
        if len(ts) != n_samples:
            raise ValueError("NetCDF v2 channels must have equal lengths")
        if float(ts.t0.value).hex() != t0.hex() or float(ts.dt.value).hex() != dt.hex():
            raise ValueError("NetCDF v2 channels must have identical t0 and dt")
    return n_samples, t0, dt


def _time_coord_name(ds):
    """Return the name of the time coordinate, or *None*.

    Prefers an explicitly named coordinate (``time``/``Time``/``TIME``/``t``).
    Only if none is present does it fall back to a datetime64 coordinate, in
    which case it warns -- and warns more loudly when the choice is ambiguous
    (several datetime64 coordinates) -- so a wrong-axis guess is never silent.
    Pass ``time_coord=`` to the reader to select the axis explicitly.
    """
    import warnings

    for name in ("time", "Time", "TIME", "t"):
        if name in ds.coords:
            return name
    # Fallback: datetime64 coordinate(s)
    datetime_coords = [
        name
        for name, coord in ds.coords.items()
        if np.issubdtype(coord.dtype, np.datetime64)
    ]
    if not datetime_coords:
        return None
    chosen = datetime_coords[0]
    if len(datetime_coords) > 1:
        # Genuinely ambiguous: the "first datetime64" guess may pick the wrong
        # axis.  Warn instead of choosing silently.
        warnings.warn(
            f"NetCDF4 file has no standard time coordinate and multiple "
            f"datetime64 coordinates {datetime_coords}; guessing '{chosen}'. "
            f"Pass time_coord=... to choose the time axis explicitly.",
            UserWarning,
            stacklevel=2,
        )
    return chosen


def _legacy_timing(ds, tc) -> tuple[float, float]:
    """Decode the unversioned, datetime-based NetCDF representation."""
    time_vals = ds[tc].values
    if len(time_vals) == 0:
        raise ValueError("NetCDF time coordinate must not be empty")
    if np.issubdtype(np.asarray(time_vals).dtype, np.datetime64):
        import datetime as _dt

        t0_dt64 = time_vals[0]
        t0_unix_ns = (t0_dt64 - np.datetime64("1970-01-01T00:00:00", "ns")).astype(
            np.int64
        )
        t0_datetime = _dt.datetime.fromtimestamp(t0_unix_ns / 1e9, tz=_dt.UTC)
        t0 = datetime_to_gps(t0_datetime)
        dt = (
            float(np.median(np.diff(time_vals.astype("datetime64[ns]").astype(np.int64))))
            / 1e9
            if len(time_vals) > 1
            else 1.0
        )
    else:
        numeric = np.asarray(time_vals, dtype=np.float64)
        t0 = float(numeric[0])
        dt = float(np.median(np.diff(numeric))) if len(numeric) > 1 else 1.0
    return t0, dt


def read_timeseriesdict_netcdf4(
    source,
    *,
    channels=None,
    unit=None,
    time_coord=None,
    **kwargs,
) -> TimeSeriesDict:
    """Read a NetCDF4 file into a TimeSeriesDict.

    Parameters
    ----------
    source : str, path-like, or list of str/path-like
        Path to a ``.nc`` file, or a list of paths.  When a list is
        given, variables found in several files are concatenated along
        the time axis and variables unique to one file are merged in.
    channels : iterable of str, optional
        Variable names to read.  If *None*, all variables with a time
        dimension are loaded.
    unit : str, optional
        Physical unit override applied to every channel.
    time_coord : str, optional
        Name of the time coordinate.  Auto-detected if *None*.
    start, end : float, optional
        GPS bounds.  The file is read in full and the result is cropped, so
        this returns exactly ``read(source).crop(start, end)``.
    **kwargs
        Additional keyword arguments forwarded to ``xarray.open_dataset``.

    """
    # Taken out before the multi-source dispatch so the crop happens once, on
    # the merged result, rather than per file — the two agree for disjoint
    # files but only the former is the documented oracle.
    start, end = pop_time_selection(kwargs)

    multi = expand_multi_source(source)
    if multi is not None:
        return apply_time_selection(
            read_multi_dict(
                read_timeseriesdict_netcdf4,
                multi,
                "nc",
                channels=channels,
                unit=unit,
                time_coord=time_coord,
                **kwargs,
            ),
            start,
            end,
        )

    xr = _import_xarray()

    # gwpy's registry may pass a file-like object; extract the path.
    if hasattr(source, "name"):
        source = source.name

    # Strip gwpy-injected kwargs that xarray.open_dataset does not accept.
    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}
    ds = xr.open_dataset(str(source), **xr_kwargs)
    try:
        is_v2 = ds.attrs.get("gwexpy_netcdf_schema_version") == _NETCDF_SCHEMA_VERSION
        if is_v2:
            tc = "sample"
            if tc not in ds.coords:
                raise ValueError("NetCDF v2 file is missing int64 sample coordinate")
            if ds[tc].dtype != np.dtype("int64"):
                raise ValueError("NetCDF v2 sample coordinate must be int64")
            t0, dt = _read_v2_timing(ds)
        else:
            warnings.warn(
                "Reading unversioned legacy NetCDF; timing precision is limited.",
                RuntimeWarning,
                stacklevel=2,
            )
            tc = time_coord or _time_coord_name(ds)
            if tc is None:
                raise ValueError(
                    "No time coordinate found in the NetCDF4 file. "
                    "Specify one explicitly via time_coord='...'."
                )
            t0, dt = _legacy_timing(ds, tc)

        tsd = TimeSeriesDict()
        available = sorted(name for name, da in ds.data_vars.items() if tc in da.dims)
        var_names = _normalize_channels(channels, available)
        for var in var_names:
            da = ds[var]

            data = da.values
            # Handle multi-dimensional variables: flatten non-time dims
            if data.ndim > 1:
                time_axis = list(da.dims).index(tc)
                # Move time axis first, then flatten remaining
                data = np.moveaxis(data, time_axis, 0)
                data = data.reshape(data.shape[0], -1)
                # Create one channel per flattened index
                for i in range(data.shape[1]):
                    ch_name = f"{var}_{i}" if data.shape[1] > 1 else var
                    var_unit = unit or da.attrs.get("units") or da.attrs.get("unit")
                    ts = TimeSeries(
                        data[:, i],
                        t0=t0,
                        dt=dt,
                        name=ch_name,
                        channel=ch_name,
                    )
                    ts = apply_unit(ts, var_unit) if var_unit else ts
                    tsd[ch_name] = ts
            else:
                var_unit = unit or da.attrs.get("units") or da.attrs.get("unit")
                ts = TimeSeries(
                    data,
                    t0=t0,
                    dt=dt,
                    name=var,
                    channel=var,
                )
                ts = apply_unit(ts, var_unit) if var_unit else ts
                tsd[var] = ts

        set_provenance(
            tsd,
            {
                "format": "nc",
                "time_coord": tc,
                "channels": list(tsd.keys()),
                "unit_source": "override" if unit else "file",
            },
        )
        return apply_time_selection(tsd, start, end)
    finally:
        ds.close()


def read_timeseries_netcdf4(source, **kwargs) -> TimeSeries:
    """Read the sole selected NetCDF4 time-series variable."""
    tsd = read_timeseriesdict_netcdf4(source, **kwargs)
    if not tsd:
        raise ValueError("No time-series variables found in NetCDF4 file")
    if len(tsd) != 1:
        raise ValueError(
            "NetCDF4 single-series reader requires exactly one selected channel"
        )
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_netcdf4(source, **kwargs) -> TimeSeriesMatrix:
    """Read a NetCDF4 file and convert its channels to a matrix.

    ``start``/``end`` are honoured by cropping the assembled matrix, matching
    ``read(source).crop(start, end)``.
    """
    start, end = pop_time_selection(kwargs)

    if isinstance(source, (list, tuple)):
        sources = list(source)
        if not sources:
            raise ValueError("no NetCDF4 files provided")
        matrices = [read_timeseriesmatrix_netcdf4(s, **kwargs) for s in sources]
        merged = matrices[0]
        for mat in matrices[1:]:
            merged = merged.append(mat, inplace=False, gap="pad", pad=np.nan)
        return apply_time_selection(merged, start, end)

    xr = _import_xarray()

    if hasattr(source, "name"):
        source = source.name

    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}
    ds = xr.open_dataset(str(source), **xr_kwargs)
    try:
        is_v2 = ds.attrs.get("gwexpy_netcdf_schema_version") == _NETCDF_SCHEMA_VERSION
        if is_v2:
            tc = "sample"
            if tc not in ds.coords or ds[tc].dtype != np.dtype("int64"):
                raise ValueError("NetCDF v2 file is missing int64 sample coordinate")
            t0, dt = _read_v2_timing(ds)
        else:
            warnings.warn(
                "Reading unversioned legacy NetCDF; timing precision is limited.",
                RuntimeWarning,
                stacklevel=2,
            )
            tc = kwargs.get("time_coord") or _time_coord_name(ds)
            if tc is None:
                raise ValueError(
                    "No time coordinate found in the NetCDF4 file. "
                    "Specify one explicitly via time_coord='...'."
                )
            t0, dt = _legacy_timing(ds, tc)

        matrix_vars = []
        for var_name, da in ds.data_vars.items():
            row_raw = da.attrs.get("gwexpy_row_key")
            col_raw = da.attrs.get("gwexpy_col_key")
            if row_raw is None or col_raw is None or tc not in da.dims:
                continue
            if da.attrs.get("gwexpy_key_format") == "json":
                row_key = _decode_netcdf_key(row_raw)
                col_key = _decode_netcdf_key(col_raw)
            else:
                row_key = str(row_raw)
                col_key = str(col_raw)
            row_index = da.attrs.get("gwexpy_row_index")
            col_index = da.attrs.get("gwexpy_col_index")
            matrix_vars.append((row_key, col_key, row_index, col_index, da))

        if not matrix_vars:
            tsd = read_timeseriesdict_netcdf4(source, **kwargs)
            return apply_time_selection(tsd.to_matrix(), start, end)

        if is_v2:
            if any(row_index is None or col_index is None for _, _, row_index, col_index, _ in matrix_vars):
                raise ValueError("NetCDF v2 matrix cell is missing row/column index")
            matrix_vars.sort(key=lambda item: (int(item[2]), int(item[3])))
        row_keys = list(OrderedDict.fromkeys(row for row, _, _, _, _ in matrix_vars))
        col_keys = list(OrderedDict.fromkeys(col for _, col, _, _, _ in matrix_vars))

        first = matrix_vars[0][4]
        unit = first.attrs.get("units") or first.attrs.get("unit")
        n_samples = len(ds[tc])
        data = np.empty(
            (len(row_keys), len(col_keys), n_samples), dtype=np.asarray(first.values).dtype
        )
        for row_key, col_key, _, _, da in matrix_vars:
            i = row_keys.index(row_key)
            j = col_keys.index(col_key)
            data[i, j, :] = np.asarray(da.values)

        matrix = TimeSeriesMatrix(
            data,
            t0=t0,
            dt=dt,
            unit=unit,
        )
        if row_keys != list(matrix.row_keys()) or col_keys != list(matrix.col_keys()):
            from gwexpy.types.metadata import MetaData, MetaDataDict

            matrix.rows = MetaDataDict(
                OrderedDict((key, MetaData()) for key in row_keys),
                expected_size=len(row_keys),
                key_prefix="row",
            )
            matrix.cols = MetaDataDict(
                OrderedDict((key, MetaData()) for key in col_keys),
                expected_size=len(col_keys),
                key_prefix="col",
            )
        return apply_time_selection(matrix, start, end)
    finally:
        ds.close()


# -- Writer --------------------------------------------------------------------


def write_timeseriesdict_netcdf4(tsd, target, **kwargs):
    """Write a TimeSeriesDict to a NetCDF4 file.

    Version 2 stores a shared integer ``sample`` coordinate and exact timing
    metadata.  It intentionally avoids a datetime axis, whose nanosecond
    quantization changes non-binary sampling intervals such as 0.1 seconds.
    """
    n_samples, t0_gps, dt_sec = _validate_v2_series(tsd)
    xr = _import_xarray()

    data_vars = {}
    row_indices: OrderedDict[object, int] = OrderedDict()
    col_indices: OrderedDict[object, int] = OrderedDict()
    for key in tsd:
        if isinstance(key, tuple) and len(key) == 2:
            row_indices.setdefault(key[0], len(row_indices))
            col_indices.setdefault(key[1], len(col_indices))
    for key, ts in tsd.items():
        attrs: dict[str, object] = {}
        if ts.unit is not None:
            attrs["units"] = str(ts.unit)
        var_name = _encode_netcdf_var_name(key)
        if isinstance(key, tuple) and len(key) == 2:
            attrs["gwexpy_row_key"] = json.dumps(_to_json_native(key[0]))
            attrs["gwexpy_col_key"] = json.dumps(_to_json_native(key[1]))
            attrs["gwexpy_key_format"] = "json"
            attrs["gwexpy_row_index"] = row_indices[key[0]]
            attrs["gwexpy_col_index"] = col_indices[key[1]]
        data_vars[var_name] = xr.DataArray(
            np.asarray(ts.value),
            dims=["sample"],
            attrs=attrs,
        )

    ds = xr.Dataset(
        data_vars,
        coords={"sample": np.arange(n_samples, dtype=np.int64)},
        attrs=_v2_timing_attrs(t0_gps, dt_sec),
    )
    ds.to_netcdf(str(target), **kwargs)


def write_timeseries_netcdf4(ts, target, **kwargs):
    """Write one ``TimeSeries`` to a NetCDF4 file."""
    write_timeseriesdict_netcdf4(
        TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs
    )


def write_timeseriesmatrix_netcdf4(tsm, target, **kwargs):
    """Write a TimeSeriesMatrix to a NetCDF4 file preserving row/col keys.

    Each matrix cell is written as a variable keyed by a ``(row_key, col_key)``
    tuple so that ``gwexpy_row_key``/``gwexpy_col_key`` attributes are encoded
    and the full matrix structure survives a write→read roundtrip.
    """
    from gwexpy.timeseries import TimeSeries, TimeSeriesDict

    row_keys = list(tsm.row_keys())
    col_keys = list(tsm.col_keys())
    n_rows, n_cols, n_samples = tsm.shape

    tsd: TimeSeriesDict = TimeSeriesDict()
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            cell_data = np.asarray(tsm[i, j])
            ts = TimeSeries(
                cell_data,
                t0=tsm.t0,
                dt=tsm.dt,
                unit=tsm.unit if hasattr(tsm, "unit") else None,
            )
            tsd[(rk, ck)] = ts

    write_timeseriesdict_netcdf4(tsd, target, **kwargs)


# -- Registration --------------------------------------------------------------

register_timeseries_format(
    "nc",
    aliases=("netcdf4",),
    reader_dict=read_timeseriesdict_netcdf4,
    reader_single=read_timeseries_netcdf4,
    reader_matrix=read_timeseriesmatrix_netcdf4,
    writer_dict=write_timeseriesdict_netcdf4,
    writer_single=write_timeseries_netcdf4,
    writer_matrix=write_timeseriesmatrix_netcdf4,
    extension="nc",
)
