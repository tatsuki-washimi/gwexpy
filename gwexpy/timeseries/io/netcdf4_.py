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
from fractions import Fraction
from functools import partial
from os import PathLike

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
_LEGACY_WARNING = "Reading unversioned legacy NetCDF; timing precision is limited."


def _record_or_warn_legacy_read(marker: list[bool] | None) -> None:
    """Emit a legacy warning once, or mark it for a multi-source parent."""
    if marker is not None:
        marker[0] = True
        return
    warnings.warn(_LEGACY_WARNING, RuntimeWarning, stacklevel=3)


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
    gps_time = Fraction(gps_seconds) + Fraction(gps_nanoseconds, 1_000_000_000)
    if abs(gps_time - Fraction.from_float(t0)) > Fraction(1, 2_000_000_000):
        raise ValueError("NetCDF v2 GPS timing disagrees with t0 float metadata")
    return t0, dt


def _validate_v2_sample_coordinate(ds) -> None:
    """Reject a v2 file whose sample coordinate contradicts its axis schema."""
    if "sample" not in ds.coords:
        raise ValueError("NetCDF v2 file is missing int64 sample coordinate")
    sample = ds["sample"]
    if sample.dtype != np.dtype("int64"):
        raise ValueError("NetCDF v2 sample coordinate must be int64")
    values = np.asarray(sample.values)
    expected = np.arange(values.size, dtype=np.int64)
    if not np.array_equal(values, expected):
        raise ValueError("NetCDF v2 sample coordinate must be exactly 0..N-1")


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
            float(
                np.median(np.diff(time_vals.astype("datetime64[ns]").astype(np.int64)))
            )
            / 1e9
            if len(time_vals) > 1
            else 1.0
        )
    else:
        numeric = np.asarray(time_vals, dtype=np.float64)
        t0 = float(numeric[0])
        dt = float(np.median(np.diff(numeric))) if len(numeric) > 1 else 1.0
    return t0, dt


def _homogeneous_unit(units) -> str:
    """Return one cell unit, rejecting a matrix with mixed cell units."""
    names = {str(unit) for unit in units}
    if len(names) != 1:
        raise ValueError("NetCDF matrix cells have mixed units")
    return names.pop()


def _validate_matrix_cells(matrix_vars, *, require_indices: bool):
    """Validate the complete rectangular cell and key/index topology."""
    rows: dict[object, int] = {}
    cols: dict[object, int] = {}
    row_indices: dict[int, object] = {}
    col_indices: dict[int, object] = {}
    cells = set()
    has_indices = [
        row is not None or col is not None for _, _, row, col, _ in matrix_vars
    ]
    if require_indices or any(has_indices):
        if not all(has_indices):
            raise ValueError("NetCDF matrix cell is missing row/column index")
        for row_key, col_key, row_index, col_index, _ in matrix_vars:
            for axis, key, index, forward, reverse in (
                ("row", row_key, row_index, rows, row_indices),
                ("column", col_key, col_index, cols, col_indices),
            ):
                if (
                    isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    or index < 0
                ):
                    raise ValueError(
                        f"NetCDF matrix {axis} index must be a nonnegative integer"
                    )
                if key in forward and forward[key] != index:
                    raise ValueError(
                        f"NetCDF matrix {axis} key has conflicting indices"
                    )
                if index in reverse and reverse[index] != key:
                    raise ValueError(f"NetCDF matrix {axis} index has conflicting keys")
                forward[key] = index
                reverse[index] = key
        for axis, reverse in (("row", row_indices), ("column", col_indices)):
            if set(reverse) != set(range(len(reverse))):
                raise ValueError(
                    f"NetCDF matrix {axis} indices are sparse or out of range"
                )
        row_keys = [row_indices[index] for index in range(len(row_indices))]
        col_keys = [col_indices[index] for index in range(len(col_indices))]
    else:
        row_keys = list(OrderedDict.fromkeys(row for row, _, _, _, _ in matrix_vars))
        col_keys = list(OrderedDict.fromkeys(col for _, col, _, _, _ in matrix_vars))
    for row_key, col_key, _, _, _ in matrix_vars:
        cell = (row_key, col_key)
        if cell in cells:
            raise ValueError("duplicate NetCDF matrix cell")
        cells.add(cell)
    if len(cells) != len(row_keys) * len(col_keys):
        raise ValueError("missing NetCDF matrix cell")
    return row_keys, col_keys


def read_timeseriesdict_netcdf4(
    source,
    *,
    channels=None,
    unit=None,
    time_coord=None,
    _legacy_warning_marker: list[bool] | None = None,
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
        legacy_warning_marker = [False]
        result = read_multi_dict(
            partial(
                read_timeseriesdict_netcdf4,
                _legacy_warning_marker=legacy_warning_marker,
            ),
            multi,
            "nc",
            channels=channels,
            unit=unit,
            time_coord=time_coord,
            **kwargs,
        )
        if legacy_warning_marker[0]:
            _record_or_warn_legacy_read(_legacy_warning_marker)
        return apply_time_selection(result, start, end)

    xr = _import_xarray()

    # gwpy's registry may pass a file-like object; extract the path.
    if hasattr(source, "name") and not isinstance(source, (str, PathLike)):
        source = source.name

    # Strip gwpy-injected kwargs that xarray.open_dataset does not accept.
    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}
    ds = xr.open_dataset(str(source), **xr_kwargs)
    try:
        schema_version = ds.attrs.get("gwexpy_netcdf_schema_version")
        if schema_version is not None and schema_version != _NETCDF_SCHEMA_VERSION:
            raise ValueError(f"unsupported NetCDF schema version: {schema_version}")
        is_v2 = schema_version == _NETCDF_SCHEMA_VERSION
        if is_v2:
            tc = "sample"
            _validate_v2_sample_coordinate(ds)
            t0, dt = _read_v2_timing(ds)
        else:
            _record_or_warn_legacy_read(_legacy_warning_marker)
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


def read_timeseriesmatrix_netcdf4(
    source, *, _legacy_warning_marker: list[bool] | None = None, **kwargs
) -> TimeSeriesMatrix:
    """Read a NetCDF4 file and convert its channels to a matrix.

    ``start``/``end`` are honoured by cropping the assembled matrix, matching
    ``read(source).crop(start, end)``.
    """
    start, end = pop_time_selection(kwargs)

    if isinstance(source, (list, tuple)):
        sources = list(source)
        if not sources:
            raise ValueError("no NetCDF4 files provided")
        legacy_warning_marker = [False]
        matrices = [
            read_timeseriesmatrix_netcdf4(
                source_item,
                _legacy_warning_marker=legacy_warning_marker,
                **kwargs,
            )
            for source_item in sources
        ]
        merged = matrices[0]
        for mat in matrices[1:]:
            merged = merged.append(mat, inplace=False, gap="pad", pad=np.nan)
        if legacy_warning_marker[0]:
            _record_or_warn_legacy_read(_legacy_warning_marker)
        return apply_time_selection(merged, start, end)

    xr = _import_xarray()

    if hasattr(source, "name") and not isinstance(source, (str, PathLike)):
        source = source.name

    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}
    ds = xr.open_dataset(str(source), **xr_kwargs)
    try:
        schema_version = ds.attrs.get("gwexpy_netcdf_schema_version")
        if schema_version is not None and schema_version != _NETCDF_SCHEMA_VERSION:
            raise ValueError(f"unsupported NetCDF schema version: {schema_version}")
        is_v2 = schema_version == _NETCDF_SCHEMA_VERSION
        if is_v2:
            tc = "sample"
            _validate_v2_sample_coordinate(ds)
            t0, dt = _read_v2_timing(ds)
        else:
            _record_or_warn_legacy_read(_legacy_warning_marker)
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
            matrix_attrs = any(
                name in da.attrs
                for name in (
                    "gwexpy_row_key",
                    "gwexpy_col_key",
                    "gwexpy_row_index",
                    "gwexpy_col_index",
                )
            )
            if matrix_attrs and (
                row_raw is None or col_raw is None or tc not in da.dims
            ):
                raise ValueError(
                    "NetCDF matrix cell is missing row/column metadata or time axis"
                )
            if not matrix_attrs:
                if "gwexpy_matrix_rows" in ds.attrs and tc in da.dims:
                    raise ValueError("NetCDF declared matrix has an unmarked cell")
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
            if "gwexpy_matrix_rows" in ds.attrs or "gwexpy_matrix_columns" in ds.attrs:
                raise ValueError("NetCDF declared matrix has no cells")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tsd = read_timeseriesdict_netcdf4(source, **kwargs)
            if not tsd:
                raise ValueError("NetCDF file has no time-series cells")
            # The file has one validated shared axis.  Construct it directly;
            # generic collection alignment can lose exact v2 timing and units.
            values = np.stack([np.asarray(series.value) for series in tsd.values()])[
                :, np.newaxis, :
            ]
            # A matrix exposes one physical unit.  Reject an unmarked legacy
            # multichannel file whose units cannot be represented by that unit.
            unit = _homogeneous_unit(series.unit for series in tsd.values())
            matrix = TimeSeriesMatrix(
                values,
                t0=t0,
                dt=dt,
                unit=unit,
                channel_names=list(tsd),
            )
            return apply_time_selection(matrix, start, end)

        row_keys, col_keys = _validate_matrix_cells(matrix_vars, require_indices=is_v2)
        declared_rows = ds.attrs.get("gwexpy_matrix_rows")
        declared_cols = ds.attrs.get("gwexpy_matrix_columns")
        if (declared_rows is None) != (declared_cols is None):
            raise ValueError("NetCDF matrix is missing declared row/column dimension")
        if declared_rows is not None:
            if not isinstance(declared_rows, (int, np.integer)) or declared_rows != len(
                row_keys
            ):
                raise ValueError(
                    "NetCDF matrix declared row dimension disagrees with cells"
                )
            if not isinstance(declared_cols, (int, np.integer)) or declared_cols != len(
                col_keys
            ):
                raise ValueError(
                    "NetCDF matrix declared column dimension disagrees with cells"
                )
        # Older v2 files did not declare dimensions.  Their observed rectangle
        # is validated, but deletion of an entire final row/column is unknowable.

        unit = _homogeneous_unit(
            da.attrs.get("units") or da.attrs.get("unit") or ""
            for _, _, _, _, da in matrix_vars
        )
        n_samples = len(ds[tc])
        cell_values = [np.asarray(da.values) for _, _, _, _, da in matrix_vars]
        dtype = np.result_type(*(values.dtype for values in cell_values))

        def lossless(values) -> bool:
            if not np.can_cast(values.dtype, dtype, casting="safe"):
                return False
            # NumPy calls int64 -> float64 a "safe" dtype cast even when an
            # individual integer exceeds the float64 exact-integer range.
            with np.errstate(over="ignore", invalid="ignore"):
                converted = values.astype(dtype)
                roundtrip = converted.astype(values.dtype)
            return bool(np.array_equal(values, roundtrip, equal_nan=True))

        if not all(lossless(values) for values in cell_values):
            raise ValueError(
                "NetCDF matrix cell dtypes have no safe common representation"
            )
        data = np.empty(
            (len(row_keys), len(col_keys), n_samples),
            dtype=dtype,
        )
        for (row_key, col_key, _, _, _), values in zip(matrix_vars, cell_values):
            i = row_keys.index(row_key)
            j = col_keys.index(col_key)
            data[i, j, :] = values

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


def write_timeseriesdict_netcdf4(
    tsd, target, *, _matrix_shape: tuple[int, int] | None = None, **kwargs
):
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
    if _matrix_shape is not None:
        ds.attrs["gwexpy_matrix_rows"] = _matrix_shape[0]
        ds.attrs["gwexpy_matrix_columns"] = _matrix_shape[1]
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
                x0=tsm.x0,
                dt=tsm.dt,
                xunit=tsm.xunit,
                unit=tsm.unit if hasattr(tsm, "unit") else None,
            )
            tsd[(rk, ck)] = ts

    write_timeseriesdict_netcdf4(tsd, target, _matrix_shape=(n_rows, n_cols), **kwargs)


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
