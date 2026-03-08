"""
NetCDF4 reader/writer for gwexpy (via xarray).

Reads variables that have a ``time`` dimension and converts them to
:class:`~gwexpy.timeseries.TimeSeries`.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    filter_by_channels,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

logger = logging.getLogger(__name__)


def _unit_to_seconds_scale(units: str) -> float | None:
    units_l = units.strip().lower()
    if units_l.startswith(("second", "sec", "s ", "sinc")):
        return 1.0
    if units_l.startswith(("millisecond", "msec", "ms")):
        return 1e-3
    if units_l.startswith(("microsecond", "usecond", "us")):
        return 1e-6
    if units_l.startswith(("nanosecond", "nsec", "ns")):
        return 1e-9
    if units_l.startswith(("minute", "min")):
        return 60.0
    if units_l.startswith(("hour", "hr", "h ")):
        return 3600.0
    if units_l.startswith(("day", "d ")):
        return 86400.0
    return None


def _infer_time_axis(time_vals, coord_attrs):
    """Infer ``(t0, dt)`` from a NetCDF time coordinate."""
    import datetime as _dt

    vals = np.asarray(time_vals)

    if np.issubdtype(vals.dtype, np.datetime64):
        t0_dt64 = vals[0]
        t0_unix_ns = (t0_dt64 - np.datetime64("1970-01-01T00:00:00", "ns")).astype(
            np.int64
        )
        t0_datetime = _dt.datetime.fromtimestamp(t0_unix_ns / 1e9, tz=_dt.UTC)
        t0 = datetime_to_gps(t0_datetime)
        if len(vals) > 1:
            diffs_ns = np.diff(vals.astype("datetime64[ns]").astype(np.int64))
            dt = float(np.median(diffs_ns)) / 1e9
        else:
            dt = 1.0
        return t0, dt

    units = str(coord_attrs.get("units", "")).strip()
    calendar = str(coord_attrs.get("calendar", "standard")).strip() or "standard"

    # Numeric coordinate (either absolute values, or relative values with
    # units such as "seconds since ...").
    try:
        numeric = vals.astype(np.float64)
        if len(numeric) > 1:
            dt = float(np.median(np.diff(numeric)))
        else:
            dt = 1.0

        m = re.match(r"^\s*([A-Za-z]+)\s+since\s+(.+)$", units)
        if m:
            scale = _unit_to_seconds_scale(m.group(1))
            if scale is not None:
                ref_text = m.group(2).strip()
                # numpy datetime64 either fails or emits a deprecation warning
                # for timezone-bearing reference strings; keep numeric fallback
                # in those cases to avoid requiring optional cftime.
                if re.search(r"(?:\s|T)(?:Z|[+-]\d{2}:?\d{2})$", ref_text):
                    ref = None
                else:
                    try:
                        ref = np.datetime64(ref_text)
                    except ValueError:
                        ref = None

                if ref is not None:
                    ref_unix = (
                        ref - np.datetime64("1970-01-01T00:00:00", "ns")
                    ).astype(np.int64) / 1e9
                    t0_datetime = _dt.datetime.fromtimestamp(
                        ref_unix + float(numeric[0]) * scale, tz=_dt.UTC
                    )
                    return datetime_to_gps(t0_datetime), dt * scale

        return float(numeric[0]), dt
    except (TypeError, ValueError):
        pass

    # cftime/object coordinate fallback.
    try:
        import cftime

        unix_seconds = np.asarray(
            cftime.date2num(
                list(vals),
                units="seconds since 1970-01-01 00:00:00",
                calendar=calendar,
            ),
            dtype=np.float64,
        )
        t0_datetime = _dt.datetime.fromtimestamp(float(unix_seconds[0]), tz=_dt.UTC)
        t0 = datetime_to_gps(t0_datetime)
        if len(unix_seconds) > 1:
            dt = float(np.median(np.diff(unix_seconds)))
        else:
            dt = 1.0
        return t0, dt
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(
            "Unsupported NetCDF time coordinate type. Provide a numeric time "
            "axis or datetime64 values (or install cftime for CFTime decoding)."
        ) from exc


def _import_xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray is required for reading/writing NetCDF4 files. "
            "Install with `pip install xarray netcdf4`."
        ) from exc
    return xr


def _time_coord_name(ds):
    """Return the name of the time coordinate, or *None*."""
    for name in ("time", "Time", "TIME", "t"):
        if name in ds.coords:
            return name
    # Fallback: first datetime64 coordinate
    for name, coord in ds.coords.items():
        if np.issubdtype(coord.dtype, np.datetime64):
            return name
    return None


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
    source : str or path-like
        Path to a ``.nc`` file.
    channels : iterable of str, optional
        Variable names to read.  If *None*, all variables with a time
        dimension are loaded.
    unit : str, optional
        Physical unit override applied to every channel.
    time_coord : str, optional
        Name of the time coordinate.  Auto-detected if *None*.
    """
    xr = _import_xarray()

    # gwpy's registry may pass a file-like object; extract the path.
    if hasattr(source, "name"):
        source = source.name

    # Strip gwpy-injected kwargs that xarray.open_dataset does not accept.
    _gwpy_keys = {"start", "end", "pad", "gap", "nproc", "scaled"}
    xr_kwargs = {k: v for k, v in kwargs.items() if k not in _gwpy_keys}
    ds = xr.open_dataset(str(source), **xr_kwargs)
    try:
        tc = time_coord or _time_coord_name(ds)
        if tc is None:
            raise ValueError(
                "No time coordinate found in the NetCDF4 file. "
                "Specify one explicitly via time_coord='...'."
            )

        time_vals = ds[tc].values
        t0, dt = _infer_time_axis(time_vals, ds[tc].attrs)

        tsd = TimeSeriesDict()

        var_names = list(channels) if channels else list(ds.data_vars)
        for var in var_names:
            if var not in ds.data_vars:
                logger.debug("Variable '%s' not found in dataset, skipping", var)
                continue
            da = ds[var]
            if tc not in da.dims:
                logger.debug(
                    "Variable '%s' has no time dimension '%s', skipping", var, tc
                )
                continue

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
                        data[:, i].astype(np.float64),
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
                    data.astype(np.float64),
                    t0=t0,
                    dt=dt,
                    name=var,
                    channel=var,
                )
                ts = apply_unit(ts, var_unit) if var_unit else ts
                tsd[var] = ts

        # filter_by_channels matches against the *final* dict keys.
        # Multi-dimensional variables are flattened to "var_0", "var_1", …
        # so we must accept both the original name and the suffixed names.
        if channels:
            expanded = set(channels)
            for ch in channels:
                expanded.update(k for k in tsd if k == ch or k.startswith(f"{ch}_"))
            tsd = TimeSeriesDict(filter_by_channels(tsd, expanded))

        set_provenance(
            tsd,
            {
                "format": "netcdf4",
                "time_coord": tc,
                "channels": list(tsd.keys()),
                "unit_source": "override" if unit else "file",
            },
        )
        return tsd
    finally:
        ds.close()


def read_timeseries_netcdf4(source, **kwargs) -> TimeSeries:
    tsd = read_timeseriesdict_netcdf4(source, **kwargs)
    if not tsd:
        raise ValueError("No time-series variables found in NetCDF4 file")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_netcdf4(source, **kwargs) -> TimeSeriesMatrix:
    tsd = read_timeseriesdict_netcdf4(source, **kwargs)
    return tsd.to_matrix()


# -- Writer --------------------------------------------------------------------


def write_timeseriesdict_netcdf4(tsd, target, **kwargs):
    """Write a TimeSeriesDict to a NetCDF4 file.

    Each channel becomes a variable with a shared ``time`` coordinate
    reconstructed from ``t0`` and ``dt``.
    """
    xr = _import_xarray()
    import datetime as _dt

    if not tsd:
        raise ValueError("Cannot write empty TimeSeriesDict to NetCDF4")

    first_key = next(iter(tsd.keys()))
    first = tsd[first_key]
    n_samples = first.shape[0]
    dt_sec = float(first.dt.value)
    t0_gps = float(first.t0.value)

    # Reconstruct datetime64 time axis from GPS t0
    from gwpy.time import from_gps

    t0_dt = from_gps(t0_gps).to_pydatetime()
    if t0_dt.tzinfo is None:
        t0_dt = t0_dt.replace(tzinfo=_dt.UTC)
    t0_ns = np.datetime64(t0_dt, "ns")
    dt_ns = np.timedelta64(int(round(dt_sec * 1e9)), "ns")
    time_axis = t0_ns + np.arange(n_samples) * dt_ns

    data_vars = {}
    for key, ts in tsd.items():
        attrs = {}
        if ts.unit is not None:
            attrs["units"] = str(ts.unit)
        data_vars[key] = xr.DataArray(
            np.asarray(ts.value, dtype=np.float64),
            dims=["time"],
            attrs=attrs,
        )

    ds = xr.Dataset(data_vars, coords={"time": time_axis})
    ds.to_netcdf(str(target), **kwargs)


def write_timeseries_netcdf4(ts, target, **kwargs):
    write_timeseriesdict_netcdf4(
        TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs
    )


# -- Registration --------------------------------------------------------------

for _fmt in ("netcdf4", "nc"):
    io_registry.register_reader(
        _fmt, TimeSeriesDict, read_timeseriesdict_netcdf4, force=True
    )
    io_registry.register_reader(
        _fmt, TimeSeries, read_timeseries_netcdf4, force=True
    )
    io_registry.register_reader(
        _fmt, TimeSeriesMatrix, read_timeseriesmatrix_netcdf4, force=True
    )
    io_registry.register_writer(
        _fmt, TimeSeriesDict, write_timeseriesdict_netcdf4, force=True
    )
    io_registry.register_writer(
        _fmt, TimeSeries, write_timeseries_netcdf4, force=True
    )

io_registry.register_identifier(
    "netcdf4",
    TimeSeriesDict,
    lambda *args, **kwargs: str(args[1]).lower().endswith(".nc"),
)
io_registry.register_identifier(
    "netcdf4",
    TimeSeries,
    lambda *args, **kwargs: str(args[1]).lower().endswith(".nc"),
)
