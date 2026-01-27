from __future__ import annotations

import numpy as np
from gwpy.time import LIGOTimeGPS

from ._optional import require_optional


def to_xarray(ts, time_coord="datetime"):
    """
    TimeSeries -> xarray.DataArray
    """
    xr = require_optional("xarray")

    data = ts.value
    attrs = {
        "unit": str(ts.unit),
        "name": str(ts.name),
        "channel": str(ts.channel),
        "epoch": float(ts.t0.value if hasattr(ts.t0, "value") else ts.t0),
        "time_coord": time_coord,
    }

    times_gps = ts.times.value
    if time_coord == "datetime":
        from astropy.time import Time

        t_vals = Time(times_gps, format="gps").to_datetime()
    elif time_coord == "seconds":
        from astropy.time import Time

        t_vals = Time(times_gps, format="gps").unix
    elif time_coord == "gps":
        t_vals = times_gps
    else:
        raise ValueError("time_coord must be 'datetime'|'seconds'|'gps'")

    da = xr.DataArray(
        data,
        dims=("time",),
        coords={"time": t_vals},
        name=ts.name,
        attrs=attrs,
    )
    return da


def from_xarray(cls, da, unit=None):
    """DataArray -> TimeSeries"""
    require_optional("xarray")

    val = da.values
    t_coord = da.coords["time"].values

    time_coord = da.attrs.get("time_coord")
    if np.issubdtype(t_coord.dtype, np.datetime64):
        from astropy.time import Time

        t_obj = Time(t_coord, format="datetime64")
        t0_gps = float(t_obj[0].gps)
        dt = float(t_obj[1].gps - t0_gps) if len(t_coord) > 1 else 1.0
    elif time_coord == "seconds":
        from astropy.time import Time

        t_obj = Time(t_coord, format="unix")
        t0_gps = float(t_obj[0].gps)
        dt = float(t_obj[1].gps - t0_gps) if len(t_coord) > 1 else 1.0
    else:
        t0_gps = float(t_coord[0])
        dt = float(t_coord[1] - t0_gps) if len(t_coord) > 1 else 1.0

    _unit = unit or da.attrs.get("unit")
    name = da.name or da.attrs.get("name")

    return cls(val, t0=LIGOTimeGPS(t0_gps), dt=dt, unit=_unit, name=name)
