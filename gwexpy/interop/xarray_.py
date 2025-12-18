
from ._optional import require_optional
import numpy as np
from gwpy.time import LIGOTimeGPS


def to_xarray(ts, time_coord="datetime"):
    """
    TimeSeries -> xarray.DataArray
    """
    xr = require_optional("xarray")

    data = ts.value
    attrs = {"unit": str(ts.unit), "name": str(ts.name), "channel": str(ts.channel), "epoch": float(ts.t0)}

    times_gps = ts.times.value
    if time_coord == "datetime":
        from astropy.time import Time
        t_vals = Time(times_gps, format="gps").to_datetime64()
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

    if np.issubdtype(t_coord.dtype, np.datetime64):
        from astropy.time import Time
        t_obj = Time(t_coord, format="datetime64")
        t0 = t_obj[0].gps
        dt = t_obj[1].gps - t0 if len(t_coord) > 1 else 1
    else:
        t0 = float(t_coord[0])
        dt = float(t_coord[1] - t0) if len(t_coord) > 1 else 1

    _unit = unit or da.attrs.get("unit")
    name = da.name or da.attrs.get("name")

    return cls(val, t0=t0, dt=dt, unit=_unit, name=name)
