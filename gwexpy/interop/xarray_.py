
from ._optional import require_optional
from ._time import gps_to_datetime_utc, datetime_utc_to_gps, gps_to_unix, unix_to_gps
import numpy as np

def to_xarray(ts, time_coord="datetime"):
    """
    TimeSeries -> xarray.DataArray
    
    Parameters
    ----------
    time_coord : "datetime" | "seconds" | "gps"
    """
    xr = require_optional("xarray")
    
    data = ts.value
    # Metadata
    attrs = {"unit": str(ts.unit), "name": str(ts.name), "channel": str(ts.channel)}
    
    # Time coords
    times_gps = ts.times.value
    if time_coord == "datetime":
        from astropy.time import Time
        # xarray handles datetime64[ns]
        # astropy Time -> datetime64
        t_vals = Time(times_gps, format='gps').to_datetime64()
    elif time_coord == "seconds":
        from astropy.time import Time
        t_vals = Time(times_gps, format='gps').unix
    else:
        t_vals = times_gps
        
    da = xr.DataArray(
        data,
        dims=("time",),
        coords={"time": t_vals},
        name=ts.name,
        attrs=attrs
    )
    return da

def from_xarray(cls, da, unit=None):
    """DataArray -> TimeSeries"""
    xr = require_optional("xarray")
    
    val = da.values
    t_coord = da.coords["time"].values
    
    # Infer t0, dt
    if np.issubdtype(t_coord.dtype, np.datetime64):
        # Convert to GPS
        from astropy.time import Time
        t_obj = Time(t_coord, format='datetime64')
        t0 = t_obj[0].gps
        if len(t_coord) > 1:
            dt = t_obj[1].gps - t0
        else:
            dt = 1
    else:
        # assume gps or seconds
        t0 = t_coord[0]
        dt = t_coord[1] - t0 if len(t_coord) > 1 else 1
        
    if isinstance(dt, LIGOTimeGPS):
        dt = float(dt)
        
    _unit = unit or da.attrs.get("unit")
    name = da.name or da.attrs.get("name")
    
    return cls(val, t0=t0, dt=dt, unit=_unit, name=name)
