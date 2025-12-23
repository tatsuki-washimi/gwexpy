
from ._optional import require_optional

def to_netcdf4(ts, ds, var_name, dim_time="time", time_units=None, overwrite=False):
    """
    Write to netCDF4 Dataset.
    
    ds: netCDF4.Dataset (writable)
    """
    require_optional("netCDF4")
    
    if var_name in ds.variables:
        if not overwrite:
            raise ValueError(f"Variable {var_name} exists")
        # Reuse or error? netCDF usually implies defining structure first.
        # Minimal impl: overwrite implies maybe simpler to expect user handled file mode. 
        # But we can try to fill.
        
    # Define dimension if not exists
    if dim_time not in ds.dimensions:
        ds.createDimension(dim_time, ts.size) # or None for unlimited
        
    # Create variable
    if var_name not in ds.variables:
        v = ds.createVariable(var_name, ts.dtype, (dim_time,))
    else:
        v = ds.variables[var_name]

    v[:] = ts.value
    
    # Metadata attributes
    v.t0 = ts.t0.value
    v.dt = ts.dt.value
    v.units = str(ts.unit)
    if ts.name:
        v.long_name = str(ts.name)

def from_netcdf4(cls, ds, var_name):
    """
    Read from netCDF4 Dataset.
    """
    v = ds.variables[var_name]
    data = v[:] 
    
    # Check masked array
    import numpy as np
    if np.ma.is_masked(data):
        data = data.filled(np.nan) # or specific fill
        
    t0 = getattr(v, "t0", 0)
    dt = getattr(v, "dt", 1)
    unit = getattr(v, "units", "")
    name = getattr(v, "long_name", var_name)
    
    return cls(data, t0=t0, dt=dt, unit=unit, name=name)
