"""Object-level NetCDF4 bridge helpers.

These helpers bridge a live ``TimeSeries`` object to a writable
``netCDF4.Dataset`` and back. They are distinct from direct file I/O such as
``TimeSeries.read(..., format="nc")`` / ``.write(..., format="nc")``.
"""
from __future__ import annotations

from ._optional import require_optional


def to_netcdf4(ts, ds, var_name, dim_time="time", time_units=None, overwrite=False):
    """Write to netCDF4 Dataset.

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
        ds.createDimension(dim_time, ts.size)  # or None for unlimited

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
    # Persist ``channel`` (guarded against the falsy/None case) so it can be
    # recovered on read -- keeps to_netcdf4/from_netcdf4 round-trip symmetric.
    if ts.channel:
        v.channel = str(ts.channel)


def from_netcdf4(
    cls, ds, var_name, *, unit=None, channel=None, name=None, t0=None, dt=None
):
    """Read from a netCDF4 dataset.

    Metadata absent from the variable can be supplied explicitly; an explicit
    argument takes priority over the stored attribute. Missing ``t0``/``dt``
    fall back to ``0``/``1`` with a :class:`UserWarning` rather than silently.
    """
    from .base import resolve_meta, resolve_timing

    v = ds.variables[var_name]
    data = v[:]

    # Check masked array
    import numpy as np

    if np.ma.is_masked(data):
        data = data.filled(np.nan)  # or specific fill

    final_t0, final_dt = resolve_timing(
        t0,
        dt,
        source=f"netCDF4 variable '{var_name}'",
        inferred_t0=getattr(v, "t0", None),
        inferred_dt=getattr(v, "dt", None),
    )
    final_unit = resolve_meta(unit, getattr(v, "units", ""))
    final_name = resolve_meta(name, getattr(v, "long_name", var_name))
    final_channel = resolve_meta(channel, getattr(v, "channel", None))

    return cls(
        data,
        t0=final_t0,
        dt=final_dt,
        unit=final_unit,
        name=final_name,
        channel=final_channel,
    )
