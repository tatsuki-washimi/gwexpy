from ._optional import require_optional


def to_hdf5(ts, group, path, overwrite=False, compression=None, compression_opts=None):
    """
    Write TimeSeries to HDF5 group.
    wrapper for ts.write(..., format='hdf5') usually, but here we implement direct
    low-level if strict control is needed, OR delegate.
    """
    require_optional("h5py")

    # Check overwrite
    if path in group:
        if overwrite:
            del group[path]
        else:
            raise OSError(f"Path {path} exists in HDF5 group")

    # Use gwpy's write if available on the object?
    # No, we want to write *into* an open h5py object.
    # gwpy.io.hdf5 usually expects a filename or file object.

    # Manual write for maximum control/interop
    dset = group.create_dataset(
        path, data=ts.value, compression=compression, compression_opts=compression_opts
    )

    # Metadata attributes (gwpy compatible names)
    dset.attrs["t0"] = ts.t0.value
    dset.attrs["dt"] = ts.dt.value
    dset.attrs["unit"] = str(ts.unit)
    if ts.name:
        dset.attrs["name"] = str(ts.name)
    if ts.channel:
        dset.attrs["channel"] = str(ts.channel)


def from_hdf5(cls, group, path):
    """Read TimeSeries from HDF5 group."""
    require_optional("h5py")

    dset = group[path]
    data = dset[()]

    attrs = dset.attrs
    t0 = attrs.get("t0", 0)
    dt = attrs.get("dt", 1)
    unit = attrs.get("unit", "")
    name = attrs.get("name", None)

    return cls(data, t0=t0, dt=dt, unit=unit, name=name)
