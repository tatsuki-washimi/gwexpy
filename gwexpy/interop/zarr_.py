from __future__ import annotations

from ._optional import require_optional


def to_zarr(ts, store, path, chunks=None, compressor=None, overwrite=False):
    """
    Write to Zarr array.
    """
    zarr = require_optional("zarr")

    mode = "w" if overwrite else "w-"

    # Open group or array
    # If store is path string, use open_group/open_array
    # We assume 'store' can be valid input to zarr.open

    # Save array
    open_kwargs = {
        "store": store,
        "mode": mode,
        "path": path,
        "shape": ts.shape,
        "dtype": ts.dtype,
        "chunks": chunks,
    }
    if compressor is not None:
        open_kwargs["compressor"] = compressor
        # Prefer Zarr v2 when an explicit compressor is requested (Zarr v3 uses codecs).
        open_kwargs["zarr_format"] = 2
    try:
        arr = zarr.open_array(**open_kwargs)
    except TypeError as exc:
        if "zarr_format" in open_kwargs and "zarr_format" in str(exc):
            open_kwargs.pop("zarr_format")
            arr = zarr.open_array(**open_kwargs)
        else:
            raise
    arr[:] = ts.value

    # Attributes
    arr.attrs["t0"] = ts.t0.value
    arr.attrs["dt"] = ts.dt.value
    arr.attrs["unit"] = str(ts.unit)
    if ts.name:
        arr.attrs["name"] = str(ts.name)


def from_zarr(cls, store, path):
    """
    Read from Zarr array.
    """
    zarr = require_optional("zarr")

    arr = zarr.open_array(store=store, mode="r", path=path)
    data = arr[:]  # Load into memory? Or keep as zarr array (array-like)?
    # GWpy usually expects in-memory numpy. Loading.

    t0 = arr.attrs.get("t0", 0)
    dt = arr.attrs.get("dt", 1)
    unit = arr.attrs.get("unit", "")
    name = arr.attrs.get("name", None)

    return cls(data, t0=t0, dt=dt, unit=unit, name=name)
