from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar

from ._optional import require_optional

if TYPE_CHECKING:
    import h5py

    from gwexpy.timeseries import TimeSeries

T = TypeVar("T", bound="TimeSeries")


def to_hdf5(
    ts: TimeSeries,
    group: h5py.Group,
    path: str,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Any = None,
) -> None:
    """Write TimeSeries to HDF5 group.

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


def from_hdf5(
    cls: type[T],
    group: h5py.Group,
    path: str,
    *,
    unit: Optional[str] = None,
    channel: Optional[str] = None,
    name: Optional[str] = None,
    t0: Optional[float] = None,
    dt: Optional[float] = None,
) -> T:
    """Read TimeSeries from HDF5 group.

    The ``channel`` attribute written by :func:`to_hdf5` is restored (closing a
    write-but-not-read round-trip gap). Metadata absent from the dataset can be
    supplied explicitly; an explicit argument takes priority over the stored
    attribute. Missing ``t0``/``dt`` fall back to ``0``/``1`` with a
    :class:`UserWarning` rather than silently.
    """
    require_optional("h5py")
    from .base import resolve_meta, resolve_timing

    dset = group[path]
    data = dset[()]

    attrs = dset.attrs
    final_t0, final_dt = resolve_timing(
        t0,
        dt,
        source=f"HDF5 dataset '{path}'",
        inferred_t0=attrs["t0"] if "t0" in attrs else None,
        inferred_dt=attrs["dt"] if "dt" in attrs else None,
    )
    final_unit = resolve_meta(unit, attrs["unit"] if "unit" in attrs else "")
    final_name = resolve_meta(name, attrs["name"] if "name" in attrs else None)
    final_channel = resolve_meta(
        channel, attrs["channel"] if "channel" in attrs else None
    )

    return cls(
        data,
        t0=final_t0,
        dt=final_dt,
        unit=final_unit,
        name=final_name,
        channel=final_channel,
    )
