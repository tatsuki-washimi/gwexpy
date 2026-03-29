from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py

if TYPE_CHECKING:
    from gwexpy.histogram import Histogram


def write_hdf5_dataset(hist: Histogram, f: h5py.Group | h5py.File, path: str | None = "data") -> None:
    """
    Write a Histogram to an HDF5 group or file.

    Parameters
    ----------
    hist : Histogram
        The histogram instance to write.
    f : h5py.Group or h5py.File
        The target HDF5 object.
    path : str or None
        The path within the HDF5 file to store the data. If None, write directly to f.
    """
    # Create the internal group to store the arrays and attributes
    group = f.create_group(path) if path else f

    # Store essential arrays
    group.create_dataset("values", data=hist.values.value)
    group.create_dataset("edges", data=hist.edges.value)

    # Optional arrays
    if hist.cov is not None:
        group.create_dataset("cov", data=hist.cov.value)
    if hist.sumw2 is not None:
        group.create_dataset("sumw2", data=hist.sumw2.value)

    # Standard metadata mapping aligned with gwexpy patterns
    group.attrs["unit"] = str(hist.unit)
    group.attrs["xunit"] = str(hist.xunit)

    if hist.name:
        group.attrs["name"] = hist.name
    if getattr(hist, "channel", None):
        group.attrs["channel"] = str(getattr(hist.channel, "name", hist.channel))


def read_hdf5_dataset(cls: type[Histogram], f: h5py.Group | h5py.File, path: str | None = "data") -> Histogram:
    """
    Read a Histogram from an HDF5 group or file.

    Parameters
    ----------
    cls : type
        The Histogram class.
    f : h5py.Group or h5py.File
        The source HDF5 object.
    path : str or None
        The path within the HDF5 file to read the data from. If None, read directly from f.

    Returns
    -------
    Histogram
        The loaded Histogram object.
    """
    import h5py

    group = f[path] if path else f
    if not isinstance(group, h5py.Group):
        raise TypeError(f"Target path {path} is not an HDF5 Group.")

    values = group["values"][:]
    edges = group["edges"][:]

    kwargs: dict[str, Any] = {}
    if "cov" in group:
        kwargs["cov"] = group["cov"][:]
    if "sumw2" in group:
        kwargs["sumw2"] = group["sumw2"][:]

    unit = group.attrs.get("unit")
    if isinstance(unit, bytes):
        unit = unit.decode("utf-8")

    xunit = group.attrs.get("xunit")
    if isinstance(xunit, bytes):
        xunit = xunit.decode("utf-8")

    name = group.attrs.get("name")
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    channel = group.attrs.get("channel")
    if isinstance(channel, bytes):
        channel = channel.decode("utf-8")

    return cls(
        values=values,
        edges=edges,
        unit=unit,
        xunit=xunit,
        name=name,
        channel=channel,
        **kwargs
    )
