"""HDF5 writing of GWpy-style series with an irregular (non-uniform) x-axis.

GWpy's series writer always requests ``dx``, which is undefined for an
irregular axis, so the write fails before anything is stored.  GWpy's reader,
however, already rebuilds a series from an ``xindex`` dataset attribute.  The
helpers here write that representation: the exact axis samples in ``xindex``
and the axis unit in ``xunit``, with no ``dx``.

Regular-axis series are never routed through this module's layout; their
on-disk attributes (``x0``, ``dx``, ``xunit``) are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
from gwpy.types.io.hdf5 import write_hdf5_array, write_hdf5_series

# Lowest HDF5 file-format bound that provides dense attribute storage.  The
# default ("earliest") bound limits an attribute to one 64 KiB object-header
# message, i.e. fewer than 8192 float64 axis samples.
IRREGULAR_AXIS_LIBVER = "v108"

__all__ = [
    "IRREGULAR_AXIS_LIBVER",
    "has_irregular_axis",
    "irregular_axis_attrs",
    "raise_for_attribute_limit",
    "write_hdf5_series_any_axis",
]


def has_irregular_axis(series: Any) -> bool:
    """Return `True` if ``series`` carries an explicit, non-uniform x-axis."""
    xindex = getattr(series, "_xindex", None)
    return xindex is not None and not xindex.regular


def irregular_axis_attrs(series: Any) -> dict[str, Any]:
    """Return the index attributes to store for an irregular-axis series.

    Only the axis unit is listed: the axis samples themselves are written by
    GWpy's metadata writer as the ``xindex`` attribute, and ``dx`` is
    undefined.
    """
    return {"xunit": str(series.xunit)}


def raise_for_attribute_limit(
    series: Any,
    h5g: h5py.Group,
    path: str | None,
    exc: OSError,
) -> None:
    """Explain an HDF5 attribute-size failure for a long irregular axis.

    Returns without raising when ``exc`` is some other failure, so the caller
    can re-raise the original exception.

    Raises
    ------
    OSError
        If the axis did not fit in an attribute because the file uses the
        'earliest' HDF5 format bound.  The dataset created before the axis was
        rejected is removed first, so no dataset with a default axis is left.

    """
    if h5g.file.libver[0] != "earliest" or "too large" not in str(exc):
        return
    name = path if path is not None else series.name
    if name is not None and name in h5g:
        del h5g[name]
    msg = (
        f"Cannot store the irregular x-axis of {type(series).__name__} "
        f"{series.name!r} ({series.xindex.size} samples) in an HDF5 file "
        "opened with the 'earliest' format bound, which limits an attribute "
        "to 64 KiB. Pass a file path, or open the file with "
        f"h5py.File(..., libver={IRREGULAR_AXIS_LIBVER!r})."
    )
    raise OSError(msg) from exc


def _write_irregular(
    series: Any,
    h5g: h5py.Group,
    path: str | None,
    **kwargs: Any,
) -> h5py.Dataset:
    """Write ``series`` with its irregular axis into an open HDF5 group."""
    try:
        return write_hdf5_array(
            series, h5g, path=path, attrs=irregular_axis_attrs(series), **kwargs
        )
    except OSError as exc:
        raise_for_attribute_limit(series, h5g, path, exc)
        raise


def write_hdf5_series_any_axis(
    series: Any,
    output: Any,
    path: str | None = None,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> h5py.Dataset:
    """Write a series with a regular or irregular x-axis to HDF5.

    A series with a regular axis, or any call with explicit ``attrs``, is
    delegated to GWpy's writer and keeps its on-disk layout (``x0``, ``dx``
    and ``xunit`` dataset attributes).

    A series with an irregular axis is written with the exact axis samples in
    the ``xindex`` dataset attribute and the axis unit in ``xunit``; ``dx`` is
    not written because it is undefined.  Values, axis samples, ``unit`` and
    ``name`` are stored without conversion.

    Parameters
    ----------
    series : `gwpy.types.Series`
        The series to write.
    output : `str`, `pathlib.Path`, `file`, `h5py.Group`
        Target file path, or an open group in which to create the dataset.
    path : `str`, optional
        Dataset path inside the file; defaults to ``series.name``.
    attrs : `dict`, optional
        Caller-supplied dataset attributes replacing the default axis
        attributes, as in GWpy.
    **kwargs
        ``append``, ``overwrite``, ``compression`` and other keywords accepted
        by `gwpy.types.io.hdf5.write_hdf5_array`.

    Returns
    -------
    dataset : `h5py.Dataset`
        The newly created dataset.

    Raises
    ------
    OSError
        If the target file exists and neither ``append`` nor ``overwrite`` is
        set, or if a long irregular axis cannot be stored in an already open
        file that uses the 'earliest' HDF5 format bound.

    """
    if attrs is not None or not has_irregular_axis(series):
        return write_hdf5_series(series, output, path=path, attrs=attrs, **kwargs)

    if isinstance(output, h5py.HLObject):
        return _write_irregular(series, output, path, **kwargs)

    # Same file-opening contract as ``gwpy.io.hdf5.with_write_hdf5``, with a
    # format bound that can hold an axis longer than 64 KiB as an attribute.
    append = kwargs.get("append", False)
    overwrite = kwargs.get("overwrite", False)
    if (
        isinstance(output, (str, os.PathLike))
        and Path(output).exists()
        and not (overwrite or append)
    ):
        msg = f"File exists: {output}"
        raise OSError(msg)
    with h5py.File(output, "a" if append else "w", libver=IRREGULAR_AXIS_LIBVER) as h5f:
        return _write_irregular(series, h5f, path, **kwargs)
