"""HDF5 input/output for `FrequencySeries`.

Regular-axis series are written by GWpy's writer unchanged.  GWpy's writer
fails for an irregular frequency axis because it always requests ``dx``; the
writer registered here stores such an axis as the ``xindex`` dataset attribute
(the representation GWpy's reader already understands) together with ``xunit``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
from gwpy.frequencyseries.io import hdf5 as _gwpy_hdf5
from gwpy.frequencyseries.io.hdf5 import *  # noqa: F403
from gwpy.io.registry import default_registry as _io_registry
from gwpy.types.io.hdf5 import write_hdf5_array as _write_hdf5_array
from gwpy.types.io.hdf5 import write_hdf5_series as _write_hdf5_series

from ..frequencyseries import FrequencySeries

__all__ = getattr(
    _gwpy_hdf5,
    "__all__",
    [name for name in dir(_gwpy_hdf5) if not name.startswith("_")],
)

# Lowest HDF5 file-format bound that provides dense attribute storage.  The
# default ("earliest") bound limits an attribute to one 64 KiB object-header
# message, i.e. fewer than 8192 float64 axis samples.
_IRREGULAR_AXIS_LIBVER = "v108"


def _has_irregular_axis(series: Any) -> bool:
    """Return `True` if ``series`` carries an explicit, non-uniform x-axis."""
    xindex = getattr(series, "_xindex", None)
    return xindex is not None and not xindex.regular


def _write_irregular(
    series: Any,
    h5g: h5py.Group,
    path: str | None,
    **kwargs: Any,
) -> h5py.Dataset:
    """Write ``series`` with its irregular axis into an open HDF5 group."""
    attrs = {"xunit": str(series.xunit)}
    try:
        return _write_hdf5_array(series, h5g, path=path, attrs=attrs, **kwargs)
    except OSError as exc:
        if h5g.file.libver[0] != "earliest" or "too large" not in str(exc):
            raise
        # The dataset was created before the axis attribute was rejected; do
        # not leave a dataset that would read back with a default axis.
        name = path if path is not None else series.name
        if name is not None and name in h5g:
            del h5g[name]
        msg = (
            f"Cannot store the irregular frequency axis of "
            f"{type(series).__name__} {series.name!r} "
            f"({series.xindex.size} samples) in an HDF5 file opened with the "
            "'earliest' format bound, which limits an attribute to 64 KiB. "
            "Pass a file path, or open the file with "
            f"h5py.File(..., libver={_IRREGULAR_AXIS_LIBVER!r})."
        )
        raise OSError(msg) from exc


def write_hdf5_frequencyseries(
    series: Any,
    output: Any,
    path: str | None = None,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> h5py.Dataset:
    """Write a `FrequencySeries` to HDF5.

    A series with a regular frequency axis, or any call with explicit
    ``attrs``, is delegated to GWpy's writer and keeps its on-disk layout
    (``x0``, ``dx`` and ``xunit`` dataset attributes).

    A series with an irregular frequency axis is written with the exact axis
    samples in the ``xindex`` dataset attribute and the axis unit in
    ``xunit``; ``dx`` is not written because it is undefined.  Values, axis
    samples, ``unit`` and ``name`` are stored without conversion.

    Parameters
    ----------
    series : `FrequencySeries`
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
    if attrs is not None or not _has_irregular_axis(series):
        return _write_hdf5_series(series, output, path=path, attrs=attrs, **kwargs)

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
    with h5py.File(
        output, "a" if append else "w", libver=_IRREGULAR_AXIS_LIBVER
    ) as h5f:
        return _write_irregular(series, h5f, path, **kwargs)


_io_registry.register_writer(
    "hdf5", FrequencySeries, write_hdf5_frequencyseries, force=True
)
