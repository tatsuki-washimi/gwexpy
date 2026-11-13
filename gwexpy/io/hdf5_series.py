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
import numpy as np
from astropy import units as u
from gwpy.segments import Segment
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


def needs_explicit_axis(series: Any) -> bool:
    """Return whether ``x0``/``dx`` would change the stored coordinates.

    GWpy's ``Index.regular`` uses an absolute tolerance, so it can classify a
    non-uniform index as regular.  HDF5's compact ``x0``/``dx`` layout is safe
    only when GWpy's reader reconstructs every coordinate exactly.  Otherwise
    the exact index must be stored in the existing ``xindex`` attribute.
    """
    xindex = getattr(series, "_xindex", None)
    if xindex is None:
        return False
    if not xindex.regular:
        return True
    try:
        start = xindex[0]
        step = xindex[1] - start
        reconstructed = type(xindex).define(start, step, xindex.size)
    except (AttributeError, IndexError, TypeError, ValueError):
        # If affine metadata cannot be obtained, retain the only complete
        # representation already available: the explicit coordinates.
        return True
    return not np.array_equal(reconstructed.value, xindex.value)


def _explicit_axis_span(series: Any) -> Any:
    """Return the covered interval from explicit coordinates, not ``dx``."""
    xindex = series.xindex
    if xindex.size == 1:
        coordinate = float(xindex.value[0])
        return Segment(coordinate, coordinate)
    if xindex.size == 0:
        raise ValueError("cannot determine an interval from an empty xindex")
    span = series.span
    last_step = xindex.value[-1] - xindex.value[-2]
    return type(span)(float(xindex.value[0]), float(xindex.value[-1] + last_step))


def irregular_axis_attrs(series: Any) -> dict[str, Any]:
    """Return the index attributes to store for an irregular-axis series.

    Set the axis unit and samples explicitly. GWpy's metadata writer omits an
    ``xindex`` attribute when its tolerance classifies the index as regular.
    """
    return {
        "xunit": str(series.xunit),
        "xindex": np.asarray(series.xindex.value),
    }


def _explicit_axis_attrs(series: Any, attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Combine caller attributes with the authoritative explicit axis data."""
    axis_attrs = irregular_axis_attrs(series)
    combined = dict(attrs or {})

    if "dx" in combined:
        raise ValueError("attrs['dx'] cannot describe an explicit HDF5 xindex")
    if "xindex" in combined and not np.array_equal(
        np.asarray(combined["xindex"]), axis_attrs["xindex"]
    ):
        raise ValueError("attrs['xindex'] conflicts with the series coordinates")
    if "xunit" in combined:
        try:
            supplied_xunit = u.Unit(combined["xunit"])
        except (TypeError, ValueError) as exc:
            raise ValueError("attrs['xunit'] must match the series axis unit") from exc
        if supplied_xunit != series.xunit:
            raise ValueError("attrs['xunit'] conflicts with the series axis unit")
    if "x0" in combined:
        supplied_x0 = combined["x0"]
        try:
            if hasattr(supplied_x0, "to_value"):
                supplied_x0 = supplied_x0.to_value(series.xunit)
            if float(supplied_x0) != float(series.xindex.value[0]):
                raise ValueError("attrs['x0'] conflicts with the series coordinates")
        except (TypeError, u.UnitConversionError) as exc:
            raise ValueError(
                "attrs['x0'] conflicts with the series coordinates"
            ) from exc

    combined.update(axis_attrs)
    return combined


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
    attrs = kwargs.pop("attrs", None)
    if attrs is None:
        attrs = irregular_axis_attrs(series)
    try:
        dataset = write_hdf5_array(series, h5g, path=path, attrs=attrs, **kwargs)
    except OSError as exc:
        raise_for_attribute_limit(series, h5g, path, exc)
        raise
    # The explicit xindex is authoritative. Avoid retaining an affine stride
    # that would invite readers to reconstruct a different axis.
    if "dx" in dataset.attrs:
        del dataset.attrs["dx"]
    return dataset


def write_hdf5_series_any_axis(
    series: Any,
    output: Any,
    path: str | None = None,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> h5py.Dataset:
    """Write a series with a regular or irregular x-axis to HDF5.

    A series whose coordinates are reproduced exactly by ``x0``/``dx`` is
    delegated to GWpy's writer and keeps its on-disk layout (``x0``, ``dx``
    and ``xunit`` dataset attributes). Caller-supplied attributes are passed
    through in that case.

    A series that cannot be represented exactly by ``x0``/``dx`` is written
    with its exact axis samples in the ``xindex`` dataset attribute and the
    axis unit in ``xunit``. ``dx`` is omitted. Non-conflicting caller
    attributes are retained. Values, axis samples, ``unit`` and ``name`` are
    stored without conversion.

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
    if not needs_explicit_axis(series):
        return write_hdf5_series(series, output, path=path, attrs=attrs, **kwargs)

    explicit_attrs = _explicit_axis_attrs(series, attrs)
    explicit_kwargs = {**kwargs, "attrs": explicit_attrs}

    if isinstance(output, h5py.HLObject):
        return _write_irregular(series, output, path, **explicit_kwargs)

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
        return _write_irregular(series, h5f, path, **explicit_kwargs)
