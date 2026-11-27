"""Coordinate checks before replacing explicit coordinates with affine metadata.

Two different questions arise when an interop converter meets a coordinate
array:

* A target that can carry explicit coordinates (a GWexpy series built from
  Control FRD, Polars, SimPEG or a ROOT ``TGraph``) should keep the original
  samples unless an origin and cadence rebuild every one of them exactly.
  :func:`exactly_affine_axis` answers this, using the same criterion as the
  HDF5 writers (``Index.define(x0, dx, n)`` compared with ``np.array_equal``).
* A target that can only carry an origin and cadence (LAL and PyCBC frequency
  series, fixed-width ROOT histogram binning) cannot hold any other axis.
  Floating-point round-off in how the source axis was generated (for example
  ``numpy.linspace``) must not make an ordinary uniform axis unexportable, but
  a genuinely non-uniform axis must not be silently linearized.
  :func:`affine_representable_axis` answers this with the least-squares
  regularity criterion of :meth:`gwexpy.types.axis.AxisDescriptor.is_regular`
  at ``max_deviation=0``: only deviations within the dtype representation
  floor are accepted.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from gwpy.types import Index

from gwexpy.types.axis import AxisDescriptor


def _increasing_finite_1d(values: np.ndarray) -> bool:
    return bool(
        values.ndim == 1
        and np.all(np.isfinite(values))
        and (values.size < 2 or np.all(np.diff(values) > 0))
    )


def exactly_affine_axis(
    coordinates: np.ndarray,
    *,
    start: float | None = None,
    step: float | None = None,
) -> bool:
    """Return whether GWpy ``Index.define`` reconstructs every coordinate.

    Tolerant interval comparisons are insufficient for converters that emit
    only an origin and cadence. Compare the actual reconstructed coordinates
    before selecting that representation.
    """
    values = np.asarray(coordinates, dtype=float)
    if not _increasing_finite_1d(values):
        return False
    if values.size < 2:
        return True
    origin = float(values[0]) if start is None else float(start)
    cadence = float(values[1] - values[0]) if step is None else float(step)
    if not np.isfinite(origin) or not np.isfinite(cadence) or cadence <= 0:
        return False
    try:
        rebuilt = Index.define(origin, cadence, values.size)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.array_equal(rebuilt.value, values))


def affine_representable_axis(coordinates: np.ndarray) -> bool:
    """Return whether an origin and cadence represent the axis up to round-off.

    The axis must be one-dimensional, finite and strictly increasing. Its
    deviation from the least-squares uniform grid must not exceed the float
    representation floor (``AxisDescriptor.is_regular(max_deviation=0.0)``),
    so a non-uniform axis is rejected regardless of the unit it is expressed
    in.
    """
    values = np.asarray(coordinates, dtype=float)
    if not _increasing_finite_1d(values):
        return False
    if values.size < 3:
        return True
    descriptor = AxisDescriptor("axis", u.Quantity(values, u.dimensionless_unscaled))
    return bool(descriptor.is_regular(max_deviation=0.0))
