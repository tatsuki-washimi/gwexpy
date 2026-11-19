"""Exact checks before replacing explicit coordinates with affine metadata."""

from __future__ import annotations

import numpy as np
from gwpy.types import Index


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
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        return False
    if values.size < 2:
        return True
    if not np.all(np.diff(values) > 0):
        return False
    origin = float(values[0]) if start is None else float(start)
    cadence = float(values[1] - values[0]) if step is None else float(step)
    if not np.isfinite(origin) or not np.isfinite(cadence) or cadence <= 0:
        return False
    try:
        rebuilt = Index.define(origin, cadence, values.size)
    except (TypeError, ValueError, OverflowError):
        return False
    return np.array_equal(rebuilt.value, values)
