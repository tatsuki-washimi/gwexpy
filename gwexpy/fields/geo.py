"""Georeferencing provenance metadata and spatial-bounds helpers for fields."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["GeoMetadata", "geo_bounds"]


@dataclass(frozen=True)
class GeoMetadata:
    """Immutable georeferencing provenance attached to raster-derived fields.

    Axis coordinate arrays on the field remain the single source of truth for
    grid positions. ``transform`` records the read-time GDAL 6-coefficient
    geotransform ``(x_ul, dx, row_rotation, y_ul, col_rotation, dy)`` as frozen
    provenance and is not mutated on slicing. Use :func:`geo_bounds` to derive
    current outer cell bounds directly from the field's spatial axes.
    """

    crs: str | None = None
    transform: tuple[float, ...] | None = None
    vertical_datum: str = "geoid"
    nodata: float | None = None
    band: int | None = None
    source: tuple[str, ...] = ()
    source_format: str | None = None
    active: bool = True
    local_origin: tuple[float, float] | None = None
    history: tuple[str, ...] = ()

    def with_history(self, entry: str) -> GeoMetadata:
        """Return a new :class:`GeoMetadata` with *entry* appended to history."""
        return replace(self, history=(*self.history, str(entry)))

    def deactivated(
        self,
        reason: str,
        *,
        local_origin: tuple[float, float] | None = None,
    ) -> GeoMetadata:
        """Return a deactivated provenance-only copy with *reason* recorded."""
        origin = self.local_origin if local_origin is None else local_origin
        return replace(
            self,
            active=False,
            local_origin=origin,
            history=(*self.history, str(reason)),
        )


def _axis_edge_bounds(
    values: np.ndarray, fallback_step: float | None
) -> tuple[float, float]:
    """Extrapolate pixel-center coordinates along a 1D axis to outer cell edges."""
    if values.size == 0:
        raise ValueError("Cannot compute geo_bounds on an empty spatial axis.")
    if values.size == 1:
        half = 0.5 * abs(float(fallback_step)) if fallback_step is not None else 0.0
        center = float(values[0])
        return center - half, center + half
    edge_start = float(values[0]) - 0.5 * float(values[1] - values[0])
    edge_end = float(values[-1]) + 0.5 * float(values[-1] - values[-2])
    return min(edge_start, edge_end), max(edge_start, edge_end)


def geo_bounds(field: Any) -> tuple[float, float, float, float]:
    """Return ``(xmin, ymin, xmax, ymax)`` extrapolated from pixel-center axes."""
    x_index = getattr(field, "_axis1_index", None)
    y_index = getattr(field, "_axis2_index", None)
    if x_index is None or y_index is None:
        raise ValueError("Field does not expose '_axis1_index' and '_axis2_index'.")

    x_vals = np.asarray(getattr(x_index, "value", x_index), dtype=float)
    y_vals = np.asarray(getattr(y_index, "value", y_index), dtype=float)

    geo = getattr(field, "geo", None)
    tf = getattr(geo, "transform", None) if geo is not None else None
    dx_fallback = float(tf[1]) if tf is not None and len(tf) >= 6 else None
    dy_fallback = float(tf[5]) if tf is not None and len(tf) >= 6 else None

    xmin, xmax = _axis_edge_bounds(x_vals, dx_fallback)
    ymin, ymax = _axis_edge_bounds(y_vals, dy_fallback)
    return (xmin, ymin, xmax, ymax)
