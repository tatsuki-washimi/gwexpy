"""Layered 2D field plotting and animation built on GWpy's :class:`Plot`."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
import numpy as np
from astropy import units as u
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ._detectors import DETECTORS
from ._geo_utils import (
    EARTH_MEAN_RADIUS_M,
    ZORDER_BASEMAP,
    ZORDER_MARKERS,
    ZORDER_SCALAR,
    ZORDER_TERRAIN,
    ZORDER_VECTOR,
    GeoPlotContext,
    compute_hillshade,
    compute_shaded_relief,
    convert_axis_values,
    get_geo_context,
    local_xy_to_lonlat,
    lonlat_to_local_xy,
    resolve_geo_context,
    set_geo_context,
)
from ._label_utils import format_axis_label, format_label_with_unit
from ._tiles import PROVIDERS, auto_zoom, deg2tile, fetch_tile, tile_extent_deg
from .plot import Plot

__all__ = ["FieldPlot"]

_LABEL_NOT_PROVIDED = object()


def _resolve_display_axes(
    field: Any,
    x: str | None,
    y: str | None,
    slice_kwargs: dict[str, Any],
    *,
    exclude: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Select default display axes preferring non-singleton dimensions.

    ``FieldBase.get_slice()`` picks the first two axes not in ``fixed_coords``,
    which selects a singleton time axis on ``(1, nx, ny, 1)`` DEM fields.
    Selecting the first two axes with size > 1 avoids that pitfall without
    altering ``get_slice()`` itself.
    """
    if x is not None and y is not None:
        return x, y

    all_axes = [
        field._axis0_name,
        field._axis1_name,
        field._axis2_name,
        field._axis3_name,
    ]
    excluded = set(slice_kwargs.keys())
    if exclude:
        excluded.update(exclude)

    sizes = {all_axes[i]: len(getattr(field, f"_axis{i}_index")) for i in range(4)}
    non_singleton = [
        ax_name
        for ax_name in all_axes
        if ax_name not in excluded and sizes[ax_name] > 1
    ]
    fallback = [ax_name for ax_name in all_axes if ax_name not in excluded]

    if x is None and y is None:
        if len(non_singleton) >= 2:
            return non_singleton[0], non_singleton[1]
        if len(fallback) >= 2:
            return fallback[0], fallback[1]
        return x, y

    if x is None:
        pool = [a for a in non_singleton if a != y] or [a for a in fallback if a != y]
        return (pool[0] if pool else None), y

    pool = [a for a in non_singleton if a != x] or [a for a in fallback if a != x]
    return x, (pool[0] if pool else None)


def _format_frame_title(axis: str, frame_val: Any) -> str:
    if isinstance(frame_val, u.Quantity):
        return f"{axis} = {frame_val:.3g}"
    return f"{axis} = {frame_val}"


def _convert_scalar_or_array(
    values: Any,
    source_unit: u.UnitBase,
    target_unit: u.UnitBase,
) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(getattr(values, "value", values), dtype=float))
    val_unit = getattr(values, "unit", source_unit)
    if (
        val_unit is not None
        and val_unit != u.dimensionless_unscaled
        and target_unit != u.dimensionless_unscaled
        and val_unit != target_unit
    ):
        arr = np.atleast_1d(
            np.asarray((arr * val_unit).to_value(target_unit), dtype=float)
        )
    return arr


class FieldPlot(Plot):
    """Enhanced plot class for visualizing ScalarField, VectorField, and TensorField.

    Provides a builder-style interface for composing layered 2D geographic and
    physical-field visualizations (basemap tiles, DEM terrain relief/contours,
    scalar pcolormesh overlays, vector quiver/streamline layers, and markers)
    as well as static-background animations.

    Layer ``zorder`` conventions (all overridable via keyword arguments):

    - basemap tiles (:meth:`add_basemap`): ``0.0``
    - terrain relief / contours (:meth:`add_terrain`): ``1.0``
    - scalar fields (:meth:`add_scalar`): ``2.0``
    - vector fields (:meth:`add_vector`): ``3.0``
    - markers / detectors (:meth:`add_markers`, :meth:`add_detector`): ``4.0``

    Attributes
    ----------
    last_field_colorbar
        Most recent colorbar returned by :meth:`add_scalar` when
        ``colorbar=True`` (or by :meth:`add_terrain` when ``colorbar=True``).

    """

    def __init__(self, *args, **kwargs):
        """Initialize FieldPlot.

        If a field is passed as the first argument, it initializes the plot
        but does not automatically render it (unless method is implicit).
        Typically usage is:

        >>> plot = FieldPlot()
        >>> plot.add_scalar(field, z=0)

        Or via field method:
        >>> field.plot(z=0)
        """
        super().__init__(*args, **kwargs)
        self.last_field_colorbar = None

    def add_scalar(
        self,
        field,
        x=None,
        y=None,
        slice_kwargs=None,
        *,
        colorbar: bool = True,
        **kwargs,
    ):
        """Add a scalar field slice to the plot.

        Parameters
        ----------
        field : ScalarField
            The field to plot.
        x : str, optional
            Name of X-axis.
        y : str, optional
            Name of Y-axis.
        slice_kwargs : dict, optional
            Fixed coordinates for slicing (e.g. ``{'z': 0, 't': 10}``).
        colorbar : bool, optional
            If ``True`` (default), create a colorbar and update
            :attr:`last_field_colorbar`. Set to ``False`` when overlaying
            scalar layers where an extra colorbar is undesirable.
        **kwargs
            Arguments passed to ``ax.pcolormesh`` (``cmap``, ``vmin``,
            ``vmax``, ``alpha``, ``zorder``, etc.). Default ``zorder`` is
            ``2.0``.

        Returns
        -------
        matplotlib.collections.QuadMesh
            The rendered pcolormesh artist.

        """
        if slice_kwargs is None:
            slice_kwargs = {}

        data, x_idx, y_idx, x_name, y_name = field.get_slice(
            x_axis=x, y_axis=y, **slice_kwargs
        )

        ax = self.gca()
        ctx = resolve_geo_context(
            ax,
            field,
            x_name,
            y_name,
            x_unit=getattr(x_idx, "unit", None),
            y_unit=getattr(y_idx, "unit", None),
        )
        x_vals = convert_axis_values(x_idx, ctx.axis_units[0])
        y_vals = convert_axis_values(y_idx, ctx.axis_units[1])

        kwargs.setdefault("zorder", ZORDER_SCALAR)
        shading = kwargs.pop("shading", "auto")
        label = kwargs.pop("label", _LABEL_NOT_PROVIDED)

        mesh = ax.pcolormesh(x_vals, y_vals, data.value.T, shading=shading, **kwargs)

        if not ax.get_xlabel():
            ax.set_xlabel(format_axis_label(x_name, ctx.axis_units[0]))
        if not ax.get_ylabel():
            ax.set_ylabel(format_axis_label(y_name, ctx.axis_units[1]))

        if colorbar:
            if label is _LABEL_NOT_PROVIDED:
                cbar_label = format_label_with_unit(
                    getattr(field, "name", None), getattr(field, "unit", None)
                )
            elif label is None:
                cbar_label = None
            elif label == "":
                cbar_label = ""
            else:
                cbar_label = format_label_with_unit(label, getattr(field, "unit", None))
            cbar_kwargs = {"label": cbar_label} if cbar_label is not None else {}
            self.last_field_colorbar = self.colorbar(mesh, **cbar_kwargs)

        return mesh

    def add_vector(
        self, field, x=None, y=None, mode="quiver", slice_kwargs=None, **kwargs
    ):
        """Add a vector field slice to the plot.

        Parameters
        ----------
        field : VectorField
            The field to plot.
        x : str, optional
            Name of the x-axis to display.
        y : str, optional
            Name of the y-axis to display.
        mode : {'quiver', 'streamline'}
            Plotting mode.
        slice_kwargs : dict, optional
            Fixed coordinates used to extract the displayed slice.
        **kwargs
            Additional keyword arguments passed to the plotting backend.
            Default ``zorder`` is ``3.0``.

        Notes
        -----
        When plotting on ``angle``-domain axes (degrees/radians), vector arrow
        lengths are scaled in display coordinates rather than physical meters.
        Convert fields with ``to_local_cartesian()`` for physically consistent
        vector geometry.

        """
        if slice_kwargs is None:
            slice_kwargs = {}

        first_comp = list(field.values())[0]
        _, x_idx, y_idx, x_name, y_name = first_comp.get_slice(
            x_axis=x, y_axis=y, **slice_kwargs
        )

        if x_name in field and y_name in field:
            u_field = field[x_name]
            v_field = field[y_name]
        else:
            keys = list(field.keys())
            if len(keys) >= 2:
                u_field = field[keys[0]]
                v_field = field[keys[1]]
            else:
                raise ValueError(
                    f"Cannot map vector components to axes {x_name}, {y_name}"
                )

        u_data, _, _, _, _ = u_field.get_slice(
            x_axis=x_name, y_axis=y_name, **slice_kwargs
        )
        v_data, _, _, _, _ = v_field.get_slice(
            x_axis=x_name, y_axis=y_name, **slice_kwargs
        )

        ax = self.gca()
        ctx = resolve_geo_context(
            ax,
            first_comp,
            x_name,
            y_name,
            x_unit=getattr(x_idx, "unit", None),
            y_unit=getattr(y_idx, "unit", None),
        )
        x_vals = convert_axis_values(x_idx, ctx.axis_units[0])
        y_vals = convert_axis_values(y_idx, ctx.axis_units[1])

        kwargs.setdefault("zorder", ZORDER_VECTOR)
        u_val = u_data.value.T
        v_val = v_data.value.T

        if mode == "quiver":
            stride = kwargs.pop("stride", 1)
            mesh = ax.quiver(
                x_vals[::stride],
                y_vals[::stride],
                u_val[::stride, ::stride],
                v_val[::stride, ::stride],
                **kwargs,
            )
        elif mode == "streamline":
            mesh = ax.streamplot(x_vals, y_vals, u_val, v_val, **kwargs)
        else:
            raise ValueError(f"Unknown vector mode: {mode}")

        if not ax.get_xlabel():
            ax.set_xlabel(format_axis_label(x_name, ctx.axis_units[0]))
        if not ax.get_ylabel():
            ax.set_ylabel(format_axis_label(y_name, ctx.axis_units[1]))

        return mesh

    def add_terrain(
        self,
        dem,
        x=None,
        y=None,
        *,
        style: str = "hillshade",
        azdeg: float = 315.0,
        altdeg: float = 45.0,
        vert_exag: float = 1.0,
        levels=None,
        cmap=None,
        colorbar: bool = False,
        slice_kwargs=None,
        **kwargs,
    ):
        """Render a DEM :class:`~gwexpy.fields.ScalarField` as a terrain background layer.

        Parameters
        ----------
        dem : ScalarField
            Digital elevation model field (angle-domain ``lon``/``lat`` or
            local-Cartesian ``x``/``y``).
        x, y : str, optional
            Axis names to display. When omitted, the first two axes with
            size > 1 are chosen automatically (avoiding the singleton time/z
            axes of ``(1, nx, ny, 1)`` DEMs).
        style : {'hillshade', 'shaded_relief', 'contour', 'contourf'}
            Terrain rendering mode.
        azdeg, altdeg : float, optional
            Illumination azimuth and altitude in degrees for relief shading.
        vert_exag : float, optional
            Vertical exaggeration factor for hillshade/shaded relief.
        levels : int or sequence of float, optional
            Contour levels (only for ``'contour'`` and ``'contourf'``).
        cmap : str or Colormap, optional
            Colormap name or object. Defaults to ``'gray'`` for
            ``'hillshade'`` and ``'terrain'`` for ``'shaded_relief'``.
        colorbar : bool, optional
            Whether to attach a colorbar (default ``False`` for background).
        slice_kwargs : dict, optional
            Fixed coordinates for slicing.
        **kwargs
            Additional artist keyword arguments (e.g., ``alpha``, ``zorder``).
            Default ``zorder`` is ``1.0``.

        Returns
        -------
        matplotlib.image.AxesImage or matplotlib.contour.QuadContourSet
            The created terrain artist.

        Notes
        -----
        On ``angle``-domain axes, horizontal grid spacing ``dx`` uses the
        tangent-plane ``cos(lat_center)`` approximation. When ``dem.unit`` is
        length-equivalent (e.g., meters), angular spacing is scaled to meters
        via mean Earth radius so ``vert_exag=1.0`` reflects physical slope.

        """
        valid_styles = {"hillshade", "shaded_relief", "contour", "contourf"}
        if style not in valid_styles:
            raise ValueError(
                f"Unknown terrain style '{style}'. Expected one of {sorted(valid_styles)}."
            )

        slice_kw = dict(slice_kwargs) if slice_kwargs is not None else {}
        x_resolved, y_resolved = _resolve_display_axes(dem, x, y, slice_kw)
        data, x_idx, y_idx, x_name, y_name = dem.get_slice(
            x_axis=x_resolved, y_axis=y_resolved, **slice_kw
        )

        ax = self.gca()
        ctx = resolve_geo_context(
            ax,
            dem,
            x_name,
            y_name,
            x_unit=getattr(x_idx, "unit", None),
            y_unit=getattr(y_idx, "unit", None),
        )
        x_vals = convert_axis_values(x_idx, ctx.axis_units[0])
        y_vals = convert_axis_values(y_idx, ctx.axis_units[1])
        z_arr = np.asarray(data.value.T, dtype=float)

        # Keep horizontal coordinate ascending so extent left < right
        if len(x_vals) >= 2 and x_vals[0] > x_vals[-1]:
            x_vals = x_vals[::-1]
            z_arr = z_arr[:, ::-1]

        y_descending = len(y_vals) >= 2 and y_vals[0] > y_vals[-1]
        kwargs.setdefault("zorder", ZORDER_TERRAIN)

        artist: Any
        if style in {"hillshade", "shaded_relief"}:
            hx = 0.5 * abs(float(x_vals[1] - x_vals[0])) if len(x_vals) >= 2 else 0.5
            hy = 0.5 * abs(float(y_vals[1] - y_vals[0])) if len(y_vals) >= 2 else 0.5
            extent = (
                float(x_vals[0] - hx),
                float(x_vals[-1] + hx),
                float(min(y_vals[0], y_vals[-1]) - hy),
                float(max(y_vals[0], y_vals[-1]) + hy),
            )
            origin = "upper" if y_descending else "lower"

            dem_unit = getattr(dem, "unit", u.dimensionless_unscaled)
            if ctx.kind == "angle":
                x_unit = (
                    ctx.axis_units[0]
                    if ctx.axis_units[0] != u.dimensionless_unscaled
                    else u.deg
                )
                y_unit = (
                    ctx.axis_units[1]
                    if ctx.axis_units[1] != u.dimensionless_unscaled
                    else u.deg
                )
                lat_center_rad = float(
                    (0.5 * (y_vals[0] + y_vals[-1]) * y_unit).to_value(u.rad)
                )
                dx_raw = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) >= 2 else 1.0
                dy_raw = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) >= 2 else 1.0
                if dem_unit is not None and dem_unit.is_equivalent(u.m):
                    z_ls = np.asarray((z_arr * dem_unit).to_value(u.m), dtype=float)
                    dlon_rad = float((dx_raw * x_unit).to_value(u.rad))
                    dlat_rad = float((dy_raw * y_unit).to_value(u.rad))
                    dx_ls = (
                        EARTH_MEAN_RADIUS_M * float(np.cos(lat_center_rad)) * dlon_rad
                    )
                    dy_ls = EARTH_MEAN_RADIUS_M * dlat_rad
                else:
                    z_ls = z_arr
                    dx_ls = dx_raw * float(np.cos(lat_center_rad))
                    dy_ls = dy_raw
            else:
                z_ls = z_arr
                dx_ls = abs(float(x_vals[1] - x_vals[0])) if len(x_vals) >= 2 else 1.0
                dy_ls = abs(float(y_vals[1] - y_vals[0])) if len(y_vals) >= 2 else 1.0

            if not y_descending:
                dy_ls = -dy_ls

            kwargs.setdefault("aspect", "auto")
            if style == "hillshade":
                hs = compute_hillshade(
                    z_ls,
                    dx=dx_ls,
                    dy=dy_ls,
                    azdeg=azdeg,
                    altdeg=altdeg,
                    vert_exag=vert_exag,
                )
                cmap_src = "gray" if cmap is None else cmap
                cmap_obj = (
                    matplotlib.colormaps.get_cmap(cmap_src).copy()
                    if isinstance(cmap_src, str)
                    else cmap_src.copy()
                )
                cmap_obj.set_bad(alpha=0.0)
                vmin = kwargs.pop("vmin", 0.0)
                vmax = kwargs.pop("vmax", 1.0)
                artist = ax.imshow(
                    hs,
                    extent=extent,
                    origin=origin,
                    cmap=cmap_obj,
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )
                if colorbar:
                    cbar_label = format_label_with_unit(
                        getattr(dem, "name", None) or "Hillshade", None
                    )
                    cbar_kw = {"label": cbar_label} if cbar_label else {}
                    self.last_field_colorbar = self.colorbar(artist, **cbar_kw)
            else:
                cmap_src = "terrain" if cmap is None else cmap
                cmap_obj = (
                    matplotlib.colormaps.get_cmap(cmap_src)
                    if isinstance(cmap_src, str)
                    else cmap_src
                )
                vmin = kwargs.pop("vmin", None)
                vmax = kwargs.pop("vmax", None)
                rgba = compute_shaded_relief(
                    z_ls,
                    cmap=cmap_obj,
                    dx=dx_ls,
                    dy=dy_ls,
                    azdeg=azdeg,
                    altdeg=altdeg,
                    vert_exag=vert_exag,
                    vmin=vmin,
                    vmax=vmax,
                )
                artist = ax.imshow(
                    rgba,
                    extent=extent,
                    origin=origin,
                    **kwargs,
                )
                if colorbar:
                    finite_vals = z_arr[np.isfinite(z_arr)]
                    cmin = (
                        float(np.min(finite_vals))
                        if vmin is None and finite_vals.size
                        else (0.0 if vmin is None else float(vmin))
                    )
                    cmax = (
                        float(np.max(finite_vals))
                        if vmax is None and finite_vals.size
                        else (1.0 if vmax is None else float(vmax))
                    )
                    sm = ScalarMappable(
                        norm=Normalize(vmin=cmin, vmax=cmax), cmap=cmap_obj
                    )
                    sm.set_array([])
                    cbar_label = format_label_with_unit(
                        getattr(dem, "name", None) or "Elevation",
                        getattr(dem, "unit", None),
                    )
                    cbar_kw = {"label": cbar_label} if cbar_label else {}
                    self.last_field_colorbar = self.colorbar(sm, ax=ax, **cbar_kw)
        else:
            contour_kwargs = dict(kwargs)
            if levels is not None:
                contour_kwargs["levels"] = levels
            if cmap is not None:
                contour_kwargs["cmap"] = cmap
            fn = ax.contour if style == "contour" else ax.contourf
            artist = fn(x_vals, y_vals, z_arr, **contour_kwargs)
            if colorbar:
                cbar_label = format_label_with_unit(
                    getattr(dem, "name", None) or "Elevation",
                    getattr(dem, "unit", None),
                )
                cbar_kw = {"label": cbar_label} if cbar_label else {}
                self.last_field_colorbar = self.colorbar(artist, **cbar_kw)

        if not ax.get_xlabel():
            ax.set_xlabel(format_axis_label(x_name, ctx.axis_units[0]))
        if not ax.get_ylabel():
            ax.set_ylabel(format_axis_label(y_name, ctx.axis_units[1]))

        return artist

    def add_markers(
        self,
        x,
        y,
        *,
        labels=None,
        marker: str = "o",
        zorder: float = ZORDER_MARKERS,
        **kwargs,
    ):
        """Overlay point markers (and optional text labels) onto the current axes.

        Parameters
        ----------
        x, y : float, array-like, or astropy.units.Quantity
            Marker coordinates.
        labels : str or sequence of str, optional
            Text annotations placed next to each marker.
        marker : str, optional
            Matplotlib marker style (default ``'o'``).
        zorder : float, optional
            Layer drawing order (default ``4.0``).
        **kwargs
            Additional arguments passed to ``ax.scatter``.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter artist.

        """
        ax = self.gca()
        ctx = get_geo_context(ax)
        if ctx is not None:
            x_vals = _convert_scalar_or_array(
                x, getattr(x, "unit", u.dimensionless_unscaled), ctx.axis_units[0]
            )
            y_vals = _convert_scalar_or_array(
                y, getattr(y, "unit", u.dimensionless_unscaled), ctx.axis_units[1]
            )
        else:
            x_vals = np.atleast_1d(np.asarray(getattr(x, "value", x), dtype=float))
            y_vals = np.atleast_1d(np.asarray(getattr(y, "value", y), dtype=float))

        scatter_kwargs = dict(kwargs)
        if "markersize" in scatter_kwargs and "s" not in scatter_kwargs:
            scatter_kwargs["s"] = float(scatter_kwargs.pop("markersize")) ** 2

        artist = ax.scatter(
            x_vals, y_vals, marker=marker, zorder=zorder, **scatter_kwargs
        )

        if labels is not None:
            label_list = [labels] if isinstance(labels, str) else list(labels)
            for xi, yi, lbl in zip(x_vals, y_vals, label_list, strict=False):
                if lbl is not None:
                    ax.annotate(
                        str(lbl),
                        (float(xi), float(yi)),
                        xytext=(4, 4),
                        textcoords="offset points",
                        zorder=zorder,
                    )

        return artist

    def add_detector(
        self,
        name: str,
        *,
        label: bool | str = True,
        zorder: float = ZORDER_MARKERS,
        **kwargs,
    ):
        """Plot a known gravitational-wave detector on geographic or local axes.

        Parameters
        ----------
        name : str
            Detector key from :data:`gwexpy.plot._detectors.DETECTORS`
            (e.g., ``'K1'``, ``'H1'``, ``'L1'``, ``'V1'``, ``'G1'``).
        label : bool or str, optional
            If ``True``, annotate with the detector's full name. Pass a string
            to override the label text, or ``False`` to omit.
        zorder : float, optional
            Layer drawing order (default ``4.0``).
        **kwargs
            Additional arguments forwarded to :meth:`add_markers`.

        """
        if name not in DETECTORS:
            raise ValueError(
                f"Unknown detector: '{name}'. Available: {list(DETECTORS)}"
            )
        det = DETECTORS[name]
        ax = self.gca()
        ctx = get_geo_context(ax)

        if ctx is not None and ctx.kind == "plain":
            raise ValueError(
                "Cannot add geographic detector marker to a plain "
                "(non-georeferenced) field plot."
            )

        lon_deg = float(det["lon"])
        lat_deg = float(det["lat"])

        if ctx is None:
            ctx = set_geo_context(
                ax,
                GeoPlotContext(
                    kind="angle",
                    crs=None,
                    local_origin=None,
                    axis_units=(u.deg, u.deg),
                ),
            )
            px, py = lon_deg, lat_deg
            if not ax.get_xlabel():
                ax.set_xlabel(format_axis_label("lon", u.deg))
            if not ax.get_ylabel():
                ax.set_ylabel(format_axis_label("lat", u.deg))
        elif ctx.kind == "angle":
            px = (
                float((lon_deg * u.deg).to_value(ctx.axis_units[0]))
                if ctx.axis_units[0] != u.dimensionless_unscaled
                else lon_deg
            )
            py = (
                float((lat_deg * u.deg).to_value(ctx.axis_units[1]))
                if ctx.axis_units[1] != u.dimensionless_unscaled
                else lat_deg
            )
        else:
            if ctx.local_origin is None:
                raise ValueError(
                    "Cannot place detector on local-Cartesian axes without "
                    "local_origin in GeoPlotContext."
                )
            lon0, lat0 = ctx.local_origin
            px_m, py_m = lonlat_to_local_xy(lon_deg, lat_deg, lon0, lat0)
            px = (
                float((px_m * u.m).to_value(ctx.axis_units[0]))
                if ctx.axis_units[0] != u.dimensionless_unscaled
                else float(px_m)
            )
            py = (
                float((py_m * u.m).to_value(ctx.axis_units[1]))
                if ctx.axis_units[1] != u.dimensionless_unscaled
                else float(py_m)
            )

        marker = kwargs.pop("marker", "*")
        kwargs.setdefault("color", det["color"])
        if "s" not in kwargs and "markersize" not in kwargs:
            kwargs["s"] = 120

        if isinstance(label, str):
            lbl: str | None = label
        elif label:
            lbl = str(det["name"])
        else:
            lbl = None

        return self.add_markers(
            [px],
            [py],
            labels=[lbl] if lbl is not None else None,
            marker=marker,
            zorder=zorder,
            **kwargs,
        )

    def add_basemap(
        self,
        source: str = "gsi-std",
        *,
        zoom: int | None = None,
        extent: tuple[float, float, float, float] | None = None,
        alpha: float = 1.0,
        attribution: bool = True,
        cache_dir=None,
        timeout: float = 10.0,
        **kwargs,
    ) -> list[Any]:
        """Add XYZ slippy-map tiles as a background basemap layer.

        Parameters
        ----------
        source : str, optional
            Provider name (``'gsi-std'``, ``'gsi-pale'``, ``'gsi-photo'``,
            ``'gsi-relief'``, ``'osm'``) or custom ``{z}/{x}/{y}`` URL template.
        zoom : int, optional
            Tile zoom level. When ``None``, chosen automatically from the
            axes extent subject to a tile count cap.
        extent : tuple of 4 float, optional
            Explicit ``(xmin, xmax, ymin, ymax)`` bounds in current axes
            units (or degrees when axes are empty). Defaults to current limits.
        alpha : float, optional
            Tile opacity in ``[0, 1]``.
        attribution : bool, optional
            If ``True`` (default), render provider attribution text in the
            corner of the axes (required by GSI tile terms of use).
        cache_dir : str or Path, optional
            Directory for caching downloaded tiles on disk.
        timeout : float, optional
            Network timeout in seconds per tile request.
        **kwargs
            Additional keyword arguments passed to ``ax.imshow``. Default
            ``zorder`` is ``0.0``.

        Returns
        -------
        list of matplotlib.image.AxesImage
            The rendered tile image artists.

        Notes
        -----
        Tiles are placed individually at their geographic lon/lat extents
        without WebMercator-to-plate-carree warping. For local regions
        (latitude span <= ~2 deg) distortion is negligible; wider spans emit
        a ``UserWarning`` recommending :class:`~gwexpy.plot.GeoMap`.

        """
        if source in PROVIDERS:
            provider_info = PROVIDERS[source]
            url_template = str(provider_info["url"])
            attr_text = str(provider_info["attribution"])
            provider_name = source
            provider_max_zoom = int(provider_info.get("max_zoom", 18))
        elif (
            isinstance(source, str)
            and "{z}" in source
            and "{x}" in source
            and "{y}" in source
        ):
            url_template = source
            attr_text = kwargs.pop("attribution_text", None)
            provider_name = "custom"
            provider_max_zoom = 18
        else:
            raise ValueError(
                f"Unknown basemap source '{source}'. Expected one of "
                f"{list(PROVIDERS)} or a URL template containing '{{z}}/{{x}}/{{y}}'."
            )

        ax = self.gca()
        ctx = get_geo_context(ax)
        if ctx is not None and ctx.kind == "plain":
            raise ValueError(
                "Cannot add geographic basemap to a plain "
                "(non-georeferenced) field plot."
            )

        had_data = ax.has_data()
        prev_xlim = ax.get_xlim()
        prev_ylim = ax.get_ylim()

        if extent is not None:
            xmin, xmax, ymin, ymax = (float(v) for v in extent)
        else:
            if not had_data and prev_xlim == (0.0, 1.0) and prev_ylim == (0.0, 1.0):
                raise ValueError(
                    "Cannot infer basemap extent from empty axes; pass "
                    "extent=(xmin, xmax, ymin, ymax) or plot a georeferenced "
                    "layer first."
                )
            xmin, xmax = float(prev_xlim[0]), float(prev_xlim[1])
            ymin, ymax = float(prev_ylim[0]), float(prev_ylim[1])
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        if ctx is None:
            ctx = set_geo_context(
                ax,
                GeoPlotContext(
                    kind="angle",
                    crs=None,
                    local_origin=None,
                    axis_units=(u.deg, u.deg),
                ),
            )
            lon_min, lon_max, lat_min, lat_max = xmin, xmax, ymin, ymax
            coord_mode = "angle"
        elif ctx.kind == "angle":
            lon_min = (
                float((xmin * ctx.axis_units[0]).to_value(u.deg))
                if ctx.axis_units[0] != u.dimensionless_unscaled
                else xmin
            )
            lon_max = (
                float((xmax * ctx.axis_units[0]).to_value(u.deg))
                if ctx.axis_units[0] != u.dimensionless_unscaled
                else xmax
            )
            lat_min = (
                float((ymin * ctx.axis_units[1]).to_value(u.deg))
                if ctx.axis_units[1] != u.dimensionless_unscaled
                else ymin
            )
            lat_max = (
                float((ymax * ctx.axis_units[1]).to_value(u.deg))
                if ctx.axis_units[1] != u.dimensionless_unscaled
                else ymax
            )
            coord_mode = "angle"
        else:
            if ctx.local_origin is None:
                raise ValueError(
                    "Cannot add basemap to local-Cartesian axes without "
                    "local_origin; pass origin=(lon0, lat0) when converting."
                )
            lon0, lat0 = ctx.local_origin
            x_m_vals = (
                np.asarray(
                    u.Quantity([xmin, xmax], ctx.axis_units[0]).to_value(u.m),
                    dtype=np.float64,
                )
                if ctx.axis_units[0] != u.dimensionless_unscaled
                else np.array([xmin, xmax], dtype=np.float64)
            )
            y_m_vals = (
                np.asarray(
                    u.Quantity([ymin, ymax], ctx.axis_units[1]).to_value(u.m),
                    dtype=np.float64,
                )
                if ctx.axis_units[1] != u.dimensionless_unscaled
                else np.array([ymin, ymax], dtype=np.float64)
            )
            lons, lats = local_xy_to_lonlat(x_m_vals, y_m_vals, lon0, lat0)
            lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
            lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
            coord_mode = "local"

        lat_span = abs(lat_max - lat_min)
        if lat_span > 2.0:
            warnings.warn(
                f"Basemap latitude span ({lat_span:.2f} deg) exceeds 2 deg; "
                "tile-by-tile extent placement introduces WebMercator distortion. "
                "Consider using GeoMap for wide-area geodetic plots.",
                UserWarning,
                stacklevel=2,
            )

        max_tiles = int(kwargs.pop("max_tiles", 16))
        z = (
            auto_zoom(
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                max_tiles=max_tiles,
                max_zoom=provider_max_zoom,
            )
            if zoom is None
            else int(zoom)
        )

        tx_a, ty_a = deg2tile(lon_min, lat_min, z)
        tx_b, ty_b = deg2tile(lon_max, lat_max, z)
        tx0, tx1 = min(tx_a, tx_b), max(tx_a, tx_b)
        ty0, ty1 = min(ty_a, ty_b), max(ty_a, ty_b)

        zorder = kwargs.pop("zorder", ZORDER_BASEMAP)
        kwargs.setdefault("aspect", "auto")

        images: list[Any] = []
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                url = url_template.format(z=z, x=tx, y=ty)
                im = fetch_tile(
                    url,
                    timeout=timeout,
                    cache_dir=cache_dir,
                    provider=provider_name,
                )
                tlon0, tlon1, tlat0, tlat1 = tile_extent_deg(tx, ty, z)
                if coord_mode == "angle":
                    ex0 = (
                        float((tlon0 * u.deg).to_value(ctx.axis_units[0]))
                        if ctx.axis_units[0] != u.dimensionless_unscaled
                        else tlon0
                    )
                    ex1 = (
                        float((tlon1 * u.deg).to_value(ctx.axis_units[0]))
                        if ctx.axis_units[0] != u.dimensionless_unscaled
                        else tlon1
                    )
                    ey0 = (
                        float((tlat0 * u.deg).to_value(ctx.axis_units[1]))
                        if ctx.axis_units[1] != u.dimensionless_unscaled
                        else tlat0
                    )
                    ey1 = (
                        float((tlat1 * u.deg).to_value(ctx.axis_units[1]))
                        if ctx.axis_units[1] != u.dimensionless_unscaled
                        else tlat1
                    )
                else:
                    assert ctx.local_origin is not None
                    lon0, lat0 = ctx.local_origin
                    lx_m, ly_m = lonlat_to_local_xy(
                        np.array([tlon0, tlon1]),
                        np.array([tlat0, tlat1]),
                        lon0,
                        lat0,
                    )
                    ex0 = (
                        float((lx_m[0] * u.m).to_value(ctx.axis_units[0]))
                        if ctx.axis_units[0] != u.dimensionless_unscaled
                        else float(lx_m[0])
                    )
                    ex1 = (
                        float((lx_m[1] * u.m).to_value(ctx.axis_units[0]))
                        if ctx.axis_units[0] != u.dimensionless_unscaled
                        else float(lx_m[1])
                    )
                    ey0 = (
                        float((ly_m[0] * u.m).to_value(ctx.axis_units[1]))
                        if ctx.axis_units[1] != u.dimensionless_unscaled
                        else float(ly_m[0])
                    )
                    ey1 = (
                        float((ly_m[1] * u.m).to_value(ctx.axis_units[1]))
                        if ctx.axis_units[1] != u.dimensionless_unscaled
                        else float(ly_m[1])
                    )

                img_artist = ax.imshow(
                    np.asarray(im),
                    extent=(ex0, ex1, ey0, ey1),
                    origin="upper",
                    alpha=alpha,
                    zorder=zorder,
                    **kwargs,
                )
                images.append(img_artist)

        if had_data:
            ax.set_xlim(prev_xlim)
            ax.set_ylim(prev_ylim)
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if coord_mode == "angle":
            if not ax.get_xlabel():
                ax.set_xlabel(format_axis_label("lon", ctx.axis_units[0]))
            if not ax.get_ylabel():
                ax.set_ylabel(format_axis_label("lat", ctx.axis_units[1]))

        if attribution and attr_text:
            ax.text(
                0.99,
                0.01,
                attr_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
                zorder=max(float(zorder), ZORDER_MARKERS) + 0.5,
            )

        return images

    def animate_scalar(
        self,
        field,
        x=None,
        y=None,
        axis: str = "t",
        interval: int = 100,
        slice_kwargs=None,
        *,
        colorbar: bool = True,
        **kwargs,
    ) -> FuncAnimation:
        """Animate a scalar field slice while preserving static background layers.

        Creates a single ``QuadMesh`` and (optionally) a single colorbar once,
        then updates only the mesh array via ``set_array()`` and the axes title
        via ``set_title()`` on each frame. Existing basemap, terrain, and
        marker layers on the axes remain untouched.

        Parameters
        ----------
        field : ScalarField
            Field to animate.
        x, y : str, optional
            Display axis names.
        axis : str, optional
            Axis name to iterate across frames (default ``'t'``).
        interval : int, optional
            Delay between frames in milliseconds (default ``100``).
        slice_kwargs : dict, optional
            Fixed coordinates for non-animated, non-displayed axes.
        colorbar : bool, optional
            Whether to create a single colorbar (default ``True``).
        **kwargs
            Additional arguments forwarded to :meth:`add_scalar`. Note that
            ``shading='gouraud'`` is rejected with ``ValueError`` because it
            is incompatible with ``QuadMesh.set_array()``.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object (with ``ani._fig = self.figure``).

        """
        shading = kwargs.get("shading", "auto")
        if shading == "gouraud":
            raise ValueError(
                "shading='gouraud' is not supported in animate_scalar because "
                "QuadMesh.set_array() requires flat or auto shading."
            )

        all_axes = [
            field._axis0_name,
            field._axis1_name,
            field._axis2_name,
            field._axis3_name,
        ]
        try:
            loop_idx_pos = all_axes.index(axis)
        except ValueError as exc:
            raise ValueError(
                f"Animation axis '{axis}' not found in {all_axes}"
            ) from exc

        loop_axis_index = getattr(field, f"_axis{loop_idx_pos}_index")
        if len(loop_axis_index) == 0:
            raise ValueError(f"Animation axis '{axis}' is empty")

        base_slice = dict(slice_kwargs) if slice_kwargs is not None else {}
        x_resolved, y_resolved = _resolve_display_axes(
            field, x, y, base_slice, exclude={axis}
        )

        plot_kwargs = dict(kwargs)
        if "vmin" not in plot_kwargs:
            plot_kwargs["vmin"] = float(np.nanmin(field.value))
        if "vmax" not in plot_kwargs:
            plot_kwargs["vmax"] = float(np.nanmax(field.value))

        init_slice = dict(base_slice)
        init_slice[axis] = loop_axis_index[0]

        mesh = self.add_scalar(
            field,
            x=x_resolved,
            y=y_resolved,
            slice_kwargs=init_slice,
            colorbar=colorbar,
            **plot_kwargs,
        )
        ax = self.gca()
        ax.set_title(_format_frame_title(axis, loop_axis_index[0]))

        def update(frame_val):
            current_slice = dict(base_slice)
            current_slice[axis] = frame_val
            data, _, _, _, _ = field.get_slice(
                x_axis=x_resolved, y_axis=y_resolved, **current_slice
            )
            arr = np.asarray(data.value.T)
            current_arr = mesh.get_array()
            if current_arr is not None and arr.shape != current_arr.shape:
                mesh.set_array(arr.ravel())
            else:
                mesh.set_array(arr)
            ax.set_title(_format_frame_title(axis, frame_val))
            return (mesh,)

        ani = FuncAnimation(
            self.figure,
            update,
            frames=loop_axis_index,
            interval=interval,
        )
        ani._fig = self.figure  # type: ignore[attr-defined]
        return ani
