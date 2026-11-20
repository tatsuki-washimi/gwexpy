"""Geographic map plotting using PyGMT backend with Cartopy-like interface."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from gwexpy.interop._optional import require_optional

from ._detectors import DETECTORS
from ._geo_utils import get_axis_unit, get_space_domains

_SKIP_PYGMT = os.getenv("GWEXPY_SKIP_PYGMT", "").lower() in {"1", "true", "yes", "on"}
_PYGMT_IMPORT_ERROR: Exception | None = None
pygmt: Any | None = None
if not _SKIP_PYGMT:
    try:
        import pygmt as _pygmt

        pygmt = _pygmt
        HAS_PYGMT = True
    except Exception as exc:
        HAS_PYGMT = False
        _PYGMT_IMPORT_ERROR = exc
else:
    HAS_PYGMT = False
    _PYGMT_IMPORT_ERROR = RuntimeError("pygmt import skipped via GWEXPY_SKIP_PYGMT")

# Projection mapping (Friendly Name -> GMT Code)
PROJECTIONS = {
    "Mercator": "M",
    "Robinson": "N",
    "Mollweide": "W",
    "PlateCarree": "Q",  # Cylindrical Equidistant
    "Equidistant": "Q",
}

# Marker mapping (Matplotlib -> GMT)
MARKERS = {
    "o": "c",  # circle
    "s": "s",  # square
    "^": "t",  # triangle
    "d": "d",  # diamond
    "+": "+",  # plus
    "x": "x",  # cross
    "*": "a",  # star
}


__all__ = ["DETECTORS", "GeoMap", "MARKERS", "PROJECTIONS"]


def _field_to_grid(field: Any) -> Any:
    """Convert an angle-domain 2D ScalarField into a PyGMT-ready xarray.DataArray."""
    from astropy import units as u

    domains = get_space_domains(field)
    names = list(getattr(field, "axis_names", ()))
    if not names:
        names = [
            str(getattr(field, f"_axis{i}_name"))
            for i in range(4)
            if hasattr(field, f"_axis{i}_name")
        ]

    shape = tuple(getattr(field, "shape", ()))
    non_singleton = [
        names[i] for i in range(min(len(names), len(shape))) if shape[i] > 1
    ]
    display_axes = (
        non_singleton[:2]
        if len(non_singleton) >= 2
        else (names[1:3] if len(names) >= 3 else names[:2])
    )
    if len(display_axes) < 2 or not all(
        domains.get(ax_name) == "angle" for ax_name in display_axes
    ):
        raise ValueError(
            "GeoMap is for geographic coordinates (angle domain); "
            "use FieldPlot for local Cartesian fields."
        )

    require_optional("xarray")
    from gwexpy.interop.xarray_ import to_xarray_field

    da = to_xarray_field(field)
    da2 = da.squeeze(drop=True)
    if da2.ndim != 2:
        raise ValueError(
            f"GeoMap.add_field requires a 2D field after squeezing singleton "
            f"axes, got shape {da2.shape}."
        )

    dims = list(da2.dims)
    if "lat" in dims and "lon" in dims:
        lat_dim, lon_dim = "lat", "lon"
    else:
        lon_dim, lat_dim = str(dims[0]), str(dims[1])

    # Ensure coordinates are in degrees when radians were supplied
    for dim_name in (lon_dim, lat_dim):
        unit = get_axis_unit(field, dim_name)
        if (
            unit != u.dimensionless_unscaled
            and unit != u.deg
            and unit.is_equivalent(u.deg)
            and dim_name in da2.coords
        ):
            deg_vals = (np.asarray(da2.coords[dim_name].values) * unit).to_value(u.deg)
            da2 = da2.assign_coords({dim_name: deg_vals})

    if tuple(da2.dims) != (lat_dim, lon_dim):
        da2 = da2.transpose(lat_dim, lon_dim)

    if lat_dim in da2.coords and da2.coords[lat_dim].size >= 2:
        lat_vals = np.asarray(da2.coords[lat_dim].values)
        if lat_vals[0] > lat_vals[-1]:
            da2 = da2.sortby(lat_dim, ascending=True)
    if lon_dim in da2.coords and da2.coords[lon_dim].size >= 2:
        lon_vals = np.asarray(da2.coords[lon_dim].values)
        if lon_vals[0] > lon_vals[-1]:
            da2 = da2.sortby(lon_dim, ascending=True)

    return da2


class GeoMap:
    """A PyGMT wrapper that provides a Cartopy-like interface.

    Note: GeoMap targets wide-area and geodetic-projection figures. Static
    overlay plotting is supported; animation is out of scope (use
    :class:`~gwexpy.plot.FieldPlot` for animations).
    """

    def __init__(self, projection="Robinson", center_lon=0, width="15c", **kwargs):
        """Initialize a new GeoMap.

        Parameters
        ----------
        projection : str, optional
            The projection name (e.g., 'Robinson', 'Mercator', 'Mollweide').
        center_lon : float, optional
            Central longitude for the projection.
        width : str, optional
            Width of the map (GMT-style, e.g., '15c' for 15 cm).
        **kwargs
            Additional arguments:
            region: GMT region (default 'd' for global).
            frame: GMT frame (default 'afg').

        """
        if not HAS_PYGMT:
            message = "pygmt is required for GeoMap. Install with: pip install pygmt"
            if _PYGMT_IMPORT_ERROR is not None:
                message = f"{message} (details: {_PYGMT_IMPORT_ERROR})"
            raise ImportError(message)

        assert pygmt is not None
        self.fig = pygmt.Figure()

        # Setup projection
        proj_code = PROJECTIONS.get(projection, projection)

        # Create GMT projection string
        # region 'd' is -180/180/-90/90
        self.region = kwargs.get("region", "d")

        if center_lon != 0:
            self.projection = f"{proj_code}{center_lon}/{width}"
        else:
            self.projection = f"{proj_code}{width}"

        # GMT Error: Option -R: Cannot include south/north poles with Mercator projection!
        if proj_code == "M" and self.region == "d":
            self.region = [-180, 180, -80, 80]  # Trim poles for Mercator

        self.frame = kwargs.get("frame", "ag")  # auto, grid

        # Initialize base map
        self.fig.basemap(
            region=self.region, projection=self.projection, frame=self.frame
        )

    def add_coastlines(self, resolution="low", color="black", linewidth=0.5):
        """Draw coastlines (Cartopy-like API)."""
        res_map = {"low": "l", "medium": "i", "high": "h", "crude": "c", "full": "f"}
        res = res_map.get(resolution, "l")

        pen = f"{linewidth}p,{color}"
        self.fig.coast(shorelines=pen, resolution=res, area_thresh=10000)

    def fill_continents(self, color="lightgray"):
        """Fill land colors."""
        self.fig.coast(land=color)

    def fill_oceans(self, color="azure"):
        """Fill water colors."""
        self.fig.coast(water=color)

    def plot(self, x, y, color="blue", marker="o", markersize=10, label=None, **kwargs):
        """Plot points (Matplotlib-like API)."""
        gmt_marker = MARKERS.get(marker, "c")
        style = f"{gmt_marker}{markersize}p"

        self.fig.plot(
            x=x, y=y, style=style, fill=color, pen="0.5p,black", label=label, **kwargs
        )

    def text(self, x, y, text, **kwargs):
        """Add text at (x, y)."""
        # PyGMT text expects justify, font, etc.
        self.fig.text(x=x, y=y, text=text, **kwargs)

    def plot_detector(self, name, label=True, label_offset=None, **kwargs):
        """Plot a gravitational wave detector by name (e.g., 'K1').

        Parameters
        ----------
        name : str
            Name of the detector.
        label : bool, optional
            If True, plot a label above the marker.
        label_offset : float, optional
            Manual latitude offset for the label. If None, it's calculated dynamically.
        **kwargs
            Additional arguments for GeoMap.plot (e.g., color, markersize).

        """
        if name not in DETECTORS:
            raise ValueError(f"Unknown detector: {name}")

        det = DETECTORS[name]

        plot_kwargs = {"marker": "*", "markersize": 15, "color": det["color"]}
        plot_kwargs.update(kwargs)

        self.plot(det["lon"], det["lat"], **plot_kwargs)

        if label:
            if label_offset is not None:
                offset = label_offset
            else:
                # Calculate dynamic offset based on the latitude range
                if isinstance(self.region, (list, tuple, np.ndarray)):
                    lat_min, lat_max = self.region[2], self.region[3]
                    lat_span = lat_max - lat_min
                elif self.region == "d":
                    lat_span = 180.0
                else:
                    # If region is a string (e.g., 'JP'), it's likely a regional map.
                    # We assume a standard regional span (e.g., 30 degrees) for the heuristic.
                    lat_span = 30.0

                offset = max(
                    0.2, lat_span * 0.03
                )  # 3% of the span, minimum 0.2 degrees

            self.text(
                det["lon"],
                det["lat"] + offset,
                text=det["name"],
                font="10p,Helvetica-Bold,black",
                justify="CM",
            )

    def add_scale_bar(
        self, width="500k", position="jBL", offset="0.5c/0.5c", fancy=True
    ):
        """Add a scale bar to the map.

        Parameters
        ----------
        width : str, optional
            Width of the scale bar (e.g., '500k' for 500 km, '100k' for 100 km).
        position : str, optional
            Position anchor (e.g., 'jBL' for Bottom Left).
        offset : str, optional
            Offset from the anchor (e.g., '0.5c/0.5c').
        fancy : bool, optional
            If True, use a fancy scale bar.

        """
        spec = f"{position}+w{width}+o{offset}"
        if fancy:
            spec += "+f"
        self.fig.basemap(map_scale=spec)

    def add_relief(
        self,
        resolution: str = "01s",
        *,
        shading: bool | str = True,
        cmap: str = "geo",
        **kwargs,
    ):
        """Render GMT remote ``earth_relief`` topography via ``grdimage``.

        Parameters
        ----------
        resolution : str, optional
            GMT earth_relief resolution code (e.g., ``'01s'``, ``'03s'``,
            ``'15s'``, ``'01m'``). Requires network access for remote download.
        shading : bool or str, optional
            Illumination shading passed to ``fig.grdimage`` (default ``True``).
        cmap : str, optional
            GMT colormap name (default ``'geo'``).
        **kwargs
            Additional arguments forwarded to ``fig.grdimage``.

        """
        grid = f"@earth_relief_{resolution}"
        grd_kwargs = dict(kwargs)
        grd_kwargs.setdefault("region", self.region)
        if shading is not None:
            grd_kwargs.setdefault("shading", shading)
        if cmap is not None:
            grd_kwargs.setdefault("cmap", cmap)
        self.fig.grdimage(grid=grid, **grd_kwargs)

    def add_field(
        self,
        field,
        *,
        cmap: str | None = None,
        alpha: float | None = None,
        shading: bool | str | None = None,
        **kwargs,
    ):
        """Render an angle-domain :class:`~gwexpy.fields.ScalarField` via ``grdimage``.

        Parameters
        ----------
        field : ScalarField
            2D (or singleton-padded) geographic field with ``angle`` spatial domain.
        cmap : str, optional
            GMT colormap name.
        alpha : float, optional
            Opacity in ``[0, 1]`` (mapped to PyGMT ``transparency``).
        shading : bool or str, optional
            Optional shading parameter passed to ``fig.grdimage``.
        **kwargs
            Additional arguments forwarded to ``fig.grdimage``.

        """
        grid = _field_to_grid(field)
        grd_kwargs = dict(kwargs)
        if cmap is not None:
            grd_kwargs.setdefault("cmap", cmap)
        if shading is not None:
            grd_kwargs.setdefault("shading", shading)
        if alpha is not None:
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            grd_kwargs.setdefault("transparency", (1.0 - alpha_val) * 100.0)
        self.fig.grdimage(grid=grid, **grd_kwargs)

    def add_colorbar(self, label: str | None = None, **kwargs):
        """Add a GMT colorbar to the figure.

        Parameters
        ----------
        label : str, optional
            Colorbar axis label.
        **kwargs
            Additional arguments forwarded to ``fig.colorbar``.

        """
        cb_kwargs = dict(kwargs)
        if label is not None and "frame" not in cb_kwargs:
            cb_kwargs["frame"] = f"af+l{label}"
        self.fig.colorbar(**cb_kwargs)

    def show(self):
        """Display the plot."""
        self.fig.show()

    def save(self, filename, **kwargs):
        """Save the plot."""
        self.fig.savefig(filename, **kwargs)
