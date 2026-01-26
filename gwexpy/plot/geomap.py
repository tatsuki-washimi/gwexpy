"""Geographic map plotting using PyGMT backend with Cartopy-like interface."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

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

# 検出器座標データベース (簡易版)
DETECTORS = {
    "K1": {"lon": 137.31, "lat": 36.41, "name": "KAGRA", "color": "darkgreen"},
    "H1": {"lon": -119.41, "lat": 46.45, "name": "LIGO Hanford", "color": "darkred"},
    "L1": {"lon": -90.77, "lat": 30.56, "name": "LIGO Livingston", "color": "darkblue"},
    "V1": {"lon": 10.5, "lat": 43.63, "name": "Virgo", "color": "darkorange"},
    "G1": {"lon": 9.80, "lat": 52.24, "name": "GEO600", "color": "darkmagenta"},
}

# 投影法のマッピング (Friendly Name -> GMT Code)
PROJECTIONS = {
    "Mercator": "M",
    "Robinson": "N",
    "Mollweide": "W",
    "PlateCarree": "Q",  # Cylindrical Equidistant
    "Equidistant": "Q",
}

# マーカーのマッピング (Matplotlib -> GMT)
MARKERS = {
    "o": "c",  # circle
    "s": "s",  # square
    "^": "t",  # triangle
    "d": "d",  # diamond
    "+": "+",  # plus
    "x": "x",  # cross
    "*": "a",  # star
}


class GeoMap:
    """A PyGMT wrapper that provides a Cartopy-like interface."""

    def __init__(self, projection="Robinson", center_lon=0, width="15c", **kwargs):
        """
        Initialize a new GeoMap.

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

        # 投影法の設定
        proj_code = PROJECTIONS.get(projection, projection)

        # GMTの投影指定を作成
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

        # ベースマップの初期化
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

    def show(self):
        """Display the plot."""
        self.fig.show()

    def save(self, filename, **kwargs):
        """Save the plot."""
        self.fig.savefig(filename, **kwargs)
