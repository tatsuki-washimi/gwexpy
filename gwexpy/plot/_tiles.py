"""Lightweight XYZ slippy-map tile utilities (urllib + PIL, zero extra deps)."""

from __future__ import annotations

import hashlib
import math
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from gwexpy._version import __version__

__all__ = [
    "PROVIDERS",
    "auto_zoom",
    "deg2tile",
    "fetch_tile",
    "tile_extent_deg",
]

_WEBMERCATOR_MAX_LAT = 85.05112877980659

PROVIDERS: dict[str, dict[str, Any]] = {
    "gsi-std": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        "attribution": "Source: GSI (cyberjapandata.gsi.go.jp)",
        "max_zoom": 18,
    },
    "gsi-pale": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        "attribution": "Source: GSI (cyberjapandata.gsi.go.jp)",
        "max_zoom": 18,
    },
    "gsi-photo": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
        "attribution": "Source: GSI (cyberjapandata.gsi.go.jp)",
        "max_zoom": 18,
    },
    "gsi-relief": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/relief/{z}/{x}/{y}.png",
        "attribution": "Source: GSI (cyberjapandata.gsi.go.jp)",
        "max_zoom": 15,
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "(c) OpenStreetMap contributors",
        "max_zoom": 19,
    },
}


def deg2tile(lon_deg: float, lat_deg: float, zoom: int) -> tuple[int, int]:
    """Convert geographic ``(lon_deg, lat_deg)`` to XYZ slippy-map ``(x, y)`` indices."""
    if zoom < 0:
        raise ValueError(f"zoom must be non-negative, got {zoom}")
    lat_clamped = max(-_WEBMERCATOR_MAX_LAT, min(_WEBMERCATOR_MAX_LAT, float(lat_deg)))
    lon_wrapped = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    # Preserve +180 boundary instead of wrapping to -180 when lon == 180
    if float(lon_deg) >= 180.0 and lon_wrapped == -180.0:
        lon_wrapped = 180.0

    n = 1 << int(zoom)
    xtile = int(math.floor((lon_wrapped + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat_clamped)
    ytile = int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))
    xtile = max(0, min(n - 1, xtile))
    ytile = max(0, min(n - 1, ytile))
    return (xtile, ytile)


def tile_extent_deg(
    xtile: int, ytile: int, zoom: int
) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lon_max, lat_min, lat_max)`` in degrees for tile ``(x, y, z)``."""
    if zoom < 0:
        raise ValueError(f"zoom must be non-negative, got {zoom}")
    n = 1 << int(zoom)
    lon_min = xtile / n * 360.0 - 180.0
    lon_max = (xtile + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n))))
    lat_min = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ytile + 1) / n)))
    )
    return (lon_min, lon_max, lat_min, lat_max)


def auto_zoom(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    max_tiles: int = 16,
    min_zoom: int = 0,
    max_zoom: int = 18,
) -> int:
    """Choose the highest zoom level whose tile grid stays within ``max_tiles``."""
    if max_tiles < 1:
        raise ValueError(f"max_tiles must be >= 1, got {max_tiles}")
    for z in range(int(max_zoom), int(min_zoom) - 1, -1):
        x0, y1 = deg2tile(lon_min, lat_min, z)
        x1, y0 = deg2tile(lon_max, lat_max, z)
        nx = abs(x1 - x0) + 1
        ny = abs(y1 - y0) + 1
        if nx * ny <= max_tiles:
            return z
    return int(min_zoom)


def fetch_tile(
    url: str,
    *,
    timeout: float = 10.0,
    cache_dir: str | Path | None = None,
    provider: str | None = None,
) -> Image.Image:
    """Fetch a single XYZ tile image over HTTP(S) with optional disk caching.

    This is the sole network entry point for basemap rendering.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(url.split("?")[0]).suffix or ".png"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = cache_root / f"{digest}{suffix}"
        if cache_path.exists():
            with Image.open(cache_path) as cached_im:
                return cached_im.convert("RGBA")

    provider_label = f"provider '{provider}'" if provider else "tile server"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"gwexpy/{__version__} (+https://github.com/tatsuki-washimi/gwexpy)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(
            f"Failed to fetch tile from {provider_label} ({url}); "
            f"network may be offline or unavailable: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Failed to fetch tile from {provider_label} ({url}); "
            f"network may be offline or unavailable: {exc}"
        ) from exc

    if cache_path is not None:
        cache_path.write_bytes(raw)

    with Image.open(BytesIO(raw)) as im:
        return im.convert("RGBA")
