"""Shared gravitational-wave detector coordinate registry for plotting."""

from __future__ import annotations

from typing import Any

__all__ = ["DETECTORS"]

# Detector coordinate database (longitude/latitude in WGS84 degrees)
DETECTORS: dict[str, dict[str, Any]] = {
    "K1": {"lon": 137.31, "lat": 36.41, "name": "KAGRA", "color": "darkgreen"},
    "H1": {"lon": -119.41, "lat": 46.45, "name": "LIGO Hanford", "color": "darkred"},
    "L1": {"lon": -90.77, "lat": 30.56, "name": "LIGO Livingston", "color": "darkblue"},
    "V1": {"lon": 10.5, "lat": 43.63, "name": "Virgo", "color": "darkorange"},
    "G1": {"lon": 9.80, "lat": 52.24, "name": "GEO600", "color": "darkmagenta"},
}
