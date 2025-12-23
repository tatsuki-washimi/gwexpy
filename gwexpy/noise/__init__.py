from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gwinc_ import from_pygwinc
    from .obspy_ import from_obspy
    from .colored import power_law, white_noise, pink_noise, red_noise
    from .magnetic import schumann_resonance, geomagnetic_background
    from .peaks import lorentzian_line, gaussian_line, voigt_line

__all__ = [
    "from_pygwinc",
    "from_obspy",
    "power_law",
    "white_noise",
    "pink_noise",
    "red_noise",
    "schumann_resonance",
    "geomagnetic_background",
    "lorentzian_line",
    "gaussian_line",
    "voigt_line",
]


def __getattr__(name: str) -> Any:
    if name == "from_pygwinc":
        from .gwinc_ import from_pygwinc
        return from_pygwinc
    if name == "from_obspy":
        from .obspy_ import from_obspy
        return from_obspy
    if name in ("power_law", "white_noise", "pink_noise", "red_noise"):
        from .colored import power_law, white_noise, pink_noise, red_noise
        if name == "power_law":
            return power_law
        if name == "white_noise":
            return white_noise
        if name == "pink_noise":
            return pink_noise
        if name == "red_noise":
            return red_noise
    if name in ("schumann_resonance", "geomagnetic_background"):
        from .magnetic import schumann_resonance, geomagnetic_background
        if name == "schumann_resonance":
            return schumann_resonance
        if name == "geomagnetic_background":
            return geomagnetic_background
    if name in ("lorentzian_line", "gaussian_line", "voigt_line"):
        from .peaks import lorentzian_line, gaussian_line, voigt_line
        if name == "lorentzian_line":
            return lorentzian_line
        if name == "gaussian_line":
            return gaussian_line
        if name == "voigt_line":
            return voigt_line

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
