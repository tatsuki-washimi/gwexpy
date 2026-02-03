"""
gwexpy.numerics — Single source of truth for numerical stability.

This module centralises all numerical constants and scaling utilities
used throughout gwexpy, replacing ad-hoc "Death Floats" (``1e-12``,
``1e-20``, etc.) with values derived from machine precision or the
physical scale of the data.

Submodules
----------
constants
    Dtype-aware epsilon values for variance, PSD, coherence, etc.
scaling
    :class:`AutoScaler` context manager for safe internal normalisation.
"""

from __future__ import annotations

from .constants import (
    EPS_COHERENCE,
    EPS_PSD,
    EPS_VARIANCE,
    SAFE_FLOOR,
    SAFE_FLOOR_STRAIN,
    eps_for_dtype,
)
from .scaling import AutoScaler, get_safe_epsilon, safe_epsilon, safe_log_scale

__all__ = [
    "EPS_VARIANCE",
    "EPS_PSD",
    "EPS_COHERENCE",
    "SAFE_FLOOR",
    "SAFE_FLOOR_STRAIN",
    "eps_for_dtype",
    "AutoScaler",
    "get_safe_epsilon",
    "safe_epsilon",
    "safe_log_scale",
]
