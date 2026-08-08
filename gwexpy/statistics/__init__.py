"""gwexpy.statistics - Statistical analysis tools for gravitational wave data."""

from __future__ import annotations

from .dq_flag import to_segments
from .gauch import GauChResult, compute_gauch
from .rayleigh_test import rayleigh_pvalue
from .student_t_indicator import compute_student_t_nu

__all__ = [
    "compute_gauch",
    "GauChResult",
    "rayleigh_pvalue",
    "compute_student_t_nu",
    "to_segments",
]
