"""gwexpy.statistics - Statistical analysis tools for gravitational wave data."""

from __future__ import annotations

from .gauch import compute_gauch, GauChResult
from .rayleigh_test import rayleigh_pvalue
from .student_t_indicator import compute_student_t_nu
from .dq_flag import to_segments

__all__ = [
    "compute_gauch",
    "GauChResult",
    "rayleigh_pvalue",
    "compute_student_t_nu",
    "to_segments",
]
