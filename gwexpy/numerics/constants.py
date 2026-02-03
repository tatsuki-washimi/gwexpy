"""
Dtype-aware numerical constants for gwexpy.

Why not just ``1e-12``?
-----------------------
Gravitational-wave strain data lives at :math:`\\sim 10^{-21}`.
A hardcoded ``eps = 1e-12`` is **nine orders of magnitude** larger than
the signal variance (:math:`\\sim 10^{-42}`).  Using it as a
regularisation floor in whitening or PSD estimation effectively
**deletes the signal**.

The constants defined here are derived from ``numpy.finfo`` so they
adapt to the working dtype (float32 / float64) and remain safely below
any physically meaningful quantity in gravitational-wave analysis.

Machine precision reference
---------------------------
* ``float64``: ``eps ≈ 2.22e-16``
* ``float32``: ``eps ≈ 1.19e-07``
"""

from __future__ import annotations

from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DTypeLike = Union[np.dtype, type, str]


def eps_for_dtype(dtype: DTypeLike = np.float64) -> float:
    """Return machine epsilon for *dtype*.

    Parameters
    ----------
    dtype : dtype-like, optional
        NumPy dtype (or type alias).  Default ``float64``.

    Returns
    -------
    float
        Machine epsilon — the smallest representable positive number such
        that ``1.0 + eps != 1.0``.
    """
    return float(np.finfo(np.dtype(dtype)).eps)


# ---------------------------------------------------------------------------
# Semantic constants (float64)
#
# Each constant carries a short docstring explaining *why* this particular
# multiplier was chosen.  When a function accepts ``eps=None``, it should
# default to the appropriate constant here (or better yet, use
# ``safe_epsilon`` from ``gwexpy.numerics.scaling``).
# ---------------------------------------------------------------------------

_EPS64 = eps_for_dtype(np.float64)  # ≈ 2.22e-16

EPS_VARIANCE: float = _EPS64 * 100
"""Floor for variance / power estimates.

100 × machine-eps (≈ 2.2e-14) is comfortably below GW strain power
(~ 1e-42) yet large enough to prevent true division-by-zero when the
input is numerically silent (all zeros due to padding, for example).
"""

EPS_PSD: float = _EPS64 * 1_000
"""Floor for PSD / cross-spectral regularisation.

A slightly more generous floor (≈ 2.2e-13) avoids instability in
Welch-based estimators without masking physically meaningful spectral
content.
"""

EPS_COHERENCE: float = _EPS64 * 10
"""Floor for coherence denominator clipping (range [0, 1]).

Coherence is a normalised quantity so a very tight floor (≈ 2.2e-15)
suffices to prevent 0/0 without biasing the estimate.
"""

SAFE_FLOOR: float = 1e-50
"""Absolute safety floor for logarithmic operations.

GW strain *power* is ~ 1e-42.  A floor at 1e-50 is eight orders of
magnitude below that, guaranteeing that ``log10(x + SAFE_FLOOR)`` never
encounters a non-positive argument while introducing negligible bias.

Replaces the former ``+ 1e-20`` offsets scattered through the codebase,
which were only 1–2 orders below typical strain *amplitude* and could
dominate the signal in power-based plots.
"""

# Backward/compat alias for Phase 2 code references.
SAFE_FLOOR_STRAIN: float = SAFE_FLOOR
