"""
gwexpy.types.mixin
------------------

Common mixins for gwexpy types.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RegularityMixin:
    """Mixin to provide regularity checking for series types."""

    @property
    def is_regular(self) -> bool:
        """Return True if this series has a regular grid (constant spacing)."""
        # 1. Check for 'dt' attribute (standard for regular TimeSeries/FrequencySeries)
        dt = getattr(self, "dt", None)
        if dt is not None:
             try:
                 # If dt is a Quantity with value > 0, it's regular by definition in GWpy
                 return float(dt) > 0
             except (TypeError, ValueError):
                 pass

        # 2. Check underlying index accurately
        try:
            idx = getattr(self, "xindex", None)
            if idx is None:
                return True
            
            # Use GWpy's own assessment if available and looks healthy
            if hasattr(idx, "regular") and idx.regular is True:
                return True

            # Manual fallback check with very relaxed tolerances
            vals = np.asarray(idx)
            if len(vals) < 2:
                return True
            diffs = np.diff(vals)
            # Extremely relaxed to handle GBD/simulated data noise
            return np.allclose(diffs, diffs[0], atol=1e-4, rtol=1e-3)
        except (AttributeError, ValueError, TypeError):
            return False

    def _check_regular(self, method_name: str | None = None):
        """Helper to ensure the series is regular before applying certain transforms."""
        if not self.is_regular:
            method = method_name or "This method"
            # Try to identify if it is Time or Frequency via class name or properties
            cls_name = self.__class__.__name__
            if "Time" in cls_name:
                extra = "Consider using .asfreq() or .interpolate() to regularized the series first."
                msg = f"{method} requires a regular sample rate (constant dt). {extra}"
            elif "Frequency" in cls_name or "Spectrogram" in cls_name:
                msg = f"{method} requires a regular frequency grid."
            else:
                msg = f"{method} requires a regular grid (constant spacing)."

            raise ValueError(msg)


class PhaseMethodsMixin:
    """Mixin to provide unified phase and angle methods."""

    def phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any:
        """
        Calculate the phase of the data.

        Parameters
        ----------
        unwrap : `bool`, optional
            If `True`, unwrap the phase to remove discontinuities.
            Default is `False`.
        deg : `bool`, optional
            If `True`, return the phase in degrees.
            Default is `False` (radians).
        **kwargs
            Additional arguments passed to the underlying calculation.

        Returns
        -------
        `Series` or `Matrix` or `Collection`
            The phase of the data.
        """
        if deg:
            return self.degree(unwrap=unwrap, **kwargs)  # type: ignore[attr-defined]
        return self.radian(unwrap=unwrap, **kwargs)  # type: ignore[attr-defined]

    def angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any:
        """Alias for `phase(unwrap=unwrap, deg=deg)`."""
        return self.phase(unwrap=unwrap, deg=deg, **kwargs)
