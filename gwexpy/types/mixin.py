"""
gwexpy.types.mixin
------------------

Common mixins for gwexpy types.
"""

from typing import Optional
import numpy as np

class RegularityMixin:
    """Mixin to provide regularity checking for series types."""

    @property
    def is_regular(self) -> bool:
        """Return True if this series has a regular grid (constant spacing)."""
        try:
            # Use underlying index safely to avoid triggering GWpy AttributeErrors on irregular series
            idx = getattr(self, "xindex", None)
            if idx is None:
                return True
            if hasattr(idx, "regular"):
                 return idx.regular

            # Manual check
            vals = np.asarray(idx)
            if len(vals) < 2:
                return True
            diffs = np.diff(vals)
            # Use same tolerances as original implementations
            return np.allclose(diffs, diffs[0], atol=1e-12, rtol=1e-10)
        except (AttributeError, ValueError, TypeError):
            return False

    def _check_regular(self, method_name: Optional[str] = None):
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
