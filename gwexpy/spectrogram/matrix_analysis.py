from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


class SpectrogramMatrixAnalysisMixin:
    """Analysis methods for SpectrogramMatrix (Phase calculations)."""

    def radian(self, unwrap: bool = False) -> Any:
        """
        Calculate the phase of the matrix in radians.
        """
        val = np.angle(self)
        if unwrap:
            # Unwrap along time axis (usually axis -2 in SpectrogramMatrix: (..., Time, Freq))
            val = np.unwrap(val, axis=-2)

        new_meta = self.meta.copy() if self.meta is not None else None
        if new_meta is not None:
             for m in new_meta.flat:
                  m.unit = u.rad
                  if m.name:
                       m.name += "_phase"

        # Use constructor of the class (SpectrogramMatrix)
        new_mat = self.__class__(
            val,
            times=self.times,
            frequencies=self.frequencies,
            unit=u.rad,
            name=self.name + "_phase" if self.name else "phase",
            meta=new_meta
        )
        return new_mat

    def degree(self, unwrap: bool = False) -> Any:
        """
        Calculate the phase of the matrix in degrees.
        """
        p = self.radian(unwrap=unwrap)
        val = np.rad2deg(np.asarray(p))

        new_meta = p.meta.copy() if p.meta is not None else None
        if new_meta is not None:
             for m in new_meta.flat:
                  m.unit = u.deg
                  if m.name and "_phase" in m.name:
                       m.name += "_deg"

        new_mat = self.__class__(
            val,
            times=p.times,
            frequencies=p.frequencies,
            unit=u.deg,
            name=p.name + "_deg" if p.name else "phase_deg",
            meta=new_meta
        )
        return new_mat
