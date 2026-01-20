from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


class SpectrogramMatrixAnalysisMixin:
    """Analysis methods for SpectrogramMatrix (Phase calculations)."""

    def radian(self, unwrap: bool = False) -> Any:
        """
        Calculate the phase of the matrix in radians.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase along the time axis (axis=-2).

        Returns
        -------
        SpectrogramMatrix
            A new matrix with phase in radians.
        """
        # Copy to preserve all attributes (times, frequencies, rows, cols, epoch, etc.)
        new = self.copy()

        # Calculate phase values
        val = np.angle(self.view(np.ndarray))
        if unwrap:
            # Unwrap along time axis (Time is axis -2 for SpectrogramMatrix)
            val = np.unwrap(val, axis=-2)

        # If original was complex, Ensure new is real-valued to hold phase
        if np.iscomplexobj(new):
            # new.real is a method in SeriesMatrix types
            new = new.real()

        # Update values
        new.view(np.ndarray)[:] = val

        # Update metadata units/names
        if new.meta is not None:
            # metadata.copy() copies the array and provides new MetaData objects
            new.meta = new.meta.copy()
            for m in new.meta.flat:
                # We already have new MetaData objects from meta.copy()
                m.unit = u.rad
                if m.name:
                    if ".real" in m.name:
                         m.name = m.name.replace(".real", "_phase")
                    else:
                         m.name += "_phase"
                else:
                    m.name = "phase"

        # Update global name and unit
        if self.name:
            new.name = self.name + "_phase"
        else:
            new.name = "phase"
        
        new.unit = u.rad

        return new

    def degree(self, unwrap: bool = False) -> Any:
        """
        Calculate the phase of the matrix in degrees.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase along the time axis (axis=-2).

        Returns
        -------
        SpectrogramMatrix
            A new matrix with phase in degrees.
        """
        # Reuse radian implementation which handles unwrap, metadata preservation and real-casting
        p = self.radian(unwrap=unwrap)

        # Convert values to degrees
        val = np.rad2deg(p.view(np.ndarray))

        # Create final object (p already has correct metadata structure)
        new = p
        new.view(np.ndarray)[:] = val

        if new.meta is not None:
            # We already copied meta in radian(), but we need to change unit to deg
            for m in new.meta.flat:
                m.unit = u.deg
                if m.name and "_phase" in m.name:
                    m.name = m.name.replace("_phase", "_phase_deg")

        if self.name:
            new.name = self.name + "_phase_deg"
        else:
            new.name = "phase_deg"
        
        new.unit = u.deg

        return new
