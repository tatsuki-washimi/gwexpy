from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from gwexpy.types.metadata import MetaDataMatrix


class _SpectrogramMatrixLike(Protocol):
    """Protocol defining the interface expected by SpectrogramMatrixAnalysisMixin."""

    meta: MetaDataMatrix | None
    name: str | None
    unit: u.Unit | None
    times: Any
    frequencies: Any

    def copy(self) -> _SpectrogramMatrixLike: ...
    def view(self, dtype: type) -> np.ndarray: ...
    @property
    def real(self) -> _SpectrogramMatrixLike: ...
    def radian(self, unwrap: bool = False) -> Any: ...


class SpectrogramMatrixAnalysisMixin:
    """Analysis methods for SpectrogramMatrix (Phase calculations)."""

    def radian(self: _SpectrogramMatrixLike, unwrap: bool = False) -> Any:
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
        if np.iscomplexobj(new.view(np.ndarray)):
            # new.real returns the real-valued view for SeriesMatrix types
            new = new.real

        # Restore axis metadata in case copy/real dropped it
        new.times = self.times
        new.frequencies = self.frequencies

        # Update values
        new.view(np.ndarray)[:] = val

        # Update metadata units/names
        if new.meta is not None:
            # metadata.copy() copies the array and provides new MetaData objects
            # Use temporary variable to satisfy type checker
            temp_meta = new.meta.copy()
            for m in temp_meta.flat:
                # We already have new MetaData objects from meta.copy()
                m.unit = u.rad
                if m.name:
                    if ".real" in m.name:
                        m.name = m.name.replace(".real", "_phase")
                    else:
                        m.name += "_phase"
                else:
                    m.name = "phase"
            new.meta = temp_meta

        # Update global name and unit
        if self.name:
            new.name = self.name + "_phase"
        else:
            new.name = "phase"

        new.unit = u.rad

        return new

    def degree(self: _SpectrogramMatrixLike, unwrap: bool = False) -> Any:
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
