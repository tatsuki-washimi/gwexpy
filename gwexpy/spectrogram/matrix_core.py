from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from gwpy.types.index import Index


class SpectrogramMatrixCoreMixin:
    """
    Core properties and metadata for SpectrogramMatrix.
    Overrides SeriesMatrix defaults to handle 4D (or 3D) Spectrogram data structure.

    Structure
    ---------
    - 3D: (Batch, Time, Freq)
    - 4D: (Row, Col, Time, Freq)

    Axis Invariants
    ---------------
    SpectrogramMatrix assumes a fixed axis layout:

    - **Time axis**: always ``shape[-2]`` (second-to-last dimension)
    - **Frequency axis**: always ``shape[-1]`` (last dimension)

    The ``xindex`` (accessed as ``.times``) must have length equal to ``shape[-2]``,
    and ``frequencies`` must have length equal to ``shape[-1]``.

    Structural Operations (Limitation)
    ----------------------------------
    Because of these fixed axis semantics, **transpose / swapaxes / moveaxis**
    operations that reorder dimensions are **not currently supported**. Applying
    such operations breaks the xindex/frequencies length validation and produces
    semantically invalid results.

    TODO: Future 4D support may introduce axis-swap-aware metadata tracking,
    allowing transpose to update xindex/frequencies accordingly.
    """


    @property
    def _x_axis_index(self) -> int:
        """
        Index of the time axis.
        For Spectrogram (Time, Freq), time is -2.
        """
        return -2

    @property
    def times(self) -> Any:
        """Time array (xindex)."""
        return self.xindex

    @times.setter
    def times(self, value: Any) -> None:
        self.xindex = value

    @property
    def t0(self) -> Any:
        """Start time (x0)."""
        return self.x0

    @property
    def dt(self) -> Any:
        """Time spacing (dx)."""
        return self.dx

    @property
    def frequencies(self) -> Any:
        """Frequency array (yindex)."""
        return getattr(self, "_frequencies", None)

    @frequencies.setter
    def frequencies(self, value: Any) -> None:
        """Set frequency array."""
        if value is None:
            self._frequencies = None
            return

        if isinstance(value, Index):
            fi = value
        elif isinstance(value, u.Quantity):
            fi = value
        else:
            fi = np.asarray(value)

        # Check length against last dimension (frequency axis)
        try:
            n_freqs = self._value.shape[-1]
            suppress = getattr(
                self, "_suppress_xindex_check", False
            )  # reuse suppression flag or add new one?
            if (not suppress) and hasattr(fi, "__len__") and len(fi) != n_freqs:
                raise ValueError(
                    f"frequencies length mismatch: expected {n_freqs}, got {len(fi)}"
                )
        except (IndexError, AttributeError):
            pass

        self._frequencies = fi
        # Reset cached freq props if we had any (df, f0)
        for attr in ("_df", "_f0"):
            if hasattr(self, attr):
                delattr(self, attr)

    # Add y-axis specific properties (df, f0) similar to dx/x0
    @property
    def f0(self) -> Any:
        try:
            return self._f0
        except AttributeError:
            if self.frequencies is not None and len(self.frequencies) > 0:
                self._f0 = self.frequencies[0]
            else:
                self._f0 = None  # or 0 Hz
            return self._f0

    @property
    def df(self) -> Any:
        try:
            return self._df
        except AttributeError:
            if self.frequencies is not None and len(self.frequencies) > 1:
                # Assume regular if calculating blindly, or check regularity
                if (
                    hasattr(self.frequencies, "regular")
                    and not self.frequencies.regular
                ):
                    raise AttributeError("Irregular frequencies, df undefined")
                df = self.frequencies[1] - self.frequencies[0]
                if not isinstance(df, u.Quantity):
                    funit = getattr(self.frequencies, "unit", u.Hz)  # default to Hz?
                    df = u.Quantity(df, funit)
                self._df = df
            else:
                self._df = None
            return self._df

    def _get_series_kwargs(self, xindex, meta):
        """Arguments to construct a Spectrogram element."""
        return {
            "times": xindex,
            "frequencies": self.frequencies,
            "unit": meta.unit,
            "name": meta.name,
            "epoch": getattr(self, "epoch", None),
        }

    def _get_meta_for_constructor(self, data, xindex):
        """Arguments to construct a SpectrogramMatrix."""
        return {
            "data": data,
            "times": xindex,  # Map xindex to times
            "frequencies": self.frequencies,
            "rows": getattr(self, "rows", None),
            "cols": getattr(self, "cols", None),
            "meta": getattr(self, "meta", None),
            "name": getattr(self, "name", None),
            "epoch": getattr(self, "epoch", None),
            "unit": getattr(self, "unit", None),
        }
