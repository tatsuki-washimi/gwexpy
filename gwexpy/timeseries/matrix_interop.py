from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .matrix import TimeSeriesMatrix


class TimeSeriesMatrixInteropMixin:
    """Interoperability methods for TimeSeriesMatrix (Neo, MNE, etc.)."""

    def to_neo(self, units: Any = None) -> Any:
        """
        Convert to neo.AnalogSignal.

        Returns
        -------
        neo.core.AnalogSignal
        """
        from gwexpy.interop import to_neo

        return to_neo(self, units=units)

    @classmethod
    def from_neo(cls, sig: Any) -> Any:
        """
        Create TimeSeriesMatrix from neo.AnalogSignal.

        Parameters
        ----------
        sig : neo.core.AnalogSignal
            Input signal.

        Returns
        -------
        TimeSeriesMatrix
        """
        from gwexpy.interop import from_neo

        return from_neo(cls, sig)

    def to_mne(self, info: Any = None) -> Any:
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray

        self_typed = cast("TimeSeriesMatrix", self)
        tsd = self_typed.to_dict_flat()
        if not isinstance(tsd, self_typed.dict_class):
            tsd = self_typed.dict_class(tsd)
        return to_mne_rawarray(tsd, info=info)
