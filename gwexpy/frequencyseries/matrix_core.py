from __future__ import annotations

from typing import Any


class FrequencySeriesMatrixCoreMixin:
    """Core properties and basic operations for FrequencySeriesMatrix."""

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def df(self) -> Any:
        """Frequency spacing (dx)."""
        return self.dx

    @property
    def f0(self) -> Any:
        """Starting frequency (x0)."""
        return self.x0

    @property
    def frequencies(self) -> Any:
        """Frequency array (xindex)."""
        return self.xindex

    @frequencies.setter
    def frequencies(self, value: Any) -> None:
        self.xindex = value

    def _repr_string_(self) -> str:
        if self.size > 0:
            u_meta = self.meta[0, 0].unit
        else:
            u_meta = None
        return (
            f"<FrequencySeriesMatrix shape={self.shape}, df={self.df}, unit={u_meta}>"
        )

    def _get_series_kwargs(self, xindex, meta):
        return {
            "frequencies": xindex,
            "unit": meta.unit,
            "name": meta.name,
            "channel": meta.channel,
            "epoch": getattr(self, "epoch", None),
        }

    def _get_meta_for_constructor(self, data, xindex):
        """Arguments to construct a FrequencySeriesMatrix."""
        return {
            "data": data,
            "frequencies": xindex,
            "rows": getattr(self, "rows", None),
            "cols": getattr(self, "cols", None),
            "meta": getattr(self, "meta", None),
            "name": getattr(self, "name", None),
            "epoch": getattr(self, "epoch", None),
            "unit": getattr(self, "unit", None),
        }
