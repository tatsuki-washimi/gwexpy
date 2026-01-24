from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from astropy import units as u

    from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


class _FrequencySeriesMatrixCoreLike(Protocol):
    """Protocol defining the interface expected by FrequencySeriesMatrixCoreMixin."""

    dx: u.Quantity | float | None
    x0: u.Quantity | float | None
    xindex: Any
    meta: MetaDataMatrix
    shape: tuple[int, ...]
    size: int
    epoch: Any
    rows: MetaDataDict
    cols: MetaDataDict
    name: str | None
    unit: u.Unit | None

    @property
    def df(self) -> Any: ...


class FrequencySeriesMatrixCoreMixin:
    """Core properties and basic operations for FrequencySeriesMatrix."""

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def df(self: _FrequencySeriesMatrixCoreLike) -> Any:
        """Frequency spacing (dx)."""
        return self.dx

    @property
    def f0(self: _FrequencySeriesMatrixCoreLike) -> Any:
        """Starting frequency (x0)."""
        return self.x0

    @property
    def frequencies(self: _FrequencySeriesMatrixCoreLike) -> Any:
        """Frequency array (xindex)."""
        return self.xindex

    @frequencies.setter
    def frequencies(self: _FrequencySeriesMatrixCoreLike, value: Any) -> None:
        self.xindex = value

    def _repr_string_(self: _FrequencySeriesMatrixCoreLike) -> str:
        if self.size > 0:
            u_meta = self.meta[0, 0].unit
        else:
            u_meta = None
        return (
            f"<FrequencySeriesMatrix shape={self.shape}, df={self.df}, unit={u_meta}>"
        )

    def _get_series_kwargs(self: _FrequencySeriesMatrixCoreLike, xindex, meta):
        return {
            "frequencies": xindex,
            "unit": meta.unit,
            "name": meta.name,
            "channel": meta.channel,
            "epoch": getattr(self, "epoch", None),
        }

    def _get_meta_for_constructor(self: _FrequencySeriesMatrixCoreLike, data, xindex):
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
