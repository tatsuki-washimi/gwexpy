from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from astropy import units as u


@runtime_checkable
class HistogramProtocol(Protocol):
    """Protocol for objects behaving like a Histogram."""

    @property
    def values(self) -> u.Quantity: ...

    @property
    def edges(self) -> u.Quantity: ...

    @property
    def unit(self) -> u.UnitBase: ...

    @property
    def xunit(self) -> u.UnitBase: ...

    @property
    def nbins(self) -> int: ...

    @property
    def name(self) -> str | None: ...

    @property
    def channel(self) -> Any: ...

    @property
    def cov(self) -> u.Quantity | None: ...

    @property
    def sumw2(self) -> u.Quantity | None: ...

    def __call__(self, **kwargs: Any) -> Any: ...
