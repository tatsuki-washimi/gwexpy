from __future__ import annotations

from gwpy.types.series import Series as GwpySeries

from ._stats import StatisticalMethodsMixin
from ._uncertainty import UncertaintyMixin

__all__ = ["Series"]


class Series(UncertaintyMixin, StatisticalMethodsMixin, GwpySeries):
    """1D series with unified statistical methods and optional uncertainty metadata."""

    pass


# Dynamic import from gwpy (to keep other symbols if any)
# But usually series.py just has Series and some helpers.
