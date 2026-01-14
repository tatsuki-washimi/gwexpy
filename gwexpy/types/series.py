from gwpy.types.series import Series as GwpySeries

from ._stats import StatisticalMethodsMixin

__all__ = ["Series"]


class Series(StatisticalMethodsMixin, GwpySeries):
    """
    1D Series with unified statistical methods.
    """

    pass


# Dynamic import from gwpy (to keep other symbols if any)
# But usually series.py just has Series and some helpers.
