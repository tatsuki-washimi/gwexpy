
import numpy as np

class StatisticalMethodsMixin:
    """
    Mixin class providing statistical methods with ignore_nan support.
    """

    def _apply_stat_func(self, func_nan, func_raw, ignore_nan, **kwargs):
        # Extract data and unit
        data = np.asarray(self)
        unit = getattr(self, "unit", None)
        
        func = func_nan if ignore_nan else func_raw
        
        # Pull out arguments that numpy functions expect
        # This is a bit generic but works for mean, std, var, min, max, median
        res = func(data, **kwargs)
        
        if unit is not None:
            from astropy.units import Quantity
            return Quantity(res, unit=unit)
        return res

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, ignore_nan=True):
        return self._apply_stat_func(np.nanmean, np.mean, ignore_nan, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True):
        return self._apply_stat_func(np.nanstd, np.std, ignore_nan, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True):
        return self._apply_stat_func(np.nanvar, np.var, ignore_nan, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True):
        return self._apply_stat_func(np.nanmin, np.min, ignore_nan, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True):
        return self._apply_stat_func(np.nanmax, np.max, ignore_nan, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def median(self, axis=None, out=None, overwrite_input=False, keepdims=False, ignore_nan=True):
        # overwrite_input is only in median/nanmedian
        kwargs = dict(axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)
        return self._apply_stat_func(np.nanmedian, np.median, ignore_nan, **kwargs)

    def rms(self, axis=None, keepdims=False, ignore_nan=True):
        func = np.nanmean if ignore_nan else np.mean
        data = np.asarray(self)
        val = np.sqrt(func(np.square(data), axis=axis, keepdims=keepdims))
        unit = getattr(self, "unit", None)
        if unit is not None:
             from astropy.units import Quantity
             return Quantity(val, unit=unit)
        return val
