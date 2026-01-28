from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


class StatisticalMethodsMixin:
    """
    Mixin class providing statistical methods with ignore_nan support.

    This mixin works for both 1D (TimeSeries) and N-D (Matrix) data.
    For matrices, use axis parameter to specify the reduction axis.
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

    def mean(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        where=True,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        return self._apply_stat_func(
            np.nanmean,
            np.mean,
            ignore_nan,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
            **kwargs,
        )

    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        where=True,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        return self._apply_stat_func(
            np.nanstd,
            np.std,
            ignore_nan,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            **kwargs,
        )

    def var(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        where=True,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        return self._apply_stat_func(
            np.nanvar,
            np.var,
            ignore_nan,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            **kwargs,
        )

    def min(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        return self._apply_stat_func(
            np.nanmin,
            np.min,
            ignore_nan,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            **kwargs,
        )

    def max(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        return self._apply_stat_func(
            np.nanmax,
            np.max,
            ignore_nan,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            **kwargs,
        )

    def median(
        self,
        axis=None,
        out=None,
        overwrite_input=False,
        keepdims=False,
        **kwargs,
    ):
        ignore_nan = kwargs.pop("ignore_nan", True)
        # overwrite_input is only in median/nanmedian
        base_kwargs = dict(
            axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims
        )
        base_kwargs.update(kwargs)
        return self._apply_stat_func(np.nanmedian, np.median, ignore_nan, **base_kwargs)

    def rms(self, axis=None, keepdims=False, ignore_nan=True):
        """Compute root-mean-square."""
        func = np.nanmean if ignore_nan else np.mean
        data = np.asarray(self)
        val = np.sqrt(func(np.square(data), axis=axis, keepdims=keepdims))
        unit = getattr(self, "unit", None)
        if unit is not None:
            from astropy.units import Quantity

            return Quantity(val, unit=unit)
        return val

    def skewness(self, axis=None, nan_policy="propagate"):
        """
        Compute the skewness of the data.

        Skewness is a measure of the asymmetry of the probability distribution
        of a real-valued random variable about its mean.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to compute skewness. If None, compute over the
            flattened array.
        nan_policy : str, optional
            How to handle NaNs: 'propagate', 'raise', or 'omit'.

        Returns
        -------
        float or ndarray
            The skewness value(s).
        """
        data = np.asarray(self)
        return scipy_stats.skew(data, axis=axis, nan_policy=nan_policy)

    def kurtosis(self, axis=None, fisher=True, nan_policy="propagate"):
        """
        Compute the kurtosis (Fisher or Pearson) of the data.

        Kurtosis is a measure of the "tailedness" of the probability distribution.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to compute kurtosis. If None, compute over the
            flattened array.
        fisher : bool, optional
            If True, Fisher's definition is used (normal ==> 0.0).
            If False, Pearson's definition is used (normal ==> 3.0).
        nan_policy : str, optional
            How to handle NaNs: 'propagate', 'raise', or 'omit'.

        Returns
        -------
        float or ndarray
            The kurtosis value(s).
        """
        data = np.asarray(self)
        return scipy_stats.kurtosis(
            data, axis=axis, fisher=fisher, nan_policy=nan_policy
        )
