from __future__ import annotations

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy. Install with: pip install scipy"
    ) from _exc


_NO_VALUE = getattr(np, "_NoValue")


class StatisticalMethodsMixin:
    """Mixin class providing statistical methods with ignore_nan support.

    This mixin works for both 1D (TimeSeries) and N-D (Matrix) data.
    For matrices, use axis parameter to specify the reduction axis.
    """

    def _apply_stat_func(
        self,
        func_nan,
        func_raw,
        ignore_nan,
        *,
        result_unit_power=1,
        **kwargs,
    ):
        # Extract data and unit
        data = np.asarray(self)
        unit = getattr(self, "unit", None)

        func = func_nan if ignore_nan else func_raw

        # Pull out arguments that numpy functions expect
        # This is a bit generic but works for mean, std, var, min, max, median
        res = func(data, **kwargs)

        if unit is not None:
            from astropy.units import Quantity

            unit = unit**result_unit_power
            return Quantity(res, unit=unit)
        return res

    def _call_parent_stat(self, method, *args, **kwargs):
        parent = super()
        return getattr(parent, method)(*args, **kwargs)

    def _has_gwpy_quantity_parent(self):
        from astropy.units import Quantity

        return isinstance(self, Quantity)

    def mean(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        *,
        where=True,
        ignore_nan=False,
    ):
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat(
                "mean",
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                where=where,
            )
        return self._apply_stat_func(
            np.nanmean,
            np.mean,
            ignore_nan,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=True,
        ignore_nan=False,
    ):
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat(
                "std",
                axis=axis,
                dtype=dtype,
                out=out,
                ddof=ddof,
                keepdims=keepdims,
                where=where,
            )
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
        )

    def var(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=True,
        ignore_nan=False,
    ):
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat(
                "var",
                axis=axis,
                dtype=dtype,
                out=out,
                ddof=ddof,
                keepdims=keepdims,
                where=where,
            )
        return self._apply_stat_func(
            np.nanvar,
            np.var,
            ignore_nan,
            result_unit_power=2,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def min(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=_NO_VALUE,
        where=_NO_VALUE,
        *,
        ignore_nan=False,
    ):
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims}
        if initial is not _NO_VALUE:
            kwargs["initial"] = initial
        if where is not _NO_VALUE:
            kwargs["where"] = where
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat("min", **kwargs)
        return self._apply_stat_func(
            np.nanmin,
            np.min,
            ignore_nan,
            **kwargs,
        )

    def max(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=_NO_VALUE,
        where=_NO_VALUE,
        *,
        ignore_nan=False,
    ):
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims}
        if initial is not _NO_VALUE:
            kwargs["initial"] = initial
        if where is not _NO_VALUE:
            kwargs["where"] = where
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat("max", **kwargs)
        return self._apply_stat_func(
            np.nanmax,
            np.max,
            ignore_nan,
            **kwargs,
        )

    def median(
        self,
        axis=None,
        **kwargs,
    ):
        """Compute the median.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to compute the median. If None, compute over the
            flattened array.
        ignore_nan : bool, optional
            If True, use ``numpy.nanmedian`` and ignore NaNs. The default is
            False, matching GWpy and NumPy NaN propagation.
        **kwargs
            Passed to the GWpy implementation, or to ``numpy.nanmedian`` when
            ``ignore_nan=True``.

        Returns
        -------
        Any
            The median value(s). If the object carries a unit, the result is
            returned with the same unit where applicable.

        """
        ignore_nan = kwargs.pop("ignore_nan", False)
        if not ignore_nan and self._has_gwpy_quantity_parent():
            return self._call_parent_stat("median", axis=axis, **kwargs)
        return self._apply_stat_func(
            np.nanmedian,
            np.median,
            ignore_nan,
            axis=axis,
            **kwargs,
        )

    def rms(self, axis=None, keepdims=False, ignore_nan=True):
        """Compute the Root Mean Square (RMS) value.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to compute RMS. If None, compute over the flattened array.
        keepdims : bool, optional
            If True, the reduced axes are left in the result as dimensions with size one.
        ignore_nan : bool, optional
            If True (default), NaNs are ignored during computation.

        Returns
        -------
        Quantity or float
            The RMS value(s). Returns `~astropy.units.Quantity` if the object has a unit.

        """
        func = np.nanmean if ignore_nan else np.mean
        data = np.asarray(self)
        val = np.sqrt(func(np.square(data), axis=axis, keepdims=keepdims))
        unit = getattr(self, "unit", None)
        if unit is not None:
            from astropy.units import Quantity

            return Quantity(val, unit=unit)
        return val

    def skewness(self, axis=None, nan_policy="propagate"):
        """Compute the skewness of the data.

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
        """Compute the kurtosis (Fisher or Pearson) of the data.

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
