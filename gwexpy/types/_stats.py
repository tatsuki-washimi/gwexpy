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
        unit = getattr(self, "unit", None)
        func = func_nan if ignore_nan else func_raw

        if (
            ignore_nan
            and unit is not None
            and kwargs.get("out") is None
            and self._has_gwpy_quantity_parent()
        ):

            def nan_reduction(*args, **function_kwargs):
                return func_nan(*args, **function_kwargs)

            nan_reduction.__name__ = func_raw.__name__
            wrap_function = getattr(self, "_wrap_function")
            result = wrap_function(
                nan_reduction,
                unit=unit**result_unit_power,
                **kwargs,
            )
            return self._normalize_reduction_result(
                result,
                axis=kwargs.get("axis"),
                keepdims=kwargs.get("keepdims", False),
            )

        # Non-GWpy matrix types and explicit ``out`` extension routes use raw
        # NumPy arrays deliberately.  The Quantity failure preflight below is
        # the sole exception to their pre-existing mutation behavior.
        data = np.asarray(self)

        # A Quantity ``out`` makes NumPy's nan reductions return a
        # dimensionless Quantity because ``data`` is deliberately unitless.
        # Validate the final wrapping conversion before NumPy can overwrite
        # the caller's values and unit on a conversion failure.
        if ignore_nan and unit is not None:
            from astropy.units import Quantity, dimensionless_unscaled

            if isinstance(kwargs.get("out"), Quantity):
                Quantity(0, unit=dimensionless_unscaled).to(unit**result_unit_power)

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
        result = getattr(parent, method)(*args, **kwargs)
        if kwargs.get("out") is not None:
            return result
        return self._normalize_reduction_result(
            result,
            axis=kwargs.get("axis"),
            keepdims=kwargs.get("keepdims", False),
        )

    def _normalize_reduction_result(self, result, *, axis, keepdims):
        """Restore axis invariants after a successful parent reduction."""
        from .array import Array
        from .array2d import Array2D
        from .array3d import Array3D
        from .array4d import Array4D

        if isinstance(self, Array2D):
            return self._normalize_array2d_result(result, axis=axis)

        reduced_axes = self._normalized_reduction_axes(axis)
        if type(self) is Array:
            if keepdims:
                result._axis_names = list(self.axis_names)
            else:
                result._axis_names = [
                    name
                    for index, name in enumerate(self.axis_names)
                    if index not in reduced_axes
                ]
            return result

        if not isinstance(self, (Array3D, Array4D)):
            return result

        from astropy.units import dimensionless_unscaled

        source_axes = tuple(self.axes)
        if keepdims or result.ndim == self.ndim:
            for index, source_axis in enumerate(source_axes):
                setattr(result, f"_axis{index}_name", source_axis.name)
                if index in reduced_axes:
                    index_values = np.arange(result.shape[index])
                    axis_index = index_values * dimensionless_unscaled
                else:
                    axis_index = source_axis.index
                setattr(result, f"_axis{index}_index", axis_index)
            result._axis_names = [source_axis.name for source_axis in source_axes]
            return result

        surviving_axes = [
            source_axis
            for index, source_axis in enumerate(source_axes)
            if index not in reduced_axes
        ]
        return self._rebuild_reduced_result(result, surviving_axes)

    def _normalize_array2d_result(self, result, *, axis):
        """Match GWpy's implicit-index behavior without changing outcomes."""
        from gwpy.types.array2d import Array2D as GwpyArray2D

        explicit = tuple(
            f"_{attribute}" in self.__dict__ for attribute in ("xindex", "yindex")
        )

        if isinstance(result, GwpyArray2D):
            for attribute, is_explicit in zip(
                ("xindex", "yindex"), explicit, strict=True
            ):
                if not is_explicit and f"_{attribute}" in result.__dict__:
                    delattr(result, attribute)
            return result

        if getattr(result, "ndim", 0) != 1:
            return result

        reduced_axes = self._normalized_reduction_axes(axis)
        surviving_axes = [
            position
            for position in range(getattr(self, "ndim"))
            if position not in reduced_axes
        ]
        if (
            len(surviving_axes) == 1
            and not explicit[surviving_axes[0]]
            and "_xindex" in result.__dict__
        ):
            delattr(result, "xindex")
        return result

    def _normalized_reduction_axes(self, axis):
        ndim = getattr(self, "ndim")
        if axis is None:
            return tuple(range(ndim))
        axes = axis if isinstance(axis, tuple) else (axis,)
        return tuple(int(item) % ndim for item in axes)

    def _rebuild_reduced_result(self, result, surviving_axes):
        from astropy.units import Quantity

        metadata = {
            name: value
            for name in ("name", "epoch", "channel")
            if (value := getattr(result, name, None)) is not None
        }
        values = result.value
        unit = result.unit

        if not surviving_axes:
            return Quantity(values, unit=unit, copy=False)

        if len(surviving_axes) == 1:
            from .series import Series

            rebuilt = Series(
                values,
                unit=unit,
                xindex=surviving_axes[0].index,
                copy=False,
                **metadata,
            )
            rebuilt.xindex.info.name = surviving_axes[0].name
            return rebuilt

        if len(surviving_axes) == 2:
            from .plane2d import Plane2D

            return Plane2D(
                values,
                unit=unit,
                axis1_name=surviving_axes[0].name,
                axis2_name=surviving_axes[1].name,
                xindex=surviving_axes[0].index,
                yindex=surviving_axes[1].index,
                copy=False,
                **metadata,
            )

        if len(surviving_axes) == 3:
            from .array3d import Array3D

            return Array3D(
                values,
                unit=unit,
                axis_names=[axis_.name for axis_ in surviving_axes],
                axis0=surviving_axes[0].index,
                axis1=surviving_axes[1].index,
                axis2=surviving_axes[2].index,
                copy=False,
                **metadata,
            )

        return result

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
