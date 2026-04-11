from __future__ import annotations

from typing import Any, cast

import numpy as np
from astropy import units as u

from gwexpy.fitting.mixin import FittingMixin
from gwexpy.types.mixin._plot_mixin import PlotMixin

from ._core import HistogramCoreMixin
from ._rebin import HistogramRebinMixin


class Histogram(
    PlotMixin,
    FittingMixin,
    HistogramRebinMixin,
    HistogramCoreMixin,
):
    """A 1D Histogram representation with physical units and uncertainty.

    `Histogram` stores a 1-dimensional distribution of values across a
    set of bin edges. It maintains physical consistency using
    `astropy.units` and provides robust uncertainty tracking via
    covariance matrices and sum-of-weights-squared.

    Parameters
    ----------
    values : array-like or `~astropy.units.Quantity`
        The counts or integrated values in each bin.

    edges : array-like or `~astropy.units.Quantity`
        The bin boundaries (length n_bins + 1).

    unit : `str`, `~astropy.units.Unit`, optional
        Unit for `values`.

    xunit : `str`, `~astropy.units.Unit`, optional
        Unit for `edges`.

    cov : array-like or `~astropy.units.Quantity`, optional
        Covariance matrix for the bin values.

    sumw2 : array-like or `~astropy.units.Quantity`, optional
        Sum of squared weights per bin for statistical error tracking.

    **kwargs
        Additional attributes like `name`, `channel`, `underflow`, etc.

    Notes
    -----
    The uncertainty is tracked in two ways:

    1. `sumw2`: Statistical (uncorrelated) variance per bin.
    2. `cov`: Full covariance matrix. The diagonal of `cov` must stay
       consistent with `sumw2`.

    Examples
    --------
    >>> from gwexpy.histogram import Histogram
    >>> h = Histogram([1, 2], [0, 1, 2])
    >>> h
    <Histogram (nbins=2, unit=)>

    """

    def __init__(
        self,
        values: Any,
        edges: Any,
        unit: Any = None,
        xunit: Any = None,
        cov: Any = None,
        sumw2: Any = None,
        underflow: Any = None,
        overflow: Any = None,
        underflow_sumw2: Any = None,
        overflow_sumw2: Any = None,
        name: str | None = None,
        channel: Any | None = None,
    ):
        """Initialize a histogram from values and bin edges.

        Parameters
        ----------
        values : array-like or Quantity
            Total events or integrated quantity in each bin (Length: n_bins).
        edges : array-like or Quantity
            Bin boundaries (Length: n_bins + 1). Must be strictly monotonically increasing.
        unit : str or astropy.units.Unit, optional
            Unit for `values` if it's not a Quantity.
        xunit : str or astropy.units.Unit, optional
            Unit for `edges` if it's not a Quantity.
        cov : array-like or Quantity, optional
            Covariance matrix (n_bins, n_bins) denoting uncertainties and correlations.
        sumw2 : array-like or Quantity, optional
            Sum of squares of weights for each bin (n_bins) for uncorrelated errors.
        underflow : float or Quantity, optional
            Total value below the first bin edge.
        overflow : float or Quantity, optional
            Total value above the last bin edge.
        underflow_sumw2 : float or Quantity, optional
            Sum of squared weights for the underflow region.
        overflow_sumw2 : float or Quantity, optional
            Sum of squared weights for the overflow region.
        name : str, optional
            Name of this histogram.
        channel : any, optional
            Linked data channel.

        """
        # Validate values
        values_q = u.Quantity(values, unit=unit) if unit else u.Quantity(values)
        if values_q.unit == u.dimensionless_unscaled and not hasattr(values, "unit"):
            if unit is not None:
                values_q = values_q * u.Unit(unit)

        # Validate edges
        edges_q = u.Quantity(edges, unit=xunit) if xunit else u.Quantity(edges)
        if edges_q.unit == u.dimensionless_unscaled and not hasattr(edges, "unit"):
            if xunit is not None:
                edges_q = edges_q * u.Unit(xunit)

        val_arr = values_q.value
        edg_arr = edges_q.value

        if val_arr.ndim != 1:
            raise ValueError(f"Histogram values must be 1D, got {val_arr.ndim}D.")
        if edg_arr.ndim != 1:
            raise ValueError(f"Histogram edges must be 1D, got {edg_arr.ndim}D.")

        n_bins = len(val_arr)
        if len(edg_arr) != n_bins + 1:
            raise ValueError(
                f"edges length ({len(edg_arr)}) must be values length + 1 ({n_bins + 1})."
            )

        if not np.all(np.diff(edg_arr) > 0):
            raise ValueError("Histogram edges must be strictly monotonically increasing.")

        self._values = values_q
        self._edges = edges_q

        # Validation for cov
        if cov is not None:
            cov_q = u.Quantity(cov, unit=values_q.unit**2)
            if cov_q.shape != (n_bins, n_bins):
                raise ValueError(
                    f"cov shape must be ({n_bins}, {n_bins}), got {cov_q.shape}"
                )
            self._cov = cov_q
        else:
            self._cov = None

        # Validation for sumw2
        if sumw2 is not None:
            sw2_q = u.Quantity(sumw2, unit=values_q.unit**2)
            if sw2_q.shape != (n_bins,):
                raise ValueError(f"sumw2 shape must be ({n_bins},), got {sw2_q.shape}")
            self._sumw2 = sw2_q
        else:
            self._sumw2 = None

        # Validation for underflow/overflow
        self._underflow = u.Quantity(underflow or 0.0, unit=values_q.unit)
        self._overflow = u.Quantity(overflow or 0.0, unit=values_q.unit)

        if underflow_sumw2 is not None:
            self._underflow_sumw2 = u.Quantity(underflow_sumw2, unit=values_q.unit**2)
        else:
            self._underflow_sumw2 = (
                None if sumw2 is None else u.Quantity(0.0, unit=values_q.unit**2)
            )

        if overflow_sumw2 is not None:
            self._overflow_sumw2 = u.Quantity(overflow_sumw2, unit=values_q.unit**2)
        else:
            self._overflow_sumw2 = (
                None if sumw2 is None else u.Quantity(0.0, unit=values_q.unit**2)
            )

        self.name = name
        self.channel = channel

    def fill(self, data: Any, weights: Any = None) -> Histogram:
        """Fill the histogram with new data points.

        Calculates occurrence counts for the given data within current
        edges and increments existing values.

        Parameters
        ----------
        data : array-like or `~astropy.units.Quantity`
            Data points to add.

        weights : array-like or `~astropy.units.Quantity`, optional
            Weights for each data point.

        Returns
        -------
        Histogram
            A new Histogram object with updated values and uncertainties.

        Notes
        -----
        Updates both `sumw2` and the diagonal of `cov` to maintain
        statistical consistency.

        Examples
        --------
        >>> h = Histogram([1, 2], [0, 1, 2])
        >>> h = h.fill([0.5, 1.5, 1.5])
        >>> h.values
        <Quantity [2., 4.]>

        """
        import numpy as np

        data_arr = np.asarray(data)
        if hasattr(data, "unit"):
            data_arr = u.Quantity(data).to(self.xunit).value

        # Prepare weights: accept Quantity or array-like or scalar
        weights_arr = None
        if weights is not None:
            # If weights is a Quantity, convert to histogram value unit
            if hasattr(weights, "unit"):
                try:
                    weights_arr = u.Quantity(weights).to(self.unit).value
                except Exception as e:
                    raise ValueError(
                        f"weights unit {getattr(weights, 'unit', None)} is not convertible to histogram unit {self.unit}"
                    ) from e
            else:
                weights_arr = np.asarray(weights)

            # If scalar weight, broadcast to data length
            if np.ndim(weights_arr) == 0:
                weights_arr = np.full_like(data_arr, float(weights_arr))
            elif weights_arr.shape != data_arr.shape:
                raise ValueError(
                    f"weights shape {weights_arr.shape} does not match data shape {data_arr.shape}"
                )

        # 2. Calculate values increment
        hist, _ = np.histogram(data_arr, bins=self.edges.value, weights=weights_arr)
        new_values = self.values.value + hist

        # 2b. Calculate underflow/overflow increment
        # Data below first edge or above last edge
        first_edge = self.edges.value[0]
        last_edge = self.edges.value[-1]
        under_mask = data_arr < first_edge
        over_mask = data_arr > last_edge

        under_inc = (
            np.sum(weights_arr[under_mask])
            if weights_arr is not None
            else np.sum(under_mask)
        )
        over_inc = (
            np.sum(weights_arr[over_mask])
            if weights_arr is not None
            else np.sum(over_mask)
        )

        new_underflow = self.underflow.value + under_inc
        new_overflow = self.overflow.value + over_inc

        # 3. Calculate variance increment (sum of squared weights)
        w_eff = np.ones_like(data_arr) if weights_arr is None else weights_arr
        hist_sw2, _ = np.histogram(data_arr, bins=self.edges.value, weights=w_eff**2)

        under_sw2_inc = np.sum(w_eff[under_mask] ** 2)
        over_sw2_inc = np.sum(w_eff[over_mask] ** 2)

        # 4. Update sumw2 (uncorrelated statistical variance)
        new_sw2 = None
        new_under_sw2 = None
        new_over_sw2 = None

        if self.sumw2 is not None or weights is not None:
            old_sw2_val = (
                self.sumw2.value
                if self.sumw2 is not None
                else np.zeros_like(hist_sw2)
            )
            new_sw2 = old_sw2_val + hist_sw2

            old_under_sw2 = (
                self.underflow_sumw2.value
                if self.underflow_sumw2 is not None
                else 0.0
            )
            old_over_sw2 = (
                self.overflow_sumw2.value
                if self.overflow_sumw2 is not None
                else 0.0
            )
            new_under_sw2 = old_under_sw2 + under_sw2_inc
            new_over_sw2 = old_over_sw2 + over_sw2_inc

        # 5. Update cov (include new statistical variance in diagonal)
        new_cov = self.cov
        if self.cov is not None:
            # Consistent with Double Management Rule:
            # Diagonal components of cov must track the statistical variance in sumw2.
            new_cov_val = self.cov.value.copy()
            if hist_sw2 is not None:
                new_cov_val[np.diag_indices_from(new_cov_val)] += hist_sw2
            new_cov = u.Quantity(new_cov_val, unit=self.unit**2)

        kwargs: dict[str, Any] = {}
        if new_sw2 is not None:
            kwargs["sumw2"] = u.Quantity(new_sw2, unit=self.unit**2)
        if new_under_sw2 is not None:
            kwargs["underflow_sumw2"] = u.Quantity(new_under_sw2, unit=self.unit**2)
        if new_over_sw2 is not None:
            kwargs["overflow_sumw2"] = u.Quantity(new_over_sw2, unit=self.unit**2)
        if new_cov is not None:
            kwargs["cov"] = new_cov

        return self.__class__(
            values=new_values,
            edges=self.edges,
            underflow=new_underflow,
            overflow=new_overflow,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            **kwargs,
        )

    @classmethod
    def read(cls, source: Any, *args: Any, **kwargs: Any) -> Histogram:
        """Read data into a Histogram."""
        fmt = kwargs.get("format")
        if fmt in ("hdf5", "h5", "hdf"):
            import h5py

            from .io._hdf5 import read_hdf5_dataset

            if isinstance(source, h5py.Group):
                path = kwargs.get("path", "data")
                return read_hdf5_dataset(cls, source, path=path)

            from gwexpy.io.hdf5_collection import ensure_hdf5_file

            with ensure_hdf5_file(source, mode="r") as h5f:
                path = kwargs.get("path", "data")
                return read_hdf5_dataset(cls, h5f, path=path)

        # Fallback to astropy io registry
        from astropy.io import registry

        return cast("Histogram", registry.read(cls, source, *args, **kwargs))

    @classmethod
    def from_root(cls, obj: Any) -> Histogram:
        """Create a Histogram from a ROOT TH1 object."""
        from gwexpy.interop.root_ import from_root

        return cast("Histogram", from_root(cls, obj))

    def to_th1d(self) -> Any:
        """Convert this Histogram to a ROOT TH1D."""
        from gwexpy.interop.root_ import to_th1d

        return to_th1d(self)

    def write(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        """Write Histogram to file."""
        fmt = kwargs.get("format")
        if fmt in ("hdf5", "h5", "hdf"):
            import h5py

            from .io._hdf5 import write_hdf5_dataset

            path = kwargs.pop("path", "data")
            if isinstance(target, h5py.Group):
                write_hdf5_dataset(self, target, path=path)
                return target

            from gwexpy.io.hdf5_collection import ensure_hdf5_file

            mode = kwargs.pop("mode", None)
            overwrite = kwargs.pop("overwrite", False)
            with ensure_hdf5_file(target, mode=mode, overwrite=overwrite) as h5f:
                write_hdf5_dataset(self, h5f, path=path)
            return target

        from astropy.io import registry

        return registry.write(self, target, *args, **kwargs)
