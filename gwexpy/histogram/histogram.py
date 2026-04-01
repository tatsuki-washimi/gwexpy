from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
    """
    A unified 1D Histogram representation preserving astropy.units.

    Unlike TimeSeries/FrequencySeries, Histogram is not an instance of ndarray,
    but instead a composite holding `values`, `edges`, `cov`, and `sumw2` attributes.
    All properties expose proper astropy Quantities to maintain physical consistency.

    Notes
    -----
    **Statistical Error Tracking (Double Management Rule):**
    `Histogram` maintains two attributes for error tracking:
    1. `sumw2`: The sum of squared weights per bin. This tracks the uncorrelated
       statistical variance (e.g., from counting statistics).
    2. `cov`: The full covariance matrix. Its diagonal elements SHOULD include the
       statistical variance tracked by `sumw2`.
    When filling the histogram, both are updated to ensure consistency.
    
    """

    def __init__(
        self,
        values: Any,
        edges: Any,
        unit: Any = None,
        xunit: Any = None,
        cov: Any = None,
        sumw2: Any = None,
        name: str | None = None,
        channel: Any | None = None,
    ):
        """
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

        self.name = name
        self.channel = channel

    def fill(self, data: Any, weights: Any = None) -> Histogram:
        """
        Fill the histogram with new data points.

        This method calculates the occurrence counts for the given data
        within the existing bin edges and adds them to the current values.

        Statistical Error Tracking (Double Management Rule):
        It updates both `sumw2` and the diagonal of `cov` if they are present.
        The diagonal of `cov` is updated with the new statistical variance
        derived from weights (or counts if weights are None).

        Parameters
        ----------
        data : array-like or Quantity
            Data points to be added to the histogram.
        weights : array-like or Quantity, optional
            Weights for each data point. If a Quantity, it is converted to
            the histogram's unit (`self.unit`).

        Returns
        -------
        Histogram
            A new Histogram object with updated values and uncertainties.
        """
        # 1. Prepare data and weights in the correct units
        data_arr = np.asarray(data)
        if hasattr(data, "unit"):
            data_arr = u.Quantity(data).to(self.xunit).value

        w_arr = weights
        if weights is not None:
            if hasattr(weights, "unit"):
                w_arr = u.Quantity(weights).to(self.unit).value
            else:
                w_arr = np.asarray(weights)

        # 2. Calculate values increment
        hist, _ = np.histogram(data_arr, bins=self.edges.value, weights=w_arr)
        new_values = self.values.value + hist

        # 3. Calculate variance increment (sum of squared weights)
        w_eff = np.ones_like(data_arr) if w_arr is None else w_arr
        hist_sw2, _ = np.histogram(data_arr, bins=self.edges.value, weights=w_eff**2)

        # 4. Update sumw2 (uncorrelated statistical variance)
        new_sw2 = None
        if self.sumw2 is not None or weights is not None:
            old_sw2_val = (
                self.sumw2.value
                if self.sumw2 is not None
                else np.zeros_like(hist_sw2)
            )
            new_sw2 = old_sw2_val + hist_sw2

        # 5. Update cov (include new statistical variance in diagonal)
        new_cov = self.cov
        if self.cov is not None:
            # Consistent with Double Management Rule:
            # Diagonal components of cov must track the statistical variance in sumw2.
            new_cov_val = self.cov.value.copy()
            new_diag = np.diag(new_cov_val) + hist_sw2
            np.fill_diagonal(new_cov_val, new_diag)
            new_cov = u.Quantity(new_cov_val, unit=self.unit**2)

        kwargs: dict[str, Any] = {}
        if new_sw2 is not None:
            kwargs["sumw2"] = new_sw2
        if new_cov is not None:
            kwargs["cov"] = new_cov

        return self.__class__(
            values=new_values,
            edges=self.edges,
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
