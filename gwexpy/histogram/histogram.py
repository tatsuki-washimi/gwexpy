from __future__ import annotations

from typing import Any

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
        It also updates sumw2 if statistical error tracking is enabled.

        Parameters
        ----------
        data : array-like or Quantity
            Data points to be added to the histogram.
        weights : array-like, optional
            Weights for each data point.

        Returns
        -------
        Histogram
            A new Histogram object with updated values and uncertainties.
        """
        import numpy as np

        data_arr = np.asarray(data)
        if hasattr(data, "unit"):
            data_arr = u.Quantity(data).to(self.xunit).value

        # Calculate hist inside existing bin edges
        hist, _ = np.histogram(data_arr, bins=self.edges.value, weights=weights)

        new_values = self.values.value + hist

        # update sumw2 for error tracking
        new_sw2 = None
        if self.sumw2 is not None or weights is not None:
            w_arr = np.ones_like(data_arr) if weights is None else np.asarray(weights)
            hist_sw2, _ = np.histogram(data_arr, bins=self.edges.value, weights=w_arr**2)
            old_sw2 = self.sumw2.value if self.sumw2 is not None else np.zeros_like(hist_sw2)
            new_sw2 = old_sw2 + hist_sw2

        # covariance tracking - for independent items, we only update diagonal (sumw2 covers this)
        new_cov = self.cov

        kwargs = {}
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
            **kwargs
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

        return registry.read(cls, source, *args, **kwargs)  # type: ignore[no-any-return]

    @classmethod
    def from_root(cls, obj: Any) -> Histogram:
        """Create a Histogram from a ROOT TH1 object."""
        from gwexpy.interop.root_ import from_root

        return from_root(cls, obj)  # type: ignore[no-any-return, no-untyped-call]

    def to_th1d(self) -> Any:
        """Convert this Histogram to a ROOT TH1D."""
        from gwexpy.interop.root_ import to_th1d

        return to_th1d(self)  # type: ignore[no-untyped-call]

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
