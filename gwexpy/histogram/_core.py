from __future__ import annotations

import copy
from typing import Any

import numpy as np
from astropy import units as u


class HistogramCoreMixin:
    """Core Mixin for Histogram.

    Provides composite attributes for values, edges, and covariance,
    handling astropy units transparently without inheriting from ndarray.
    """

    # These properties are defined in Histogram.__init__:
    _values: u.Quantity
    _edges: u.Quantity
    _cov: u.Quantity | None
    _sumw2: u.Quantity | None

    @property
    def values(self) -> u.Quantity:
        """The total values in each bin (bin total)."""
        return self._values

    @property
    def value(self) -> u.Quantity:
        """Alias for values (bin total)."""
        return self._values

    @property
    def edges(self) -> u.Quantity:
        """The bin edges of the histogram (length: n_bins + 1)."""
        return self._edges

    @property
    def xindex(self) -> u.Quantity:
        """The coordinates of the bins (centers). Used by fitting and plotting."""
        return self.bin_centers

    @property
    def errors(self) -> u.Quantity | None:
        """1-sigma statistical errors (derived from cov or sumw2)."""
        if self._cov is not None:
            # cov trace provides errors
            v = np.sqrt(np.diag(self._cov.value))
            return v * self.unit
        if getattr(self, "_sumw2", None) is not None:
            return np.sqrt(self._sumw2)
        return None  # No statistical uncertainties available

    @property
    def cov(self) -> u.Quantity | None:
        """Covariance matrix of the bins (n_bins, n_bins)."""
        return getattr(self, "_cov", None)

    @property
    def sumw2(self) -> u.Quantity | None:
        """Sum of squares of weights for each bin (n_bins)."""
        return getattr(self, "_sumw2", None)

    @property
    def unit(self) -> u.UnitBase:
        """The physical unit of the bin values."""
        return self._values.unit

    @property
    def xunit(self) -> u.UnitBase:
        """The physical unit of the bin edges."""
        return self._edges.unit

    @property
    def nbins(self) -> int:
        """Number of bins."""
        return int(self._values.size)

    @property
    def bin_widths(self) -> u.Quantity:
        """The width of each bin."""
        return np.diff(self.edges)

    @property
    def bin_centers(self) -> u.Quantity:
        """The center of each bin."""
        return self.edges[:-1] + self.bin_widths / 2

    def copy(self, deep: bool = True) -> Any:
        """Return a copy of the histogram."""
        if not deep:
            return copy.copy(self)

        kwargs = {
            "values": self.values.copy(),
            "edges": self.edges.copy(),
        }
        if self.cov is not None:
            kwargs["cov"] = self.cov.copy()
        if self.sumw2 is not None:
            kwargs["sumw2"] = self.sumw2.copy()

        kwargs["name"] = getattr(self, "name", None)
        kwargs["channel"] = getattr(self, "channel", None)

        from typing import cast
        cls_any = cast(Any, self.__class__)
        return cls_any(**kwargs)

    def crop(self, start: Any | None = None, end: Any | None = None) -> Any:
        """
        Crop the histogram to a specific range [start, end].
        
        Returns a new Histogram containing only the bins that fall entirely 
        within the specified range.

        Parameters
        ----------
        start : float or Quantity, optional
            The lower limit of the range. If None, use the low edge of the first bin.
        end : float or Quantity, optional
            The upper limit of the range. If None, use the high edge of the last bin.

        Returns
        -------
        Histogram
            A new Histogram object containing the subset of bins.
        """
        from astropy import units as u
        
        edges = self.edges.value
        
        q_start = u.Quantity(start, unit=self.xunit).value if start is not None else -np.inf
        q_end = u.Quantity(end, unit=self.xunit).value if end is not None else np.inf
        
        # Mask for bins that are completely within [q_start, q_end]
        mask = (edges[:-1] >= q_start) & (edges[1:] <= q_end)
        
        if not np.any(mask):
            raise ValueError(f"No bins found in range [{start}, {end}]")
            
        # Get start and end indices of the contiguous block
        indices = np.where(mask)[0]
        i0, i1 = indices[0], indices[-1]
        
        new_values = self.values[i0 : i1 + 1]
        new_edges = self.edges[i0 : i1 + 2]
        
        kwargs = {
            "values": new_values,
            "edges": new_edges,
            "name": getattr(self, "name", None),
            "channel": getattr(self, "channel", None),
        }
        
        if self.cov is not None:
            kwargs["cov"] = self.cov[i0 : i1 + 1, i0 : i1 + 1]
        if self.sumw2 is not None:
            kwargs["sumw2"] = self.sumw2[i0 : i1 + 1]
            
        return self.__class__(**kwargs)

    def to_density(self) -> u.Quantity:
        """Return the physical density (values / bin_widths)."""
        return self.values / self.bin_widths

    def mean(self) -> u.Quantity:
        """Compute the weighted mean of the histogram.

        Calculated as sum(bin_center * counts) / sum(counts).

        Returns
        -------
        Quantity
            The mean value with xunit.
        """
        w = self.values.value
        x = self.bin_centers.value
        if np.sum(w) == 0:
            return np.nan * self.xunit
        return (np.sum(w * x) / np.sum(w)) * self.xunit

    def var(self, ddof: int = 0) -> u.Quantity:
        """Compute the weighted variance of the histogram.

        Parameters
        ----------
        ddof : int, optional
            Delta Degrees of Freedom. The divisor used in calculations
            is ``N - ddof``, where ``N`` is the sum of bin counts.

        Returns
        -------
        Quantity
            The variance value with xunit**2.
        """
        w = self.values.value
        x = self.bin_centers.value
        w_sum = np.sum(w)
        if w_sum <= ddof:
            return np.nan * (self.xunit**2)
        mu = self.mean().value
        v = np.sum(w * (x - mu) ** 2) / (w_sum - ddof)
        return v * (self.xunit**2)

    def std(self, ddof: int = 0) -> u.Quantity:
        """Compute the weighted standard deviation of the histogram.

        Parameters
        ----------
        ddof : int, optional
            Delta Degrees of Freedom.

        Returns
        -------
        Quantity
            The standard deviation value with xunit.
        """
        return np.sqrt(self.var(ddof=ddof))

    def quantile(self, q: float) -> u.Quantity:
        """Compute the q-th quantile of the distribution.

        Uses linear interpolation on the cumulative distribution function (CDF)
        calculated from the bin contents.

        Parameters
        ----------
        q : float
            Quantile level, between 0 and 1.

        Returns
        -------
        Quantity
            The quantile value with xunit.
        """
        if not (0 <= q <= 1):
            raise ValueError("q must be between 0 and 1.")
        
        w = self.values.value
        w_sum = np.sum(w)
        if w_sum == 0:
            return np.nan * self.xunit
            
        cdf = np.cumsum(w) / w_sum
        # Prepend 0 to CDF for the low edge of the first bin
        cdf_full = np.concatenate(([0.0], cdf))
        edges = self.edges.value
        
        # Linearly interpolate
        return np.interp(q, cdf_full, edges) * self.xunit

    def median(self) -> u.Quantity:
        """Compute the median of the histogram.

        Returns
        -------
        Quantity
            The median value with xunit.
        """
        return self.quantile(0.5)

    def min(self) -> u.Quantity:
        """Return the lower edge of the first bin with non-zero content.

        Returns
        -------
        Quantity
            The lower boundary with xunit. Returns NaN if the histogram is empty.
        """
        w = self.values.value
        idx = np.where(w > 0)[0]
        if len(idx) == 0:
            return np.nan * self.xunit
        return self.edges[idx[0]]

    def max(self) -> u.Quantity:
        """Return the upper edge of the last bin with non-zero content.

        Returns
        -------
        Quantity
            The upper boundary with xunit. Returns NaN if the histogram is empty.
        """
        w = self.values.value
        idx = np.where(w > 0)[0]
        if len(idx) == 0:
            return np.nan * self.xunit
        return self.edges[idx[-1] + 1]

    @classmethod
    def from_density(
        cls,
        density: Any,
        edges: Any,
        unit: Any = None,
        xunit: Any = None,
        **kwargs: Any
    ) -> Any:
        """Create a Histogram from density values."""
        density_q = u.Quantity(density, unit=unit)
        edges_q = u.Quantity(edges, unit=xunit)
        bin_widths = np.diff(edges_q)

        values = density_q * bin_widths
        return cls(values=values, edges=edges_q, **kwargs)

    def __len__(self) -> int:
        return self.nbins

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        ns = f" '{name}'" if name else ""
        return f"<{self.__class__.__name__}{ns} (nbins={self.nbins}, unit={self.unit})>"
