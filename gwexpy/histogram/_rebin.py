from __future__ import annotations

from typing import Any

import numpy as np


def compute_A_matrix(old_edges: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    """
    Compute intersection fraction matrix A for rebinning.

    A[i, j] represents the fraction of old bin `j` that falls into new bin `i`.
    Assuming `values` represents the *total amount* in each bin (not density),
    the new values are given by: y_new = A @ y_old

    Parameters
    ----------
    old_edges : np.ndarray
        Array of old bin boundaries of shape (N + 1,).
    new_edges : np.ndarray
        Array of new bin boundaries of shape (M + 1,).

    Returns
    -------
    np.ndarray
        Matrix A of shape (M, N).
    """
    old_edges = np.asarray(old_edges)
    new_edges = np.asarray(new_edges)

    if not np.all(np.diff(old_edges) > 0):
        raise ValueError("old_edges must be strictly monotonically increasing.")
    if not np.all(np.diff(new_edges) > 0):
        raise ValueError("new_edges must be strictly monotonically increasing.")

    n_old = len(old_edges) - 1
    n_new = len(new_edges) - 1
    old_widths = np.diff(old_edges)

    A = np.zeros((n_new, n_old), dtype=float)

    # For each new bin, figure out overlap with all old bins
    for i in range(n_new):
        low_new = new_edges[i]
        high_new = new_edges[i + 1]

        # Overlap computation (vectorized over old bins)
        low_overlap = np.maximum(low_new, old_edges[:-1])
        high_overlap = np.minimum(high_new, old_edges[1:])

        # Size of the overlap
        overlap = np.maximum(0.0, high_overlap - low_overlap)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = overlap / old_widths
            # If old_width is 0 (should not happen for strictly increasing), frac is 0.
            np.nan_to_num(frac, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        A[i, :] = frac

    return A


class HistogramRebinMixin:
    """Mixin to provide rebinning and integration functionality using the A matrix."""

    def rebin(self, new_edges: Any, xunit: Any = None, **kwargs: Any) -> Any:
        """
        Rebin the histogram into new bins using a fraction intersection matrix.

        Values and covariances are scaled proportionally, maintaining the physical
        summed quantities. Assumes the internal values represent bin totals.

        Parameters
        ----------
        new_edges : array-like or Quantity
            New bin edges to project onto.
        xunit : str or astropy.units.Unit, optional
            The unit of new_edges if it is not a Quantity.

        Returns
        -------
        Histogram
            A new rebinned Histogram.
        """
        from astropy import units as u
        from typing import cast

        h = cast(Any, self)

        if hasattr(new_edges, "unit"):
            new_edges_q = u.Quantity(new_edges).to(h.xunit)
        elif xunit is not None:
            new_edges_q = u.Quantity(new_edges, unit=xunit).to(h.xunit)
        else:
            new_edges_q = u.Quantity(new_edges, unit=h.xunit)

        # Ensure valid arrays
        old_val = h.edges.value
        new_val = new_edges_q.value

        if new_val[0] < old_val[0] or new_val[-1] > old_val[-1]:
            # Extrapolation region will just sum 0 overlap, which is mathematically
            # correct (assuming 0 counts outside range), but we should warn or be aware.
            pass

        # Calculate Transform Matrix
        A = compute_A_matrix(old_val, new_val)

        # Propagate total values
        y_new_value = A @ h.values.value

        # Propagate covariance if present
        cov_new = None
        if getattr(h, "cov", None) is not None:
            cov_val = h.cov.value
            # Covariance propagation based on linear combination: C_y = A * C_x * A.T
            cov_new_val = A @ cov_val @ A.T
            cov_new = u.Quantity(cov_new_val, unit=h.cov.unit)

        # Propagate sumw2 if present
        sumw2_new = None
        if getattr(h, "sumw2", None) is not None:
            # For purely independent poisson-like bins, sumw2 propagates similarly to variance.
            # However, sumw2 = diagonal of C_x if uncorrelated.
            # A_ij^2 * sumw2_j -> new sumw2_i
            sumw2_new_val = (A**2) @ h.sumw2.value
            sumw2_new = u.Quantity(sumw2_new_val, unit=h.sumw2.unit)

        cls_any = cast(Any, h.__class__)
        return cls_any(
            values=u.Quantity(y_new_value, unit=h.unit),
            edges=new_edges_q,
            cov=cov_new,
            sumw2=sumw2_new,
            name=getattr(h, "name", None),
            channel=getattr(h, "channel", None),
        )

    def integral(
        self,
        start: Any | None = None,
        end: Any | None = None,
        xunit: Any = None,
        return_error: bool = False,
    ) -> Any:
        """
        Integrate the histogram over the interval [start, end].

        If start or end is None, the boundaries of the histogram are used.

        Parameters
        ----------
        start, end : float or Quantity, optional
            Integration limits.
        xunit : str or astropy.units.Unit, optional
            Unit for start/end if they are not Quantities.
        return_error : bool, optional
            If True, return (integral, error). Otherwise return only integral.

        Returns
        -------
        Quantity or (Quantity, Quantity)
            The integral value, and optionally its uncertainty.
        """
        from astropy import units as u
        from typing import cast

        h = cast(Any, self)

        if start is None:
            start_val = h.edges.value[0]
        elif xunit is not None:
            start_val = u.Quantity(start, unit=xunit).to(h.xunit).value
        elif hasattr(start, "unit"):
            start_val = start.to(h.xunit).value
        else:
            start_val = float(start)

        if end is None:
            end_val = h.edges.value[-1]
        elif xunit is not None:
            end_val = u.Quantity(end, unit=xunit).to(h.xunit).value
        elif hasattr(end, "unit"):
            end_val = end.to(h.xunit).value
        else:
            end_val = float(end)

        # To integrate, we rebin into a single bin [start, end]
        pseudo_new_edges = np.array([start_val, end_val])
        A = compute_A_matrix(h.edges.value, pseudo_new_edges)

        # Propagate values and errors to the single pseudo-bin
        total_val = (A @ h.values.value)[0]

        if not return_error:
            return u.Quantity(total_val, unit=h.unit)

        var_total = 0.0
        if getattr(h, "cov", None) is not None:
            var_total = (A @ h.cov.value @ A.T)[0, 0]
        elif getattr(h, "sumw2", None) is not None:
            var_total = ((A**2) @ h.sumw2.value)[0]

        return u.Quantity(total_val, unit=h.unit), u.Quantity(
            np.sqrt(var_total), unit=h.unit
        )
