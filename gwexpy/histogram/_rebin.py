import functools
from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.utils.logger import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=128)
def _compute_A_matrix_cached(
    old_edges_tuple: tuple[float, ...], new_edges_tuple: tuple[float, ...]
) -> np.ndarray:
    """Internal cached implementation of compute_A_matrix using tuples as keys."""
    old_edges = np.array(old_edges_tuple)
    new_edges = np.array(new_edges_tuple)

    n_old = len(old_edges) - 1
    n_new = len(new_edges) - 1
    old_widths = np.diff(old_edges)

    A = np.zeros((n_new, n_old), dtype=float)

    for i in range(n_new):
        low_new = new_edges[i]
        high_new = new_edges[i + 1]

        # Overlap computation (vectorized over old bins)
        low_overlap = np.maximum(low_new, old_edges[:-1])
        high_overlap = np.minimum(high_new, old_edges[1:])

        # Size of the overlap
        overlap = high_overlap - low_overlap
        # Robust check: if high_overlap is slightly smaller due to float error, set to 0
        overlap = np.maximum(0.0, overlap)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = overlap / old_widths

            # If frac is very close to 1.0 or 0.0, snap it to prevent dispersion
            frac[np.isclose(frac, 1.0, atol=1e-14, rtol=0)] = 1.0
            frac[np.isclose(frac, 0.0, atol=1e-14, rtol=0)] = 0.0

            # Clamp to [0, 1] to stay within physical bounds
            np.clip(frac, 0.0, 1.0, out=frac)
            np.nan_to_num(frac, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        A[i, :] = frac

    return A


def compute_A_matrix(old_edges: Any, new_edges: Any) -> np.ndarray:
    """
    Compute intersection fraction matrix A for rebinning.

    A[i, j] represents the fraction of old bin `j` that falls into new bin `i`.
    Assuming `values` represents the *total amount* in each bin (not density),
    the new values are given by: y_new = A @ y_old

    This function caches results to improve performance in repetitive rebinning tasks.

    Parameters
    ----------
    old_edges : array-like or Quantity
        Array of old bin boundaries of shape (N + 1,).
    new_edges : array-like or Quantity
        Array of new bin boundaries of shape (M + 1,).

    Returns
    -------
    np.ndarray
        Matrix A of shape (M, N).
    """
    # Convert to plain array and then to tuple for hashing
    old_arr = np.asarray(old_edges, dtype=float)
    new_arr = np.asarray(new_edges, dtype=float)

    if not np.all(np.diff(old_arr) > 0):
        # We need this check here too or in the cached part?
        # Better here to fail early and keep clean cache.
        raise ValueError("old_edges must be strictly monotonically increasing.")
    if not np.all(np.diff(new_arr) > 0):
        raise ValueError("new_edges must be strictly monotonically increasing.")

    # Call cached version
    return _compute_A_matrix_cached(tuple(old_arr), tuple(new_arr))


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

        Notes
        -----
        If `new_edges` extend beyond the current histogram range, the regions
        outside are assumed to have zero content and a warning is issued.
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
            logger.warning(
                "New bin edges [%.3g, %.3g] extend beyond original range [%.3g, %.3g]. "
                "Regions outside the original range will have zero content.",
                new_val[0],
                new_val[-1],
                old_val[0],
                old_val[-1],
            )

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
        The result is calculated by identifying the fraction of each bin that
        overlaps with the specified range. Regions outside the existing bin
        edges are assumed to have zero content.

        Parameters
        ----------
        start, end : float or Quantity, optional
            Integration limits. If start > end, ValueError is raised.
            If start == end, the result is 0.
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

        # Validation
        if start_val > end_val:
            raise ValueError(
                f"Integration start ({start_val}) cannot be greater than end ({end_val})."
            )

        if start_val == end_val:
            val = u.Quantity(0.0, unit=h.unit)
            if return_error:
                return val, val
            return val

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
