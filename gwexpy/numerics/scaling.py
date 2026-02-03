"""
Data-adaptive scaling utilities for gwexpy.

The core problem
----------------
Functions like ``whiten(X, eps=1e-12)`` assume the data lives at O(1).
Gravitational-wave strain is O(1e-21), so a fixed ``eps`` can be
**larger** than the signal itself.

:func:`safe_epsilon` and :class:`AutoScaler` solve this by computing
an epsilon *relative to the data's own scale*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import SAFE_FLOOR

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["safe_epsilon", "get_safe_epsilon", "AutoScaler", "safe_log_scale"]


def safe_epsilon(
    data: ArrayLike,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = SAFE_FLOOR,
) -> float:
    """Return an epsilon appropriate for *data*'s scale.

    .. math::

        \\varepsilon = \\max\\bigl(\\text{abs\\_tol},\\;
        \\operatorname{std}(\\text{data}) \\times \\text{rel\\_tol}\\bigr)

    Parameters
    ----------
    data : array-like
        Input data whose scale determines the epsilon.
    rel_tol : float, optional
        Relative tolerance multiplied by ``std(data)``.  Default ``1e-6``.
    abs_tol : float, optional
        Absolute floor, used when the data is numerically silent
        (all zeros).  Default :data:`~gwexpy.numerics.constants.SAFE_FLOOR`.

    Returns
    -------
    float
        A positive epsilon scaled to the data.
    """
    arr = np.asarray(data, dtype=float)
    scale = float(np.std(arr))
    return max(abs_tol, scale * rel_tol)


def get_safe_epsilon(
    data: ArrayLike,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = SAFE_FLOOR,
) -> float:
    """Alias for :func:`safe_epsilon` (backward/compat)."""
    return safe_epsilon(data, rel_tol=rel_tol, abs_tol=abs_tol)


class AutoScaler:
    """Context manager for safe internal normalisation.

    Many numerical algorithms (whitening, ICA, MCMC) work best when the
    data is O(1).  ``AutoScaler`` normalises the input on entry and
    rescales the output on exit, hiding the bookkeeping from the caller.

    Parameters
    ----------
    data : array-like
        Input data to be normalised.
    eps : float or None, optional
        If given, used as the minimum permissible scale.  When ``None``
        (default), :data:`~gwexpy.numerics.constants.SAFE_FLOOR` is
        used — a value (1e-50) well below any physical GW quantity,
        so that even O(1e-21) strain is correctly normalised.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.numerics import AutoScaler
    >>> x = np.array([1e-21, 2e-21, 3e-21])
    >>> with AutoScaler(x) as sc:
    ...     normed = sc.normalize()       # O(1) data
    ...     result = normed * 2           # process at O(1)
    ...     output = sc.denormalize(result)  # back to original scale
    >>> np.allclose(output, x * 2)
    True
    """

    def __init__(
        self,
        data: ArrayLike,
        eps: float | None = None,
    ) -> None:
        self._data = np.asarray(data, dtype=float)
        self._eps = eps if eps is not None else SAFE_FLOOR
        self._scale: float = self._compute_scale()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _compute_scale(self) -> float:
        s = float(np.std(self._data))
        if s < self._eps:
            # Data is effectively silent — fall back to unit scale so
            # downstream code receives zeros rather than garbage.
            return 1.0
        return s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def scale(self) -> float:
        """The computed (or fallback) scale factor."""
        return self._scale

    def normalize(self) -> NDArray[np.floating]:
        """Return ``data / scale``."""
        return self._data / self._scale

    def denormalize(self, result: ArrayLike) -> NDArray[np.floating]:
        """Return ``result * scale``, restoring the original magnitude."""
        return np.asarray(result, dtype=float) * self._scale

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> AutoScaler:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def __repr__(self) -> str:
        return f"AutoScaler(scale={self._scale:.4e})"


def safe_log_scale(
    data: ArrayLike,
    *,
    dynamic_range_db: float = 200.0,
    factor: float = 10.0,
) -> NDArray[np.floating]:
    """Return ``factor * log10`` of *data* with a dynamic floor.

    The floor is computed as ``max(SAFE_FLOOR, max(abs(data)) * 10^{-dynamic_range_db/10})``.
    This keeps low-amplitude data distinguishable down to the ``SAFE_FLOOR`` scale.
    """
    arr = np.asarray(data, dtype=float)
    abs_arr = np.abs(arr)
    finite_mask = np.isfinite(abs_arr)
    if not np.any(finite_mask):
        return np.full_like(abs_arr, -np.inf)
    max_val = float(np.nanmax(abs_arr[finite_mask]))
    floor_ratio = 10.0 ** (-dynamic_range_db / 10.0)
    floor_value = max(SAFE_FLOOR, max_val * floor_ratio)
    return factor * np.log10(np.maximum(abs_arr, floor_value))
