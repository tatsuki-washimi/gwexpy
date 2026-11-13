"""Finite-sample Monte Carlo p-values shared by the statistics module.

With ``N`` simulated null statistics the observed statistic is one of
``N + 1`` exchangeable values under the null hypothesis, so it is ranked
among ``N + 1`` positions, not ``N``. Counting the observed statistic itself
gives the one-sided p-value

    p = (r + 1) / (N + 1),

where ``r`` is the number of simulated statistics at least as extreme as the
observed one. For independent draws from a continuous null this satisfies
``P(p <= alpha) <= alpha`` for every ``alpha``; the raw proportion ``r / N``
does not, and it reports ``p = 0`` when the observed statistic lies beyond
every simulated value (North, Curtis & Sham 2002; Phipson & Smyth 2010).

The value is a valid test p-value. It is not an unbiased estimate of the
underlying tail probability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["two_sided_rank_pvalue"]


def two_sided_rank_pvalue(sorted_null: np.ndarray, observed: ArrayLike) -> np.ndarray:
    """Two-sided Monte Carlo p-value, doubling the smaller one-sided tail.

    ``p = min(1, 2 * (min(r_upper, r_lower) + 1) / (N + 1))`` with
    ``r_upper = #{T_sim >= T_obs}`` and ``r_lower = #{T_sim <= T_obs}``.

    Parameters
    ----------
    sorted_null : numpy.ndarray
        The ``N`` simulated null statistics, sorted ascending.
    observed : array-like
        Observed statistic(s), dimensionless and on the same scale as
        `sorted_null`.

    Returns
    -------
    numpy.ndarray
        Dimensionless p-values in ``[2 / (N + 1), 1]``, with the shape of
        `observed`. A simulated statistic equal to the observed one is
        counted in both tails. Non-finite `observed` values are not handled
        here; callers mask them.

    """
    n_trials = len(sorted_null)
    upper = n_trials - np.searchsorted(sorted_null, observed, side="left")
    lower = np.searchsorted(sorted_null, observed, side="right")
    p_values = 2.0 * (np.minimum(upper, lower) + 1) / (n_trials + 1)
    return np.asarray(np.minimum(p_values, 1.0), dtype=float)
