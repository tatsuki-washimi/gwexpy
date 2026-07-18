"""gwexpy.statistics.rayleigh_test - Rayleigh statistic test p-values."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

try:
    import scipy  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.statistics. Install with: pip install scipy"
    ) from _exc

from ..spectrogram import Spectrogram

if TYPE_CHECKING:
    pass


def rayleigh_pvalue(
    rayleigh_spec: Spectrogram,
    n_samples: int,
    n_monte_carlo: int = 1000,
) -> Spectrogram:
    """Convert Rayleigh statistic spectrogram to p-value spectrogram.

    Rayleigh statistic is around 1 for Gaussian noise.

    Parameters
    ----------
    rayleigh_spec : Spectrogram
        The output of TimeSeries.rayleigh_spectrogram().
    n_samples : int
        Number of segments used to compute the Rayleigh statistic.
    n_monte_carlo : int, default=1000
        Number of trials for background distribution.

    Returns
    -------
    Spectrogram (p-values)

    """
    # 1. Get background distribution of Rayleigh statistic for n_samples
    dist = _get_rayleigh_stat_null_distribution(n_samples, n_monte_carlo)

    # 2. Compute p-values (both-sided)
    # R is concentrated around 1.
    # Small p-value means R is significantly different from 1.

    r_vals = rayleigh_spec.value

    # Vectorized p-value calculation (both-sided)
    # count of dist >= r
    upper_counts = len(dist) - np.searchsorted(dist, r_vals, side='left')
    # count of dist <= r
    lower_counts = np.searchsorted(dist, r_vals, side='right')

    p_vals = 2.0 * np.minimum(upper_counts, lower_counts) / len(dist)

    # Clip p-values to [0, 1]
    p_vals = np.clip(p_vals, 0.0, 1.0).astype(float)

    # A non-finite Rayleigh statistic sorts to the end of `dist`, so
    # searchsorted would make min(upper, lower)=0 -> p=0.0, i.e. a false
    # maximal-significance detection that triggers a spurious veto in
    # to_segments. Report NaN for those bins instead (issue #459 / S6).
    nonfinite = ~np.isfinite(r_vals)
    if np.any(nonfinite):
        n_bad = int(np.count_nonzero(nonfinite))
        warnings.warn(
            f"rayleigh_pvalue: {n_bad} non-finite Rayleigh statistic value(s) "
            "set to NaN p-value (excluded from veto) instead of a false p=0",
            RuntimeWarning,
            stacklevel=2,
        )
        p_vals[nonfinite] = np.nan

    return Spectrogram(
        p_vals,
        times=rayleigh_spec.times,
        frequencies=rayleigh_spec.frequencies,
        unit="",
        name=f"p-value({rayleigh_spec.name})",
    )


_RAYLEIGH_STAT_CACHE: dict[tuple[int, int], np.ndarray] = {}

def _get_rayleigh_stat_null_distribution(n: int, n_trials: int = 1000) -> np.ndarray:
    """Generate the null distribution of the Rayleigh statistic for `n` segments."""
    # Entry-point guards: an empty sample (n<=0) makes np.std/np.mean return NaN
    # -> all-NaN distribution -> p=0 for every bin; an empty trial set
    # (n_trials<=0) makes the later len(dist) division a silent 0/0 -> all-NaN
    # p-values (issue #459).
    if n <= 0:
        raise ValueError(f"n_samples must be >= 1, got {n}")
    if n_trials <= 0:
        raise ValueError(f"n_monte_carlo must be >= 1, got {n_trials}")

    # Key the cache by (n, n_trials): keying by n alone silently returns a
    # low-resolution distribution when a later caller requests more trials.
    key = (n, n_trials)
    if key not in _RAYLEIGH_STAT_CACHE:
        # Simulate Rayleigh statistic (matching GWpy implementation)
        # In GWpy, rayleigh_spectrogram computes:
        # std(ASD_i) / (ASD_mean * sqrt( (4-pi)/pi ))
        # where ASD_mean is the mean of n ASDs.

        null_stats = np.zeros(n_trials)
        const = np.sqrt((4.0 - np.pi) / np.pi)

        for i in range(n_trials):
            # Rayleigh(sigma=1) samples; floor the uniform draw away from 0 so
            # log(0)=-inf cannot inject an Inf into the null distribution.
            u = np.clip(np.random.rand(n), np.finfo(float).tiny, 1.0)
            s = np.sqrt(-2.0 * np.log(u))
            # Rayleigh statistic R
            r = np.std(s) / (np.mean(s) * const)
            null_stats[i] = r

        _RAYLEIGH_STAT_CACHE[key] = np.sort(null_stats)

    return _RAYLEIGH_STAT_CACHE[key]
