"""gwexpy.statistics.rayleigh_test - Rayleigh statistic test p-values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from ..spectrogram import Spectrogram

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


def rayleigh_pvalue(
    rayleigh_spec: Spectrogram,
    n_samples: int,
    n_monte_carlo: int = 1000,
) -> Spectrogram:
    """
    Convert Rayleigh statistic spectrogram to p-value spectrogram.
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
    p_vals = np.clip(p_vals, 0.0, 1.0)
    
    return Spectrogram(
        p_vals,
        times=rayleigh_spec.times,
        frequencies=rayleigh_spec.frequencies,
        unit="",
        name=f"p-value({rayleigh_spec.name})",
    )


_RAYLEIGH_STAT_CACHE: dict[int, np.ndarray] = {}

def _get_rayleigh_stat_null_distribution(n: int, n_trials: int = 1000) -> np.ndarray:
    """
    Generate the null distribution of the Rayleigh statistic for n segments.
    """
    if n not in _RAYLEIGH_STAT_CACHE:
        # Simulate Rayleigh statistic (matching GWpy implementation)
        # In GWpy, rayleigh_spectrogram computes:
        # std(ASD_i) / (ASD_mean * sqrt( (4-pi)/pi ))
        # where ASD_mean is the mean of n ASDs.
        
        null_stats = np.zeros(n_trials)
        const = np.sqrt((4.0 - np.pi) / np.pi)
        
        for i in range(n_trials):
            # Rayleigh(sigma=1) samples
            s = np.sqrt(-2.0 * np.log(np.random.rand(n)))
            # Rayleigh statistic R
            r = np.std(s) / (np.mean(s) * const)
            null_stats[i] = r
            
        _RAYLEIGH_STAT_CACHE[n] = np.sort(null_stats)
        
    return _RAYLEIGH_STAT_CACHE[n]
