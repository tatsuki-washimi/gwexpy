"""gwexpy.statistics.gauch - GauCh (Modified Kolmogorov-Smirnov test)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..spectrogram import Spectrogram

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


class GauChResult:
    """Result of the GauCh test."""

    def __init__(
        self,
        pvalue_map: Spectrogram,
        statistic_map: Spectrogram,
        n_samples: int,
        **metadata: Any,
    ):
        self.pvalue_map = pvalue_map
        self.statistic_map = statistic_map
        self.n_samples = n_samples
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<GauChResult n_samples={self.n_samples}>"


def compute_gauch(
    ts: TimeSeries,
    fftlength: float,
    stride: float | None = None,
    window: int = 40,
    overlap: float | None = None,
    n_monte_carlo: int = 1000,
) -> GauChResult:
    """Compute GauCh (Modified KS test) for the given TimeSeries using a sliding window.

    Parameters
    ----------
    ts : TimeSeries
        Input time series for the GauCh analysis.
    fftlength : float
        Length of each FFT segment in seconds.
    stride : float, optional
        Step between FFT segments in seconds.
    window : int, default=40
        Number of segments used for each GauCh test.
    overlap : float, optional
        Overlap between FFT segments in seconds (alternative to stride).
    n_monte_carlo : int, default=1000
        Number of Monte Carlo trials for background distribution.

    Returns
    -------
    GauChResult

    """
    if stride is None:
        if overlap is None:
            stride = fftlength
        else:
            stride = fftlength - overlap

    # 1. Compute Spectrogram
    # The spectrogram will have many segments.
    spec = ts.spectrogram(fftlength=fftlength, stride=stride)
    asds = np.sqrt(spec.value)  # (n_times, n_freqs)

    n_times, n_freqs = asds.shape
    if n_times < window:
        raise ValueError(f"Too few segments ({n_times}) for window size {window}.")

    # 2. Sliding Window GauCh
    # Output will have (n_times - window + 1) time steps.
    n_out = n_times - window + 1
    statistic_map = np.zeros((n_out, n_freqs))

    for i in range(n_out):
        window_asds = asds[i : i + window, :] # (window, n_freqs)

        # Vectorized sigma2 estimation across frequencies
        sigma2 = np.mean(window_asds**2, axis=0) / 2.0

        # A dead bin (DC, Nyquist, a notched line, or a zero test signal) has
        # sigma2==0, which would turn the tcdf below into a silent 0/0 (NaN) or
        # x/0 path. Mark such bins NaN explicitly so they propagate to a NaN
        # statistic / p-value rather than a misleading number (issue #459).
        sigma2 = np.where(sigma2 > 0, sigma2, np.nan)

        # Sort along the temporal axis for each frequency bin
        sorted_window = np.sort(window_asds, axis=0) # (window, n_freqs)

        # Vectorized ECDF and TCDF calculation
        # ecdf is (window, 1) broadcasted to (window, n_freqs)
        ecdf = (np.arange(1, window + 1) / window)[:, np.newaxis]

        # tcdf = 1 - exp(-x^2 / 2sigma^2)
        # sorted_window is (window, n_freqs), sigma2 is (n_freqs,)
        with np.errstate(invalid="ignore"):
            tcdf = 1.0 - np.exp(-(sorted_window**2) / (2.0 * sigma2))

        # statistic is max|ecdf - tcdf| along temporal axis
        dn = np.max(np.abs(ecdf - tcdf), axis=0) # (n_freqs,)
        statistic_map[i, :] = dn

    # 3. Vectorized p-value calculation
    # Pre-cache Lilliefors distribution for 'window'
    _get_rayleigh_lilliefors_pvalue(0.1, window, n_monte_carlo)
    dist = _LILLIEFORS_CACHE[(window, n_monte_carlo)]

    # p-value is (count of dist >= dn) / len(dist)
    # searchsorted returns index where dn would be inserted to maintain order.
    # idx = number of elements < dn.
    # so count(dist >= dn) = len(dist) - idx.
    # Floor at 1/len(dist): a Monte-Carlo p-value of exactly 0 is invalid and
    # makes -log10(p)=Inf downstream (issue #459).
    indices = np.searchsorted(dist, statistic_map)
    pvalue_map = np.maximum((len(dist) - indices) / len(dist), 1.0 / len(dist))

    # A dead bin produced a NaN statistic above; searchsorted would map it to a
    # spurious p-value, so restore NaN explicitly for those bins.
    nonfinite = ~np.isfinite(statistic_map)
    if np.any(nonfinite):
        pvalue_map[nonfinite] = np.nan

    # 3. Create Resulting Spectrograms
    # Adjust times for the output maps (center of the window)
    out_times = spec.times[window // 2 : window // 2 + n_out]

    res_p = Spectrogram(pvalue_map, frequencies=spec.frequencies, times=out_times)
    res_s = Spectrogram(statistic_map, frequencies=spec.frequencies, times=out_times)

    return GauChResult(
        pvalue_map=res_p,
        statistic_map=res_s,
        n_samples=window,
        fftlength=fftlength,
        stride=stride,
    )


_LILLIEFORS_CACHE: dict[tuple[int, int], np.ndarray] = {}

def _get_rayleigh_lilliefors_pvalue(dn: float, n: int, n_trials: int = 1000) -> float:
    """Get the p-value and manage the cached null distribution."""
    # Entry-point guards mirroring rayleigh_test: an empty sample (n<=0) makes
    # np.mean/np.max return NaN and an empty trial set (n_trials<=0) makes the
    # later len(dist) division a silent 0/0 (issue #459).
    if n <= 0:
        raise ValueError(f"window/n must be >= 1, got {n}")
    if n_trials <= 0:
        raise ValueError(f"n_monte_carlo must be >= 1, got {n_trials}")

    # Key the cache by (n, n_trials): keying by n alone silently returns a
    # low-resolution distribution when a later caller requests more trials,
    # discarding their explicit n_trials (issue #459).
    key = (n, n_trials)
    if key not in _LILLIEFORS_CACHE:
        # Generate distribution of Dn under H0
        null_dns = np.zeros(n_trials)
        for i in range(n_trials):
            # Floor the uniform draw away from 0 so log(0)=-inf cannot inject
            # an Inf that permanently corrupts the cached null distribution.
            u = np.clip(np.random.rand(n), np.finfo(float).tiny, 1.0)
            null_sample = np.sqrt(-2.0 * np.log(u))
            s2_est = np.mean(null_sample**2) / 2.0
            sorted_null = np.sort(null_sample)
            null_ecdf = np.arange(1, n + 1) / n
            null_tcdf = 1.0 - np.exp(-(sorted_null**2) / (2.0 * s2_est))
            null_dns[i] = np.max(np.abs(null_ecdf - null_tcdf))
        _LILLIEFORS_CACHE[key] = np.sort(null_dns)

    dist = _LILLIEFORS_CACHE[key]
    # Floor at 1/len(dist): a Monte-Carlo p-value of exactly 0 is invalid.
    return float(max(np.sum(dist >= dn) / len(dist), 1.0 / len(dist)))
