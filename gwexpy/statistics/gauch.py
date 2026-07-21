"""gwexpy.statistics.gauch - GauCh (Modified Kolmogorov-Smirnov test)."""
from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
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
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
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
    rng : numpy.random.Generator, optional
        Generator for the Lilliefors null distribution. See
        `_get_lilliefors_null_distribution` for the reproducibility
        contract; takes priority over `seed` when both are given.
    seed : int, optional
        Seed for the null distribution's Generator, used only when `rng`
        is not given.

    Returns
    -------
    GauChResult

    """
    if rng is not None and seed is not None:
        warnings.warn(
            "compute_gauch: both rng and seed were given; seed is "
            "ignored because rng takes priority",
            UserWarning,
            stacklevel=2,
        )

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
    # Get the (possibly cached) Lilliefors null distribution directly --
    # calling _get_rayleigh_lilliefors_pvalue() as a cache-populating side
    # effect and then reading _LILLIEFORS_CACHE[(window, n_monte_carlo)]
    # would silently pick up a stale/unrelated cache entry (or KeyError) once
    # rng/seed can bypass that cache (#464), so fetch the array directly.
    dist = _get_lilliefors_null_distribution(window, n_monte_carlo, rng=rng, seed=seed)

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

    metadata: dict[str, Any] = {"n_monte_carlo": n_monte_carlo}
    if rng is not None:
        metadata["rng_provided"] = True
        if seed is not None:
            metadata["seed_unused"] = True
    elif seed is not None:
        metadata["seed"] = seed

    return GauChResult(
        pvalue_map=res_p,
        statistic_map=res_s,
        n_samples=window,
        fftlength=fftlength,
        stride=stride,
        **metadata,
    )


_LILLIEFORS_CACHE: dict[tuple[int, int], np.ndarray] = {}
_LILLIEFORS_CACHE_LOCK = threading.Lock()


def _simulate_lilliefors_null(
    n: int, n_trials: int, draw_uniform: Callable[[int], np.ndarray]
) -> np.ndarray:
    """Draw `n_trials` Lilliefors Dn statistics under H0, each from `n` samples.

    `draw_uniform(size)` must return `size` uniform(0, 1) samples -- either a
    `numpy.random.Generator.random` bound method or the legacy
    `numpy.random.random` global-state function.
    """
    null_dns = np.zeros(n_trials)
    for i in range(n_trials):
        # Floor the uniform draw away from 0 so log(0)=-inf cannot inject
        # an Inf that permanently corrupts the cached null distribution.
        u = np.clip(draw_uniform(n), np.finfo(float).tiny, 1.0)
        null_sample = np.sqrt(-2.0 * np.log(u))
        s2_est = np.mean(null_sample**2) / 2.0
        sorted_null = np.sort(null_sample)
        null_ecdf = np.arange(1, n + 1) / n
        null_tcdf = 1.0 - np.exp(-(sorted_null**2) / (2.0 * s2_est))
        null_dns[i] = np.max(np.abs(null_ecdf - null_tcdf))
    return np.sort(null_dns)


def _get_lilliefors_null_distribution(
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate the Lilliefors null distribution of Dn for window size `n`.

    Reproducibility contract (#464), mirroring
    `gwexpy.statistics.rayleigh_test._get_rayleigh_stat_null_distribution`:

    - Neither `rng` nor `seed`: backward-compatible non-deterministic
      behavior, drawn from the legacy global `numpy.random` state (so
      `numpy.random.seed(...)` still controls it, as it always has) and
      cached by `(n, n_trials)` for reuse across calls (cache population is
      serialized, so this path is thread-safe).
    - `seed=`: reproducible across calls with the same `(n, n_trials, seed)`,
      via a dedicated `numpy.random.Generator` independent of the legacy
      global state.
    - `rng=`: follows the given Generator's own state.
    - Both given: `rng` is used and `seed` is ignored.

    Passing either `rng` or `seed` bypasses the shared cache.
    """
    # Entry-point guards mirroring rayleigh_test: an empty sample (n<=0) makes
    # np.mean/np.max return NaN and an empty trial set (n_trials<=0) makes the
    # later len(dist) division a silent 0/0 (issue #459).
    if n <= 0:
        raise ValueError(f"window/n must be >= 1, got {n}")
    if n_trials <= 0:
        raise ValueError(f"n_monte_carlo must be >= 1, got {n_trials}")

    if rng is not None or seed is not None:
        effective_rng = rng if rng is not None else np.random.default_rng(seed)
        return _simulate_lilliefors_null(n, n_trials, effective_rng.random)

    # Key the cache by (n, n_trials): keying by n alone silently returns a
    # low-resolution distribution when a later caller requests more trials,
    # discarding their explicit n_trials (issue #459).
    key = (n, n_trials)
    with _LILLIEFORS_CACHE_LOCK:
        if key not in _LILLIEFORS_CACHE:
            # Legacy global state, not a fresh Generator: preserves the
            # pre-#464 contract that `numpy.random.seed(...)` controls the
            # no-args path.
            _LILLIEFORS_CACHE[key] = _simulate_lilliefors_null(
                n, n_trials, np.random.random
            )
        return _LILLIEFORS_CACHE[key]


def _get_rayleigh_lilliefors_pvalue(
    dn: float,
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> float:
    """Get the p-value using the (possibly cached) Lilliefors null distribution."""
    dist = _get_lilliefors_null_distribution(n, n_trials, rng=rng, seed=seed)
    # Floor at 1/len(dist): a Monte-Carlo p-value of exactly 0 is invalid.
    return float(max(np.sum(dist >= dn) / len(dist), 1.0 / len(dist)))
