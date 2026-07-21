"""gwexpy.statistics.rayleigh_test - Rayleigh statistic test p-values."""
from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
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
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
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
    rng : numpy.random.Generator, optional
        Generator for the Monte Carlo null distribution. See
        `_get_rayleigh_stat_null_distribution` for the reproducibility
        contract; takes priority over `seed` when both are given.
    seed : int, optional
        Seed for the null distribution's Generator, used only when `rng`
        is not given.

    Returns
    -------
    Spectrogram (p-values)
        ``n_monte_carlo`` (and ``seed`` or ``rng_provided``/``seed_unused``,
        as applicable) are attached as attributes on the returned instance
        for provenance. These are one-off analysis attributes local to this
        instance, not persisted metadata: they do not survive `.copy()`,
        slicing, arithmetic, or serialization.

    """
    if rng is not None and seed is not None:
        warnings.warn(
            "rayleigh_pvalue: both rng and seed were given; seed is "
            "ignored because rng takes priority",
            UserWarning,
            stacklevel=2,
        )

    # 1. Get background distribution of Rayleigh statistic for n_samples
    dist = _get_rayleigh_stat_null_distribution(
        n_samples, n_monte_carlo, rng=rng, seed=seed
    )

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

    result = Spectrogram(
        p_vals,
        times=rayleigh_spec.times,
        frequencies=rayleigh_spec.frequencies,
        unit="",
        name=f"p-value({rayleigh_spec.name})",
    )
    result.n_monte_carlo = n_monte_carlo
    if rng is not None:
        result.rng_provided = True
        if seed is not None:
            result.seed_unused = True
    elif seed is not None:
        result.seed = seed
    return result


_RAYLEIGH_STAT_CACHE: dict[tuple[int, int], np.ndarray] = {}
_RAYLEIGH_STAT_CACHE_LOCK = threading.Lock()


def _simulate_rayleigh_null(
    n: int, n_trials: int, draw_uniform: Callable[[int], np.ndarray]
) -> np.ndarray:
    """Draw `n_trials` Rayleigh statistics, each from `n` Rayleigh(sigma=1) samples.

    `draw_uniform(size)` must return `size` uniform(0, 1) samples -- either a
    `numpy.random.Generator.random` bound method or the legacy
    `numpy.random.random` global-state function, so the caller controls
    whether draws come from an explicit Generator or the seedable global
    state.
    """
    null_stats = np.zeros(n_trials)
    const = np.sqrt((4.0 - np.pi) / np.pi)

    for i in range(n_trials):
        # Rayleigh(sigma=1) samples; floor the uniform draw away from 0 so
        # log(0)=-inf cannot inject an Inf into the null distribution.
        u = np.clip(draw_uniform(n), np.finfo(float).tiny, 1.0)
        s = np.sqrt(-2.0 * np.log(u))
        # Rayleigh statistic R
        r = np.std(s) / (np.mean(s) * const)
        null_stats[i] = r

    return np.sort(null_stats)


def _get_rayleigh_stat_null_distribution(
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate the null distribution of the Rayleigh statistic for `n` segments.

    Reproducibility contract (#464):

    - Neither `rng` nor `seed`: backward-compatible non-deterministic
      behavior, drawn from the legacy global `numpy.random` state (so
      `numpy.random.seed(...)` still controls it, as it always has) and
      cached by `(n, n_trials)` for reuse across calls (cache population is
      serialized, so this path is thread-safe).
    - `seed=`: reproducible across calls with the same `(n, n_trials, seed)`,
      via a dedicated `numpy.random.Generator` independent of the legacy
      global state.
    - `rng=`: follows the given Generator's own state (not reproducible by
      itself unless the caller seeds and re-supplies the Generator).
    - Both given: `rng` is used and `seed` is ignored.

    Passing either `rng` or `seed` bypasses the shared cache -- a
    cached-by-`(n, n_trials)` entry could otherwise silently hand back a
    distribution drawn under a different (or no) seed than the one just
    requested.
    """
    # Entry-point guards: an empty sample (n<=0) makes np.std/np.mean return NaN
    # -> all-NaN distribution -> p=0 for every bin; an empty trial set
    # (n_trials<=0) makes the later len(dist) division a silent 0/0 -> all-NaN
    # p-values (issue #459).
    if n <= 0:
        raise ValueError(f"n_samples must be >= 1, got {n}")
    if n_trials <= 0:
        raise ValueError(f"n_monte_carlo must be >= 1, got {n_trials}")

    if rng is not None or seed is not None:
        effective_rng = rng if rng is not None else np.random.default_rng(seed)
        return _simulate_rayleigh_null(n, n_trials, effective_rng.random)

    # Key the cache by (n, n_trials): keying by n alone silently returns a
    # low-resolution distribution when a later caller requests more trials.
    key = (n, n_trials)
    with _RAYLEIGH_STAT_CACHE_LOCK:
        if key not in _RAYLEIGH_STAT_CACHE:
            # Legacy global state, not a fresh Generator: preserves the
            # pre-#464 contract that `numpy.random.seed(...)` controls the
            # no-args path.
            _RAYLEIGH_STAT_CACHE[key] = _simulate_rayleigh_null(
                n, n_trials, np.random.random
            )
        return _RAYLEIGH_STAT_CACHE[key]
