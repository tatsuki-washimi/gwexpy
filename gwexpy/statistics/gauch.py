"""gwexpy.statistics.gauch - GauCh (Gaussianity check by a KS test on ASD samples)."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from scipy import signal, special
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.statistics. Install with: pip install scipy"
    ) from _exc

from ..spectrogram import Spectrogram
from ..spectrogram.provenance import analysis_provenance
from ._monte_carlo import upper_tail_rank_pvalue

if TYPE_CHECKING:
    from ..timeseries import TimeSeries

_RAYLEIGH = "rayleigh"
_HALFNORMAL = "halfnormal"


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


def _segment_asds(
    data: np.ndarray, sample_rate: float, nfft: int, hop: int
) -> np.ndarray:
    """One Hann-windowed ASD per segment, without averaging.

    Parameters
    ----------
    data : numpy.ndarray
        Real time-domain samples.
    sample_rate : float
        Sampling rate in Hz.
    nfft : int
        Segment length in samples.
    hop : int
        Step between segment starts in samples.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_segments, nfft // 2 + 1)``, in input units per ``Hz**0.5``.
        Each row is the one-sided density ``sqrt(2 |X(f)|**2 / (fs * sum(w**2)))``
        of one segment after removing its mean, i.e. what
        ``TimeSeries.asd(fftlength, window="hann")`` returns for that segment
        alone. The DC and Nyquist rows are not doubled.

    """
    n_segments = (len(data) - nfft) // hop + 1
    if n_segments < 1:
        return np.empty((0, nfft // 2 + 1))
    index = np.arange(nfft)[np.newaxis, :] + hop * np.arange(n_segments)[:, np.newaxis]
    _, psd = signal.periodogram(
        np.asarray(data, dtype=float)[index],
        fs=sample_rate,
        window="hann",
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    return np.sqrt(psd)


def _ks_statistic(samples: np.ndarray, real_only: np.ndarray) -> np.ndarray:
    """Two-sided KS statistic per column, with the scale fitted per column.

    Parameters
    ----------
    samples : numpy.ndarray
        Amplitude samples, shape ``(n, n_columns)``.
    real_only : numpy.ndarray
        Boolean mask of length ``n_columns``. Columns flagged here are tested
        against a half-normal amplitude (``sigma**2 = mean(x**2)``), the
        others against a Rayleigh amplitude (``sigma**2 = mean(x**2) / 2``).

    Returns
    -------
    numpy.ndarray
        Dimensionless ``D = max(D+, D-)`` per column, where
        ``D+ = max_i(i/n - F(x_(i)))`` and ``D- = max_i(F(x_(i)) - (i-1)/n)``.
        A column whose mean square is not positive gives NaN (issue #459).

    """
    n = samples.shape[0]
    ordered = np.sort(samples, axis=0)
    mean_square = np.mean(samples**2, axis=0)
    mean_square = np.where(mean_square > 0, mean_square, np.nan)

    with np.errstate(invalid="ignore"):
        cdf = 1.0 - np.exp(-(ordered**2) / mean_square)
        if np.any(real_only):
            cdf[:, real_only] = special.erf(
                ordered[:, real_only] / np.sqrt(2.0 * mean_square[real_only])
            )

    rank = np.arange(1, n + 1)[:, np.newaxis]
    d_plus = np.max(rank / n - cdf, axis=0)
    d_minus = np.max(cdf - (rank - 1) / n, axis=0)
    # np.max propagates the NaN of a dead column; np.maximum keeps it.
    return np.maximum(d_plus, d_minus)


def _resolve_hop(
    fftlength: float, stride: float | None, overlap: float | None, sample_rate: float
) -> tuple[float, int, int]:
    """Return ``(stride_seconds, nfft, hop)`` and reject uncalibrated overlaps."""
    if stride is None:
        stride = fftlength if overlap is None else fftlength - overlap
    nfft = int(round(fftlength * sample_rate))
    hop = int(round(stride * sample_rate))
    if nfft < 2:
        raise ValueError(f"fftlength must span at least 2 samples, got {nfft}.")
    if hop < 1:
        raise ValueError(f"stride must span at least 1 sample, got {hop}.")
    # The null distribution treats the ASD samples of one bin as independent.
    # That holds for disjoint segments, and was measured to hold for Hann
    # segments at 50% overlap (#879). Other overlaps are not calibrated.
    # For odd nfft, either nearest integer hop represents half a segment.
    # Even nfft requires exactly nfft // 2; a +/-1 hop is not 50%.
    if hop < nfft and abs(2 * hop - nfft) > 1:
        raise ValueError(
            "compute_gauch: segment overlap must be 0 or 50% of fftlength "
            f"(stride {fftlength / 2:g} s or >= {fftlength:g} s), got an overlap "
            f"of {(nfft - hop) / sample_rate:g} s. The null distribution is not "
            "calibrated for other overlaps."
        )
    return stride, nfft, hop


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
    """Compute GauCh, a KS test of Gaussianity on ASD samples, in a sliding window.

    Parameters
    ----------
    ts : TimeSeries
        Input time series for the GauCh analysis.
    fftlength : float
        Length of each FFT segment in seconds.
    stride : float, optional
        Step between FFT segment starts in seconds. Defaults to
        ``fftlength - overlap``, or to `fftlength` when `overlap` is not
        given. Must be `fftlength` or more, or half of `fftlength`.
    window : int, default=40
        Number of consecutive ASD samples used for each GauCh test.
    overlap : float, optional
        Overlap between FFT segments in seconds (alternative to stride).
        Must be 0 or half of `fftlength`. Ignored when `stride` is given.
    n_monte_carlo : int, default=1000
        Number of Monte Carlo trials for background distribution.
    rng : numpy.random.Generator, optional
        Generator for the null distributions. See
        `_get_lilliefors_null_distribution` for the reproducibility
        contract; takes priority over `seed` when both are given.
    seed : int, optional
        Seed for the null distributions' Generators, used only when `rng`
        is not given.

    Returns
    -------
    GauChResult
        ``pvalue_map`` and ``statistic_map`` are dimensionless
        `Spectrogram` objects. Row ``i`` covers ASD samples ``i`` to
        ``i + window - 1`` and is stamped with the start time of sample
        ``i + window // 2``; the time step is the stride.

    Raises
    ------
    ValueError
        If the segments overlap by anything other than 0 or 50%, or if the
        series holds fewer than `window` segments.

    Notes
    -----
    Each ASD sample is one Hann-windowed FFT segment with its mean removed;
    no periodograms are averaged. For Gaussian noise the amplitude of an
    interior frequency bin is Rayleigh distributed. The DC bin and, for an
    even FFT length in samples, the Nyquist bin have a purely real FFT
    coefficient, so their amplitude is half-normal. An odd FFT length has no
    Nyquist bin and its last bin is an interior bin.

    Per bin, the scale is fitted from the window (``sigma**2 = mean(x**2)/2``
    for Rayleigh, ``mean(x**2)`` for half-normal) and the standard two-sided
    Kolmogorov-Smirnov statistic ``D = max(D+, D-)`` is taken against the
    fitted distribution. Fitting the scale changes the null distribution of
    ``D``, so it is simulated: independent amplitude samples, the same
    statistic, the scale refitted per trial. The p-value is one-sided,
    ``p = (r + 1) / (N + 1)`` with ``N = n_monte_carlo`` and ``r`` the number
    of simulated statistics ``>=`` the observed one (#507).

    With ``overlap=fftlength / 2`` and ``window=2 * interval / fftlength - 1``
    this is the configuration of the KAGRA GauCh analysis.

    Through v0.2.9 the ASD samples came from ``TimeSeries.spectrogram()``,
    which averages periodograms, the statistic used only the right ECDF
    endpoints, every bin used the Rayleigh model, and the p-value was
    ``max(r / N, 1 / N)``. On white Gaussian noise that rejected about 57%
    of interior bins at ``alpha = 0.05`` (#879). All outputs changed.

    Multiple testing over frequency bins is not corrected here.

    """
    if rng is not None and seed is not None:
        warnings.warn(
            "compute_gauch: both rng and seed were given; seed is "
            "ignored because rng takes priority",
            UserWarning,
            stacklevel=2,
        )

    stride_was_provided = stride is not None
    sample_rate = float(ts.sample_rate.to_value("Hz"))
    stride, nfft, hop = _resolve_hop(fftlength, stride, overlap, sample_rate)

    # 1. One ASD sample per FFT segment: (n_times, n_freqs)
    asds = _segment_asds(np.asarray(ts.value), sample_rate, nfft, hop)
    n_times, n_freqs = asds.shape
    if n_times < window:
        raise ValueError(f"Too few segments ({n_times}) for window size {window}.")

    # DC always, Nyquist only for an even FFT length.
    real_only = np.zeros(n_freqs, dtype=bool)
    real_only[0] = True
    if nfft % 2 == 0:
        real_only[-1] = True

    # 2. Sliding-window statistic: (n_times - window + 1, n_freqs)
    n_out = n_times - window + 1
    statistic_map = np.empty((n_out, n_freqs))
    for i in range(n_out):
        statistic_map[i, :] = _ks_statistic(asds[i : i + window, :], real_only)

    # 3. p-values against the null of the matching amplitude model
    pvalue_map = np.empty_like(statistic_map)
    rayleigh_null = _get_lilliefors_null_distribution(
        window, n_monte_carlo, rng=rng, seed=seed
    )
    pvalue_map[:, ~real_only] = upper_tail_rank_pvalue(
        rayleigh_null, statistic_map[:, ~real_only]
    )
    halfnormal_null = _get_halfnormal_null_distribution(
        window, n_monte_carlo, rng=rng, seed=seed
    )
    pvalue_map[:, real_only] = upper_tail_rank_pvalue(
        halfnormal_null, statistic_map[:, real_only]
    )

    # A dead bin has a NaN statistic; searchsorted would map it to a spurious
    # p-value, so restore NaN explicitly for those bins.
    nonfinite = ~np.isfinite(statistic_map)
    if np.any(nonfinite):
        pvalue_map[nonfinite] = np.nan

    # 4. Output axes: segment start times, stamped at the window centre
    time_unit = ts.times.unit
    start = float(ts.t0.to_value(time_unit))
    step = (hop / sample_rate * ts.dt.unit).to_value(time_unit)
    out_times = (start + step * (window // 2 + np.arange(n_out))) * time_unit
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate) * ts.sample_rate.unit

    res_p = Spectrogram(pvalue_map, frequencies=frequencies, times=out_times)
    res_s = Spectrogram(statistic_map, frequencies=frequencies, times=out_times)

    provenance_parameters: dict[str, Any] = {
        "fftlength": fftlength,
        "stride": stride,
        "window": window,
        "n_monte_carlo": n_monte_carlo,
    }
    if stride_was_provided and overlap is not None:
        provenance_parameters["overlap_ignored"] = True
    else:
        provenance_parameters["overlap"] = overlap
    provenance = analysis_provenance(
        "compute_gauch",
        provenance_parameters,
        seed=seed,
        rng_provided=rng is not None,
        seed_unused=rng is not None and seed is not None,
    )
    res_p.provenance = provenance
    res_s.provenance = provenance

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
_HALFNORMAL_CACHE: dict[tuple[int, int], np.ndarray] = {}
_LILLIEFORS_CACHE_LOCK = threading.Lock()

# Stream label that keeps the seeded half-normal null independent of the
# seeded Rayleigh null, which uses ``default_rng(seed)`` directly.
_HALFNORMAL_STREAM = 1


def _simulate_lilliefors_null(
    n: int,
    n_trials: int,
    draw_uniform: Callable[[int], np.ndarray],
    model: str = _RAYLEIGH,
) -> np.ndarray:
    """Draw `n_trials` two-sided KS statistics under H0, each from `n` samples.

    `draw_uniform(size)` must return `size` uniform(0, 1) samples -- either a
    `numpy.random.Generator.random` bound method or the legacy
    `numpy.random.random` global-state function. The scale is refitted for
    every trial, as it is for the observed statistic.
    """
    real_only = np.array([model == _HALFNORMAL])
    null_dns = np.zeros(n_trials)
    for i in range(n_trials):
        # Floor the uniform draw away from 0 so log(0)=-inf cannot inject
        # an Inf that permanently corrupts the cached null distribution.
        u = np.clip(draw_uniform(n), np.finfo(float).tiny, 1.0)
        if model == _HALFNORMAL:
            null_sample = np.sqrt(2.0) * special.erfinv(u)
        else:
            null_sample = np.sqrt(-2.0 * np.log(u))
        null_dns[i] = _ks_statistic(null_sample[:, np.newaxis], real_only)[0]
    return np.sort(null_dns)


def _validate_null_request(n: int, n_trials: int) -> None:
    # Entry-point guards mirroring rayleigh_test: an empty sample (n<=0) makes
    # np.mean/np.max return NaN and an empty trial set (n_trials<=0) makes the
    # later len(dist) division a silent 0/0 (issue #459).
    if n <= 0:
        raise ValueError(f"window/n must be >= 1, got {n}")
    if n_trials <= 0:
        raise ValueError(f"n_monte_carlo must be >= 1, got {n_trials}")


def _cached_null(
    cache: dict[tuple[int, int], np.ndarray], n: int, n_trials: int, model: str
) -> np.ndarray:
    # Key the cache by (n, n_trials): keying by n alone silently returns a
    # low-resolution distribution when a later caller requests more trials,
    # discarding their explicit n_trials (issue #459).
    key = (n, n_trials)
    with _LILLIEFORS_CACHE_LOCK:
        if key not in cache:
            # Legacy global state, not a fresh Generator: preserves the
            # pre-#464 contract that `numpy.random.seed(...)` controls the
            # no-args path.
            cache[key] = _simulate_lilliefors_null(n, n_trials, np.random.random, model)
        return cache[key]


def _get_lilliefors_null_distribution(
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate the null distribution of the Rayleigh-model statistic for size `n`.

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
    _validate_null_request(n, n_trials)

    if rng is not None or seed is not None:
        effective_rng = rng if rng is not None else np.random.default_rng(seed)
        return _simulate_lilliefors_null(n, n_trials, effective_rng.random)

    return _cached_null(_LILLIEFORS_CACHE, n, n_trials, _RAYLEIGH)


def _get_halfnormal_null_distribution(
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate the null distribution of the half-normal-model statistic.

    Used for the DC and Nyquist bins. Follows the reproducibility contract of
    `_get_lilliefors_null_distribution`, with its own cache. With `seed=` it
    uses a Generator stream separate from the Rayleigh null's, so the two
    nulls are independent; with `rng=` it continues the given Generator.
    """
    _validate_null_request(n, n_trials)

    if rng is not None:
        return _simulate_lilliefors_null(n, n_trials, rng.random, _HALFNORMAL)
    if seed is not None:
        stream = np.random.default_rng([_HALFNORMAL_STREAM, seed])
        return _simulate_lilliefors_null(n, n_trials, stream.random, _HALFNORMAL)

    return _cached_null(_HALFNORMAL_CACHE, n, n_trials, _HALFNORMAL)


def _get_rayleigh_lilliefors_pvalue(
    dn: float,
    n: int,
    n_trials: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> float:
    """Get the Rayleigh-model p-value ``(r + 1) / (N + 1)`` for statistic `dn`."""
    dist = _get_lilliefors_null_distribution(n, n_trials, rng=rng, seed=seed)
    return float(upper_tail_rank_pvalue(dist, dn))
