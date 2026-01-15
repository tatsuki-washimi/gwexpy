"""
Spectral estimation helpers for PSD and bootstrap statistics.

PSD estimation in this module delegates to GWpy's Welch-style estimator with
one-sided density normalization. For a regularly sampled series x[n] with
sample spacing dt, the PSD Sxx(f) is defined so that the variance satisfies
var(x) ~= integral_0^{f_N} Sxx(f) df (Parseval). The frequency axis is uniquely
determined by dt and the FFT length: df = 1 / fftlength and f_N = 1 / (2 * dt).

NaN samples are rejected because FFT-based averaging propagates NaNs and
invalidates the normalization; callers must pre-clean data instead.
"""

import os
import warnings

import numpy as np
from scipy.signal import get_window

from gwexpy.frequencyseries import FrequencySeries

# Optional Numba import
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range

    # Create a dummy njit decorator that just returns the function
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(parallel=True)
def _bootstrap_resample_jit(data, all_indices, use_median, ignore_nan):
    n_boot = all_indices.shape[0]
    n_freq = data.shape[1]
    resampled_stats = np.zeros((n_boot, n_freq), dtype=data.dtype)

    for i in prange(n_boot):
        indices = all_indices[i]

        # Iterate over frequency bins to save memory (avoid creating full (T, F) sample)
        for f in range(n_freq):
            # Extract column using fancy indexing
            col = data[indices, f]

            if use_median:
                if ignore_nan:
                    resampled_stats[i, f] = np.nanmedian(col)
                else:
                    resampled_stats[i, f] = np.median(col)
            else:  # mean
                if ignore_nan:
                    resampled_stats[i, f] = np.nanmean(col)
                else:
                    resampled_stats[i, f] = np.mean(col)
    return resampled_stats


def _bootstrap_resample_py(data, all_indices, use_median, ignore_nan):
    n_boot = all_indices.shape[0]
    n_freq = data.shape[1]
    resampled_stats = np.zeros((n_boot, n_freq), dtype=data.dtype)

    for i in range(n_boot):
        indices = all_indices[i]
        for f in range(n_freq):
            col = data[indices, f]
            if use_median:
                resampled_stats[i, f] = (
                    np.nanmedian(col) if ignore_nan else np.median(col)
                )
            else:
                resampled_stats[i, f] = np.nanmean(col) if ignore_nan else np.mean(col)
    return resampled_stats


def estimate_psd(
    timeseries,
    *,
    fftlength=None,
    overlap=None,
    window="hann",
    method="median",
    **kwargs,
):
    """
    Estimate the one-sided PSD for a regular TimeSeries.

    This wraps ``TimeSeries.psd`` without changing the underlying algorithm.
    It enforces NaN-free input and checks that fftlength does not exceed the
    data duration so that the frequency axis (df = 1 / fftlength) is well-defined.

    Parameters
    ----------
    timeseries : gwexpy.timeseries.TimeSeries
        Input series for PSD estimation.
    fftlength : float or Quantity, optional
        FFT length in seconds. Defines df = 1 / fftlength.
    overlap : float or Quantity, optional
        Overlap between segments in seconds.
    window : str or array_like, optional
        Window function name or samples.
    method : str, optional
        PSD estimation method name registered in GWpy.

    Returns
    -------
    FrequencySeries
        One-sided PSD with unit (input unit)**2 / Hz.
    """
    data = np.asarray(getattr(timeseries, "value", timeseries))
    if np.isnan(data).any():
        raise ValueError("estimate_psd does not allow NaN samples.")

    def _to_seconds(value):
        if value is None:
            return None
        if hasattr(value, "to"):
            try:
                return float(value.to("s").value)
            except Exception:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    dt_sec = _to_seconds(getattr(timeseries, "dt", None))
    fftlength_sec = _to_seconds(fftlength)
    if dt_sec is not None and fftlength_sec is not None:
        duration = dt_sec * len(timeseries)
        if duration < fftlength_sec:
            raise ValueError("fftlength must not exceed data duration.")

    res = timeseries.psd(
        fftlength=fftlength,
        overlap=overlap,
        window=window,
        method=method,
        **kwargs,
    )
    if isinstance(res, FrequencySeries):
        return res
    return res.view(FrequencySeries)


def calculate_correlation_factor(window, nperseg, noverlap, n_blocks):
    """
    Calculate the variance inflation factor for Welch's method with overlap.

    This computes the correction factor by numerically calculating the normalized
    squared autocorrelation of the window function. The correction accounts for
    reduced effective degrees of freedom due to correlated (overlapping) segments.

    References:
    - Bendat, J. S., & Piersol, A. G., "Random Data".
    - Percival, D. B., & Walden, A. T., "Spectral Analysis for Physical Applications".
    - Ingram, A. (2019), "Error formulae for the energy-dependent cross-spectrum".

    Formula::

        factor = sqrt(1 + 2 * sum_{k=1}^{M-1} (1 - k/M) * abs(rho_window(k * S))**2)

    where rho_window is the normalized autocorrelation of the window.

    Parameters
    ----------
    window : str, tuple, or array_like
        Window function used for FFT.
    nperseg : int
        Length of each segment (N_fft).
    noverlap : int
        Number of overlapping samples.
    n_blocks : int
        Number of segments available for averaging.

    Returns
    -------
    float
        Multiplicative correction factor for the standard error.
    """
    if window is None or nperseg is None or noverlap is None or n_blocks <= 1:
        return 1.0

    try:
        if isinstance(window, (list, np.ndarray)):
            win_array = np.asarray(window)
            if len(win_array) != nperseg:
                return 1.0
        else:
            win_array = get_window(window, nperseg)
    except Exception:
        warnings.warn(
            f"Could not generate window '{window}'. Assuming independent segments (factor=1.0)."
        )
        return 1.0

    step = nperseg - noverlap
    if step <= 0:
        return 1.0

    energy = np.sum(win_array**2)
    if energy == 0:
        return 1.0

    rho_sq_weighted_sum = 0.0
    max_lag = int(np.ceil(nperseg / step))

    for k in range(1, min(n_blocks, max_lag)):
        shift = k * step
        if shift >= nperseg:
            break

        autocorr = np.sum(win_array[:-shift] * win_array[shift:])
        rho = autocorr / energy
        weight = 1.0 - (k / n_blocks)
        rho_sq_weighted_sum += weight * (rho**2)

    vif = 1.0 + 2.0 * rho_sq_weighted_sum
    return np.sqrt(vif)


def _infer_overlap_ratio(spectrogram):
    """Infer overlap ratio from spectrogram metadata."""
    stride = getattr(spectrogram, "dt", None)
    resolution = getattr(spectrogram, "df", None)
    if stride is None or resolution is None:
        return None
    try:
        stride_val = stride.value
        duration = 1.0 / resolution.value
    except Exception:
        return None
    if stride_val <= 0 or duration <= 0:
        return None
    return duration / stride_val


def bootstrap_spectrogram(
    spectrogram,
    n_boot=1000,
    method="median",
    average=None,
    ci=0.68,
    window="hann",
    nperseg=None,
    noverlap=None,
    block_size=None,
    rebin_width=None,
    return_map=False,
    ignore_nan=True,
):
    """
    Estimate robust ASD from a spectrogram using bootstrap resampling.

    Error bars are corrected for correlation between overlapping segments
    based on the window function's autocorrelation.

    Performance Note:
    This function utilizes Numba for JIT compilation and parallelization
    if available, significantly accelerating the resampling process.

    Parameters
    ----------
    spectrogram : gwpy.spectrogram.Spectrogram
    n_boot : int
    method : str
        'median' (default) or 'mean'.
    average : str, optional
        Alias for method ('median' or 'mean') for compatibility.
    ci : float
    window : str or array, optional
    nperseg : int, optional
    noverlap : int, optional
    block_size : int, optional
        Size of blocks for block bootstrap. If None, perform standard bootstrap.
    rebin_width : float, optional
        Width to rebin frequencies (in Hz) before bootstrapping.
    return_map : bool, optional
        If True, return a BifrequencyMap of the covariance.
    ignore_nan : bool, optional
        If True, ignore NaNs during bootstrap averaging. Default is True.

    Returns
    -------
    FrequencySeries or (FrequencySeries, BifrequencyMap)
    """
    data = spectrogram.value
    frequencies = spectrogram.frequencies.value
    n_time = data.shape[0]

    if n_time < 2:
        raise ValueError("Spectrogram must have at least 2 time bins.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1.")

    if average is not None:
        method = average

    avg = method.lower()
    if avg not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    # 1. Frequency Rebinning
    if rebin_width is not None and rebin_width > 0:
        df = (
            spectrogram.df.value if hasattr(spectrogram.df, "value") else spectrogram.df
        )
        bin_size = int(rebin_width / df)

        if bin_size > 1:
            n_freq = data.shape[1]
            # Truncate to multiple of bin_size
            n_freq_new = n_freq // bin_size
            if n_freq_new * bin_size != n_freq:
                data = data[:, : n_freq_new * bin_size]
                frequencies = frequencies[: n_freq_new * bin_size]

            # Rebin: reshape and mean
            # (Time, Freq) -> (Time, FreqNew, BinSize) -> mean(axis=2)
            data = data.reshape(n_time, n_freq_new, bin_size)
            if ignore_nan:
                data = np.nanmean(data, axis=2)
            else:
                data = np.mean(data, axis=2)

            # Update frequency axis (centers)
            frequencies = frequencies.reshape(n_freq_new, bin_size)
            frequencies = np.mean(frequencies, axis=1)

    # Update n_freq after potentially rebinning/truncating
    n_freq = data.shape[1]

    use_median = avg == "median"

    # 2. Block Bootstrap
    if block_size is not None and block_size > 1:
        if block_size >= n_time:
            # Just one block? or simple bootstrap of blocks?
            # If block size is entire duration, only 1 possible block.
            # Assume user knows what they are doing, but warn?
            pass

        # Create list of all possible start indices for blocks
        # Overlapping blocks? Moving block bootstrap usually allows overlapping.
        # "Circular block bootstrap" vs "Moving block bootstrap".
        # Let's assume standard moving block bootstrap: blocks can start at any index i where i+block <= n
        num_possible_blocks = n_time - block_size + 1

        # We need to construct a sample of length n_time (approx)
        num_blocks_needed = int(np.ceil(n_time / block_size))

        # Randomly choose start indices
        start_indices = np.random.randint(
            0, num_possible_blocks, (n_boot, num_blocks_needed)
        )

        # Construct full indices
        # This is tricky to do fully vectorized for all_indices shaped (n_boot, n_time)
        # because the last block might be truncated.
        # But we can pre-allocate.

        all_indices = np.zeros((n_boot, n_time), dtype=int)

        # Vectorized block expansion is complex, let's do semi-vectorized.
        # Or construct a large array of shape (n_boot, num_blocks_needed * block_size)
        # taking care of wrapping? No wrapping for Moving Block.

        # Let's create an template of offsets: [0, 1, ..., block_size-1]
        offsets = np.arange(block_size)

        # shape: (n_boot, num_blocks_needed, block_size)
        # block_starts: (n_boot, num_blocks_needed, 1)
        block_indices = (
            start_indices[..., np.newaxis] + offsets[np.newaxis, np.newaxis, :]
        )

        # Flatten to (n_boot, num_blocks_needed * block_size)
        flat_indices = block_indices.reshape(n_boot, -1)

        # Truncate to n_time
        all_indices = flat_indices[:, :n_time]

    else:
        # Generate all bootstrap indices at once using NumPy (ensures reproducibility with seed)
        all_indices = np.random.randint(0, n_time, (n_boot, n_time))

    use_numba = HAS_NUMBA and os.environ.get("NUMBA_DISABLE_JIT", "0") not in (
        "1",
        "true",
        "True",
    )
    if use_numba:
        try:
            resampled_stats = _bootstrap_resample_jit(
                data, all_indices, use_median, ignore_nan
            )
        except Exception:
            resampled_stats = _bootstrap_resample_py(
                data, all_indices, use_median, ignore_nan
            )
    else:
        resampled_stats = _bootstrap_resample_py(
            data, all_indices, use_median, ignore_nan
        )

    if avg == "median":
        if ignore_nan:
            center = np.nanmedian(resampled_stats, axis=0)
        else:
            center = np.median(resampled_stats, axis=0)
    else:
        if ignore_nan:
            center = np.nanmean(resampled_stats, axis=0)
        else:
            center = np.mean(resampled_stats, axis=0)

    alpha = (1 - ci) / 2
    if ignore_nan:
        p_low = np.nanpercentile(resampled_stats, 100 * alpha, axis=0)
        p_high = np.nanpercentile(resampled_stats, 100 * (1 - alpha), axis=0)
    else:
        p_low = np.percentile(resampled_stats, 100 * alpha, axis=0)
        p_high = np.percentile(resampled_stats, 100 * (1 - alpha), axis=0)

    err_low = center - p_low
    err_high = p_high - center

    # Overlap Ratio Correction (not applied if block bootstrap is used?
    # Block bootstrap accounts for serial correlation implicitly if block size is large enough.
    # But calculate_correlation_factor corrects for OVERLAP of FFTs in Welch.
    # If we are resampling blocks of overlapping segments, do we still need it?
    # Generally, bootstrap on dependent data should capture variance correctly if blocks are long enough.
    # If block_size is used, we might treat factor as 1.0 or let it apply?
    # Standard practice: If block bootstrap is used, it should capture the variance.
    # But usually factor > 1 only if N_effective < N_actual.
    # With block bootstrap, we resample N_actual points. The variance of the mean of dependent vars
    # is captured by the block structure.
    # However, individual segments in Welch are correlated.
    # Let's err on side of caution: If block_size is specified, we assume it handles correlation -> factor=1.
    # If not, we use the analytical correction.

    if block_size is not None and block_size > 1:
        factor = 1.0
    else:
        overlap_ratio = _infer_overlap_ratio(spectrogram)
        factor = 1.0

        if nperseg is None:
            if overlap_ratio is not None and overlap_ratio > 1:
                dummy_step = 100
                dummy_nperseg = int(round(dummy_step * overlap_ratio))
                if dummy_nperseg > dummy_step:
                    dummy_noverlap = dummy_nperseg - dummy_step
                    factor = calculate_correlation_factor(
                        window, dummy_nperseg, dummy_noverlap, n_time
                    )
        else:
            if noverlap is None and overlap_ratio is not None and overlap_ratio > 1:
                inferred_noverlap = int(round(nperseg - (nperseg / overlap_ratio)))
                noverlap = max(0, inferred_noverlap)
            if noverlap is None:
                noverlap = 0
            factor = calculate_correlation_factor(window, nperseg, noverlap, n_time)

    err_low *= factor
    err_high *= factor

    name = f"{spectrogram.name} (Bootstrap {avg})"
    # Use re-calculated frequencies
    from astropy import units as u

    out_freqs = u.Quantity(frequencies, unit=spectrogram.frequencies.unit)

    fs = FrequencySeries(
        center,
        frequencies=out_freqs,
        unit=spectrogram.unit,
        name=name,
    )
    fs.error_low = FrequencySeries(
        err_low,
        frequencies=out_freqs,
        unit=spectrogram.unit,
        name="Error Low",
    )
    fs.error_high = FrequencySeries(
        err_high,
        frequencies=out_freqs,
        unit=spectrogram.unit,
        name="Error High",
    )  # Fixed Error High name

    if return_map:
        from gwexpy.frequencyseries import BifrequencyMap

        if ignore_nan:
            # masked covariance? np.cov doesn't support nan directly well in all versions
            # Use pandas or manual? Or just mask?
            # Simple approach: nan -> mean (impute) or just let np.cov handle if recent enough?
            # np.cov does NOT handle NaNs.
            # If robust is needed, maybe skip map or warning?
            # Hack: zero-fill or mean-fill for covariance calculation if NaNs exist
            # But if ignore_nan=True, we likely have NaNs.
            # Let's fill NaNs with column means for covariance purpose
            stats_filled = resampled_stats.copy()
            col_mean = np.nanmean(stats_filled, axis=0)
            inds = np.where(np.isnan(stats_filled))
            stats_filled[inds] = np.take(col_mean, inds[1])
            cov_matrix = np.cov(stats_filled, rowvar=False)
        else:
            cov_matrix = np.cov(resampled_stats, rowvar=False)

        bfm = BifrequencyMap.from_points(
            cov_matrix,
            f2=out_freqs,
            f1=out_freqs,
            unit=spectrogram.unit**2,
            name=f"Covariance of {spectrogram.name}",
        )
        return fs, bfm

    return fs
