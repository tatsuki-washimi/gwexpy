import numpy as np
import warnings
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
            else: # mean
                if ignore_nan:
                    resampled_stats[i, f] = np.nanmean(col)
                else:
                    resampled_stats[i, f] = np.mean(col)
    return resampled_stats

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

    Formula:
        factor = sqrt(1 + 2 * sum_{k=1}^{M-1} (1 - k/M) * |rho_window(k * S)|^2)
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
    average="median",
    ci=0.68,
    window="hann",
    nperseg=None,
    noverlap=None,
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
    average : str
    ci : float
    window : str or array, optional
    nperseg : int, optional
    noverlap : int, optional
    ignore_nan : bool, optional
        If True, ignore NaNs during bootstrap averaging. Default is True.

    Returns
    -------
    FrequencySeries
    """
    data = spectrogram.value
    n_time = data.shape[0]

    if n_time < 2:
        raise ValueError("Spectrogram must have at least 2 time bins.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1.")

    avg = average.lower()
    if avg not in {"median", "mean"}:
        raise ValueError("average must be 'median' or 'mean'.")

    use_median = (avg == "median")

    # Generate all bootstrap indices at once using NumPy (ensures reproducibility with seed)
    all_indices = np.random.randint(0, n_time, (n_boot, n_time))

    if HAS_NUMBA:
        resampled_stats = _bootstrap_resample_jit(data, all_indices, use_median, ignore_nan)
    else:
        # Fallback to pure Python/NumPy implementation
        resampled_stats = np.zeros((n_boot, data.shape[1]))
        for i in range(n_boot):
            indices = all_indices[i]
            sample = data[indices]
            if use_median:
                if ignore_nan:
                    resampled_stats[i] = np.nanmedian(sample, axis=0)
                else:
                    resampled_stats[i] = np.median(sample, axis=0)
            else:
                if ignore_nan:
                    resampled_stats[i] = np.nanmean(sample, axis=0)
                else:
                    resampled_stats[i] = np.mean(sample, axis=0)

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
    fs = FrequencySeries(
        center,
        frequencies=spectrogram.frequencies,
        unit=spectrogram.unit,
        name=name,
    )
    fs.error_low = FrequencySeries(
        err_low,
        frequencies=spectrogram.frequencies,
        unit=spectrogram.unit,
        name="Error Low",
    )
    fs.error_high = FrequencySeries(
        err_high,
        frequencies=spectrogram.frequencies,
        unit=spectrogram.unit,
        name="Error High",
    )

    return fs
