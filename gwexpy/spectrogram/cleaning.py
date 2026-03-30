"""gwexpy.spectrogram.cleaning - Spectrogram cleaning algorithms."""

from __future__ import annotations

import numpy as np
try:
    from scipy.ndimage import median_filter
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.spectrogram.cleaning. Install with: pip install scipy"
    ) from _exc


def threshold_clean(
    data: np.ndarray,
    threshold: float = 5.0,
    fill: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove outlier pixels using MAD-based threshold.

    For each frequency bin, compute the median and MAD (Median Absolute
    Deviation). Pixels exceeding ``median + threshold * MAD`` are flagged
    and replaced.

    Parameters
    ----------
    data : ndarray, shape (ntimes, nfreqs)
        Input spectrogram values.
    threshold : float
        Number of MAD above median to flag as outlier.
    fill : {'median', 'nan', 'zero', 'interpolate'}
        How to fill flagged pixels.

    Returns
    -------
    cleaned : ndarray
        Cleaned data array.
    mask : ndarray (bool)
        True where pixels were flagged as outliers.
    """
    cleaned = data.copy()
    med = np.median(data, axis=0)
    mad = np.median(np.abs(data - med[np.newaxis, :]), axis=0)
    # Scale MAD to approximate standard deviation
    mad_scaled = mad * 1.4826

    # Avoid division issues with constant columns
    mad_scaled = np.where(mad_scaled > 0, mad_scaled, 1.0)

    deviation = np.abs(data - med[np.newaxis, :]) / mad_scaled[np.newaxis, :]
    mask = deviation > threshold

    if fill == "median":
        # Replace outliers with column median (vectorized)
        cleaned = np.where(mask, med[np.newaxis, :], cleaned)
    elif fill == "nan":
        cleaned[mask] = np.nan
    elif fill == "zero":
        cleaned[mask] = 0.0
    elif fill == "interpolate":
        # Linear interpolation along time axis for each frequency
        for j in range(data.shape[1]):
            col_mask = mask[:, j]
            if not np.any(col_mask):
                continue
            good = ~col_mask
            if np.sum(good) < 2:
                cleaned[col_mask, j] = med[j]
                continue
            indices = np.arange(data.shape[0])
            cleaned[col_mask, j] = np.interp(
                indices[col_mask], indices[good], data[good, j]
            )
    else:
        raise ValueError(
            f"Unknown fill method: {fill!r}. "
            "Choose from 'median', 'nan', 'zero', 'interpolate'."
        )

    return cleaned, mask


def rolling_median_clean(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Normalize slow trends using a rolling median along the time axis.

    Parameters
    ----------
    data : ndarray, shape (ntimes, nfreqs)
        Input spectrogram values.
    window_size : int
        Window size in time bins for the median filter.

    Returns
    -------
    normalized : ndarray
        Data divided by the rolling median trend.
    """
    # Apply 1D median filter along time axis for each frequency bin
    trend = median_filter(data, size=(window_size, 1), mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = data / trend
        normalized[~np.isfinite(normalized)] = np.nan
    return normalized


def line_removal_clean(
    data: np.ndarray,
    persistence_threshold: float = 0.8,
    amplitude_threshold: float = 3.0,
) -> tuple[np.ndarray, list[int]]:
    """
    Detect and remove persistent narrowband lines.

    A frequency bin is flagged as a persistent line if it exceeds
    ``amplitude_threshold * global_median`` for more than
    ``persistence_threshold`` fraction of time bins.

    Parameters
    ----------
    data : ndarray, shape (ntimes, nfreqs)
        Input spectrogram values.
    persistence_threshold : float
        Fraction of time bins where a frequency must be elevated to be
        considered a persistent line (0.0 to 1.0).
    amplitude_threshold : float
        Factor above global median to consider a bin elevated.

    Returns
    -------
    cleaned : ndarray
        Data with persistent lines replaced by column median.
    line_indices : list of int
        Frequency bin indices that were identified as lines.
    """
    cleaned = data.copy()
    global_med = np.median(data)
    ntimes = data.shape[0]

    # For each frequency bin, check what fraction of time bins exceed threshold
    elevated = data > amplitude_threshold * global_med
    persistence = np.sum(elevated, axis=0) / ntimes

    line_indices = list(np.where(persistence > persistence_threshold)[0])

    # Replace line bins with their time-median value
    for idx in line_indices:
        cleaned[:, idx] = np.median(data[:, idx])

    return cleaned, line_indices
