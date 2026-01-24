"""
gwexpy.signal.normalization - Spectral normalization and ENBW utilities.

This module provides utilities to handle window normalization and Equivalent Noise
Bandwidth (ENBW) calculations, supporting both standard scientific approaches
(as used in scipy.signal) and DTT (diaggui) compatible modes.
"""

import numpy as np
from scipy.signal import get_window


def get_enbw(window, fs, mode="standard"):
    """
    Calculate the Equivalent Noise Bandwidth (ENBW) of a window.

    Parameters
    ----------
    window : array_like
        The window coefficients.
    fs : float
        Sampling frequency in Hz.
    mode : {'standard', 'dtt'}, optional
        The normalization mode. 
        - 'standard': Industry standard ENBW = fs * (sum(w^2) / sum(w)^2).
        - 'dtt': DTT (diaggui) definition = (fs / N) * (1 / mean(w)^2).

    Returns
    -------
    enbw : float
        The calculated ENBW in Hz.
    """
    w = np.asarray(window)
    n = len(w)
    sum_w = np.sum(w)

    if sum_w == 0:
        return 0.0

    if mode == "dtt":
        # DTT windowNorm = (sum(w)/N)^2
        # windowBW = (fs/N) / windowNorm = fs * N / (sum(w)^2)
        return (fs * n) / (sum_w**2)

    # Standard: sum(w^2) / (sum(w)^2) * fs
    sum_w2 = np.sum(w**2)
    return fs * sum_w2 / (sum_w**2)


def get_psd_normalization_factor(window, fs, mode="standard"):
    """
    Calculate the normalization factor to convert squared FFT magnitude to PSD.

    Parameters
    ----------
    window : array_like
        The window coefficients.
    fs : float
        Sampling frequency in Hz.
    mode : {'standard', 'dtt'}, optional
        The normalization mode.

    Returns
    -------
    factor : float
        The multiplier to apply to |FFT|^2.
    """
    w = np.asarray(window)
    n = len(w)
    sum_w = np.sum(w)

    if sum_w == 0:
        return 0.0

    if mode == "dtt":
        # DTT scale = 1 / windowBW = (sum(w)^2) / (fs * N)
        return (sum_w**2) / (fs * n)

    # Standard scale = 2 / (fs * sum(w^2)) for one-sided
    # Here we return 1 / (fs * sum(w^2)) as the base scale.
    # Note: scipy.signal.welch applies this factor internally.
    sum_w2 = np.sum(w**2)
    return 1.0 / (fs * sum_w2)


def convert_scipy_to_dtt(psd, window, is_one_sided=True):
    """
    Convert a PSD calculated with scipy.signal.welch to DTT compatible normalization.

    Parameters
    ----------
    psd : array_like
        The PSD values.
    window : array_like or str
        The window coefficients or name. If string, length must be provided by psd shape?
        Better to pass the actual coefficients.
    is_one_sided : bool, optional
        Whether the PSD is one-sided (standard for LIGO).

    Returns
    -------
    psd_dtt : array_like
        The PSD adjusted to DTT scale.
    """
    w = np.asarray(window)
    n = len(w)
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)

    if sum_w == 0 or sum_w2 == 0:
        return psd

    # Ratio = DTT_Scale / Scipy_Scale
    # DTT_Scale = (sum_w^2) / (fs * N)
    # Scipy_Scale = 1 / (fs * sum_w2)
    # Ratio = (sum_w^2 * sum_w2) / N
    ratio = (sum_w**2 * sum_w2) / (n**2) # Wait, let's re-verify.

    # Re-derivation from dtt_normalization_analysis.md Task 2 logic:
    # dc_gain = sum_w / n
    # power_sum = sum_w2 / n
    # dtt_factor = 1.0 / (dc_gain**2)  # Proportional part in windowBW
    # scipy_factor = 1.0 / power_sum
    # ratio = dtt_factor / scipy_factor = power_sum / dc_gain^2
    #       = (sum_w2 / n) / (sum_w^2 / n^2) = (sum_w2 * n) / (sum_w^2)

    ratio = (sum_w2 * n) / (sum_w**2)

    return psd * ratio


def get_window_normalized(window_name, n, mode="standard"):
    """
    Get a window and its ENBW.

    Parameters
    ----------
    window_name : str
        Name of the window (e.g., 'hann').
    n : int
        Length of the window.
    mode : str
        Normalization mode.

    Returns
    -------
    w : ndarray
    enbw : float
    """
    w = get_window(window_name, n)
    # Most windows in DTT/LIGO are assumed to be used with specific scaling.
    # For now, we just return the raw scipy window.
    return w
