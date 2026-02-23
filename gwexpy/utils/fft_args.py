"""FFT argument parsing utilities for gwexpy public API.

This module provides utilities to parse fftlength/overlap arguments
from gwexpy's public API and convert them to sample counts for internal use.

The API follows GWpy conventions:
- fftlength and overlap are specified in seconds (or time-like Quantity)
- int and float are treated as seconds (GWpy-compatible)
- Quantity must have time-like units (convertible to seconds)
- Default values match GWpy behavior
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from astropy import units as u


def parse_fftlength_or_overlap(
    value: Union[None, float, int, u.Quantity],
    sample_rate: Optional[float] = None,
    arg_name: str = "fftlength",
) -> tuple[Optional[float], Optional[int]]:
    """
    Parse fftlength or overlap argument for gwexpy public API.

    This function converts user-provided values (in seconds or time-like
    Quantity) into both seconds and sample counts for internal use.

    Parameters
    ----------
    value : None, float, int, or Quantity
        Input value. If numeric (float or int), treated as seconds.
        If Quantity, must have time-like units (convertible to seconds).
        If None, returns (None, None).
    sample_rate : float, optional
        Sample rate in Hz for converting seconds to samples.
        If None, samples will be None (only seconds are returned).
    arg_name : str, optional
        Argument name for error messages (default: "fftlength").

    Returns
    -------
    tuple
        (seconds: Optional[float], samples: Optional[int])
        - seconds: None if value is None, otherwise value in seconds (float)
        - samples: None if sample_rate is None, otherwise rounded integer sample count

    Raises
    ------
    ValueError
        If Quantity has non-time units, or if seconds is negative,
        or if sample_rate is required but invalid.

    Examples
    --------
    >>> # Parse seconds (float)
    >>> parse_fftlength_or_overlap(1.0, sample_rate=256)
    (1.0, 256)

    >>> # Parse seconds (int, treated as seconds, not samples)
    >>> parse_fftlength_or_overlap(2, sample_rate=128)
    (2.0, 256)

    >>> # Parse Quantity
    >>> from astropy import units as u
    >>> parse_fftlength_or_overlap(1.5 * u.s, sample_rate=1024)
    (1.5, 1536)

    >>> # No sample_rate (only seconds)
    >>> parse_fftlength_or_overlap(1.0)
    (1.0, None)

    >>> # None input
    >>> parse_fftlength_or_overlap(None, sample_rate=256)
    (None, None)
    """
    if value is None:
        return (None, None)

    # Handle Quantity (must be time-like)
    if isinstance(value, u.Quantity):
        try:
            seconds = float(value.to(u.s).value)
        except u.UnitConversionError:
            raise ValueError(
                f"{arg_name}: expected a time-like Quantity (e.g. 1.0*u.s); "
                f"got unit '{value.unit}'."
            )
    else:
        # Treat numeric (int or float) as seconds (GWpy-compatible)
        seconds = float(value)

    # Validate non-negative
    if seconds < 0:
        raise ValueError(f"{arg_name}: negative time ({seconds}) is not allowed.")

    # Convert to samples if sample_rate is provided
    if sample_rate is None:
        samples = None
    else:
        # Normalize sample_rate (handle Quantity if needed)
        if hasattr(sample_rate, "to"):
            # Type narrowing: sample_rate is Quantity here
            sr_qty = sample_rate  # type: ignore[misc]
            if not sr_qty.unit.is_equivalent(u.Hz):  # type: ignore[attr-defined]
                raise ValueError(
                    "sample_rate must be a frequency (Hz) Quantity or numeric Hz."
                )
            sr = float(sr_qty.to(u.Hz).value)  # type: ignore[attr-defined]
        else:
            sr = float(sample_rate)

        if sr <= 0:
            raise ValueError(
                f"sample_rate is required to convert seconds to samples ({arg_name} -> nperseg). "
                "Provide a TimeSeries with a valid sample_rate."
            )

        # Convert to samples: max(1, round(seconds * sample_rate))
        samples = max(1, int(round(seconds * sr)))

    return (seconds, samples)


def check_deprecated_kwargs(**kwargs):
    """
    Check for deprecated nperseg usage and raise TypeError.

    Note: noverlap is now supported as a valid parameter when used with nfft.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to check.

    Raises
    ------
    TypeError
        If nperseg is found in kwargs.

    Examples
    --------
    >>> check_deprecated_kwargs(fftlength=1.0, overlap=0.5)  # OK
    >>> check_deprecated_kwargs(nfft=1024, noverlap=512)  # OK (new API)
    >>> check_deprecated_kwargs(nperseg=256)  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: nperseg is removed from the public API. Use fftlength (seconds) or nfft (samples) instead.
    """
    if "nperseg" in kwargs:
        raise TypeError(
            "nperseg is removed from the public API. "
            "Use fftlength (seconds) or nfft (samples) instead."
        )
    if "noverlap" in kwargs and "nfft" not in kwargs:
        raise TypeError(
            "noverlap is removed from the public API. "
            "Use overlap (seconds) or nfft/noverlap (samples) instead."
        )


def validate_and_convert_fft_params(
    fftlength: Union[None, float, int, u.Quantity] = None,
    overlap: Union[None, float, int, u.Quantity] = None,
    nfft: Optional[int] = None,
    noverlap: Optional[int] = None,
    sample_rate: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    """
    Validate and convert FFT parameters from either time-based or sample-based specification.

    This function supports two mutually exclusive parameter sets:
    1. Time-based (high-level): fftlength (seconds) and overlap (seconds)
    2. Sample-based (low-level): nfft (samples) and noverlap (samples)

    Parameters
    ----------
    fftlength : float, int, Quantity, or None
        FFT segment length in seconds. Cannot be used with nfft.
    overlap : float, int, Quantity, or None
        Overlap length in seconds. Cannot be used with noverlap.
    nfft : int or None
        FFT segment length in samples. Cannot be used with fftlength.
    noverlap : int or None
        Overlap length in samples. Cannot be used with overlap.
    sample_rate : float or None
        Sample rate in Hz. Required when converting nfft/noverlap to fftlength/overlap.

    Returns
    -------
    tuple
        (fftlength_sec: Optional[float], overlap_sec: Optional[float])
        Both values in seconds. Returns (None, None) if no parameters specified.

    Raises
    ------
    ValueError
        If both fftlength and nfft are specified.
        If both overlap and noverlap are specified.
        If fftlength is used with noverlap, or nfft is used with overlap.
        If nfft/noverlap is specified but sample_rate is not provided.
    TypeError
        If nfft or noverlap is not an integer.

    Examples
    --------
    >>> # Time-based specification
    >>> validate_and_convert_fft_params(fftlength=1.0, overlap=0.5)
    (1.0, 0.5)

    >>> # Sample-based specification
    >>> validate_and_convert_fft_params(nfft=1024, noverlap=512, sample_rate=1024.0)
    (1.0, 0.5)

    >>> # Error: cannot mix
    >>> validate_and_convert_fft_params(fftlength=1.0, noverlap=512)
    Traceback (most recent call last):
        ...
    ValueError: Cannot use noverlap (samples) with fftlength (seconds). Use overlap instead.
    """
    # Check for mutually exclusive parameters
    if fftlength is not None and nfft is not None:
        raise ValueError(
            "Cannot specify both fftlength and nfft. "
            "Use fftlength (seconds) for time-based or nfft (samples) for sample-based specification."
        )

    if overlap is not None and noverlap is not None:
        raise ValueError(
            "Cannot specify both overlap and noverlap. "
            "Use overlap (seconds) for time-based or noverlap (samples) for sample-based specification."
        )

    # Check for invalid mixing
    if fftlength is not None and noverlap is not None:
        raise ValueError(
            "Cannot use noverlap (samples) with fftlength (seconds). "
            "Use overlap instead."
        )

    if nfft is not None and overlap is not None:
        raise ValueError(
            "Cannot use overlap (seconds) with nfft (samples). "
            "Use noverlap instead."
        )

    # Convert sample-based to time-based if needed
    if nfft is not None or noverlap is not None:
        if sample_rate is None:
            raise ValueError(
                "sample_rate is required when using nfft or noverlap. "
                "Ensure the input data has a valid sample_rate attribute."
            )

        # Type check
        if nfft is not None and not isinstance(nfft, (int, np.integer)):
            raise TypeError(f"nfft must be an integer (sample count), got {type(nfft)}")
        if noverlap is not None and not isinstance(noverlap, (int, np.integer)):
            raise TypeError(f"noverlap must be an integer (sample count), got {type(noverlap)}")

        # Convert to seconds
        if nfft is not None:
            fftlength = float(nfft) / float(sample_rate)
        if noverlap is not None:
            overlap = float(noverlap) / float(sample_rate)

    # Parse time-based parameters (already in seconds or None)
    if fftlength is not None:
        fftlength_sec, _ = parse_fftlength_or_overlap(fftlength, arg_name="fftlength")
    else:
        fftlength_sec = None

    if overlap is not None:
        overlap_sec, _ = parse_fftlength_or_overlap(overlap, arg_name="overlap")
    else:
        overlap_sec = None

    return (fftlength_sec, overlap_sec)


def get_default_overlap(fftlength: Optional[float], window: str = "hann") -> Optional[float]:
    """
    Get the recommended overlap for a given window function (GWpy-compatible).

    Parameters
    ----------
    fftlength : float or None
        FFT length in seconds. If None, returns None.
    window : str, optional
        Window function name (default: 'hann').

    Returns
    -------
    float or None
        Recommended overlap in seconds, or None if fftlength is None.

    Notes
    -----
    GWpy's default overlap behavior:
    - For 'hann' and 'hamming': 50% overlap (fftlength / 2)
    - For 'boxcar' (uniform): 0% overlap
    - For other windows: 50% overlap (conservative default)

    This matches GWpy's internal recommendation from scipy.signal.get_window.

    Examples
    --------
    >>> get_default_overlap(1.0, window='hann')
    0.5
    >>> get_default_overlap(2.0, window='boxcar')
    0.0
    >>> get_default_overlap(None, window='hann')
    """
    if fftlength is None:
        return None

    # Normalize window name
    window_lower = str(window).lower()

    # Recommended overlaps (GWpy-compatible)
    if window_lower in ("hann", "hanning", "hamming", "blackman", "blackmanharris"):
        return fftlength / 2.0
    elif window_lower in ("boxcar", "rectangular", "uniform"):
        return 0.0
    else:
        # Conservative default: 50% overlap
        return fftlength / 2.0
