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
    Check for deprecated nperseg/noverlap usage and raise TypeError.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to check.

    Raises
    ------
    TypeError
        If nperseg or noverlap is found in kwargs.

    Examples
    --------
    >>> check_deprecated_kwargs(fftlength=1.0, overlap=0.5)  # OK
    >>> check_deprecated_kwargs(nperseg=256)  # Raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: nperseg is removed from the public API. Specify fftlength in seconds instead.
    """
    if "nperseg" in kwargs:
        raise TypeError(
            "nperseg is removed from the public API. "
            "Specify fftlength in seconds instead."
        )
    if "noverlap" in kwargs:
        raise TypeError(
            "noverlap is removed from the public API. "
            "Specify overlap in seconds instead."
        )


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
