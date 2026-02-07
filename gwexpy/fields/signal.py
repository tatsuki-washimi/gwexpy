"""Signal processing utilities for ScalarField.

This module provides spectral analysis, correlation, and coherence tools
for 4D field data, following the Phase 3 signal processing extension plan.

Key features:
- PSD estimation (Welch method)
- Frequency-space mapping
- Cross-correlation and time delay estimation
- Coherence analysis

All functions preserve axis metadata and units, returning gwexpy-compatible
data containers (FrequencySeries, TimeSeries, ScalarField).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from astropy.units import Quantity

    from gwexpy.fields import ScalarField
    from gwexpy.fields.collections import FieldDict
    from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
    from gwexpy.timeseries import TimeSeries

__all__ = [
    "spectral_density",
    "compute_psd",
    "freq_space_map",
    "compute_freq_space",
    "compute_xcorr",
    "time_delay_map",
    "coherence_map",
]


def _validate_regular_time_axis(field: ScalarField) -> tuple[float, u.Unit]:
    """Validate that ScalarField has a regular time axis suitable for FFT/PSD.

    Returns
    -------
    tuple
        (dt_value, dt_unit): Time step value and unit.

    Raises
    ------
    ValueError
        If axis0_domain is not 'time' or axis is irregular.
    """
    if field.axis0_domain != "time":
        raise ValueError(
            f"Signal processing requires axis0_domain='time', "
            f"got '{field.axis0_domain}'"
        )

    from ..types.axis import AxisDescriptor

    ax = AxisDescriptor(field._axis0_name, field._axis0_index)
    if not ax.regular:
        raise ValueError(
            "Time axis must be regularly spaced for spectral analysis. "
            "Consider resampling the data to a uniform time grid."
        )

    if ax.size < 2:
        raise ValueError("Time axis must have at least 2 points for spectral analysis.")

    dt = ax.delta
    if dt is None:
        raise ValueError(
            "Time axis must be regularly spaced for spectral analysis. "
            "Consider resampling the data to a uniform time grid."
        )
    dt_value = getattr(dt, "value", dt)
    dt_unit = getattr(dt, "unit", u.dimensionless_unscaled)
    return dt_value, dt_unit


def _validate_axis_for_spectral(
    field: ScalarField, axis: int | str
) -> tuple[int, float, u.Unit, str]:
    """Validate axis for spectral density computation.

    Returns
    -------
    tuple
        (axis_int, delta_value, delta_unit, domain): axis index, sample spacing,
        spacing unit, and current domain ('time' or 'real').
    """
    from ..types.axis import AxisDescriptor

    # Resolve axis name to index
    if isinstance(axis, str):
        axis_int = field._get_axis_index(axis)
    else:
        axis_int = axis

    if axis_int < 0 or axis_int > 3:
        raise ValueError(f"axis must be 0-3, got {axis_int}")

    # Get axis info
    axis_mapping = {
        0: (field._axis0_name, field._axis0_index, field._axis0_domain),
        1: (
            field._axis1_name,
            field._axis1_index,
            field._space_domains.get(field._axis1_name, "real"),
        ),
        2: (
            field._axis2_name,
            field._axis2_index,
            field._space_domains.get(field._axis2_name, "real"),
        ),
        3: (
            field._axis3_name,
            field._axis3_index,
            field._space_domains.get(field._axis3_name, "real"),
        ),
    }
    ax_name, ax_index, domain = axis_mapping[axis_int]

    # Validate regular spacing
    ax_desc = AxisDescriptor(ax_name, ax_index)
    if not ax_desc.regular:
        raise ValueError(
            f"Axis '{ax_name}' must be regularly spaced for spectral analysis. "
            f"Consider resampling to a uniform grid."
        )
    if ax_desc.size < 2:
        raise ValueError(f"Axis '{ax_name}' must have at least 2 points.")

    delta = ax_desc.delta
    if delta is None:
        raise ValueError(f"Axis '{ax_name}' does not have a defined spacing.")

    # Check domain is appropriate
    if axis_int == 0:
        if domain == "frequency":
            raise ValueError(
                "spectral_density requires axis0_domain='time'. "
                "Use ifft_time first to transform to time domain."
            )
    else:
        if domain == "k":
            raise ValueError(
                f"spectral_density requires axis '{ax_name}' in 'real' domain. "
                f"Use ifft_space first to transform to real space."
            )

    delta_value = getattr(delta, "value", delta)
    delta_unit = getattr(delta, "unit", u.dimensionless_unscaled)
    return axis_int, delta_value, delta_unit, domain


def spectral_density(
    field: ScalarField,
    axis: int | str = 0,
    *,
    method: Literal["welch", "fft"] = "welch",
    fftlength: float | None = None,
    overlap: float | None = None,
    nfft: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    average: Literal["mean", "median"] = "mean",
    **kwargs,
) -> ScalarField:
    """Compute spectral density along any axis.

    This is the generalized spectral density function that works on:
    - Time axis (axis=0): Returns power spectral density vs frequency
    - Space axes (axis=1,2,3): Returns spatial spectral density vs wavenumber

    The output maintains 4D shape, with the specified axis transformed
    from its original domain (time->frequency, position->wavenumber).

    Parameters
    ----------
    field : ScalarField
        Input 4D field. For time axis, axis0_domain must be 'time'.
        For space axes, the axis must be in 'real' domain.
    axis : int or str
        Axis to transform. Can be integer (0-3) or axis name.
        Default is 0 (time axis).
    method : {'welch', 'fft'}
        Estimation method. 'welch' for averaged periodogram,
        'fft' for direct FFT magnitude squared. Default 'welch'.
    fftlength : float, optional
        Segment length in axis units (seconds for time axis, spatial
        units for space axes). Default min(256, axis_length) samples.
        Cannot be used with `nfft`.
    overlap : float, optional
        Overlap in axis units. Default fftlength / 2.
        Cannot be used with `noverlap`.
    nfft : int, optional
        FFT segment length in samples. Alternative to `fftlength`.
        Cannot be used with `fftlength`.
    noverlap : int, optional
        Overlap length in samples. Must be used with `nfft`.
        Cannot be used with `overlap`.
    window : str
        Window function. Default 'hann'.
    detrend : str or bool
        Detrending: 'constant', 'linear', or False. Default 'constant'.
    scaling : {'density', 'spectrum'}
        'density' divides by frequency/wavenumber resolution,
        'spectrum' gives total power in each bin. Default 'density'.
    average : {'mean', 'median'}
        Averaging for Welch. Default 'mean'.
    **kwargs
        Additional keyword arguments. Passing the removed ``nperseg``
        or ``noverlap`` parameters will raise :class:`TypeError`.

    Returns
    -------
    ScalarField
        Spectral density field with the specified axis transformed:
        - axis0: frequency (if axis=0) or unchanged
        - axis0_domain: 'frequency' (if axis=0)
        - space_domains: 'k' for transformed spatial axis

    Raises
    ------
    ValueError
        If axis is already in spectral domain, or is irregular.

    Examples
    --------
    >>> from astropy import units as u
    >>> from gwexpy.fields import ScalarField
    >>> # Time PSD (all spatial points)
    >>> psd_field = spectral_density(field, axis=0)
    >>> psd_field.axis0_domain  # 'frequency'

    >>> # Spatial wavenumber spectrum along x
    >>> kx_spec = spectral_density(field, axis='x')

    Notes
    -----
    For time axis (axis=0), this is equivalent to applying Welch PSD
    estimation to every spatial point independently.

    For spatial axes, this computes the spatial power spectrum,
    where the output unit is ``field.unit**2 / (1/dx.unit)`` assuming
    density scaling.

    The wavenumber definition follows the convention k = 1/λ (not angular).
    To get angular wavenumber (2π/λ), multiply the output axis by 2π.
    """
    from scipy.signal import welch

    from ..utils.fft_args import (
        check_deprecated_kwargs,
        get_default_overlap,
        parse_fftlength_or_overlap,
        validate_and_convert_fft_params,
    )

    check_deprecated_kwargs(**kwargs)

    # Validate and convert nfft/noverlap to fftlength/overlap if needed
    # Get sample_rate from field axis for conversion
    axis_int, delta_value, delta_unit, domain = _validate_axis_for_spectral(field, axis)
    sample_rate_for_conversion = 1.0 / delta_value if delta_value else None

    fftlength, overlap = validate_and_convert_fft_params(
        fftlength=fftlength,
        overlap=overlap,
        nfft=nfft,
        noverlap=noverlap,
        sample_rate=sample_rate_for_conversion,
    )

    axis_int, delta, delta_unit, domain = _validate_axis_for_spectral(field, axis)

    n_axis = field.shape[axis_int]
    fs = 1.0 / delta  # Sampling frequency (temporal or spatial)

    # Convert fftlength/overlap to samples
    fftlength_sec, nperseg = parse_fftlength_or_overlap(fftlength, fs, "fftlength")
    overlap_sec, noverlap_calc = parse_fftlength_or_overlap(overlap, fs, "overlap")

    if nperseg is None:
        nperseg = min(256, n_axis)
    if noverlap_calc is None:
        if fftlength_sec is not None:
            overlap_sec_default = get_default_overlap(fftlength_sec, window=window)
            if overlap_sec_default is not None:
                noverlap_calc = max(1, int(round(overlap_sec_default * fs)))
            else:
                noverlap_calc = 0
        else:
            noverlap_calc = nperseg // 2

    # Apply Welch along specified axis
    if method == "welch":
        freqs, psd = welch(
            field.value,
            fs=fs,
            axis=axis_int,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap_calc,
            detrend=detrend,
            scaling=scaling,
            average=average,
            return_onesided=True,  # Real-valued input assumed
        )
    else:  # fft method
        # Direct FFT-based PSD
        n = field.shape[axis_int]
        fft_result = np.fft.rfft(field.value, axis=axis_int)
        psd = (np.abs(fft_result) ** 2) / n**2

        # One-sided correction
        slices = [slice(None)] * 4
        slices[axis_int] = slice(1, -1 if n % 2 == 0 else None)
        psd[tuple(slices)] *= 2

        if scaling == "density":
            df = fs / n
            psd = psd / df

        freqs = np.fft.rfftfreq(n, d=delta)

    # Determine output unit
    freq_unit = 1 / delta_unit
    if field.unit is not None:
        if scaling == "density":
            psd_unit = field.unit**2 / freq_unit
        else:
            psd_unit = field.unit**2
    else:
        psd_unit = u.dimensionless_unscaled

    # Build output ScalarField
    freq_axis = freqs * freq_unit

    # Determine new axis names and domains
    if axis_int == 0:
        new_axis0_domain = "frequency"
        new_axis_names = ["f", field._axis1_name, field._axis2_name, field._axis3_name]
        new_space_domains = dict(field._space_domains)
        new_axes = {
            0: freq_axis,
            1: field._axis1_index,
            2: field._axis2_index,
            3: field._axis3_index,
        }
    else:
        new_axis0_domain = field._axis0_domain
        new_axis_names = list(field.axis_names)
        new_space_domains = dict(field._space_domains)

        # Rename spatial axis to k-variant
        old_name = new_axis_names[axis_int]
        new_name = f"k{old_name}"
        new_axis_names[axis_int] = new_name

        # Update domain
        if old_name in new_space_domains:
            del new_space_domains[old_name]
        new_space_domains[new_name] = "k"

        new_axes = {
            0: field._axis0_index,
            1: field._axis1_index,
            2: field._axis2_index,
            3: field._axis3_index,
        }
        new_axes[axis_int] = freq_axis

    from gwexpy.fields import ScalarField as SF

    return SF(
        psd,
        unit=psd_unit,
        axis0=new_axes[0],
        axis1=new_axes[1],
        axis2=new_axes[2],
        axis3=new_axes[3],
        axis_names=new_axis_names,
        axis0_domain=new_axis0_domain,
        space_domain=new_space_domains,
    )


def _extract_timeseries_1d(
    field: ScalarField,
    point: tuple[Quantity, Quantity, Quantity],
) -> tuple[np.ndarray, float, u.Unit]:
    """Extract 1D time series at a spatial point.

    Returns
    -------
    tuple
        (data_1d, fs, data_unit): 1D array, sampling frequency, and unit.
    """
    from gwexpy.plot._coord import nearest_index

    dt_value, dt_unit = _validate_regular_time_axis(field)
    fs = 1.0 / dt_value  # sampling frequency in 1/dt_unit

    x_val, y_val, z_val = point
    i1 = nearest_index(field._axis1_index, x_val)
    i2 = nearest_index(field._axis2_index, y_val)
    i3 = nearest_index(field._axis3_index, z_val)

    data_1d = field.value[:, i1, i2, i3]
    return data_1d, fs, field.unit


def compute_psd(
    field: ScalarField,
    point_or_region: (
        tuple[Quantity, Quantity, Quantity]
        | list[tuple[Quantity, Quantity, Quantity]]
        | dict[str, Any]
    ),
    *,
    fftlength: float | None = None,
    overlap: float | None = None,
    nfft: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    average: Literal["mean", "median"] = "mean",
    return_onesided: bool = True,
    **kwargs,
) -> FrequencySeries | FrequencySeriesList:
    """Compute power spectral density using Welch's method.

    Extracts time series from one or more spatial points/regions and
    estimates their PSD using scipy.signal.welch.

    Parameters
    ----------
    field : ScalarField
        Input 4D field with axis0_domain='time'.
    point_or_region : tuple, list of tuples, or dict
        Spatial location(s) to extract:

        - Single point: ``(x, y, z)`` tuple of Quantities
        - Multiple points: list of ``(x, y, z)`` tuples
        - Region dict: ``{'x': slice or value, 'y': ..., 'z': ...}``
          If region, averages over all points in the region.

    fftlength : float, optional
        Segment length in seconds. Default is min(256, len(time_axis)) samples.
    overlap : float, optional
        Overlap in seconds. Default is fftlength / 2.
    window : str
        Window function name. Default is 'hann'.
    detrend : str or bool
        Detrending method: 'constant', 'linear', or False. Default 'constant'.
    scaling : {'density', 'spectrum'}
        'density' for PSD (V²/Hz), 'spectrum' for power spectrum (V²).
        Default is 'density'.
    average : {'mean', 'median'}
        Averaging method for segments. Default is 'mean'.
    return_onesided : bool
        If True, return one-sided spectrum for real data. Default True.
    **kwargs
        Additional keyword arguments. Passing the removed ``nperseg``
        or ``noverlap`` parameters will raise :class:`TypeError`.

    Returns
    -------
    FrequencySeries or FrequencySeriesList
        - Single point: FrequencySeries with PSD values
        - Multiple points: FrequencySeriesList

    Raises
    ------
    ValueError
        If time axis is not regularly spaced or axis0_domain != 'time'.

    Examples
    --------
    >>> from astropy import units as u
    >>> psd = compute_psd(field, (1.0*u.m, 2.0*u.m, 0.0*u.m))
    >>> psd.frequencies  # Frequency axis with units

    >>> # Multiple points
    >>> points = [(0*u.m, 0*u.m, 0*u.m), (1*u.m, 0*u.m, 0*u.m)]
    >>> psd_list = compute_psd(field, points)
    """
    from scipy.signal import welch

    from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList

    from ..utils.fft_args import (
        check_deprecated_kwargs,
        get_default_overlap,
        parse_fftlength_or_overlap,
        validate_and_convert_fft_params,
    )

    check_deprecated_kwargs(**kwargs)

    dt_value, dt_unit = _validate_regular_time_axis(field)
    fs = 1.0 / dt_value

    nt = field.shape[0]

    # Validate and convert nfft/noverlap to fftlength/overlap if needed
    fftlength, overlap = validate_and_convert_fft_params(
        fftlength=fftlength,
        overlap=overlap,
        nfft=nfft,
        noverlap=noverlap,
        sample_rate=fs,
    )

    # Convert fftlength/overlap to samples
    fftlength_sec, nperseg = parse_fftlength_or_overlap(fftlength, fs, "fftlength")
    overlap_sec, noverlap_samples = parse_fftlength_or_overlap(overlap, fs, "overlap")

    if nperseg is None:
        nperseg = min(256, nt)
    if noverlap_samples is None:
        if fftlength_sec is not None:
            overlap_sec_default = get_default_overlap(fftlength_sec, window=window)
            if overlap_sec_default is not None:
                noverlap_calc = max(1, int(round(overlap_sec_default * fs)))
            else:
                noverlap_calc = 0
        else:
            noverlap_calc = nperseg // 2
    else:
        noverlap_calc = noverlap_samples

    # Determine extraction mode
    if isinstance(point_or_region, dict):
        # Region mode: extract and average
        data_1d = _extract_region_average(field, point_or_region)
        points_data = [(data_1d, "region_avg")]
    elif isinstance(point_or_region, list):
        # Multiple points
        points_data = []
        for i, pt in enumerate(point_or_region):
            data_1d, _, _ = _extract_timeseries_1d(field, pt)
            label = f"point_{i}"
            points_data.append((data_1d, label))
    else:
        # Single point
        data_1d, _, _ = _extract_timeseries_1d(field, point_or_region)
        points_data = [(data_1d, "point_0")]

    # Compute PSD for each
    results = []
    freq_unit = 1 / dt_unit

    for data_1d, label in points_data:
        freqs, psd_values = welch(
            data_1d,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap_calc,
            detrend=detrend,
            scaling=scaling,
            average=average,
            return_onesided=return_onesided,
        )

        # Determine PSD unit
        if field.unit is not None:
            if scaling == "density":
                psd_unit = field.unit**2 / freq_unit
            else:  # spectrum
                psd_unit = field.unit**2
        else:
            psd_unit = u.dimensionless_unscaled

        fs_obj = FrequencySeries(
            psd_values,
            frequencies=freqs * freq_unit,
            unit=psd_unit,
            name=label,
        )
        results.append(fs_obj)

    if len(results) == 1:
        return results[0]
    return FrequencySeriesList(results)


def _extract_region_average(
    field: ScalarField,
    region: dict[str, Any],
) -> np.ndarray:
    """Extract time series averaged over a spatial region.

    Parameters
    ----------
    region : dict
        Dictionary with keys 'x', 'y', 'z' specifying slices or values.
        Use slice(None) for full axis, or a single Quantity for a point.
    """
    from gwexpy.plot._coord import nearest_index, slice_from_index

    slices = [slice(None), slice(None), slice(None), slice(None)]  # t, x, y, z

    axes_info = [
        ("x", 1, field._axis1_index),
        ("y", 2, field._axis2_index),
        ("z", 3, field._axis3_index),
    ]

    for name, idx, axis_index in axes_info:
        if name in region:
            spec = region[name]
            if isinstance(spec, slice):
                slices[idx] = spec
            elif hasattr(spec, "unit"):
                # Single value -> convert to index
                i = nearest_index(axis_index, spec)
                slices[idx] = slice_from_index(i)
            else:
                slices[idx] = spec

    # Extract region
    extracted = field.value[tuple(slices)]

    # Average over spatial dimensions (axes 1, 2, 3 after slicing)
    # The time axis is always 0
    spatial_axes = tuple(range(1, extracted.ndim))
    return np.mean(extracted, axis=spatial_axes)


def freq_space_map(
    field: ScalarField,
    axis: str,
    at: dict[str, Quantity] | None = None,
    *,
    method: Literal["welch", "fft"] = "welch",
    fftlength: float | None = None,
    overlap: float | None = None,
    nfft: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    **kwargs,
) -> ScalarField:
    """Compute frequency-space map along a spatial axis.

    Scans along the specified spatial axis, computing PSD at each position
    while fixing other spatial coordinates. Produces a 2D map with
    frequency on one axis and spatial coordinate on the other.

    Parameters
    ----------
    field : ScalarField
        Input 4D field with axis0_domain='time'.
    axis : str
        Spatial axis to scan along ('x', 'y', or 'z').
    at : dict, optional
        Fixed values for the other two spatial axes.
        Example: ``{'y': 0*u.m, 'z': 0*u.m}`` when axis='x'.
    method : {'welch', 'fft'}
        PSD estimation method. Default is 'welch'.
    fftlength : float, optional
        Segment length in seconds. Default is min(256, nt) samples.
    overlap : float, optional
        Overlap in seconds. Default is fftlength / 2.
    window : str
        Window function. Default is 'hann'.
    detrend : str or bool
        Detrending method. Default is 'constant'.
    scaling : {'density', 'spectrum'}
        PSD scaling. Default is 'density'.
    **kwargs
        Additional keyword arguments. Passing the removed ``nperseg``
        or ``noverlap`` parameters will raise :class:`TypeError`.

    Returns
    -------
    ScalarField
        2D frequency-space map stored as ScalarField with:
        - axis0: frequency (with axis0_domain='frequency')
        - axis1: spatial coordinate
        - axis2, axis3: length-1 dummy axes

    Examples
    --------
    >>> from astropy import units as u
    >>> fsmap = freq_space_map(field, 'x', at={'y': 0*u.m, 'z': 0*u.m})
    >>> fsmap.shape  # (n_freq, n_x, 1, 1)

    See Also
    --------
    compute_psd : Single-point PSD computation.
    compute_freq_space : Alias for this function.
    """
    from scipy.signal import welch

    from gwexpy.plot._coord import nearest_index, slice_from_index

    from ..utils.fft_args import (
        check_deprecated_kwargs,
        get_default_overlap,
        parse_fftlength_or_overlap,
        validate_and_convert_fft_params,
    )

    check_deprecated_kwargs(**kwargs)

    dt_value, dt_unit = _validate_regular_time_axis(field)
    fs = 1.0 / dt_value

    nt = field.shape[0]

    # Validate and convert nfft/noverlap to fftlength/overlap if needed
    fftlength, overlap = validate_and_convert_fft_params(
        fftlength=fftlength,
        overlap=overlap,
        nfft=nfft,
        noverlap=noverlap,
        sample_rate=fs,
    )

    # Convert fftlength/overlap to samples
    fftlength_sec, nperseg = parse_fftlength_or_overlap(fftlength, fs, "fftlength")
    overlap_sec, noverlap_calc = parse_fftlength_or_overlap(overlap, fs, "overlap")

    if nperseg is None:
        nperseg = min(256, nt)
    if noverlap_calc is None:
        if fftlength_sec is not None:
            overlap_sec_default = get_default_overlap(fftlength_sec, window=window)
            if overlap_sec_default is not None:
                noverlap_calc = max(1, int(round(overlap_sec_default * fs)))
            else:
                noverlap_calc = 0
        else:
            noverlap_calc = nperseg // 2

    # Map axis name to index
    axis_int = field._get_axis_index(axis)
    if axis_int == 0:
        raise ValueError("Cannot use time axis for freq_space_map. Use compute_psd.")

    # Get axis coordinates
    axis_mapping = {
        1: (field._axis1_name, field._axis1_index),
        2: (field._axis2_name, field._axis2_index),
        3: (field._axis3_name, field._axis3_index),
    }
    scan_axis_name, scan_axis_index = axis_mapping[axis_int]
    n_scan = len(scan_axis_index)

    # Build base slice with fixed coordinates
    if at is None:
        at = {}

    slices = [slice(None)] * 4  # t, x, y, z

    for ax_int, (ax_name, ax_index) in axis_mapping.items():
        if ax_int == axis_int:
            continue  # We'll iterate over this axis

        if ax_name in at:
            idx = nearest_index(ax_index, at[ax_name])
            slices[ax_int] = slice_from_index(idx)
        elif len(ax_index) == 1:
            slices[ax_int] = slice(0, 1)
        else:
            raise ValueError(
                f"Axis '{ax_name}' has length {len(ax_index)} > 1. "
                f"Specify its value in 'at'."
            )

    # Compute PSD at each position along scan axis
    psd_list = []

    for i_pos in range(n_scan):
        slices[axis_int] = slice(i_pos, i_pos + 1)
        data_1d = field.value[tuple(slices)].squeeze()

        if method == "welch":
            freqs, psd_values = welch(
                data_1d,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap_calc,
                detrend=detrend,
                scaling=scaling,
                return_onesided=True,
            )
        else:  # fft
            fft_result = np.fft.rfft(data_1d) / len(data_1d)
            psd_values = np.abs(fft_result) ** 2
            psd_values[1:-1] *= 2  # One-sided correction
            freqs = np.fft.rfftfreq(len(data_1d), d=dt_value)

        psd_list.append(psd_values)

    # Stack into 2D array: (n_freq, n_scan)
    psd_2d = np.array(psd_list).T

    # Reshape to 4D: (n_freq, n_scan, 1, 1) or appropriate dimension
    # We need to place n_scan at the correct axis position
    shape_4d = [len(freqs), 1, 1, 1]
    shape_4d[axis_int] = n_scan
    psd_4d = psd_2d.reshape(shape_4d)

    # Build axis arrays
    freq_unit = 1 / dt_unit
    freq_axis = freqs * freq_unit

    # Dummy axes for non-scan spatial axes
    dummy_axes = {}
    for ax_int, (ax_name, ax_index) in axis_mapping.items():
        if ax_int == axis_int:
            dummy_axes[ax_int] = scan_axis_index
        else:
            # Use the single fixed value
            if ax_name in at:
                idx = nearest_index(ax_index, at[ax_name])
                dummy_axes[ax_int] = ax_index[idx : idx + 1]
            else:
                dummy_axes[ax_int] = ax_index[:1]

    # Determine unit
    if field.unit is not None:
        if scaling == "density":
            psd_unit = field.unit**2 / freq_unit
        else:
            psd_unit = field.unit**2
    else:
        psd_unit = u.dimensionless_unscaled

    from gwexpy.fields import ScalarField

    return ScalarField(
        psd_4d,
        unit=psd_unit,
        axis0=freq_axis,
        axis1=dummy_axes[1],
        axis2=dummy_axes[2],
        axis3=dummy_axes[3],
        axis_names=["f", field._axis1_name, field._axis2_name, field._axis3_name],
        axis0_domain="frequency",
        space_domain=field._space_domains,
    )


# Alias
compute_freq_space = freq_space_map


def compute_xcorr(
    field: ScalarField,
    point_a: tuple[Quantity, Quantity, Quantity],
    point_b: tuple[Quantity, Quantity, Quantity],
    *,
    max_lag: int | Quantity | None = None,
    mode: Literal["full", "same", "valid"] = "full",
    normalize: bool = True,
    detrend: bool = True,
    window: str | None = None,
) -> TimeSeries:
    """Compute cross-correlation between two spatial points.

    Calculates the normalized (or unnormalized) cross-correlation function
    between time series extracted at two spatial locations.

    Parameters
    ----------
    field : ScalarField
        Input 4D field with axis0_domain='time'.
    point_a, point_b : tuple of Quantity
        Spatial coordinates (x, y, z) for the two points.
    max_lag : int or Quantity, optional
        Maximum lag to compute. If Quantity, converted to samples.
        If None, uses all available lags.
    mode : {'full', 'same', 'valid'}
        Correlation mode (see numpy.correlate). Default is 'full'.
    normalize : bool
        If True, normalize to [-1, 1] range. Default True.
    detrend : bool
        If True, remove mean before correlation. Default True.
    window : str, optional
        Window function to apply before correlation. Default None.

    Returns
    -------
    TimeSeries
        Cross-correlation function with lag axis in time units.
        Positive lag means point_b leads point_a.

    Examples
    --------
    >>> from astropy import units as u
    >>> xcorr = compute_xcorr(field, (0*u.m, 0*u.m, 0*u.m), (1*u.m, 0*u.m, 0*u.m))
    >>> xcorr.times  # Lag axis
    >>> xcorr.value  # Correlation values

    Notes
    -----
    The correlation is computed as ``correlate(a, b, mode)``, where
    positive values in the output correspond to ``b`` leading ``a``.
    """
    from scipy.signal import correlate, get_window

    from gwexpy.timeseries import TimeSeries

    # Extract time series
    dt_value, dt_unit = _validate_regular_time_axis(field)
    data_a, _, _ = _extract_timeseries_1d(field, point_a)
    data_b, _, _ = _extract_timeseries_1d(field, point_b)

    # Detrend
    if detrend:
        data_a = data_a - np.mean(data_a)
        data_b = data_b - np.mean(data_b)

    # Apply window
    if window is not None:
        n = len(data_a)
        win = get_window(window, n)
        data_a = data_a * win
        data_b = data_b * win

    # Compute correlation
    corr = correlate(data_a, data_b, mode=mode)

    # Normalize
    if normalize:
        norm = np.sqrt(np.sum(data_a**2) * np.sum(data_b**2))
        if norm > 0:
            corr = corr / norm

    # Build lag axis
    n = len(data_a)
    if mode == "full":
        lags = np.arange(-(n - 1), n)
    elif mode == "same":
        lags = np.arange(-(n // 2), n - n // 2)
    else:  # valid
        lags = np.arange(len(corr))

    lag_times = lags * dt_value * dt_unit

    # Apply max_lag truncation
    if max_lag is not None:
        if isinstance(max_lag, u.Quantity):
            max_lag_samples = int(max_lag.to_value(dt_unit) / dt_value)
        else:
            max_lag_samples = int(max_lag)

        center = len(lags) // 2
        start = max(0, center - max_lag_samples)
        end = min(len(lags), center + max_lag_samples + 1)

        corr = corr[start:end]
        lag_times = lag_times[start:end]

    return TimeSeries(
        corr,
        times=lag_times,
        unit=u.dimensionless_unscaled if normalize else field.unit**2,
        name="xcorr",
    )


def time_delay_map(
    field: ScalarField,
    ref_point: tuple[Quantity, Quantity, Quantity],
    plane: str = "xy",
    at: dict[str, Quantity] | None = None,
    *,
    max_lag: int | Quantity | None = None,
    stride: int = 1,
    roi: dict[str, slice] | None = None,
    normalize: bool = True,
    detrend: bool = True,
) -> ScalarField:
    """Compute time delay map from a reference point to a 2D slice.

    For each point in the specified plane, computes cross-correlation
    with the reference point and extracts the lag at maximum correlation.

    Parameters
    ----------
    field : ScalarField
        Input 4D field with axis0_domain='time'.
    ref_point : tuple of Quantity
        Reference point coordinates (x, y, z).
    plane : str
        2D plane to map: 'xy', 'xz', or 'yz'. Default 'xy'.
    at : dict, optional
        Fixed value for the axis not in the plane.
        Example: ``{'z': 0*u.m}`` when plane='xy'.
    max_lag : int or Quantity, optional
        Maximum lag to search for peak. Default uses all lags.
    stride : int
        Subsample stride to reduce computation. Default 1 (no subsampling).
    roi : dict, optional
        Region of interest as dict of slices. Example:
        ``{'x': slice(10, 50), 'y': slice(20, 80)}``
    normalize : bool
        Normalize correlation. Default True.
    detrend : bool
        Detrend before correlation. Default True.

    Returns
    -------
    ScalarField
        Time delay map as ScalarField slice with delay values in time units.
        Shape matches the input plane dimensions (with stride applied).

    Examples
    --------
    >>> from astropy import units as u
    >>> delay_map = time_delay_map(
    ...     field,
    ...     ref_point=(0*u.m, 0*u.m, 0*u.m),
    ...     plane='xy',
    ...     at={'z': 0*u.m}
    ... )
    >>> delay_map.plot_map2d('xy')

    Notes
    -----
    Positive delay means the point leads the reference.
    """
    from scipy.signal import correlate

    from gwexpy.plot._coord import nearest_index

    dt_value, dt_unit = _validate_regular_time_axis(field)

    # Extract reference time series
    ref_data, _, _ = _extract_timeseries_1d(field, ref_point)
    if detrend:
        ref_data = ref_data - np.mean(ref_data)

    # Parse plane to get the two axes
    plane_map = {
        "xy": (1, 2, 3),  # scan x, y; fix z
        "xz": (1, 3, 2),  # scan x, z; fix y
        "yz": (2, 3, 1),  # scan y, z; fix x
    }
    if plane.lower() not in plane_map:
        raise ValueError(f"Invalid plane '{plane}'. Must be 'xy', 'xz', or 'yz'.")

    ax1_int, ax2_int, fix_int = plane_map[plane.lower()]

    # Get axis info
    axis_mapping = {
        1: (field._axis1_name, field._axis1_index),
        2: (field._axis2_name, field._axis2_index),
        3: (field._axis3_name, field._axis3_index),
    }

    ax1_name, ax1_index = axis_mapping[ax1_int]
    ax2_name, ax2_index = axis_mapping[ax2_int]
    fix_name, fix_index = axis_mapping[fix_int]

    # Fixed axis
    if at is None:
        at = {}
    if fix_name in at:
        fix_idx = nearest_index(fix_index, at[fix_name])
    elif len(fix_index) == 1:
        fix_idx = 0
    else:
        raise ValueError(
            f"Axis '{fix_name}' has length {len(fix_index)} > 1. "
            f"Specify its value in 'at'."
        )

    # Apply ROI and stride
    if roi is None:
        roi = {}

    ax1_slice = roi.get(ax1_name, slice(None))
    ax2_slice = roi.get(ax2_name, slice(None))

    ax1_indices = range(*ax1_slice.indices(len(ax1_index)))[::stride]
    ax2_indices = range(*ax2_slice.indices(len(ax2_index)))[::stride]

    n1 = len(ax1_indices)
    n2 = len(ax2_indices)

    # Compute delay for each point
    delay_values = np.zeros((n1, n2))
    nt = field.shape[0]

    # Max lag in samples
    if max_lag is not None:
        if isinstance(max_lag, u.Quantity):
            max_lag_samples = int(max_lag.to_value(dt_unit) / dt_value)
        else:
            max_lag_samples = int(max_lag)
    else:
        max_lag_samples = nt - 1

    for i1, idx1 in enumerate(ax1_indices):
        for i2, idx2 in enumerate(ax2_indices):
            # Build index for extraction
            slices = [slice(None), 0, 0, 0]
            slices[ax1_int] = idx1
            slices[ax2_int] = idx2
            slices[fix_int] = fix_idx

            data = field.value[tuple(slices)]
            if detrend:
                data = data - np.mean(data)

            # Cross-correlation
            corr = correlate(ref_data, data, mode="full")
            if normalize:
                norm = np.sqrt(np.sum(ref_data**2) * np.sum(data**2))
                if norm > 0:
                    corr = corr / norm

            # Find peak within max_lag
            lags = np.arange(-(nt - 1), nt)
            center = nt - 1

            start = max(0, center - max_lag_samples)
            end = min(len(lags), center + max_lag_samples + 1)

            search_corr = corr[start:end]
            search_lags = lags[start:end]

            peak_idx = np.argmax(np.abs(search_corr))
            delay_samples = search_lags[peak_idx]
            delay_values[i1, i2] = delay_samples * dt_value

    # Build output ScalarField
    # Shape: (1, n1 or 1, n2 or 1, 1) depending on plane
    shape_4d = [1, 1, 1, 1]
    shape_4d[ax1_int] = n1
    shape_4d[ax2_int] = n2
    delay_4d = delay_values.reshape(shape_4d)

    # Build axis arrays
    ax1_val = getattr(ax1_index, "value", ax1_index)
    ax1_unit = getattr(ax1_index, "unit", u.dimensionless_unscaled)
    ax2_val = getattr(ax2_index, "value", ax2_index)
    ax2_unit = getattr(ax2_index, "unit", u.dimensionless_unscaled)

    ax1_out = ax1_val[list(ax1_indices)] * ax1_unit
    ax2_out = ax2_val[list(ax2_indices)] * ax2_unit

    axes_out = {
        1: field._axis1_index[:1],
        2: field._axis2_index[:1],
        3: field._axis3_index[:1],
    }
    axes_out[ax1_int] = ax1_out
    axes_out[ax2_int] = ax2_out
    axes_out[fix_int] = fix_index[fix_idx : fix_idx + 1]

    from gwexpy.fields import ScalarField

    return ScalarField(
        delay_4d,
        unit=dt_unit,
        axis0=np.array([0.0]) * u.s,  # Dummy time axis
        axis1=axes_out[1],
        axis2=axes_out[2],
        axis3=axes_out[3],
        axis_names=["t", field._axis1_name, field._axis2_name, field._axis3_name],
        axis0_domain="time",
        space_domain=field._space_domains,
    )


def coherence_map(
    field: ScalarField,
    ref_point: tuple[Quantity, Quantity, Quantity],
    *,
    plane: str = "xy",
    at: dict[str, Quantity] | None = None,
    band: tuple[Quantity, Quantity] | None = None,
    fftlength: float | None = None,
    overlap: float | None = None,
    window: str = "hann",
    stride: int = 1,
    **kwargs,
) -> ScalarField | FieldDict:
    """Compute magnitude-squared coherence map from a reference point.

    Calculates coherence between the reference time series and all points
    in the specified 2D slice.

    Parameters
    ----------
    field : ScalarField
        Input 4D field with axis0_domain='time'.
    ref_point : tuple of Quantity
        Reference point coordinates (x, y, z).
    plane : str
        2D plane to map: 'xy', 'xz', or 'yz'. Default 'xy'.
    at : dict, optional
        Fixed value for the axis not in the plane.
    band : tuple of Quantity, optional
        Frequency band ``(fmin, fmax)`` to average coherence over.
        If None, returns coherence at all frequencies.
    fftlength : float, optional
        Segment length in seconds. Default min(256, nt) samples.
    overlap : float, optional
        Overlap in seconds. Default fftlength / 2.
    window : str
        Window function. Default 'hann'.
    stride : int
        Subsample stride. Default 1.
    **kwargs
        Additional keyword arguments. Passing the removed ``nperseg``
        or ``noverlap`` parameters will raise :class:`TypeError`.

    Returns
    -------
    ScalarField or FieldDict
        - If band is specified: ScalarField with scalar coherence values (0-1).
        - If band is None: FieldDict with keys as frequency indices,
          or a single ScalarField with frequency as axis0.

    Examples
    --------
    >>> from astropy import units as u
    >>> # Compute coherence map with time-based FFT parameters
    >>> coh_map = coherence_map(
    ...     field,
    ...     ref_point=(0*u.m, 0*u.m, 0*u.m),
    ...     plane='xy',
    ...     at={'z': 0*u.m},
    ...     band=(10*u.Hz, 100*u.Hz),
    ...     fftlength=1.0,    # 1.0 seconds
    ...     overlap=0.5,      # 0.5 seconds
    ...     window='hann'
    ... )
    >>> coh_map.plot_map2d('xy')
    """
    from scipy.signal import coherence

    from gwexpy.plot._coord import nearest_index

    from ..utils.fft_args import check_deprecated_kwargs, get_default_overlap, parse_fftlength_or_overlap

    check_deprecated_kwargs(**kwargs)

    dt_value, dt_unit = _validate_regular_time_axis(field)
    fs = 1.0 / dt_value

    nt = field.shape[0]

    # Convert fftlength/overlap to samples
    fftlength_sec, nperseg = parse_fftlength_or_overlap(fftlength, fs, "fftlength")
    overlap_sec, noverlap = parse_fftlength_or_overlap(overlap, fs, "overlap")

    if nperseg is None:
        nperseg = min(256, nt)
    if noverlap is None:
        if fftlength_sec is not None:
            overlap_sec_default = get_default_overlap(fftlength_sec, window=window)
            if overlap_sec_default is not None:
                noverlap = max(1, int(round(overlap_sec_default * fs)))
            else:
                noverlap = 0
        else:
            noverlap = nperseg // 2

    # Extract reference time series
    ref_data, _, _ = _extract_timeseries_1d(field, ref_point)

    # Parse plane
    plane_map = {
        "xy": (1, 2, 3),
        "xz": (1, 3, 2),
        "yz": (2, 3, 1),
    }
    if plane.lower() not in plane_map:
        raise ValueError(f"Invalid plane '{plane}'. Must be 'xy', 'xz', or 'yz'.")

    ax1_int, ax2_int, fix_int = plane_map[plane.lower()]

    axis_mapping = {
        1: (field._axis1_name, field._axis1_index),
        2: (field._axis2_name, field._axis2_index),
        3: (field._axis3_name, field._axis3_index),
    }

    ax1_name, ax1_index = axis_mapping[ax1_int]
    ax2_name, ax2_index = axis_mapping[ax2_int]
    fix_name, fix_index = axis_mapping[fix_int]

    # Fixed axis
    if at is None:
        at = {}
    if fix_name in at:
        fix_idx = nearest_index(fix_index, at[fix_name])
    elif len(fix_index) == 1:
        fix_idx = 0
    else:
        raise ValueError(
            f"Axis '{fix_name}' has length {len(fix_index)} > 1. "
            f"Specify its value in 'at'."
        )

    # Apply stride
    ax1_indices = range(len(ax1_index))[::stride]
    ax2_indices = range(len(ax2_index))[::stride]
    n1 = len(ax1_indices)
    n2 = len(ax2_indices)

    # Compute coherence for each point
    # First compute for one point to get frequency axis
    test_slices = [slice(None), ax1_indices[0], ax2_indices[0], fix_idx]
    test_slices = _build_extraction_slices(
        ax1_int, ax2_int, fix_int, ax1_indices[0], ax2_indices[0], fix_idx
    )
    test_data = field.value[tuple(test_slices)]
    freqs, _ = coherence(
        ref_data, test_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )

    n_freq = len(freqs)
    coh_3d = np.zeros((n_freq, n1, n2))

    for i1, idx1 in enumerate(ax1_indices):
        for i2, idx2 in enumerate(ax2_indices):
            slices = _build_extraction_slices(
                ax1_int, ax2_int, fix_int, idx1, idx2, fix_idx
            )
            data = field.value[tuple(slices)]

            _, cxy = coherence(
                ref_data, data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
            )
            coh_3d[:, i1, i2] = cxy

    freq_unit = 1 / dt_unit
    freq_axis = freqs * freq_unit

    # Band averaging if specified
    if band is not None:
        fmin, fmax = band
        fmin_val = fmin.to_value(freq_unit)
        fmax_val = fmax.to_value(freq_unit)

        mask = (freqs >= fmin_val) & (freqs <= fmax_val)
        if not mask.any():
            raise ValueError(
                f"No frequencies in band [{fmin}, {fmax}]. "
                f"Available range: [{freqs[0]}, {freqs[-1]}] {freq_unit}"
            )

        coh_band = np.mean(coh_3d[mask], axis=0)

        # Build 4D output
        shape_4d = [1, 1, 1, 1]
        shape_4d[ax1_int] = n1
        shape_4d[ax2_int] = n2
        coh_4d = coh_band.reshape(shape_4d)

        ax1_val = getattr(ax1_index, "value", ax1_index)
        ax1_unit = getattr(ax1_index, "unit", u.dimensionless_unscaled)
        ax2_val = getattr(ax2_index, "value", ax2_index)
        ax2_unit = getattr(ax2_index, "unit", u.dimensionless_unscaled)

        ax1_out = ax1_val[list(ax1_indices)] * ax1_unit
        ax2_out = ax2_val[list(ax2_indices)] * ax2_unit

        axes_out = {
            1: field._axis1_index[:1],
            2: field._axis2_index[:1],
            3: field._axis3_index[:1],
        }
        axes_out[ax1_int] = ax1_out
        axes_out[ax2_int] = ax2_out
        axes_out[fix_int] = fix_index[fix_idx : fix_idx + 1]

        from gwexpy.fields import ScalarField

        return ScalarField(
            coh_4d,
            unit=u.dimensionless_unscaled,
            axis0=np.array([0.0]) * u.s,
            axis1=axes_out[1],
            axis2=axes_out[2],
            axis3=axes_out[3],
            axis_names=["t", field._axis1_name, field._axis2_name, field._axis3_name],
            axis0_domain="time",
            space_domain=field._space_domains,
        )
    else:
        # Return with frequency axis
        shape_4d = [n_freq, 1, 1, 1]
        shape_4d[ax1_int] = n1
        shape_4d[ax2_int] = n2
        coh_4d = coh_3d.reshape(shape_4d)

        ax1_val = getattr(ax1_index, "value", ax1_index)
        ax1_unit = getattr(ax1_index, "unit", u.dimensionless_unscaled)
        ax2_val = getattr(ax2_index, "value", ax2_index)
        ax2_unit = getattr(ax2_index, "unit", u.dimensionless_unscaled)

        ax1_out = ax1_val[list(ax1_indices)] * ax1_unit
        ax2_out = ax2_val[list(ax2_indices)] * ax2_unit

        axes_out = {
            1: field._axis1_index[:1],
            2: field._axis2_index[:1],
            3: field._axis3_index[:1],
        }
        axes_out[ax1_int] = ax1_out
        axes_out[ax2_int] = ax2_out
        axes_out[fix_int] = fix_index[fix_idx : fix_idx + 1]

        from gwexpy.fields import ScalarField

        return ScalarField(
            coh_4d,
            unit=u.dimensionless_unscaled,
            axis0=freq_axis,
            axis1=axes_out[1],
            axis2=axes_out[2],
            axis3=axes_out[3],
            axis_names=["f", field._axis1_name, field._axis2_name, field._axis3_name],
            axis0_domain="frequency",
            space_domain=field._space_domains,
        )


def _build_extraction_slices(
    ax1_int: int,
    ax2_int: int,
    fix_int: int,
    idx1: int,
    idx2: int,
    fix_idx: int,
) -> list:
    """Build slice tuple for extracting 1D time series from specific spatial indices."""
    slices = [slice(None), 0, 0, 0]
    slices[ax1_int] = idx1
    slices[ax2_int] = idx2
    slices[fix_int] = fix_idx
    return slices
