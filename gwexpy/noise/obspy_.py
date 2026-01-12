"""gwexpy.noise.obspy_ - ObsPy wrapper for ASD generation.

This module provides functions to generate Amplitude Spectral Density (ASD)
from ObsPy seismic and infrasound noise models.
"""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np
from astropy import units as u

from ..frequencyseries import FrequencySeries


# Allowed quantity values for ObsPy seismic models
_OBSPY_QUANTITIES = frozenset({"displacement", "velocity", "acceleration"})


def from_obspy(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["displacement", "velocity", "acceleration"] = "acceleration",
    **kwargs: Any,
) -> FrequencySeries:
    """Get standard noise models from ObsPy.

    This function returns the Amplitude Spectral Density (ASD) of seismic
    or infrasound noise models from ObsPy.

    Parameters
    ----------
    model : str
        The ObsPy model name. Supported models:

        - ``"NHNM"``: New High Noise Model (Peterson 1993)
        - ``"NLNM"``: New Low Noise Model (Peterson 1993)
        - ``"IDCH"``: IDC Infrasound High Noise Model
        - ``"IDCL"``: IDC Infrasound Low Noise Model

    frequencies : array_like, optional
        Frequency array in Hz for interpolation. If not provided, the
        original model frequencies are used.
    quantity : {"displacement", "velocity", "acceleration"}, default "acceleration"
        Physical quantity of the ASD to return (for seismic models only):

        - ``"acceleration"``: Acceleration ASD [m/(s²·sqrt(Hz))]
        - ``"velocity"``: Velocity ASD [m/(s·sqrt(Hz))]
        - ``"displacement"``: Displacement ASD [m/sqrt(Hz)]

        .. note::
            The ``"strain"`` quantity is **not supported** for seismic noise
            models. Use ``from_pygwinc`` for strain-based detector noise.

    **kwargs
        Additional keyword arguments passed to FrequencySeries constructor.

    Returns
    -------
    FrequencySeries
        The ASD as a FrequencySeries with appropriate units.

    Raises
    ------
    ImportError
        If ObsPy is not installed.
    ValueError
        If `model` is not one of the supported models.
        If `quantity` is not one of the allowed values.
        If `quantity="strain"` is requested (not supported).

    Notes
    -----
    **Quantity Conversion (Seismic Models):**

    The NHNM and NLNM models are defined in terms of acceleration. Conversions
    to velocity and displacement are performed in the frequency domain:

    - ``velocity = acceleration / (2πf)``
    - ``displacement = acceleration / (2πf)²``

    At ``f=0``, the conversion results in ``NaN`` to avoid division by zero.

    **Unit Table:**

    ============== ========================
    Quantity       Unit
    ============== ========================
    acceleration   m / (s² · sqrt(Hz))
    velocity       m / (s · sqrt(Hz))
    displacement   m / sqrt(Hz)
    ============== ========================

    **Infrasound Models:**

    For IDCH and IDCL models, the quantity parameter is ignored as these
    return pressure ASD in units of Pa/sqrt(Hz).

    Examples
    --------
    >>> from gwexpy.noise.asd import from_obspy

    Get acceleration ASD for NLNM:

    >>> asd = from_obspy("NLNM")
    >>> asd.unit
    Unit("m / (Hz(1/2) s2)")

    Get displacement ASD:

    >>> disp_asd = from_obspy("NLNM", quantity="displacement")
    >>> disp_asd.unit
    Unit("m / Hz(1/2)")

    Strain is not supported:

    >>> from_obspy("NLNM", quantity="strain")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: quantity='strain' is not supported for seismic noise models...
    """
    try:
        from obspy.signal import spectral_estimation
    except ImportError:
        raise ImportError(
            "Please install obspy to use gwexpy.noise.asd.from_obspy"
        )

    # Validate quantity
    quantity_lower = quantity.lower()
    if quantity_lower == "strain":
        raise ValueError(
            "quantity='strain' is not supported for seismic noise models. "
            "Strain ASD is only available from gravitational wave detector "
            "models using from_pygwinc()."
        )

    if quantity_lower not in _OBSPY_QUANTITIES:
        raise ValueError(
            f"Invalid quantity '{quantity}'. "
            f"Allowed values are: 'displacement', 'velocity', 'acceleration'."
        )

    # Load model data
    model_upper = model.upper()
    if model_upper == "NHNM":
        periods, psd_db = spectral_estimation.get_nhnm()
        base_unit = u.Unit("m / s^2 / Hz^(1/2)")
        is_seismic = True
    elif model_upper == "NLNM":
        periods, psd_db = spectral_estimation.get_nlnm()
        base_unit = u.Unit("m / s^2 / Hz^(1/2)")
        is_seismic = True
    elif model_upper == "IDCH":
        periods, psd_db = spectral_estimation.get_idc_infra_hi_noise()
        base_unit = u.Unit("Pa / Hz^(1/2)")
        is_seismic = False
    elif model_upper == "IDCL":
        periods, psd_db = spectral_estimation.get_idc_infra_low_noise()
        base_unit = u.Unit("Pa / Hz^(1/2)")
        is_seismic = False
    else:
        raise ValueError(
            f"Unknown model: '{model}'. "
            f"Supported models: 'NHNM', 'NLNM', 'IDCH', 'IDCL'."
        )

    # Convert periods to frequencies and sort
    f_orig = 1.0 / periods
    idx = np.argsort(f_orig)
    f_orig = f_orig[idx]
    psd_db = psd_db[idx]
    asd_orig = np.sqrt(10 ** (psd_db / 10.0))

    # Interpolate to requested frequencies if provided
    if frequencies is not None:
        mask_orig = f_orig > 0
        mask_new = frequencies > 0

        log_f_orig = np.log10(f_orig[mask_orig])
        log_asd_orig = np.log10(asd_orig[mask_orig])

        asd = np.zeros_like(frequencies, dtype=float)
        if np.any(mask_new):
            log_f_new = np.log10(frequencies[mask_new])
            log_asd_new = np.interp(log_f_new, log_f_orig, log_asd_orig)
            asd[mask_new] = 10**log_asd_new
    else:
        frequencies = f_orig
        asd = asd_orig

    # Create FrequencySeries with base unit (acceleration for seismic)
    fs = FrequencySeries(
        asd, frequencies=frequencies, unit=base_unit, name=model_upper, **kwargs
    )

    # Apply quantity conversion for seismic models
    if is_seismic and quantity_lower != "acceleration":
        fs = _convert_seismic_quantity(fs, quantity_lower)

    return fs


def _convert_seismic_quantity(
    fs: FrequencySeries,
    quantity: str,
) -> FrequencySeries:
    """Convert seismic ASD from acceleration to velocity or displacement.

    Parameters
    ----------
    fs : FrequencySeries
        Input ASD in acceleration units [m/(s²·sqrt(Hz))].
    quantity : str
        Target quantity: "velocity" or "displacement".

    Returns
    -------
    FrequencySeries
        Converted ASD with appropriate units.

    Notes
    -----
    At f=0, the result is NaN to avoid infinity from division by zero.
    """
    f = fs.frequencies.to("Hz").value
    omega = 2 * np.pi * f

    if quantity == "velocity":
        target_unit = u.Unit("m / s / Hz^(1/2)")
        with np.errstate(divide="ignore", invalid="ignore"):
            data = fs.value / omega
            # Set f=0 to NaN (not 0 or inf)
            if len(f) > 0 and f[0] == 0:
                data[0] = np.nan

    elif quantity == "displacement":
        target_unit = u.Unit("m / Hz^(1/2)")
        with np.errstate(divide="ignore", invalid="ignore"):
            data = fs.value / (omega**2)
            # Set f=0 to NaN (not 0 or inf)
            if len(f) > 0 and f[0] == 0:
                data[0] = np.nan

    else:
        # Should not reach here due to earlier validation
        raise ValueError(f"Unknown quantity: {quantity}")

    return FrequencySeries(
        data, frequencies=fs.frequencies, unit=target_unit, name=fs.name
    )
