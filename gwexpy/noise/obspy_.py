from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ..frequencyseries import FrequencySeries

if TYPE_CHECKING:
    from typing import Any, Union


def from_obspy(
    model: str,
    frequencies: np.ndarray | None = None,
    unit: Union[str, u.Unit, None] = None,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Get standard noise models from ObsPy.

    Parameters
    ----------
    model : `str`
        Model name: 'NHNM', 'NLNM' (Seismic), 'IDCH', 'IDCL' (Infrasound).
    frequencies : `numpy.ndarray`, optional
        Frequencies to interpolate to.
    unit : `str` or `astropy.units.Unit`, optional
        Desired output unit. Automatic integration/conversion is performed.
        Base unit for Seismic is 'm/s^2/sqrt(Hz)'.
        Base unit for Infrasound is 'Pa/sqrt(Hz)'.
    **kwargs
        Additional arguments for FrequencySeries.

    Returns
    -------
    `FrequencySeries`
    """
    try:
        from obspy.signal import spectral_estimation
    except ImportError:
        raise ImportError("Please install obspy to use gwexpy.noise.from_obspy")

    model_upper = model.upper()
    if model_upper == "NHNM":
        periods, psd_db = spectral_estimation.get_nhnm()
        base_unit = u.m / u.s**2 / u.Hz**0.5
        is_seismic = True
    elif model_upper == "NLNM":
        periods, psd_db = spectral_estimation.get_nlnm()
        base_unit = u.m / u.s**2 / u.Hz**0.5
        is_seismic = True
    elif model_upper == "IDCH":
        periods, psd_db = spectral_estimation.get_idc_infra_hi_noise()
        base_unit = u.Pa / u.Hz**0.5
        is_seismic = False
    elif model_upper == "IDCL":
        periods, psd_db = spectral_estimation.get_idc_infra_low_noise()
        base_unit = u.Pa / u.Hz**0.5
        is_seismic = False
    else:
        raise ValueError(f"Unknown model: {model}. Supported: NHNM, NLNM, IDCH, IDCL")

    # Convert periods to frequencies and sort
    f_orig = 1.0 / periods
    idx = np.argsort(f_orig)
    f_orig = f_orig[idx]
    psd_db = psd_db[idx]

    # Convert PSD (dB) to ASD
    asd_orig = np.sqrt(10 ** (psd_db / 10.0))

    if frequencies is not None:
        # Log-log interpolation
        mask_orig = f_orig > 0
        mask_new = frequencies > 0

        log_f_orig = np.log10(f_orig[mask_orig])
        log_asd_orig = np.log10(asd_orig[mask_orig])

        log_f_new = np.log10(frequencies[mask_new])
        log_asd_new = np.interp(log_f_new, log_f_orig, log_asd_orig)

        asd = np.zeros_like(frequencies)
        asd[mask_new] = 10**log_asd_new
    else:
        frequencies = f_orig
        asd = asd_orig

    fs = FrequencySeries(asd, frequencies=frequencies, unit=base_unit, name=model_upper, **kwargs)

    if unit is not None:
        fs = _convert_obspy_unit(fs, unit, is_seismic)

    return fs


def _convert_obspy_unit(
    fs: FrequencySeries, target_unit: Union[str, u.Unit], is_seismic: bool
) -> FrequencySeries:
    target_unit = u.Unit(target_unit)

    if is_seismic:
        # Seismic base is acceleration [m/s^2/sqrt(Hz)]
        velocity_unit = u.m / u.s / u.Hz**0.5
        displacement_unit = u.m / u.Hz**0.5

        f = fs.frequencies.to("Hz").value
        omega = 2 * np.pi * f

        if target_unit.is_equivalent(velocity_unit):
            with np.errstate(divide="ignore", invalid="ignore"):
                data = fs.value / omega
                if f[0] == 0:
                    data[0] = 0
            res = FrequencySeries(
                data, frequencies=fs.frequencies, unit=velocity_unit, name=fs.name
            )
            return res.to(target_unit)

        if target_unit.is_equivalent(displacement_unit):
            with np.errstate(divide="ignore", invalid="ignore"):
                data = fs.value / (omega**2)
                if f[0] == 0:
                    data[0] = 0
            res = FrequencySeries(
                data, frequencies=fs.frequencies, unit=displacement_unit, name=fs.name
            )
            return res.to(target_unit)

    # For Non-seismic or cases where it's already equivalent (e.g. Pa -> hPa)
    return fs.to(target_unit)
