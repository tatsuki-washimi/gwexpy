from __future__ import annotations
from typing import Any, Union
import numpy as np
from astropy import units as u
from ..frequencyseries import FrequencySeries

def _convert_obspy_unit(
    fs: FrequencySeries, target_unit: Union[str, u.Unit], is_seismic: bool
) -> FrequencySeries:
    target_unit = u.Unit(target_unit)

    if is_seismic:
        # Seismic base is acceleration [m/s^2/sqrt(Hz)]
        velocity_unit = u.Unit("m / s / Hz^(1/2)")
        displacement_unit = u.Unit("m / Hz^(1/2)")

        f = fs.frequencies.to("Hz").value
        omega = 2 * np.pi * f

        if target_unit.is_equivalent(velocity_unit):
            with np.errstate(divide="ignore", invalid="ignore"):
                data = fs.value / omega
                if f[0] == 0:
                    data[0] = 0
            return FrequencySeries(
                data, frequencies=fs.frequencies, unit=velocity_unit, name=fs.name
            ).to(target_unit)

        if target_unit.is_equivalent(displacement_unit):
            with np.errstate(divide="ignore", invalid="ignore"):
                data = fs.value / (omega**2)
                if f[0] == 0:
                    data[0] = 0
            return FrequencySeries(
                data, frequencies=fs.frequencies, unit=displacement_unit, name=fs.name
            ).to(target_unit)

    return fs.to(target_unit)


def from_obspy(
    model: str,
    frequencies: np.ndarray | None = None,
    unit: Union[str, u.Unit, None] = None,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Get standard noise models from ObsPy.
    """
    try:
        from obspy.signal import spectral_estimation
    except ImportError:
        raise ImportError("Please install obspy to use gwexpy.noise.from_obspy")

    model_upper = model.upper()
    if model_upper in ["NHNM", "NLNM"]:
        periods, psd_db = (spectral_estimation.get_nhnm() if model_upper == "NHNM" 
                           else spectral_estimation.get_nlnm())
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
        raise ValueError(f"Unknown model: {model}")

    f_orig = 1.0 / periods
    idx = np.argsort(f_orig)
    f_orig = f_orig[idx]
    psd_db = psd_db[idx]
    asd_orig = np.sqrt(10 ** (psd_db / 10.0))

    if frequencies is not None:
        mask_orig = f_orig > 0
        mask_new = frequencies > 0
        log_f_orig = np.log10(f_orig[mask_orig])
        log_asd_orig = np.log10(asd_orig[mask_orig])
        
        asd = np.zeros_like(frequencies)
        if np.any(mask_new):
            log_f_new = np.log10(frequencies[mask_new])
            log_asd_new = np.interp(log_f_new, log_f_orig, log_asd_orig)
            asd[mask_new] = 10**log_asd_new
    else:
        frequencies = f_orig
        asd = asd_orig

    fs = FrequencySeries(asd, frequencies=frequencies, unit=base_unit, name=model_upper, **kwargs)

    if unit is not None:
        fs = _convert_obspy_unit(fs, unit, is_seismic)

    return fs
