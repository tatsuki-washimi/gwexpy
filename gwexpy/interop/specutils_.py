"""Interoperability with specutils."""

from ._optional import require_optional


def to_specutils(data, **kwargs):
    """
    Convert a gwexpy object to a specutils object.

    Parameters
    ----------
    data : FrequencySeries
        Input data.
    **kwargs
        Additional arguments for Spectrum1D constructor.

    Returns
    -------
    specutils.Spectrum1D
    """
    specutils = require_optional("specutils")
    astropy_units = require_optional("astropy.units")

    # Ensure data has unit
    value = data.value
    if data.unit is not None:
        # gwexpy/astropy unit handling
        # If data.unit is astropy unit, use it.
        # If string, try to convert? gwexpy Series usually manages units.
        # Ideally data.value * data.unit if data.unit is Quantity-compatible
        # But usually data.value is Quantity or unit is stored separately.
        # gwexpy assumes Quantity-like behavior often?
        # Actually typically we construct Quantity manually if needed.
        try:
            flux = value * data.unit
        except Exception:
            # Fallback if unit multiplication fails or simple numpy array
            flux = value * astropy_units.dimensionless_unscaled
    else:
        flux = value * astropy_units.dimensionless_unscaled

    # Spectral axis
    if hasattr(data, "frequencies"):
        spectral_axis = data.frequencies.value
        # Unit for frequencies?
        # data.frequencies might have a unit or be Quantity
        if hasattr(data.frequencies, "unit") and data.frequencies.unit is not None:
            spectral_axis = spectral_axis * data.frequencies.unit
        else:
            # Default to Hz? Or keep dimensionless?
            # Specutils strongly prefers units for spectral axis.
            # gwexpy frequencies are usually Hz.
            spectral_axis = spectral_axis * astropy_units.Hz
    else:
        raise ValueError("Input data must have frequencies attribute.")

    return specutils.Spectrum1D(flux=flux, spectral_axis=spectral_axis, **kwargs)


def from_specutils(cls, spectrum, **kwargs):
    """
    Convert a specutils object to a gwexpy object.

    Parameters
    ----------
    cls : class
        Target class (FrequencySeries).
    spectrum : specutils.Spectrum1D
        Input spectrum.

    Returns
    -------
    FrequencySeries
    """
    # Extract data
    flux = spectrum.flux
    # Should we keep unit?
    # FrequencySeries can take Quantity.

    # Extract frequencies
    spectral_axis = spectrum.spectral_axis

    return cls(flux, frequencies=spectral_axis, **kwargs)
