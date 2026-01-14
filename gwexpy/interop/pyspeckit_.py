"""Interoperability with pyspeckit."""

from ._optional import require_optional
from .base import to_plain_array


def to_pyspeckit(data, **kwargs):
    """
    Convert a gwexpy object to a pyspeckit Spectrum.

    Parameters
    ----------
    data : FrequencySeries
        Input data.
    **kwargs
        Additional arguments for pyspeckit.Spectrum.

    Returns
    -------
    pyspeckit.Spectrum
    """
    pyspeckit = require_optional("pyspeckit")

    # Extract data as plain numpy array
    d = to_plain_array(data)

    # Extract frequencies
    if hasattr(data, "frequencies"):
        xarr = to_plain_array(data.frequencies)
        # Handle units if possible?
        # pyspeckit xarr can be Quantity or have units kwarg.
        # But simplistic approach: just pass array and let user refine or pass kwargs.
    else:
        raise ValueError("Input data must have frequencies attribute.")

    return pyspeckit.Spectrum(data=d, xarr=xarr, **kwargs)


def from_pyspeckit(cls, spectrum, **kwargs):
    """
    Convert a pyspeckit Spectrum to a gwexpy object.

    Parameters
    ----------
    cls : class
        Target class (FrequencySeries).
    spectrum : pyspeckit.Spectrum
        Input spectrum.

    Returns
    -------
    FrequencySeries
    """
    # Extract data
    data = spectrum.data

    # Extract frequencies
    xarr = spectrum.xarr

    # If xarr is Quantity-like or has unit logic, preserve it?
    # pyspeckit uses its own unit system primarily but integrates with astropy units somewhat.

    # We assume xarr is the frequency axis.
    return cls(data, frequencies=xarr, **kwargs)
