"""gwexpy.noise.gwinc_ - pyGWINC wrapper for ASD generation.

This module provides functions to generate Amplitude Spectral Density (ASD)
from pyGWINC detector noise models.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..frequencyseries import FrequencySeries

# Allowed quantity values for pyGWINC
_PYGWINC_QUANTITIES = frozenset({"strain", "darm", "displacement"})


def from_pygwinc(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["strain", "darm", "displacement"] = "strain",
    **kwargs: Any,
) -> FrequencySeries:
    """Get detector noise ASD from pyGWINC.

    This function returns the Amplitude Spectral Density (ASD) of a
    gravitational wave detector noise model using pyGWINC.

    Parameters
    ----------
    model : str
        The pyGWINC model name (e.g., "aLIGO", "Aplus", "Voyager").
        "A+" is automatically converted to "Aplus".
    frequencies : array_like, optional
        Frequency array in Hz. If not provided, a default array is generated
        using `fmin`, `fmax`, and `df` from kwargs.
    quantity : {"strain", "darm", "displacement"}, default "strain"
        Physical quantity of the ASD to return:

        - ``"strain"``: Strain ASD [1/sqrt(Hz)]
        - ``"darm"``: Differential arm length ASD [m/sqrt(Hz)]
        - ``"displacement"``: Deprecated alias for "darm"

    **kwargs
        Additional keyword arguments:

        - ``fmin`` : float, default 10.0
            Minimum frequency [Hz] for default frequency array.
        - ``fmax`` : float, default 4000.0
            Maximum frequency [Hz] for default frequency array.
        - ``df`` : float, default 1.0
            Frequency step [Hz] for default frequency array.

    Returns
    -------
    FrequencySeries
        The ASD as a FrequencySeries with appropriate units.

    Raises
    ------
    ImportError
        If pyGWINC is not installed.
    ValueError
        If `quantity` is not one of the allowed values.
        If `quantity="darm"` or `"displacement"` and arm length cannot be
        retrieved from the IFO model.
        If `fmin >= fmax`.

    Notes
    -----
    **Quantity Definitions:**

    - **Strain ASD**: The detector's sensitivity expressed as equivalent
      gravitational wave strain per sqrt(Hz).
    - **DARM ASD**: Differential Arm Length Motion ASD, related to strain
      by ``darm = strain * L`` where L is the arm length.

    **Arm Length Conversion:**

    For ``quantity="darm"`` (or ``"displacement"``), the strain ASD is
    multiplied by the arm length L obtained from ``ifo.Infrastructure.Length``.

    **Frequency Array:**

    If `frequencies` is not provided, a uniform frequency array is generated:
    ``np.arange(fmin, fmax + df, df)``.

    Examples
    --------
    >>> from gwexpy.noise.asd import from_pygwinc

    Get strain ASD for aLIGO:

    >>> asd = from_pygwinc("aLIGO", fmin=10.0, fmax=1000.0, df=1.0)
    >>> asd.unit
    Unit("1 / Hz(1/2)")

    Get DARM ASD:

    >>> darm_asd = from_pygwinc("aLIGO", quantity="darm")
    >>> darm_asd.unit
    Unit("m / Hz(1/2)")
    """
    try:
        import gwinc
    except ImportError:
        raise ImportError("Please install pygwinc to use gwexpy.noise.asd.from_pygwinc")

    # Validate quantity
    quantity_lower = quantity.lower()
    if quantity_lower not in _PYGWINC_QUANTITIES:
        raise ValueError(
            f"Invalid quantity '{quantity}'. "
            f"Allowed values are: 'strain', 'darm', 'displacement'. "
            f"Note: 'velocity' and 'acceleration' are not supported for pyGWINC."
        )

    # Handle deprecated alias
    if quantity_lower == "displacement":
        quantity_lower = "darm"

    # Build frequency array
    fmin = kwargs.pop("fmin", 10.0)
    fmax = kwargs.pop("fmax", 4000.0)
    df = kwargs.pop("df", 1.0)

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    if frequencies is None:
        frequencies = np.arange(fmin, fmax + df, df).astype(float)

    # Handle model name alias
    if model == "A+":
        model = "Aplus"

    # Load budget and run
    try:
        budget = gwinc.load_budget(model)
    except RuntimeError:
        raise

    trace = budget.run(freq=frequencies)
    strain_asd = np.sqrt(trace.psd)

    # Get arm length for DARM conversion if needed
    if quantity_lower == "darm":
        try:
            ifo = budget.ifo
            arm_length = ifo.Infrastructure.Length
        except AttributeError as e:
            raise ValueError(
                f"Cannot retrieve arm length from IFO model '{model}' "
                f"for quantity='darm'. Error: {e}"
            )

        asd = strain_asd * arm_length
        unit = "m / sqrt(Hz)"
    else:
        # quantity == "strain"
        asd = strain_asd
        unit = "1 / sqrt(Hz)"

    # Remove known noise-generation parameters from kwargs
    # to avoid TypeError in FrequencySeries constructor
    for k in ["fmin", "fmax", "df"]:
        kwargs.pop(k, None)

    # If there are still problematic keys, print them (will show up in sphinx-build output/logs)
    if any(k in kwargs for k in ["fmin", "fmax", "df"]):
        print(f"CRITICAL: Key found in kwargs even after pop: {[k for k in ['fmin', 'fmax', 'df'] if k in kwargs]}")

    return FrequencySeries(
        asd, frequencies=frequencies, unit=unit, name=model, **kwargs
    )
