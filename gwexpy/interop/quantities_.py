from ._optional import require_optional


def to_quantity(series, units=None):
    """
    Convert a series (TimeSeries, FrequencySeries, etc) to a quantities.Quantity.

    Parameters
    ----------
    series : Series or array-like
        The input series with a .value and .unit attribute (or similar).
    units : str or quantities.UnitQuantity, optional
        Target units for the output Quantity.

    Returns
    -------
    quantities.Quantity
        The data wrapped as a Quantity.
    """
    pq = require_optional("quantities")

    # Get value (numpy array)
    if hasattr(series, "value"):
        val = series.value
    elif hasattr(series, "magnitude"):
        val = series.magnitude
    else:
        val = series

    # Ensure val is not an astropy Quantity (which behaves like array but carries unit)
    # If it was astropy Quantity, .value returns ndarray. But just in case.
    import numpy as np

    val = np.asarray(val)

    # Get unit

    # Get unit
    unit = getattr(series, "unit", None)

    # Convert astropy unit to string if necessary, then to pq unit
    if unit is not None:
        if hasattr(unit, "to_string"):
            u_str = unit.to_string()
        else:
            u_str = str(unit)
    else:
        u_str = "dimensionless"

    # Create Quantity
    # quantities usually parses strings well
    try:
        q = val * pq.Quantity(1.0, u_str)
    except (LookupError, ValueError):
        # Fallback if unit string isn't recognized
        # Try to map common ones or default appropriately?
        # For now, let it fail or default to dimensionless if simple
        if u_str == "dimensionless":
            q = val * pq.dimensionless
        else:
            # Try replacing known issues e.g. "count" -> "dimensionless" in some contexts?
            # But generally we expect standard physical units.
            raise ValueError(f"Could not convert unit '{u_str}' to quantities.Quantity")

    if units is not None:
        if isinstance(units, str):
            q.units = units
        else:
            # assume it is a quantities unit object
            q = q.rescale(units)

    return q


def from_quantity(cls, q, **kwargs):
    """
    Create a GWpy/GWexpy Series from a quantities.Quantity.

    Parameters
    ----------
    cls : type
        The target class (TimeSeries, FrequencySeries, etc).
    q : quantities.Quantity
        Input quantity.
    **kwargs :
        Additional arguments required by the class constructor (e.g. t0, dt, frequencies).

    Returns
    -------
    instance of cls
    """
    # Extract magnitude (numpy array)
    val = q.magnitude

    # Extract unit string
    # quantities.dimensionality.string is often formulated like "m**2/s"
    u_str = q.dimensionality.string
    if u_str == "dimensionless":
        u_str = ""

    # User might override unit in kwargs, but default to q's unit
    kwargs.setdefault("unit", u_str)

    return cls(val, **kwargs)
