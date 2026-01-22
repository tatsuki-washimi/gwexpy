"""Coordinate-to-index conversion utilities for ScalarField visualization.

This module provides functions for converting between physical coordinates
(with units) and array indices, following the ScalarField slicing convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.units import Quantity

__all__ = [
    "nearest_index",
    "slice_from_index",
    "slice_from_value",
    "select_value",
]


def nearest_index(axis_index: Quantity, value: Quantity) -> int:
    """Find the nearest index for a given coordinate value.

    Parameters
    ----------
    axis_index : Quantity
        1D array of axis coordinates with units.
    value : Quantity
        Scalar coordinate value to find.

    Returns
    -------
    int
        Index of the nearest element.

    Raises
    ------
    ValueError
        If units are incompatible.
    IndexError
        If value is outside the axis range (extrapolation needed).

    Notes
    -----
    Tie-break rule: when the value is exactly between two points,
    the smaller index is chosen (consistent with numpy searchsorted 'left').

    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np
    >>> axis = np.array([0, 1, 2, 3, 4]) * u.m
    >>> nearest_index(axis, 2.3 * u.m)
    2
    >>> nearest_index(axis, 2.5 * u.m)  # tie-break: smaller index
    2
    """
    from astropy import units as u

    # Unit compatibility check
    try:
        value_converted = value.to(axis_index.unit)
    except u.UnitConversionError as e:
        raise ValueError(
            f"Unit mismatch: cannot convert {value.unit} to {axis_index.unit}"
        ) from e

    val = value_converted.value
    axis_vals = axis_index.value

    # Range check
    axis_min = axis_vals.min()
    axis_max = axis_vals.max()

    if val < axis_min or val > axis_max:
        raise IndexError(
            f"Value {value} is outside axis range [{axis_min}, {axis_max}] {axis_index.unit}"
        )

    # Find nearest index
    # For monotonic axes, we use searchsorted for efficiency
    if len(axis_vals) > 1:
        # Check if sorted
        if axis_vals[1] >= axis_vals[0]:  # ascending
            # Use searchsorted with tie-break to left
            idx = np.searchsorted(axis_vals, val, side="left")
            if idx == len(axis_vals):
                idx = len(axis_vals) - 1
            elif idx > 0:
                # Check which neighbor is closer
                if val - axis_vals[idx - 1] < axis_vals[idx] - val:
                    idx = idx - 1
                elif val - axis_vals[idx - 1] == axis_vals[idx] - val:
                    # Tie: choose smaller index
                    idx = idx - 1
        else:  # descending - fallback to argmin
            idx = int(np.argmin(np.abs(axis_vals - val)))
    else:
        idx = 0

    return int(idx)


def slice_from_index(i: int) -> slice:
    """Convert an integer index to a slice maintaining array dimension.

    This follows the ScalarField convention of keeping axes at length=1
    instead of reducing dimensions.

    Parameters
    ----------
    i : int
        Integer index.

    Returns
    -------
    slice
        A slice object ``slice(i, i+1)`` that selects one element
        while preserving the axis dimension.

    Examples
    --------
    >>> slice_from_index(5)
    slice(5, 6, None)
    """
    return slice(i, i + 1)


def slice_from_value(
    axis_index: Quantity,
    value: Quantity,
    method: str = "nearest",
) -> slice:
    """Convert a coordinate value to a dimension-preserving slice.

    Parameters
    ----------
    axis_index : Quantity
        1D array of axis coordinates with units.
    value : Quantity
        Scalar coordinate value to find.
    method : str, optional
        Interpolation method. Currently only 'nearest' is supported.
        Default is 'nearest'.

    Returns
    -------
    slice
        A slice object selecting the nearest index while preserving dimension.

    Raises
    ------
    ValueError
        If units are incompatible or method is unsupported.
    IndexError
        If value is outside the axis range.

    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np
    >>> axis = np.array([0, 1, 2, 3, 4]) * u.m
    >>> slice_from_value(axis, 2.3 * u.m)
    slice(2, 3, None)
    """
    if method != "nearest":
        raise ValueError(f"Unsupported method '{method}'. Only 'nearest' is available.")

    idx = nearest_index(axis_index, value)
    return slice_from_index(idx)


def select_value(
    data: np.ndarray | Quantity,
    mode: str = "real",
) -> np.ndarray | Quantity:
    """Extract real/imag/abs/angle/power component from potentially complex data.

    This function provides a unified interface for extracting scalar
    components from complex-valued fields, ensuring consistent behavior
    across all visualization functions.

    Parameters
    ----------
    data : ndarray or Quantity
        Input data, possibly complex-valued.
    mode : str, optional
        Component to extract:
        - 'real': Real part (default)
        - 'imag': Imaginary part
        - 'abs': Absolute value (magnitude)
        - 'angle': Phase angle in radians
        - 'power': Squared magnitude (|data|^2)

    Returns
    -------
    ndarray or Quantity
        Real-valued result. For 'power' mode, if input has units,
        the output unit is unit^2.

    Raises
    ------
    ValueError
        If mode is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1+2j, 3+4j])
    >>> select_value(data, 'abs')
    array([2.23606798, 5.        ])
    >>> select_value(data, 'power')
    array([ 5., 25.])
    """
    from astropy import units as u

    valid_modes = ("real", "imag", "abs", "angle", "power")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

    # Handle Quantity
    if isinstance(data, u.Quantity):
        values = data.value
        unit = data.unit
        is_quantity = True
    else:
        values = np.asarray(data)
        unit = None
        is_quantity = False

    # Extract component
    if mode == "real":
        result = np.real(values)
    elif mode == "imag":
        result = np.imag(values)
    elif mode == "abs":
        result = np.abs(values)
    elif mode == "angle":
        result = np.angle(values)
        # Angle is always in radians
        return result * u.rad
    elif mode == "power":
        result = np.abs(values) ** 2
        # Power has unit^2
        if is_quantity:
            return result * (unit**2)
        return result
    else:
        # Should not reach here due to validation above
        raise ValueError(f"Invalid mode '{mode}'")

    # Return with original unit (except angle/power handled above)
    if is_quantity:
        return result * unit
    return result
