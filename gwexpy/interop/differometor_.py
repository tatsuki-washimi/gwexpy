"""Convert caller-supplied Differometor arrays without importing Differometor."""

from collections.abc import Mapping

import numpy as np
from astropy import units as u


def from_differometor_sensitivity(cls, frequencies, sensitivity, *, unit, name=None):
    """Convert one ASD array; frequencies are in Hz and unit is caller supplied."""
    if unit is None:
        raise ValueError("unit must be supplied explicitly")
    unit = u.Unit(unit)
    frequencies = u.Quantity(frequencies, u.Hz).to_value(u.Hz)
    values = np.asarray(sensitivity)
    if frequencies.ndim != 1 or values.ndim != 1 or frequencies.shape != values.shape:
        raise ValueError("frequencies and sensitivity must be equally sized 1D arrays")
    return cls(values, frequencies=frequencies, unit=unit, name=name)


def from_differometor_noise_budget(cls, frequencies, components, *, unit):
    """Convert named caller-supplied ASD components, without deriving them."""
    from gwexpy.frequencyseries import FrequencySeries

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping of names to arrays")
    if unit is None:
        raise ValueError("unit must be supplied explicitly")
    unit = u.Unit(unit)
    result = cls()
    for name, values in components.items():
        result[name] = from_differometor_sensitivity(
            FrequencySeries, frequencies, values, unit=unit, name=name
        )
    return result


def from_differometor_response(
    cls,
    frequencies,
    signal,
    port_index_map,
    *,
    output=None,
    outputs=None,
    inputs=None,
    unit,
):
    """Convert named upper-sideband signal rows, without TF normalization.

    ``port_index_map`` is the eighth element of ``differometor.run``. A row
    represents a complex upper-sideband field; it is not necessarily a
    calibrated input-to-output transfer function. The caller must supply
    the unit and establish the physical interpretation.
    """
    if inputs is not None:
        raise NotImplementedError(
            "Differometor signal rows do not identify input channels"
        )
    if (output is None) == (outputs is None):
        raise ValueError("provide exactly one of output or outputs")
    names = (
        [output]
        if output is not None
        else ([outputs] if isinstance(outputs, str) else list(outputs))
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("outputs must contain one or more unique port names")
    values = np.asarray(signal)
    if values.ndim != 2:
        raise ValueError("signal must have shape (rows, frequencies)")
    rows = []
    for name in names:
        if name not in port_index_map:
            raise KeyError(f"unknown output port {name!r}")
        index = port_index_map[name]
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < values.shape[0]
        ):
            raise ValueError(f"invalid row index for output {name!r}")
        rows.append(values[index])
    if len(rows) == 1:
        return from_differometor_sensitivity(
            cls, frequencies, rows[0], unit=unit, name=names[0]
        )
    from gwexpy.frequencyseries import FrequencySeriesMatrix

    if unit is None:
        raise ValueError("unit must be supplied explicitly")
    axis = u.Quantity(frequencies, u.Hz).to_value(u.Hz)
    if axis.ndim != 1 or values.shape[1] != len(axis):
        raise ValueError("frequencies and signal rows must have matching lengths")
    return FrequencySeriesMatrix(
        np.asarray(rows)[:, np.newaxis, :],
        frequencies=axis,
        channel_names=np.asarray(names, dtype=object)[:, np.newaxis],
        unit=u.Unit(unit),
    )
