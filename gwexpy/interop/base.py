from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from astropy.units import Quantity

T = TypeVar("T")


def to_plain_array(data: Any, copy: bool = False) -> np.ndarray:
    """Extract a plain NumPy array from common wrapper objects."""
    if hasattr(data, "value"):
        data = data.value

    if isinstance(data, Quantity):
        data = data.value

    if copy:
        return np.array(data, copy=True)
    return np.asarray(data)


def from_plain_array(
    cls: type[Any], array: Any, t0: Any, dt: Any, unit: Any = None, **kwargs: Any
) -> Any:
    """Reconstruct a gwexpy object from a plain array."""
    # Ensure data is numpy
    if hasattr(array, "numpy"):  # torch/tf
        array = array.numpy()
    elif hasattr(array, "get"):  # cupy
        array = array.get()

    return cls(array, t0=t0, dt=dt, unit=unit, **kwargs)
