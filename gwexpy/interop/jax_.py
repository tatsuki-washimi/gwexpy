"""JAX interop helpers."""

from __future__ import annotations

import numpy as np

from ._optional import require_optional

__all__ = ["to_jax", "from_jax"]


def to_jax(ts, dtype=None):
    """Convert a series-like object to a JAX array."""
    jax = require_optional("jax")
    from .base import to_plain_array

    return jax.numpy.array(to_plain_array(ts), dtype=dtype)


def from_jax(cls, array, t0, dt, unit=None):
    """Create a GWexpy object from a JAX array."""
    # jax array to numpy
    data = np.array(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
