import numpy as np

from ._optional import require_optional


def to_jax(ts, dtype=None):
    jax = require_optional("jax")
    from .base import to_plain_array

    return jax.numpy.array(to_plain_array(ts), dtype=dtype)


def from_jax(cls, array, t0, dt, unit=None):
    # jax array to numpy
    data = np.array(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
