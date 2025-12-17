
from ._optional import require_optional
import numpy as np

def to_jax(ts, dtype=None):
    jax = require_optional("jax")
    return jax.numpy.array(ts.value, dtype=dtype)

def from_jax(cls, array, t0, dt, unit=None):
    # jax array to numpy
    data = np.array(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
