
from ._optional import require_optional

def to_cupy(ts, dtype=None):
    cupy = require_optional("cupy")
    return cupy.array(ts.value, dtype=dtype)

def from_cupy(cls, array, t0, dt, unit=None):
    cupy = require_optional("cupy")
    data = cupy.asnumpy(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
