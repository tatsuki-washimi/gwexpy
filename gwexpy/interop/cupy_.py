
from ._optional import require_optional

def is_cupy_available():
    """
    Check if cupy is installed and functionally usable (CUDA environment is working).
    """
    try:
        import cupy
        # Try to get device count to ensure driver is working
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def to_cupy(obj, dtype=None):
    cupy = require_optional("cupy")
    try:
        return cupy.asarray(obj, dtype=dtype)
    except Exception as e:
        # Catch CUDA driver errors which often manifest as CUDARuntimeError
        # or other system-level errors during initialization.
        msg = str(e)
        if "cudaErrorInsufficientDriver" in msg or "CUDA driver version" in msg:
             raise RuntimeError(
                 "CuPy is installed but CUDA driver is insufficient or not found. "
                 "GPU acceleration is not available in this environment."
             ) from e
        raise e

def from_cupy(cls, array, t0, dt, unit=None):
    cupy = require_optional("cupy")
    data = cupy.asnumpy(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
