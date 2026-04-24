"""CuPy interop helpers."""

from __future__ import annotations

from ._optional import require_optional

__all__ = ["is_cupy_available", "to_cupy", "from_cupy"]


def is_cupy_available():
    """Check whether CuPy is installed and the CUDA environment is usable."""
    try:
        import cupy

        # Try to get device count to ensure driver is working
        return cupy.cuda.runtime.getDeviceCount() > 0
    except (ImportError, AttributeError, RuntimeError):
        return False


def to_cupy(obj, dtype=None):
    """Convert an object to a CuPy array."""
    cupy = require_optional("cupy")
    try:
        return cupy.asarray(obj, dtype=dtype)
    except RuntimeError as e:
        # Catch CUDA driver errors which often manifest as CUDARuntimeError
        # or other system-level errors during initialization.
        msg = str(e)
        if "cudaErrorInsufficientDriver" in msg or "CUDA driver version" in msg:
            raise RuntimeError(
                "CuPy is installed but CUDA driver is insufficient or not found. "
                "GPU acceleration is not available in this environment."
            ) from e
        raise


def from_cupy(cls, array, t0, dt, unit=None):
    """Create a GWexpy object from a CuPy array."""
    cupy = require_optional("cupy")
    data = cupy.asnumpy(array)
    return cls(data, t0=t0, dt=dt, unit=unit)
