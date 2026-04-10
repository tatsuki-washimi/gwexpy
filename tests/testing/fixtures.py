try:
    from gwpy.testing.fixtures import (
        SKIP_TEX,
        TemporaryFilename,
        TimeSeries,
        corrupt_noisy_sinusoid,
        deprecated_function,
        has_tex,
        noisy_sinusoid,
        numpy,
        pytest,
        rc_context,
        tmpfile,
        usetex,
    )
except ImportError:
    # Handle missing items in newer/different gwpy versions
    import os
    import tempfile
    from contextlib import contextmanager

    import numpy
    import pytest

    # Mock/Fallback implementations
    SKIP_TEX = "skip_tex"
    has_tex = False
    usetex = False
    @contextmanager
    def rc_context(rc=None):
        """Context manager for rc params fallback."""
        yield
    @contextmanager
    def tmpfile(*args, **kwargs):
        """Context manager for temporary file fallback."""
        yield "/tmp/fake"
    @contextmanager
    def TemporaryFilename(prefix="gwpy_"):
        """Context manager for temporary filename fallback."""
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        try:
            yield path
        finally:
            if os.path.exists(path):
                os.remove(path)
    def noisy_sinusoid(*args, **kwargs):
        """Noisy sinusoid generator fallback."""
        return numpy.zeros(100)
    def corrupt_noisy_sinusoid(*args, **kwargs):
        """Corrupt noisy sinusoid generator fallback."""
        return numpy.zeros(100)
    def deprecated_function(func):
        """Deprecated function wrapper fallback."""
        return func
    from gwexpy.timeseries import TimeSeries

__all__ = [
    "SKIP_TEX",
    "TemporaryFilename",
    "TimeSeries",
    "corrupt_noisy_sinusoid",
    "deprecated_function",
    "has_tex",
    "noisy_sinusoid",
    "numpy",
    "pytest",
    "rc_context",
    "tmpfile",
    "usetex",
]
