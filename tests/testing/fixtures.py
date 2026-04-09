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
    import pytest
    import numpy
    from contextlib import contextmanager
    import tempfile
    import os

    # Mock/Fallback implementations
    SKIP_TEX = "skip_tex"
    has_tex = False
    usetex = False
    @contextmanager
    def rc_context(rc=None): yield
    @contextmanager
    def tmpfile(*args, **kwargs): yield "/tmp/fake"
    @contextmanager
    def TemporaryFilename(prefix="gwpy_"):
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        try:
            yield path
        finally:
            if os.path.exists(path):
                os.remove(path)
    def noisy_sinusoid(*args, **kwargs): return numpy.zeros(100)
    def corrupt_noisy_sinusoid(*args, **kwargs): return numpy.zeros(100)
    def deprecated_function(func): return func
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
