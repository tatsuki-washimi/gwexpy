"""Helpers shared by the statistics functions."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import FrameType
from typing import Protocol

import numpy as np

__all__ = ["UniformRandomSource", "warn_at_user_call"]

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[1])


class UniformRandomSource(Protocol):
    """Anything that draws uniform(0, 1) samples the way a NumPy Generator does.

    Only ``random(size)`` is used, so `numpy.random.Generator` and a minimal
    stub with that single method are both accepted.
    """

    def random(self, size: int, /) -> np.ndarray:
        """Return ``size`` samples drawn uniformly from ``[0, 1)``."""
        ...


def warn_at_user_call(message: str, category: type[Warning]) -> None:
    """Emit a warning attributed to the first caller outside this package.

    A statistics function is reachable directly and through a `TimeSeries`
    method.  A fixed ``stacklevel`` is right for only one of the two paths;
    this walks the stack so both report the user's own call site.
    """
    stacklevel = 2
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_ROOT):
        frame = frame.f_back
        stacklevel += 1
    if frame is None:  # every frame is inside the package
        stacklevel = 2
    warnings.warn(message, category, stacklevel=stacklevel)
