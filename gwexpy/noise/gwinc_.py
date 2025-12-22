from __future__ import annotations
from typing import Any
import numpy as np
from ..frequencyseries import FrequencySeries

def from_pygwinc(model: str, frequencies: np.ndarray | None = None, **kwargs: Any) -> FrequencySeries:
    """
    Get detector noise ASD from pygwinc.
    """
    try:
        import gwinc
    except ImportError:
        raise ImportError("Please install pygwinc to use gwexpy.noise.from_pygwinc")

    if frequencies is None:
        fmin = kwargs.pop("fmin", 10.0)
        fmax = kwargs.pop("fmax", 4000.0)
        df = kwargs.pop("df", 1.0)
        frequencies = np.arange(fmin, fmax + df, df).astype(float)

    if model == "A+":
        model = "Aplus"

    try:
        budget = gwinc.load_budget(model)
    except RuntimeError:
        # Fallback if specific model name handling is needed or let it bubble
        raise

    trace = budget.run(freq=frequencies)
    asd = np.sqrt(trace.psd)

    return FrequencySeries(
        asd, frequencies=frequencies, unit="strain / sqrt(Hz)", name=model, **kwargs
    )
