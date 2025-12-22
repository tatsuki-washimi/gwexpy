from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..frequencyseries import FrequencySeries

if TYPE_CHECKING:
    from typing import Any


def from_pygwinc(model: str, frequencies: np.ndarray | None = None, **kwargs: Any) -> FrequencySeries:
    """
    Get detector noise ASD from pygwinc.

    Parameters
    ----------
    model : `str`
        Model name (e.g. 'aLIGO', 'A+', 'KAGRA').
    frequencies : `numpy.ndarray`, optional
        Frequencies to calculate PSD at.
        If None, generated from kwargs (fmin, fmax, df) or defaults to 10-4000Hz.
    **kwargs
        Additional arguments for frequency generation or FrequencySeries.
        fmin : `float`, optional, default: 10
        fmax : `float`, optional, default: 4000
        df : `float`, optional, default: 1.0

    Returns
    -------
    `FrequencySeries`
        ASD in strain / sqrt(Hz).
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

    # Alias handling
    model_orig = model
    if model == "A+":
        model = "Aplus"

    try:
        budget = gwinc.load_budget(model)
    except RuntimeError as e:
        # Re-raise if it's not the built-in IFO error, or if aliasing didn't help
        raise e

    trace = budget.run(freq=frequencies)

    # Resulting PSD (trace.psd) to ASD
    asd = np.sqrt(trace.psd)

    return FrequencySeries(
        asd, frequencies=frequencies, unit="strain / sqrt(Hz)", name=model_orig, **kwargs
    )
