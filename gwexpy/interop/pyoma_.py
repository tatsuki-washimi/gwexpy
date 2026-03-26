"""
gwexpy.interop.pyoma_
----------------------

Interoperability with pyOMA (Operational Modal Analysis).

pyOMA returns results as Python dicts with keys like ``"Fn"`` (natural
frequencies), ``"Zeta"`` (damping ratios), and ``"Phi"`` (mode shapes).

References
----------
https://github.com/dagghe/pyOMA
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._modal_helpers import build_frf_matrix, build_mode_dataframe
from ._optional import require_optional
from ._registry import ConverterRegistry

__all__ = ["from_pyoma_results"]


def from_pyoma_results(
    cls: type,
    results: dict,
    *,
    fs: float | None = None,
) -> Any:
    """Convert pyOMA result dict to a GWexpy type.

    Parameters
    ----------
    cls : type
        Target type.  Use ``pandas.DataFrame`` (or pass the string
        ``"DataFrame"``) for modal parameter summary, or
        ``FrequencySeriesMatrix`` for mode-shape based FRF reconstruction.
    results : dict
        pyOMA result dictionary.  Expected keys:

        - ``"Fn"`` : ndarray (n_modes,) — natural frequencies [Hz]
        - ``"Zeta"`` : ndarray (n_modes,) — damping ratios
        - ``"Phi"`` : ndarray (n_dof, n_modes) — mode-shape matrix (optional)
        - ``"Xi"`` : alias for ``"Zeta"`` in some pyOMA versions
        - ``"Freq"`` : alias for ``"Fn"`` in some versions

    fs : float, optional
        Sampling frequency [Hz].  Stored in metadata.

    Returns
    -------
    pandas.DataFrame or FrequencySeriesMatrix
    """
    pd = require_optional("pandas")

    # Extract frequencies and damping
    frequencies = np.atleast_1d(results.get("Fn", results.get("Freq", [])))
    damping = np.atleast_1d(results.get("Zeta", results.get("Xi", [])))
    mode_shapes = results.get("Phi")

    if len(frequencies) == 0:
        raise ValueError("pyOMA results dict must contain 'Fn' or 'Freq' key")

    # If cls is DataFrame-like, return modal parameters
    if cls is pd.DataFrame or (isinstance(cls, type) and issubclass(cls, pd.DataFrame)):
        return build_mode_dataframe(
            frequencies,
            damping,
            mode_shapes=np.asarray(mode_shapes) if mode_shapes is not None else None,
        )

    # Otherwise build FrequencySeriesMatrix from mode shapes if available
    if mode_shapes is None:
        raise ValueError(
            "Cannot build FrequencySeriesMatrix without mode shapes ('Phi' key)"
        )

    mode_shapes = np.asarray(mode_shapes)  # (n_dof, n_modes)
    n_dof, n_modes = mode_shapes.shape

    # Build FRF data as (n_dof, 1, n_modes) — each "frequency" bin
    # corresponds to a natural frequency, rows are DOFs, single reference.
    frf_data = mode_shapes[:, np.newaxis, :]  # (n_dof, 1, n_modes)

    FrequencySeriesMatrix = ConverterRegistry.get_constructor("FrequencySeriesMatrix")

    return build_frf_matrix(
        FrequencySeriesMatrix if cls is None else cls,
        frequencies,
        frf_data,
        name="pyOMA modes",
    )
