from __future__ import annotations

import warnings
from typing import Any, TypeVar

import numpy as np
from astropy.units import Quantity

T = TypeVar("T")


def resolve_meta(user_value: Any, source_value: Any = None) -> Any:
    """Resolve a single metadata field with ``user > source`` priority.

    Uses ``is not None`` (never ``or``) so falsy-but-valid values such as
    ``epoch=0`` or an empty string survive instead of being silently dropped.

    Parameters
    ----------
    user_value : Any
        Value explicitly supplied by the caller (``None`` means "not given").
    source_value : Any, optional
        Value recovered from the source object's metadata.

    Returns
    -------
    Any
        ``user_value`` if it is not ``None``, otherwise ``source_value``.

    """
    return user_value if user_value is not None else source_value


def resolve_timing(
    t0: Any,
    dt: Any,
    *,
    source: str,
    inferred_t0: Any = None,
    inferred_dt: Any = None,
    stacklevel: int = 3,
) -> tuple[Any, Any]:
    """Resolve ``t0``/``dt`` under a single, consistent warning policy.

    Per-field priority is **explicit user argument > value inferred from the
    source > hard default** (``t0=0.0`` / ``dt=1.0``).  Falling back to the
    hard default fabricates a GPS epoch (1980) or a unit sample spacing -- the
    T1 silent-default data-integrity trap.  When that happens a ``UserWarning``
    is emitted (mirroring the Zarr reader in ``timeseries/io/zarr_.py``) so that
    foreign sources lacking timing metadata are visible instead of silent.

    All comparisons use ``is not None`` so an explicit, falsy ``t0=0.0`` is
    honoured without triggering the warning.

    Parameters
    ----------
    t0, dt : float or None
        Explicit user-provided values; ``None`` means "not provided".
    source : str
        Human-readable description of the conversion source, used in the
        warning message (e.g. ``"dict"`` or ``"HDF5 dataset 'strain'"``).
    inferred_t0, inferred_dt : float or None
        Values inferred from the source (e.g. from a time index); ``None`` if
        inference was not possible.
    stacklevel : int
        Forwarded to :func:`warnings.warn`.

    Returns
    -------
    (t0, dt) : tuple
        The resolved timing values.

    """
    missing = []

    if t0 is not None:
        final_t0 = t0
    elif inferred_t0 is not None:
        final_t0 = inferred_t0
    else:
        final_t0 = 0.0
        missing.append("t0=0.0 (GPS epoch 1980)")

    if dt is not None:
        final_dt = dt
    elif inferred_dt is not None:
        final_dt = inferred_dt
    else:
        final_dt = 1.0
        missing.append("dt=1.0")

    if missing:
        warnings.warn(
            f"{source} is missing timing metadata; assuming "
            f"{', '.join(missing)}. Pass t0=/dt= explicitly to set the timing.",
            UserWarning,
            stacklevel=stacklevel,
        )

    return final_t0, final_dt


def to_plain_array(data: Any, copy: bool = False) -> np.ndarray:
    """Extract a plain NumPy array from common wrapper objects."""
    if hasattr(data, "value"):
        data = data.value

    if isinstance(data, Quantity):
        data = data.value

    if copy:
        return np.array(data, copy=True)
    return np.asarray(data)


def from_plain_array(
    cls: type[Any], array: Any, t0: Any, dt: Any, unit: Any = None, **kwargs: Any
) -> Any:
    """Reconstruct a gwexpy object from a plain array."""
    # Ensure data is numpy
    if hasattr(array, "numpy"):  # torch/tf
        array = array.numpy()
    elif hasattr(array, "get"):  # cupy
        array = array.get()

    return cls(array, t0=t0, dt=dt, unit=unit, **kwargs)
