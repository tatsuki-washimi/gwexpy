"""FAVOR and VOF fluid-fraction masking semantics for FLOW-3D EXODUS II."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "Flow3DMaskSet",
    "apply_flow3d_masks",
    "compute_flow3d_masks",
]


@dataclass(frozen=True)
class Flow3DMaskSet:
    """Boolean masks separating physical zeros from FAVOR-blocked/empty cells."""

    open_mask: np.ndarray
    fluid_mask: np.ndarray
    valid_mask: np.ndarray
    favor_variable: str | None
    vof_variable: str | None

    def __post_init__(self) -> None:
        for attr in ("open_mask", "fluid_mask", "valid_mask"):
            arr = np.array(getattr(self, attr), dtype=np.bool_, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, attr, arr)


def compute_flow3d_masks(
    n_items: int,
    *,
    favor_values: np.ndarray | None = None,
    favor_variable: str | None = None,
    favor_threshold: float = 0.0,
    vof_values: np.ndarray | None = None,
    vof_variable: str | None = None,
    vof_threshold: float = 0.0,
    mask_blocked: bool = True,
    empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
) -> Flow3DMaskSet:
    """Compute FAVOR open-cell and VOF fluid masks without conflating physical 0 with invalid.

    - Cells with ``favor <= favor_threshold`` (solid obstacle) are marked
      ``open_mask = False`` (invalid when ``mask_blocked=True``).
    - Cells with ``favor > favor_threshold`` and ``vof <= vof_threshold`` are
      open air/dry cells: physically valid unless ``empty_fluid_mode='nan'``.
    """
    if empty_fluid_mode not in {"keep", "nan", "zero"}:
        raise ValueError(
            f"empty_fluid_mode must be 'keep', 'nan', or 'zero', got {empty_fluid_mode!r}"
        )

    if favor_values is not None:
        fv = np.asarray(favor_values, dtype=np.float64).ravel()
        if fv.shape != (n_items,):
            raise ValueError(
                f"FAVOR array length {fv.shape} does not match entity count {n_items}"
            )
        open_mask = np.isfinite(fv) & (fv > float(favor_threshold))
    else:
        open_mask = np.ones(n_items, dtype=bool)

    if vof_values is not None:
        vv = np.asarray(vof_values, dtype=np.float64).ravel()
        if vv.shape != (n_items,):
            raise ValueError(
                f"VOF array length {vv.shape} does not match entity count {n_items}"
            )
        fluid_mask = open_mask & np.isfinite(vv) & (vv > float(vof_threshold))
    else:
        fluid_mask = open_mask.copy()

    valid_mask = open_mask.copy() if mask_blocked else np.ones(n_items, dtype=bool)
    if empty_fluid_mode == "nan" and vof_values is not None:
        valid_mask &= fluid_mask

    return Flow3DMaskSet(
        open_mask=open_mask,
        fluid_mask=fluid_mask,
        valid_mask=valid_mask,
        favor_variable=favor_variable,
        vof_variable=vof_variable,
    )


def apply_flow3d_masks(
    values: np.ndarray,
    masks: Flow3DMaskSet,
    *,
    empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
    is_mask_variable: bool = False,
) -> np.ndarray:
    """Apply ``Flow3DMaskSet`` to a scalar or vector array.

    Solid/blocked cells (``~valid_mask``) are filled with ``np.nan`` (unless
    ``is_mask_variable=True`` so inspecting ``vf`` or ``f`` itself keeps raw
    fractions!). Open cells with ``f == 0`` keep their physical value under
    ``empty_fluid_mode='keep'``, become ``0.0`` under ``'zero'``, or become
    ``NaN`` under ``'nan'``.
    """
    arr = np.array(values, dtype=np.float64, copy=True)
    if is_mask_variable:
        return arr

    if empty_fluid_mode == "zero":
        empty_open = masks.open_mask & (~masks.fluid_mask)
        if arr.ndim == 2:
            arr[empty_open, :] = 0.0
        else:
            arr[empty_open] = 0.0

    invalid = ~masks.valid_mask
    if np.any(invalid):
        if arr.ndim == 2:
            arr[invalid, :] = np.nan
        else:
            arr[invalid] = np.nan
    return arr
