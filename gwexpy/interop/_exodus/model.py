"""Native mesh-retaining data models and frozen provenance for EXODUS II."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "ExodusBlock",
    "ExodusProvenance",
    "ExodusSnapshot",
    "ExodusVariable",
    "Flow3DProvenance",
]


def _readonly(arr: Any, dtype: Any = None) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _slice_mask(
    mask: np.ndarray | None,
    item: tuple[slice, slice, slice, slice],
    source_shape: tuple[int, ...],
    name: str,
) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape != source_shape:
        raise ValueError(f"{name} shape does not match the source Field")
    return mask[item]


@dataclass(frozen=True)
class ExodusProvenance:
    """Frozen provenance attached to EXODUS II Fields via ``_gwex_exodus``."""

    source_path: str
    format_name: str
    profile_name: str
    backend: str
    title: str | None
    exodus_version: float | None
    variable_name: str
    canonical_name: str
    association: Literal["cell", "point"]
    blocks: tuple[str, ...]
    block_mode: str
    grid_mode: Literal["exact", "sample"]
    unit_str: str
    unit_source: str
    coordinate_unit: str
    time_steps: tuple[float, ...]
    interpolation_method: str | None = None
    valid_fraction: float | None = None
    valid_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.valid_mask is not None:
            object.__setattr__(self, "valid_mask", _readonly(self.valid_mask, np.bool_))

    def _slice_for_field(
        self,
        item: tuple[slice, slice, slice, slice],
        source_shape: tuple[int, ...],
        axis0: Any,
    ) -> ExodusProvenance:
        """Align selected EXODUS times and validity mask with a Field slice."""
        mask = _slice_mask(self.valid_mask, item, source_shape, "EXODUS valid_mask")
        fraction = float(np.mean(mask)) if mask is not None and mask.size else None
        return replace(
            self,
            time_steps=self.time_steps[item[0]],
            valid_mask=mask,
            valid_fraction=fraction if mask is not None else self.valid_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly dictionary representation."""
        return {
            "source_path": self.source_path,
            "format_name": self.format_name,
            "profile_name": self.profile_name,
            "backend": self.backend,
            "title": self.title,
            "exodus_version": self.exodus_version,
            "variable_name": self.variable_name,
            "canonical_name": self.canonical_name,
            "association": self.association,
            "blocks": self.blocks,
            "block_mode": self.block_mode,
            "grid_mode": self.grid_mode,
            "unit_str": self.unit_str,
            "unit_source": self.unit_source,
            "coordinate_unit": self.coordinate_unit,
            "time_steps": self.time_steps,
            "interpolation_method": self.interpolation_method,
            "valid_fraction": self.valid_fraction,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Flow3DProvenance:
    """Frozen FLOW-3D EXODUS profile provenance attached via ``_gwex_flow3d``."""

    profile_version: str
    quantity_kind: str
    unit_system: str
    favor_variable: str | None
    vof_variable: str | None
    favor_threshold: float
    vof_threshold: float
    mask_blocked: bool
    empty_fluid_mode: str
    open_cell_fraction: float | None = None
    fluid_cell_fraction: float | None = None
    open_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    fluid_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.open_mask is not None:
            object.__setattr__(self, "open_mask", _readonly(self.open_mask, np.bool_))
        if self.fluid_mask is not None:
            object.__setattr__(self, "fluid_mask", _readonly(self.fluid_mask, np.bool_))

    def _slice_for_field(
        self,
        item: tuple[slice, slice, slice, slice],
        source_shape: tuple[int, ...],
        axis0: Any,
    ) -> Flow3DProvenance:
        """Align FAVOR/VOF masks and their fractions with a Field slice."""
        open_mask = _slice_mask(self.open_mask, item, source_shape, "FLOW-3D open_mask")
        fluid_mask = _slice_mask(
            self.fluid_mask, item, source_shape, "FLOW-3D fluid_mask"
        )
        return replace(
            self,
            open_mask=open_mask,
            fluid_mask=fluid_mask,
            open_cell_fraction=(
                float(np.mean(open_mask))
                if open_mask is not None and open_mask.size
                else None
            ),
            fluid_cell_fraction=(
                float(np.mean(fluid_mask))
                if fluid_mask is not None and fluid_mask.size
                else None
            ),
        )


@dataclass(frozen=True)
class ExodusBlock:
    """Native element block geometry and connectivity from an EXODUS II dataset."""

    block_id: int
    name: str
    elem_type: str
    status: int
    connectivity: np.ndarray
    node_indices: np.ndarray
    cell_centers_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "connectivity", _readonly(self.connectivity, np.int64))
        object.__setattr__(self, "node_indices", _readonly(self.node_indices, np.int64))
        object.__setattr__(
            self, "cell_centers_m", _readonly(self.cell_centers_m, np.float64)
        )

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])


@dataclass(frozen=True)
class ExodusVariable:
    """Discovered EXODUS II variable metadata."""

    name: str
    canonical_name: str
    association: Literal["cell", "point", "global"]
    unit: u.UnitBase | None
    unit_source: str
    block_names: tuple[str, ...] = ()
    quantity_kind: str = "custom"


@dataclass(frozen=True)
class ExodusSnapshot:
    """Native mesh-retaining EXODUS II snapshot at a single time step."""

    time_s: float
    time_index: int
    points_m: np.ndarray
    blocks: Mapping[str, ExodusBlock]
    cell_data: Mapping[str, Mapping[str, np.ndarray]]
    point_data: Mapping[str, np.ndarray]
    global_data: Mapping[str, float]
    masks: Mapping[str, Mapping[str, np.ndarray]]
    provenance: Mapping[str, Any]
    pyvista_multiblock: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "points_m", _readonly(self.points_m, np.float64))
        object.__setattr__(self, "blocks", MappingProxyType(dict(self.blocks)))
        frozen_cell = {
            blk: MappingProxyType(
                {k: _readonly(v, np.float64) for k, v in vars_.items()}
            )
            for blk, vars_ in self.cell_data.items()
        }
        object.__setattr__(self, "cell_data", MappingProxyType(frozen_cell))
        object.__setattr__(
            self,
            "point_data",
            MappingProxyType(
                {k: _readonly(v, np.float64) for k, v in self.point_data.items()}
            ),
        )
        object.__setattr__(
            self, "global_data", MappingProxyType(dict(self.global_data))
        )
        frozen_masks = {
            blk: MappingProxyType({k: _readonly(v, np.bool_) for k, v in m_.items()})
            for blk, m_ in self.masks.items()
        }
        object.__setattr__(self, "masks", MappingProxyType(frozen_masks))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
