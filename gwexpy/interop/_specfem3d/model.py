"""Native mesh-retaining data models and frozen provenance for SPECFEM3D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "Specfem3DProvenance",
    "SpecfemFieldSnapshot",
]


def _readonly(arr: Any, dtype: Any = None) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Specfem3DProvenance:
    """Frozen provenance attached via ``_gwex_specfem3d`` on Fields and TimeSeries."""

    source_path: str
    source_kind: Literal[
        "gll_hdf5",
        "partition_bin",
        "vtk",
        "vtu",
        "pvtu",
        "seismogram_ascii",
        "seismogram_sac",
    ]
    field_or_channel: str
    coordinate_convention: str = "ENU"
    association: Literal["gll", "point", "cell", "station"] = "gll"
    grid_mode: Literal["exact", "sample", "timeseries"] = "exact"
    unit_str: str = ""
    t0_s: float | None = None
    dt_s: float | None = None
    dt_source: str | None = None
    nproc: int | None = None
    ngll_shape: tuple[int, int, int] | None = None
    element_id_method: str | None = None
    n_elements: int | None = None
    deduplicated_gll: bool = False
    max_shared_gll_discrepancy: float | None = None
    par_file_hash: str | None = None
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
    ) -> Specfem3DProvenance:
        """Align a sampled GLL/VTK validity mask with a Field slice."""
        mask = self.valid_mask
        if mask is not None and mask.shape != source_shape:
            raise ValueError(
                "SPECFEM3D valid_mask shape does not match the source Field"
            )
        selected = mask[item] if mask is not None else None
        fraction = (
            float(np.mean(selected)) if selected is not None and selected.size else None
        )
        times = axis0.to_value(u.s) if axis0.unit.is_equivalent(u.s) else None
        return replace(
            self,
            t0_s=(float(times[0]) if len(times) else None)
            if times is not None
            else self.t0_s,
            valid_mask=selected,
            valid_fraction=fraction if selected is not None else self.valid_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "field_or_channel": self.field_or_channel,
            "coordinate_convention": self.coordinate_convention,
            "association": self.association,
            "grid_mode": self.grid_mode,
            "unit_str": self.unit_str,
            "t0_s": self.t0_s,
            "dt_s": self.dt_s,
            "dt_source": self.dt_source,
            "nproc": self.nproc,
            "ngll_shape": self.ngll_shape,
            "element_id_method": self.element_id_method,
            "n_elements": self.n_elements,
            "deduplicated_gll": self.deduplicated_gll,
            "max_shared_gll_discrepancy": self.max_shared_gll_discrepancy,
            "par_file_hash": self.par_file_hash,
            "interpolation_method": self.interpolation_method,
            "valid_fraction": self.valid_fraction,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class SpecfemFieldSnapshot:
    """Native mesh-retaining SPECFEM3D volumetric or GLL field snapshot."""

    source_path: str
    source_kind: str
    times_s: np.ndarray
    coordinates_m: np.ndarray
    fields: Mapping[str, np.ndarray]
    field_units: Mapping[str, u.UnitBase]
    association: Literal["gll", "point", "cell"]
    cell_blocks: tuple[tuple[str, np.ndarray], ...] = ()
    element_id: np.ndarray | None = None
    partition_id: np.ndarray | None = None
    material_id: np.ndarray | None = None
    domain_id: np.ndarray | None = None
    is_pml: np.ndarray | None = None
    ibool: np.ndarray | None = None
    ngll_shape: tuple[int, int, int] | None = None
    quadrature_volume_m3: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times_s", _readonly(self.times_s, np.float64))
        object.__setattr__(
            self, "coordinates_m", _readonly(self.coordinates_m, np.float64)
        )
        object.__setattr__(
            self,
            "fields",
            MappingProxyType(
                {k: _readonly(v, np.float64) for k, v in self.fields.items()}
            ),
        )
        object.__setattr__(
            self, "field_units", MappingProxyType(dict(self.field_units))
        )
        object.__setattr__(
            self,
            "cell_blocks",
            tuple((str(k), _readonly(v, np.int64)) for k, v in self.cell_blocks),
        )
        for opt_int in (
            "element_id",
            "partition_id",
            "material_id",
            "domain_id",
            "ibool",
        ):
            val = getattr(self, opt_int)
            if val is not None:
                object.__setattr__(self, opt_int, _readonly(val, np.int64))
        if self.is_pml is not None:
            object.__setattr__(self, "is_pml", _readonly(self.is_pml, np.bool_))
        if self.quadrature_volume_m3 is not None:
            object.__setattr__(
                self,
                "quadrature_volume_m3",
                _readonly(self.quadrature_volume_m3, np.float64),
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
