"""Case discovery and unified access for SPECFEM3D Cartesian run directories."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from gwexpy.timeseries import TimeSeriesDict

from .field_hdf5 import (
    SpecfemGLLDataset,
    read_specfem_gll_hdf5,
    read_specfem_gll_snapshot,
)
from .field_vtk import read_specfem_pvtu, read_specfem_vtk, read_specfem_vtu
from .gll import EXPECTED_NGLL
from .model import SpecfemFieldSnapshot
from .par_file import ParFileData, parse_par_file_data, parse_t0
from .partitions import merge_specfem_partitions
from .seismogram_ascii import read_specfem_seismograms_ascii
from .stations import SpecfemStation, parse_stations

__all__ = ["Specfem3DCase"]


class Specfem3DCase:
    """Discovery and reader facade for a SPECFEM3D run directory or output artifact."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"SPECFEM3D path does not exist: {self.root}")

        self.par_file_path: Path | None = None
        self.stations_path: Path | None = None
        self.output_dir: Path | None = None
        self.databases_dir: Path | None = None

        if self.root.is_dir():
            if (self.root / "DATA" / "Par_file").is_file():
                self.par_file_path = self.root / "DATA" / "Par_file"
            elif (self.root / "Par_file").is_file():
                self.par_file_path = self.root / "Par_file"

            for st_rel in ("DATA/STATIONS_FILTERED", "DATA/STATIONS", "STATIONS"):
                if (self.root / st_rel).is_file():
                    self.stations_path = self.root / st_rel
                    break

            if (self.root / "OUTPUT_FILES").is_dir():
                self.output_dir = self.root / "OUTPUT_FILES"
            else:
                self.output_dir = self.root

            for db_cand in (
                self.output_dir / "DATABASES_MPI_RUN",
                self.output_dir / "DATABASES_MPI",
                self.root / "DATABASES_MPI",
                self.root,
            ):
                if db_cand.is_dir() and any(db_cand.glob("proc*_Database")):
                    self.databases_dir = db_cand
                    break

    @property
    def par_file(self) -> ParFileData | None:
        if self.par_file_path is None:
            return None
        return parse_par_file_data(self.par_file_path)

    @property
    def stations(self) -> dict[str, SpecfemStation]:
        if self.stations_path is None:
            return {}
        return parse_stations(self.stations_path)

    @property
    def t0_s(self) -> float | None:
        if self.output_dir is None:
            return None
        solver_txt = self.output_dir / "output_solver.txt"
        if solver_txt.is_file():
            t0, _ = parse_t0(solver_txt)
            return t0
        return None

    @property
    def available_seismograms(self) -> tuple[Path, ...]:
        if self.output_dir is None or not self.output_dir.is_dir():
            return ()
        exts = {".semd", ".semv", ".sema", ".semp", ".sac"}
        return tuple(
            sorted(
                p
                for p in self.output_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
        )

    def read_seismograms(
        self,
        *,
        channels: Iterable[str] | None = None,
        extensions: tuple[str, ...] = (".semd", ".semv", ".sema", ".semp"),
        epoch: Any | None = None,
        unit: Any | None = None,
    ) -> TimeSeriesDict:
        """Load seismograms from ``OUTPUT_FILES`` using ``Par_file`` and ``STATIONS`` metadata."""
        if self.output_dir is None:
            raise FileNotFoundError(
                f"No OUTPUT_FILES directory found under {self.root}"
            )
        return read_specfem_seismograms_ascii(
            self.output_dir,
            channels=channels,
            extensions=extensions,
            epoch=epoch,
            unit=unit,
            par_file=self.par_file,
            stations=self.stations,
        )

    def read_gll_hdf5(
        self,
        path: str | Path,
        *,
        profile_density_range: tuple[float, float] | None = None,
    ) -> SpecfemGLLDataset:
        """Read a canonical SPECFEM3D GLL HDF5 file."""
        return read_specfem_gll_hdf5(path, profile_density_range=profile_density_range)

    def read_field_snapshot(
        self,
        source: str | Path | None = None,
        *,
        ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
        fields: Sequence[str] | None = ("rho",),
        include_displacement: bool = False,
        association: Literal["point", "cell"] | None = None,
    ) -> SpecfemFieldSnapshot:
        """Load a volumetric or GLL field snapshot from HDF5, DATABASES_MPI, VTK, VTU, or PVTU."""
        target = Path(source).resolve() if source is not None else self.databases_dir
        if target is None:
            raise FileNotFoundError(
                f"No field source or DATABASES_MPI directory found under {self.root}"
            )
        if target.is_dir():
            par = self.par_file
            dt_s = par.dt_s if par and par.dt_s else 1.0
            t0_s = self.t0_s if self.t0_s is not None else 0.0
            return merge_specfem_partitions(
                target,
                ngll_shape=ngll_shape,
                fields=fields,
                include_displacement=include_displacement,
                dt_s=dt_s,
                t0_s=t0_s,
            )
        suf = target.suffix.lower()
        if suf in {".h5", ".hdf5"}:
            return read_specfem_gll_snapshot(target)
        if suf == ".vtk":
            return read_specfem_vtk(target, association=association)
        if suf == ".vtu":
            return read_specfem_vtu(target, association=association)
        if suf == ".pvtu":
            return read_specfem_pvtu(target, association=association)
        raise ValueError(f"Unsupported SPECFEM3D field source: {target}")
