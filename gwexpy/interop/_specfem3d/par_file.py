# Adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/specfem/provenance.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Parsers for SPECFEM3D ``Par_file``, solver timing logs, and file manifests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_T0_CROSSCHECK_ATOL",
    "ParFileData",
    "build_file_manifest",
    "crosscheck_t0_semd",
    "par_file_value_as_bool",
    "par_file_value_as_number",
    "parse_f_resolved_estimate",
    "parse_par_file",
    "parse_t0",
    "sha256_bytes",
    "sha256_file",
]

_TRUE_LITERALS: Final[frozenset[str]] = frozenset({".true.", ".t.", "true"})
_FALSE_LITERALS: Final[frozenset[str]] = frozenset({".false.", ".f.", "false"})
_INT_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"[+-]?\d+")

DEFAULT_T0_CROSSCHECK_ATOL: Final[float] = 2e-5
_START_TIME_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*start time:\s*(\S+)\s+seconds\s*$", re.MULTILINE
)
_MIN_PERIOD_RESOLVED_RE: Final[re.Pattern[str]] = re.compile(
    r"Minimum period resolved\s*=\s*(\S+)"
)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return ``'sha256:<hex>'`` digest of ``data``."""
    return f"sha256:{_sha256_hex(data)}"


def sha256_file(path: str | Path) -> str:
    """Return ``'sha256:<hex>'`` digest of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def par_file_value_as_bool(value: str) -> bool:
    """Convert a Fortran logical literal (``.true.``/``.false.``) to ``bool``."""
    normalized = value.strip().lower()
    if normalized in _TRUE_LITERALS:
        return True
    if normalized in _FALSE_LITERALS:
        return False
    raise ValueError(f"not a Fortran logical literal: {value!r}")


def par_file_value_as_number(value: str) -> int | float:
    """Convert a ``Par_file`` numeric literal (including ``1.0d-3``) to ``int`` or ``float``."""
    stripped = value.strip()
    if _INT_LITERAL_RE.fullmatch(stripped):
        return int(stripped)
    normalized = stripped.replace("D", "E").replace("d", "e")
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"not a numeric Par_file value: {value!r}") from exc


@dataclass(frozen=True)
class ParFileData:
    """Parsed SPECFEM3D ``DATA/Par_file`` metadata."""

    values: dict[str, str]
    text: str
    par_file_hash: str

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        raw = self.values.get(key)
        if raw is None:
            return default
        return par_file_value_as_bool(raw)

    def get_number(
        self, key: str, default: int | float | None = None
    ) -> int | float | None:
        raw = self.values.get(key)
        if raw is None:
            return default
        return par_file_value_as_number(raw)

    @property
    def dt_s(self) -> float | None:
        val = self.get_number("DT")
        return float(val) if val is not None else None

    @property
    def nstep(self) -> int | None:
        val = self.get_number("NSTEP")
        return int(val) if val is not None else None

    @property
    def nproc(self) -> int | None:
        val = self.get_number("NPROC")
        return int(val) if val is not None else None

    @property
    def seismogram_dt_s(self) -> float | None:
        """Return effective seismogram sample spacing ``DT * NTSTEP_BETWEEN_OUTPUT_SAMPLE``."""
        dt = self.dt_s
        if dt is None:
            return None
        step = self.get_number("NTSTEP_BETWEEN_OUTPUT_SAMPLE", 1)
        assert step is not None
        if int(step) <= 0:
            raise ValueError(
                f"NTSTEP_BETWEEN_OUTPUT_SAMPLE must be positive, got {step}"
            )
        return dt * int(step)

    @property
    def suppress_utm_projection(self) -> bool | None:
        return self.get_bool("SUPPRESS_UTM_PROJECTION")


def parse_par_file(path: str | Path) -> dict[str, object]:
    """Parse a SPECFEM3D ``DATA/Par_file`` into ``values``, ``text``, and ``par_file_hash``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        if "=" not in content:
            raise ValueError(f"{path}: expected 'KEY = value', got line: {raw_line!r}")
        key, _, value = content.partition("=")
        values[key.strip()] = value.strip()
    return {
        "values": values,
        "text": text,
        "par_file_hash": sha256_bytes(text.encode("utf-8")),
    }


def parse_par_file_data(path: str | Path) -> ParFileData:
    """Parse a SPECFEM3D ``DATA/Par_file`` into a :class:`ParFileData` object."""
    parsed = parse_par_file(path)
    raw_values = parsed["values"]
    assert isinstance(raw_values, dict)
    return ParFileData(
        values=dict(raw_values),
        text=str(parsed["text"]),
        par_file_hash=str(parsed["par_file_hash"]),
    )


def build_file_manifest(root: str | Path, files: Sequence[str]) -> tuple[str, str]:
    """Build a canonical sorted ``(manifest_text, manifest_hash)`` over ``files`` under ``root``."""
    root_path = Path(root)
    rel_paths = [Path(f).as_posix() for f in files]
    if not rel_paths:
        raise ValueError("build_file_manifest: files must be non-empty")
    duplicates = sorted({p for p in rel_paths if rel_paths.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"build_file_manifest: duplicate relative path(s): {duplicates}"
        )
    lines: list[str] = []
    for rel in sorted(rel_paths, key=lambda p: p.encode("utf-8")):
        file_path = root_path / rel
        if not file_path.is_file():
            raise ValueError(f"build_file_manifest: {file_path} is not a file")
        lines.append(f"sha256:{_sha256_hex(file_path.read_bytes())}  {rel}")
    manifest_text = "\n".join(lines) + "\n"
    return manifest_text, sha256_bytes(manifest_text.encode("utf-8"))


def parse_t0(output_solver_txt_path: str | Path) -> tuple[float, str]:
    """Extract source time shift ``t0`` from ``OUTPUT_FILES/output_solver.txt``."""
    text = Path(output_solver_txt_path).read_text(encoding="utf-8", errors="replace")
    matches = _START_TIME_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"{output_solver_txt_path}: expected exactly 1 'start time: ... "
            f"seconds' match, found {len(matches)}"
        )
    t0 = -float(matches[0])
    return t0, "output_solver_txt_start_time"


def crosscheck_t0_semd(
    semd_path: str | Path,
    dt: float,
    t0: float,
    *,
    atol: float = DEFAULT_T0_CROSSCHECK_ATOL,
) -> float:
    """Cross-check ``t0`` against a ``*.semd`` file's first (time) column."""
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    times: list[float] = []
    for line in Path(semd_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        times.append(float(stripped.split()[0]))
    if not times:
        raise ValueError(f"{semd_path}: no data rows found")

    times_arr = np.asarray(times, dtype=np.float64)
    expected = np.arange(len(times_arr), dtype=np.float64) * dt - t0
    max_deviation = float(np.abs(times_arr - expected).max())
    if max_deviation > atol:
        raise ValueError(
            f"{semd_path}: t0 crosscheck max deviation {max_deviation:.6e} s "
            f"exceeds atol {atol:.6e} s (dt={dt}, t0={t0})"
        )
    return max_deviation


def parse_f_resolved_estimate(output_generate_databases_txt_path: str | Path) -> float:
    """Extract ``f_resolved_estimate`` (Hz) from ``output_generate_databases.txt``."""
    text = Path(output_generate_databases_txt_path).read_text(
        encoding="utf-8", errors="replace"
    )
    matches = _MIN_PERIOD_RESOLVED_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"{output_generate_databases_txt_path}: expected exactly 1 "
            f"'Minimum period resolved' match, found {len(matches)}"
        )
    period_s = float(matches[0])
    if period_s <= 0:
        raise ValueError(
            f"{output_generate_databases_txt_path}: non-positive minimum "
            f"period resolved {period_s}"
        )
    return 1.0 / period_s
