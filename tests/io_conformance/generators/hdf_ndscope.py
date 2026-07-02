"""Deterministic NDScope HDF5 generator used by the IO conformance harness."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("ndscope.hdf", "manifest.json")

_SEED = 20_241
_CHANNELS = ("H1:CONFORMANCE-NDSCOPE", "L1:CONFORMANCE-NDSCOPE")
_SAMPLE_RATE = 16.0
_T0 = 1_000_000_000.0


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a deterministic NDScope-style HDF5 fixture into *output_dir*."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf_path = output_dir / "ndscope.hdf"
    manifest_path = output_dir / "manifest.json"

    rng = np.random.default_rng(_SEED)
    with h5py.File(hdf_path, "w") as handle:
        for index, channel in enumerate(_CHANNELS):
            group = handle.create_group(channel)
            group.create_dataset(
                "raw",
                data=rng.normal(loc=float(index), scale=0.1, size=32),
            )
            group.attrs["rate_hz"] = _SAMPLE_RATE
            group.attrs["gps_start"] = _T0
            group.attrs["unit"] = "m"

    manifest_path.write_text(
        json.dumps(
            {
                "channels": list(_CHANNELS),
                "files": list(GENERATED_FILES),
                "generator": "hdf_ndscope",
                "sample_rate_hz": _SAMPLE_RATE,
                "seed": _SEED,
                "t0": _T0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "hdf": hdf_path,
        "manifest": manifest_path,
    }
