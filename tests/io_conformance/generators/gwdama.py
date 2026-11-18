"""Deterministic GWDama HDF5 generator used by the IO conformance harness."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("gwdama.h5", "manifest.json")

_SEED = 60_701
_CHANNELS = ("H1:CONFORMANCE-GWDAMA", "nested/L1:CONFORMANCE-GWDAMA")
_SAMPLE_RATE = 16.0
_T0 = 1_000_000_000.0


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a deterministic GWDama-style HDF5 fixture into *output_dir*."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf_path = output_dir / "gwdama.h5"
    manifest_path = output_dir / "manifest.json"

    rng = np.random.default_rng(_SEED)
    with h5py.File(hdf_path, "w") as handle:
        handle.attrs["dama_name"] = "conformance"
        handle.attrs["time_stamp"] = "2026-01-01T00:00:00Z"
        nested_group = handle.create_group("nested")
        nested_group.attrs["site"] = "LHO"
        for index, channel in enumerate(_CHANNELS):
            ds = handle.create_dataset(
                channel,
                data=rng.normal(loc=float(index), scale=0.1, size=32),
            )
            ds.attrs["sample_rate"] = _SAMPLE_RATE
            ds.attrs["t0"] = _T0
            ds.attrs["channel"] = channel.split("/")[-1]
            ds.attrs["unit"] = "m"
            ds.attrs["calibration_factor"] = 1.25 + index

    manifest_path.write_text(
        json.dumps(
            {
                "channels": list(_CHANNELS),
                "files": list(GENERATED_FILES),
                "generator": "gwdama",
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
