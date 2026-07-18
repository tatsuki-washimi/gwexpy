"""Deterministic Zarr generator for the IO conformance harness.

Writes a Zarr group with one array per channel using the *raw* ``zarr`` library
(never gwexpy), carrying the per-array ``sample_rate``/``t0`` attributes the
gwexpy Zarr reader expects.  Skipped automatically by the harness when ``zarr``
is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("sample.zarr", "manifest.json")

_SEED = 20_260_619
_SAMPLE_RATE = 8.0
_T0 = 1_000_000_000.0
_CHANNELS = ("H1:CONFORMANCE-ZARR", "L1:CONFORMANCE-ZARR")
_N_SAMPLES = 32


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a deterministic gwexpy-readable Zarr store into *output_dir*."""

    import zarr

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / "sample.zarr"
    manifest_path = output_dir / "manifest.json"

    rng = np.random.default_rng(_SEED)
    store = zarr.open_group(str(zarr_path), mode="w")
    creator = getattr(store, "create_array", None) or store.create_dataset
    for idx, channel in enumerate(_CHANNELS):
        values = rng.normal(loc=float(idx), scale=1.0, size=_N_SAMPLES)
        arr = creator(channel, data=values.astype(np.float64))
        arr.attrs["sample_rate"] = _SAMPLE_RATE
        arr.attrs["t0"] = _T0
        arr.attrs["dt"] = 1.0 / _SAMPLE_RATE
        arr.attrs["unit"] = "m"

    manifest_path.write_text(
        json.dumps(
            {
                "generator": "zarr",
                "files": list(GENERATED_FILES),
                "channels": list(_CHANNELS),
                "sample_rate_hz": _SAMPLE_RATE,
                "t0": _T0,
                "seed": _SEED,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "zarr": zarr_path,
        "manifest": manifest_path,
    }
