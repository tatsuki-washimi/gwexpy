"""Deterministic HDF5 generator used by the IO conformance harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("sample.h5", "manifest.json")

_SEED = 15_934
_CHANNEL = "H1:CONFORMANCE-HDF5"
_SAMPLE_RATE = 8.0
_T0 = 1_000_000_000.0


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a deterministic GWpy-readable HDF5 time series into *output_dir*."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / "sample.h5"
    manifest_path = output_dir / "manifest.json"

    rng = np.random.default_rng(_SEED)
    values = rng.normal(loc=0.0, scale=1.0, size=32)
    series = TimeSeries(
        values,
        sample_rate=_SAMPLE_RATE,
        t0=_T0,
        name=_CHANNEL,
        unit="m",
    )
    series.write(h5_path, format="hdf5")

    manifest_path.write_text(
        json.dumps(
            {
                "channel": _CHANNEL,
                "files": list(GENERATED_FILES),
                "generator": "hdf5",
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
        "hdf5": h5_path,
        "manifest": manifest_path,
    }
