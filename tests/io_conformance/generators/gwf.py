"""Deterministic GWF generator used by the IO conformance harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("frame.gwf", "manifest.json")

_SEED = 8_183
_CHANNEL = "K1:CONFORMANCE-GWF"
_SAMPLE_RATE = 16.0
_T0 = 1_000_000_000.0


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a deterministic GWF frame into *output_dir* using GWpy."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gwf_path = output_dir / "frame.gwf"
    manifest_path = output_dir / "manifest.json"

    rng = np.random.default_rng(_SEED)
    series = TimeSeries(
        rng.normal(loc=0.0, scale=1.0, size=32),
        sample_rate=_SAMPLE_RATE,
        t0=_T0,
        name=_CHANNEL,
        channel=_CHANNEL,
        unit="m",
    )
    TimeSeriesDict({_CHANNEL: series}).write(gwf_path, format="gwf")

    manifest_path.write_text(
        json.dumps(
            {
                "channel": _CHANNEL,
                "files": list(GENERATED_FILES),
                "generator": "gwf",
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
        "gwf": gwf_path,
        "manifest": manifest_path,
    }
