"""Create the F4 baseline-v2 primary Linux PSS workload.

Keep the baseline-v1 fixture helper and its 16 MiB stress files unchanged.
This workload has the same 256 adjacent GWF sources with 32,768 float64
samples per source, or 64 MiB of decoded values in total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from f4_gwf_fixtures import CHANNEL, GPS_START, SAMPLE_RATE_HZ, _frame

PARTS = 256
SAMPLES_PER_PART = 32768
DECODED_SAMPLE_BYTES = PARTS * SAMPLES_PER_PART * np.dtype(np.float64).itemsize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixtures(output_dir: Path) -> dict[str, Any]:
    """Write the 64 MiB decoded workload and its file hash manifest."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for part in range(PARTS):
        first_value = part * SAMPLES_PER_PART
        values = np.arange(
            first_value, first_value + SAMPLES_PER_PART, dtype=np.float64
        )
        path = output_dir / f"stress64_{part:04d}.gwf"
        _frame(
            path,
            CHANNEL,
            values,
            GPS_START + first_value / SAMPLE_RATE_HZ,
        )
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema": "gwexpy-v025-b-f4-pss-fixtures-v2",
        "generator": "benchmarks/io/f4_gwf_pss_fixtures.py:write_fixtures",
        "channel": CHANNEL,
        "parts": PARTS,
        "samples_per_part": SAMPLES_PER_PART,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "decoded_sample_bytes": DECODED_SAMPLE_BYTES,
        "source_order": [item["name"] for item in files],
        "value_rule": "float64 arange(part_index * 32768, (part_index + 1) * 32768)",
        "time_rule": "t0 = 1000000000 + part_index * 32768 / 8 GPS seconds; all parts are adjacent",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    write_fixtures(parser.parse_args().output_dir)
