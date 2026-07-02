"""Deterministic CSV/TXT generator used by the IO conformance harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

__all__ = ["GENERATED_FILES", "generate"]

GENERATED_FILES = ("sample.csv", "sample.txt", "manifest.json")

_CSV_ROWS: tuple[tuple[str, str], ...] = (
    ("channel", "value"),
    ("H1:STRAIN", "1.0"),
    ("L1:STRAIN", "0.5"),
)

_TXT_LINES: tuple[str, ...] = (
    "generator=csv_txt",
    "kind=conformance",
    "rows=3",
)


def generate(output_dir: Path) -> dict[str, Path]:
    """Write a tiny deterministic CSV/TXT fixture set into *output_dir*."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sample.csv"
    txt_path = output_dir / "sample.txt"
    manifest_path = output_dir / "manifest.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(_CSV_ROWS)

    txt_path.write_text("\n".join(_TXT_LINES) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "generator": "csv_txt",
                "files": list(GENERATED_FILES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "txt": txt_path,
        "manifest": manifest_path,
    }
