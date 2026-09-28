"""Record public SDB read signatures using the frozen B1 wheel environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

import gwexpy
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

BASELINE_SOURCE_SHA = "1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c"
BASELINE_WHEEL_SHA256 = (
    "473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498"
)
BASELINE_SDB_FILE_SHA256 = (
    "714a7320810755c7479e5c35dc5e0344ba513fc01989b89586d77a4d6618fa6f"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return "NaN"
        if np.isposinf(number):
            return "+Inf"
        if np.isneginf(number):
            return "-Inf"
        return number
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(
    path: Path,
    column: str,
    window_unix: dict[str, int],
    fp_kind: str | None,
    fp_mode: str,
) -> dict:
    old_err = np.seterr(**({fp_kind: fp_mode} if fp_kind is not None else {}))
    try:
        start = float(Time(window_unix["start"], format="unix").gps)
        end = float(Time(window_unix["end"], format="unix").gps)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = read_timeseriesdict_sdb(
                    path, columns=[column], start=start, end=end
                )
            except Exception as exc:  # Record exact public exception class/message.
                return {
                    "outcome": "raise",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "warnings": [
                        {
                            "category": item.category.__name__,
                            "message": str(item.message),
                        }
                        for item in caught
                    ],
                }
        series = result[column]
        return {
            "outcome": "return",
            "warnings": [
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in caught
            ],
            "dtype": str(series.dtype),
            "length": len(series),
            "values": [_json_value(value) for value in series.value],
            "times_gps": [_json_value(value) for value in series.times.value],
            "t0_gps": _json_value(series.t0.value),
        }
    finally:
        np.seterr(**old_err)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("b1-public-signatures.json")
    )
    args = parser.parse_args()
    fixture_manifest = json.loads(
        (args.fixture_dir.parent / "fixture-manifest-v4.json").read_text(
            encoding="utf-8"
        )
    )
    installed_sdb_path = Path(read_timeseriesdict_sdb.__code__.co_filename)
    installed_sdb_sha256 = _sha256(installed_sdb_path)
    if installed_sdb_sha256 != BASELINE_SDB_FILE_SHA256:
        raise RuntimeError(
            "B1 installed sdb.py digest mismatch: "
            f"{installed_sdb_sha256} != {BASELINE_SDB_FILE_SHA256}"
        )
    cases: dict[str, Any] = {}
    for name, case in fixture_manifest["cases"].items():
        path = args.fixture_dir / case["name"]
        policies: list[tuple[str | None, str]] = [(None, "default")]
        if name.startswith("overflow_"):
            policies.extend([("over", "warn"), ("over", "raise")])
        elif name.startswith("underflow_"):
            policies.extend([("under", "warn"), ("under", "raise")])
        case_signatures = {}
        for kind, mode in policies:
            label = "default" if kind is None else f"{kind}_{mode}"
            case_signatures[label] = _signature(
                path, case["column"], case["window_unix"], kind, mode
            )
        cases[name] = {
            "fixture_sha256": _sha256(path),
            "column": case["column"],
            "storage_class_counts": case["storage_class_counts"],
            "window_unix": case["window_unix"],
            "signatures": case_signatures,
        }
    payload = {
        "schema": 1,
        "baseline": "B1",
        "source_sha": BASELINE_SOURCE_SHA,
        "wheel_sha256": BASELINE_WHEEL_SHA256,
        "installed_sdb_sha256_expected": BASELINE_SDB_FILE_SHA256,
        "installed_sdb_path": str(installed_sdb_path),
        "installed_sdb_sha256": installed_sdb_sha256,
        "gwexpy_module_path": str(Path(gwexpy.__file__).resolve()),
        "python": __import__("platform").python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "fixture_manifest_sha256": _sha256(
            args.fixture_dir.parent / "fixture-manifest-v4.json"
        ),
        "cases": cases,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
