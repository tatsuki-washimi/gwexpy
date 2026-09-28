"""Characterize >1 MiB native PSD faults through the frozen public harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HARNESS = Path("/tmp/gwexpy-v025-f3-frozen-harness/benchmarks/io")
sys.path.insert(0, str(HARNESS))
import f3_dttxml_harness as frozen

BASELINE = Path(
    "/tmp/gwexpy-v025-f3-frozen-harness/docs/developers/reports/"
    "v0.2.5-performance/F3/"
    "e2fbcc67ef4b08b14fb6cd8a010b28d2f23cfae55a2cf6389c2a19832845dd69/"
    "baseline-v1/fixture-manifest-small.json"
)
NAMES = (
    "unselected_payload_fault",
    "psd_unselected_short_payload_warning",
    "selected_payload_fault",
    "unselected_metadata_fault",
    "late_xml_unselected_payload_fault",
)
PADDING = b"<!--" + b"x" * 1_100_000 + b"-->"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def generate(directory: Path) -> None:
    directory.mkdir(exist_ok=False)
    source = json.loads(BASELINE.read_text())
    cases = {}
    for name in NAMES:
        original = source["cases"][name]
        payload = Path(original["path"]).read_bytes()
        start = payload.find(b"<LIGO_LW")
        assert start > 0
        padded = payload[:start] + PADDING + payload[start:]
        assert len(padded) > 1_048_576
        path = directory / f"{name}.xml"
        path.write_bytes(padded)
        cases[name] = {
            **original,
            "path": str(path),
            "sha256": sha(padded),
            "bytes": len(padded),
            "source_case_sha256": original["sha256"],
        }
    (directory / "fixture-manifest.json").write_text(
        json.dumps(
            {
                "schema": "gwexpy-f3-large-fault-extension-v1",
                "frozen_small_manifest_sha256": sha(BASELINE.read_bytes()),
                "padding_kind": "XML comment before document root",
                "padding_x_bytes": 1_100_000,
                "cases": cases,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def capture(directory: Path, output: Path) -> None:
    manifest = json.loads((directory / "fixture-manifest.json").read_text())
    result = {
        "schema": "gwexpy-f3-large-fault-public-capture-v1",
        "fixture_manifest_sha256": sha((directory / "fixture-manifest.json").read_bytes()),
        "cases": {
            name: frozen.capture_public_case(case, "native", instrument=False)
            for name, case in manifest["cases"].items()
        },
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "capture"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.directory)
    else:
        if args.output is None:
            parser.error("capture requires --output")
        capture(args.directory, args.output)
