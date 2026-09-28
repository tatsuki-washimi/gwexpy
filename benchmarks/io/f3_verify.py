"""Verify B-F3 raw captures against independent fixture oracles and each other."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import sys
from pathlib import Path
from typing import Any

try:
    from . import run
except ImportError:  # Direct execution under Python -I.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _selected_oracle(fixture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    points = fixture["points"]
    axis = fixture["frequency_axis"]
    frequencies = struct.pack(
        f"<{points}d", *(axis["f0"] + axis["df"] * i for i in range(points))
    )
    return (
        {
            "dtype": "<f4",
            "shape": [points],
            "sha256": fixture["selected_raw_sha256"],
        },
        {
            "dtype": "<f8",
            "shape": [points],
            "sha256": hashlib.sha256(frequencies).hexdigest(),
        },
    )


def _check_valid_public(
    cases: dict[str, Any], fixture: dict[str, Any], arm: str
) -> None:
    expected_values, expected_frequencies = _selected_oracle(fixture)
    expected_order = [fixture["selected_channel"], *fixture["unselected_channels"]]
    for route in ("native", "external_requested", "fallback_forced"):
        for name in (
            "many_valid",
            "many_valid_all_channels",
            "many_valid_empty_channels",
        ):
            result = cases[name][route]["outcome"]
            _expect(result["status"] == "ok", f"{arm}/{route}/{name}: not valid")
            fingerprint = result["fingerprint"]
            if name == "many_valid_empty_channels":
                _expect(
                    fingerprint["order"] == [] and fingerprint["entries"] == {},
                    f"{arm}/{route}/{name}: expected empty public dict",
                )
                continue
            _expect(
                fingerprint["order"]
                == (expected_order[:1] if name == "many_valid" else expected_order),
                f"{arm}/{route}/{name}: channel order",
            )
            selected = fingerprint["entries"][fixture["selected_channel"]]
            _expect(
                selected["values"] == expected_values
                and selected["frequencies"] == expected_frequencies
                and selected["epoch"] == fixture["epoch"],
                f"{arm}/{route}/{name}: selected PSD oracle",
            )
    gzip = cases["many_valid_gzip"]["native"]["outcome"]
    _expect(
        gzip == cases["many_valid"]["native"]["outcome"],
        f"{arm}: native gzip fingerprint differs",
    )
    oracle = fixture["tf6_oracle"]
    value_raw = struct.pack(
        "<6f",
        *(
            part
            for real, imag in zip(oracle["real"], oracle["imag"], strict=True)
            for part in (real, imag)
        ),
    )
    frequency_raw = struct.pack("<3d", *oracle["frequencies"])
    for route in ("native", "external_requested", "fallback_forced"):
        tf = cases["tf6_raw_layout"][route]["outcome"]
        _expect(tf["status"] == "ok", f"{arm}/{route}: TF6 failed")
        result = tf["fingerprint"]
        _expect(
            result["rows"] == oracle["pair"][:1]
            and result["cols"] == oracle["pair"][1:]
            and result["values"]
            == {
                "dtype": "<c8",
                "shape": [1, 1, 3],
                "sha256": hashlib.sha256(value_raw).hexdigest(),
            }
            and result["frequencies"]
            == {
                "dtype": "<f8",
                "shape": [3],
                "sha256": hashlib.sha256(frequency_raw).hexdigest(),
            },
            f"{arm}/{route}: TF6 independent complex and frequency oracle",
        )


def _check_public_parity(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    a = records["A"][0]["sample"]["cases"]
    b = records["B"][0]["sample"]["cases"]
    _expect(a.keys() == b.keys(), "B0/B1 public case set differs")
    differences = []
    for name in a:
        for route in ("native", "external_requested", "fallback_forced"):
            left = {
                key: value
                for key, value in a[name][route].items()
                if key not in ("gwexpy_import_path", "gwexpy_version")
            }
            right = {
                key: value
                for key, value in b[name][route].items()
                if key not in ("gwexpy_import_path", "gwexpy_version")
            }
            if left != right:
                differences.append(f"{name}/{route}")
    return {
        "cases_per_arm": len(a),
        "routes_per_case": 3,
        "b0_b1_exact_parity": not differences,
        "b0_b1_differences": differences,
    }


def _check_structure(
    records: dict[str, list[dict[str, Any]]], fixture: dict[str, Any], mode: str
) -> None:
    expected_values, expected_frequencies = _selected_oracle(fixture)
    for arm in ("A", "B"):
        for record in records[arm]:
            cases = record["sample"]["cases"]
            for name in ("many_valid", "many_valid_gzip"):
                if name not in cases:
                    continue
                result = cases[name]["native"]
                decoded = result["base64_observation"]
                materialized = result["series_materialization"]
                fingerprint = result["outcome"]["fingerprint"]
                selected = fingerprint["entries"][fixture["selected_channel"]]
                _expect(
                    decoded["unselected_decoded_bytes"]
                    == fixture["old_r_many_valid_unselected_decoded_bytes"]
                    and decoded["selected_decoded_bytes"]
                    == fixture["selected_raw_bytes"]
                    and decoded["unknown_decode_calls"] == 0
                    and materialized["series_constructed"] == 1
                    and materialized["unselected_series_constructed"] == 0,
                    f"{arm}/{mode}/{name}: structural baseline mismatch",
                )
                _expect(
                    fingerprint["order"] == [fixture["selected_channel"]]
                    and selected["values"] == expected_values
                    and selected["frequencies"] == expected_frequencies
                    and selected["epoch"] == fixture["epoch"],
                    f"{arm}/{mode}/{name}: selected result oracle mismatch",
                )


def _check_times(
    records: dict[str, list[dict[str, Any]]], fixture: dict[str, Any], mode: str
) -> None:
    values, frequencies = _selected_oracle(fixture)
    for arm in ("A", "B"):
        for record in records[arm]:
            sample = record["sample"]
            if mode == "memory":
                worker = sample["worker"]
                _expect(
                    sample["sample_count"] >= 2
                    and sample["peak_pss_kib"] >= sample["baseline_pss_kib"]
                    and sample["peak_rss_kib"] >= sample["baseline_rss_kib"]
                    and sample["sampled_peak_limit"],
                    f"{arm}: invalid Linux process-tree memory capture",
                )
                fingerprint = worker["fingerprint"]
                _expect(
                    worker["selection_pushdown_supported"] is False,
                    f"{arm}/{mode}: old-R unexpectedly accepts parser selection",
                )
                observed_values = fingerprint["values"]
                observed_frequencies = fingerprint["frequencies"]
            else:
                _expect(
                    sample["selection_pushdown_supported"] is False,
                    f"{arm}/{mode}: old-R unexpectedly accepts parser selection",
                )
                observed_values = sample["selected_values"]
                observed_frequencies = sample["selected_frequencies"]
            _expect(
                observed_values == values and observed_frequencies == frequencies,
                f"{arm}/{mode}: selected result differs from fixture oracle",
            )


def verify(
    small_path: Path, large_path: Path, capture_dirs: list[Path]
) -> dict[str, Any]:
    """Check every supplied mode against file hashes and public result oracles."""
    small = _read(small_path)
    large = _read(large_path)
    _expect(
        small["native_selection_scope"]["PSD"]["status"] == "eligible"
        and small["native_selection_scope"]["TS"]["status"] == "fallback_only"
        and all(
            small["native_selection_scope"][product]["status"] == "HOLD"
            for product in ("FFT", "STF", "TF6")
        ),
        "F3 native product scope changed",
    )
    digest = run.harness_digest()
    results: dict[str, Any] = {}
    public_modes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for directory in capture_dirs:
        manifest = _read(directory / "manifest.json")
        mode = manifest["mode"]
        _expect(mode not in results, f"Duplicate F3 mode {mode}")
        _expect(manifest["harness_digest"] == digest, f"{mode}: harness drift")
        expected_fixture = (
            small if mode in ("correctness", "structure_matrix") else large
        )
        expected_manifest = small_path if expected_fixture is small else large_path
        _expect(
            manifest["fixture_manifest_sha256"] == _sha(expected_manifest),
            f"{mode}: fixture manifest drift",
        )
        records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
        for index, arm in enumerate(manifest["order"]):
            item = _read(directory / f"sample-{index:02d}-{arm}.json")
            _expect(
                item["audit"]["wheel_sha256"] == manifest["arms"][arm]["wheel_sha256"],
                f"{mode}/{arm}: wheel audit mismatch",
            )
            records[arm].append(item)
        for arm in ("A", "B"):
            _expect(
                records[arm] == _read(directory / f"raw-{arm}.json"),
                f"{mode}/{arm}: raw file mismatch",
            )
            _expect(
                len(records[arm]) == manifest["samples_per_arm"],
                f"{mode}/{arm}: sample count mismatch",
            )
        if mode in ("correctness", "structure_matrix"):
            public_modes[mode] = records
            for arm in ("A", "B"):
                cases = records[arm][0]["sample"]["cases"]
                _expect(
                    cases.keys() == small["cases"].keys(), "Public case set incomplete"
                )
                _check_valid_public(cases, small, arm)
                for name, routes in cases.items():
                    product = small["cases"][name]["call_kwargs"]["products"]
                    _expect(
                        {route: item["actual_route"] for route, item in routes.items()}
                        == {
                            "native": "external" if product == "TS" else "native",
                            "external_requested": "external",
                            "fallback_forced": "fallback",
                        },
                        f"{mode}/{arm}/{name}: route identity mismatch",
                    )
                    _expect(
                        routes["native"]["native_parser_exception_applies"]
                        == (
                            name
                            in (
                                "unselected_payload_fault",
                                "psd_unselected_short_payload_warning",
                                "ts_unselected_short_payload_warning",
                                "tf6_unselected_payload_fault",
                            )
                        ),
                        f"{mode}/{arm}/{name}: #589 exception scope mismatch",
                    )
                    if name.startswith("late_xml_"):
                        native = routes["native"]
                        _expect(
                            native["outcome"]["status"] == "ok"
                            and native["outcome"]["fingerprint"]["order"] == []
                            and len(native["warnings"]) == 1
                            and native["warnings"][0]["message"].startswith(
                                "Failed to parse DTT XML:"
                            ),
                            f"{mode}/{arm}/{name}: EOF parse precedence changed",
                        )
                        if mode == "structure_matrix":
                            _expect(
                                native["base64_observation"]["total_decoded_bytes"]
                                == 0,
                                f"{mode}/{arm}/{name}: decoded before EOF parse",
                            )
                    if small["cases"][name]["oracle_category"] == (
                        "unselected_payload_semantic_error_hold"
                    ):
                        _expect(
                            routes["native"]["outcome"]["status"] == "error",
                            f"{mode}/{arm}/{name}: HOLD semantic error missing",
                        )
            summary = _check_public_parity(records)
            if mode == "structure_matrix":
                _check_structure(records, small, mode)
        elif mode == "structure_many":
            _check_structure(records, large, mode)
            for arm in ("A", "B"):
                expected = records[arm][0]["sample"]["cases"]["many_valid"]["native"]
                for record in records[arm]:
                    observed = record["sample"]["cases"]["many_valid"]["native"]
                    _expect(
                        observed == expected, f"{arm}: repeated large structure changed"
                    )
            summary = {
                "samples_per_arm": 5,
                "many_unselected_decoded_bytes": large[
                    "old_r_many_valid_unselected_decoded_bytes"
                ],
            }
        else:
            _check_times(records, large, mode)
            _expect((directory / "summary.json").is_file(), f"{mode}: no summary")
            summary = _read(directory / "summary.json")
            for arm in ("A", "B"):
                for field, values in summary[arm].items():
                    median = statistics.median(values["raw"])
                    mad = statistics.median(
                        abs(value - median) for value in values["raw"]
                    )
                    _expect(
                        values["median"] == median and values["mad"] == mad,
                        f"{mode}/{arm}/{field}: summary mismatch",
                    )
        results[mode] = summary
    if "correctness" in public_modes and "structure_matrix" in public_modes:
        for arm in ("A", "B"):
            correctness = public_modes["correctness"][arm][0]["sample"]["cases"]
            structure = public_modes["structure_matrix"][arm][0]["sample"]["cases"]
            for name, routes in correctness.items():
                for route, item in routes.items():
                    measured = structure[name][route]
                    for field in ("outcome", "warnings", "logs"):
                        _expect(
                            item[field] == measured[field],
                            f"{arm}/{name}/{route}: structure spy altered {field}",
                        )
    return {"status": "BASELINE_CANDIDATE", "harness_digest": digest, "modes": results}


def main() -> None:
    """Run verification on one B-F3 capture set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-manifest", type=Path, required=True)
    parser.add_argument("--large-manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.small_manifest, args.large_manifest, args.capture_dir)
    content = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
