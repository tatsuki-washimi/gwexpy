"""B0/B1 DTTXML baseline controller with separate public, structure, and PSS runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from . import f3_dttxml_harness as harness
    from . import run
except ImportError:  # Direct execution under Python -I.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import f3_dttxml_harness as harness
    import run


ORDER = ("A", "B", "B", "A", "B", "A", "A", "B", "A", "B")
MODES = ("correctness", "structure_matrix", "structure_many", "warm", "cold", "memory")


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(args: argparse.Namespace, arm: str, field: str) -> str:
    return str(getattr(args, f"{field}_{'a' if arm == 'A' else 'b'}"))


def _command(args: argparse.Namespace, arm: str, action: str) -> list[str]:
    return [
        _arm(args, arm, "python"),
        "-I",
        str(Path(__file__).resolve()),
        action,
        "--wheel",
        _arm(args, arm, "wheel"),
        "--version",
        _arm(args, arm, "version"),
    ]


def _timer(path: str, warmup: str, mode: str, wheel: str, version: str) -> None:
    audit = run._worker_audit(Path(wheel), version, full=False)
    if Path(path).resolve() == Path(warmup).resolve():
        raise ValueError("Timing fixture and warm-up must be distinct")
    selected_channel = "K1:F3-SELECTED"
    prepared = harness._prepare_selected_psd(selected_channel)
    if mode == "cold":
        started_wall = time.perf_counter_ns()
        started_cpu = time.process_time_ns()
        pushdown, selected, retained = harness._parser_selected_psd(
            path, selected_channel, prepared
        )
        result = {
            "outcome": "return",
            "wall_ns": time.perf_counter_ns() - started_wall,
            "cpu_ns": time.process_time_ns() - started_cpu,
            "selection_pushdown_supported": pushdown,
            "selected_values": harness._array_fingerprint(selected["data"]),
            "selected_frequencies": harness._array_fingerprint(selected["frequencies"]),
        }
        del retained
        print(json.dumps({"audit": audit, "sample": result}), flush=True)
        return
    harness._parser_selected_psd(warmup, selected_channel, prepared)
    print(json.dumps({"ready": True, "audit": audit}), flush=True)
    for command in sys.stdin:
        if command.strip() == "quit":
            return
        if command.strip() != "sample":
            raise ValueError(f"Unknown F3 timing command {command!r}")
        started_wall = time.perf_counter_ns()
        started_cpu = time.process_time_ns()
        pushdown, selected, retained = harness._parser_selected_psd(
            path, selected_channel, prepared
        )
        result = {
            "outcome": "return",
            "wall_ns": time.perf_counter_ns() - started_wall,
            "cpu_ns": time.process_time_ns() - started_cpu,
            "selection_pushdown_supported": pushdown,
            "selected_values": harness._array_fingerprint(selected["data"]),
            "selected_frequencies": harness._array_fingerprint(selected["frequencies"]),
        }
        del retained
        print(json.dumps({"audit": audit, "sample": result}), flush=True)


def _worker(args: argparse.Namespace) -> None:
    if args.action == "_audit":
        print(
            json.dumps(
                {"audit": run._worker_audit(Path(args.wheel), args.version, full=True)},
                sort_keys=True,
            )
        )
    else:
        _timer(args.path, args.warmup, args.mode, args.wheel, args.version)


def _subprocess(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter_ns()
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    result["stderr"] = completed.stderr
    result["controller_elapsed_ns"] = time.perf_counter_ns() - started
    return result


def _public(args: argparse.Namespace, arm: str, mode: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gwexpy-f3-public-") as directory:
        output = Path(directory) / "sample.json"
        command = [
            _arm(args, arm, "python"),
            "-I",
            str(Path(harness.__file__).resolve()),
            "public",
            "--manifest",
            args.manifest,
            "--output",
            str(output),
            "--mode",
            "correctness" if mode == "correctness" else "structure",
        ]
        if mode == "structure_many":
            command.extend(("--case", "many_valid", "--route", "native"))
        started = time.perf_counter_ns()
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        return {
            "sample": json.loads(output.read_text()),
            "stderr": completed.stderr,
            "controller_elapsed_ns": time.perf_counter_ns() - started,
        }


def _pss(args: argparse.Namespace, arm: str, warmup: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gwexpy-f3-pss-") as directory:
        output = Path(directory) / "sample.json"
        command = [
            _arm(args, arm, "python"),
            "-I",
            str(Path(harness.__file__).resolve()),
            "pss",
            "--manifest",
            args.manifest,
            "--warmup-path",
            warmup,
            "--output",
            str(output),
            "--sample-interval-ms",
            "5",
        ]
        started = time.perf_counter_ns()
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        return {
            "sample": json.loads(output.read_text()),
            "stderr": completed.stderr,
            "controller_elapsed_ns": time.perf_counter_ns() - started,
        }


def _start_service(
    args: argparse.Namespace, arm: str, path: str, warmup: str
) -> subprocess.Popen[str]:
    command = _command(args, arm, "_timer") + [
        "--path",
        path,
        "--warmup",
        warmup,
        "--mode",
        "warm",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line or not json.loads(line).get("ready"):
        assert process.stderr is not None
        raise RuntimeError(f"F3 warm service failed: {process.stderr.read()[-4000:]}")
    return process


def _warm_sample(process: subprocess.Popen[str]) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    started = time.perf_counter_ns()
    process.stdin.write("sample\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        assert process.stderr is not None
        raise RuntimeError(f"F3 warm service exited: {process.stderr.read()[-4000:]}")
    result = json.loads(line)
    result["stderr"] = ""
    result["controller_elapsed_ns"] = time.perf_counter_ns() - started
    return result


def _summary(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    metrics = (
        ("peak_pss_kib", "delta_pss_kib", "peak_rss_kib", "wall_ns")
        if mode == "memory"
        else ("controller_elapsed_ns", "cpu_ns")
        if mode == "cold"
        else ("wall_ns", "cpu_ns")
    )
    result = {}
    for metric in metrics:
        values = [
            record["controller_elapsed_ns"]
            if metric == "controller_elapsed_ns"
            else record["sample"]["worker"][metric]
            if mode == "memory" and metric == "wall_ns"
            else record["sample"][metric]
            for record in records
        ]
        median = statistics.median(values)
        result[metric] = {
            "raw": values,
            "median": median,
            "mad": statistics.median(abs(value - median) for value in values),
        }
    return result


def _verify_sample_identity(
    record: dict[str, Any], expected: dict[str, Any], mode: str
) -> None:
    """Check every capture imported its nominated installed wheel arm."""
    if mode in ("warm", "cold"):
        observed = record["audit"]
        path_key = "gwexpy_path"
    elif mode == "memory":
        observed = record["sample"]["worker"]
        path_key = "gwexpy_import_path"
    else:
        cases = record["sample"]["cases"]
        observed = next(iter(cases.values()))["native"]
        path_key = "gwexpy_import_path"
        for routes in cases.values():
            for item in routes.values():
                if (
                    item["gwexpy_import_path"] != observed[path_key]
                    or item["gwexpy_version"] != observed["gwexpy_version"]
                ):
                    raise RuntimeError("F3 public cases imported different wheel paths")
    if (
        observed[path_key] != expected["gwexpy_path"]
        or observed["gwexpy_version"] != expected["gwexpy_version"]
    ):
        raise RuntimeError("F3 worker imported the wrong installed wheel arm")


def _capture(args: argparse.Namespace) -> None:
    fixture = json.loads(Path(args.manifest).read_text())
    if args.mode in ("warm", "cold", "memory", "structure_many") and args.samples != 5:
        raise ValueError("F3 baseline requires five samples per arm")
    source = fixture["cases"]["many_valid"]
    harness._checked_case_hash(source)
    warmup_fixture = json.loads(Path(args.warmup_manifest).read_text())
    warmup = str(warmup_fixture["cases"]["many_valid"]["path"])
    harness._checked_case_hash(warmup_fixture["cases"]["many_valid"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        arm: _subprocess(_command(args, arm, "_audit"))["audit"] for arm in ("A", "B")
    }
    if (
        audit["A"]["python"] != audit["B"]["python"]
        or audit["A"]["distributions"] != audit["B"]["distributions"]
    ):
        raise RuntimeError("F3 Python or dependency distributions differ")
    order = ("A", "B") if args.mode in ("correctness", "structure_matrix") else ORDER
    _write_new(
        output / "manifest.json",
        {
            "schema": 1,
            "status": "UNBASELINED",
            "lane": "B-F3",
            "mode": args.mode,
            "order": order,
            "samples_per_arm": 1 if len(order) == 2 else 5,
            "fixture_manifest_sha256": _sha(Path(args.manifest)),
            "warmup_manifest_sha256": _sha(Path(args.warmup_manifest)),
            "harness_digest": run.harness_digest(),
            "host": {"platform": platform.platform(), "uname": list(platform.uname())},
            "arms": {
                arm: {
                    **audit[arm],
                    "source_sha": _arm(args, arm, "source_sha"),
                    "install_mode": "wheel-no-deps",
                }
                for arm in ("A", "B")
            },
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    services = (
        {arm: _start_service(args, arm, source["path"], warmup) for arm in ("A", "B")}
        if args.mode == "warm"
        else {}
    )
    try:
        for index, arm in enumerate(order):
            if args.mode in ("correctness", "structure_matrix", "structure_many"):
                record = _public(args, arm, args.mode)
            elif args.mode == "memory":
                record = _pss(args, arm, warmup)
            elif args.mode == "warm":
                record = _warm_sample(services[arm])
            else:
                record = _subprocess(
                    _command(args, arm, "_timer")
                    + ["--path", source["path"], "--warmup", warmup, "--mode", "cold"]
                )
            _verify_sample_identity(record, audit[arm], args.mode)
            record["audit"] = {
                key: audit[arm][key]
                for key in ("gwexpy_path", "gwexpy_version", "wheel_sha256")
            }
            if args.mode in ("warm", "cold"):
                identity = record["sample"]["selected_values"]
                if not identity["sha256"]:
                    raise RuntimeError("F3 timing worker has no selected data")
            elif args.mode == "memory":
                if record["sample"]["peak_pss_kib"] is None:
                    raise RuntimeError("F3 memory worker has no PSS sample")
            records[arm].append(record)
            _write_new(output / f"sample-{index:02d}-{arm}.json", record)
    finally:
        for process in services.values():
            assert process.stdin is not None
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.communicate(timeout=20)
    for arm in ("A", "B"):
        _write_new(output / f"raw-{arm}.json", records[arm])
    if args.mode in ("warm", "cold", "memory"):
        _write_new(
            output / "summary.json",
            {arm: _summary(records[arm], args.mode) for arm in ("A", "B")},
        )
    print(output)


def main() -> None:
    """Run isolated worker commands or collect one B-F3 baseline mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--warmup-manifest", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--mode", choices=MODES, required=True)
    capture.add_argument("--samples", type=int, default=5)
    for suffix in ("a", "b"):
        capture.add_argument(f"--python-{suffix}", required=True)
        capture.add_argument(f"--wheel-{suffix}", required=True)
        capture.add_argument(f"--version-{suffix}", required=True)
        capture.add_argument(f"--source-sha-{suffix}", required=True)
    audit = sub.add_parser("_audit")
    audit.add_argument("--wheel", required=True)
    audit.add_argument("--version", required=True)
    timer = sub.add_parser("_timer")
    timer.add_argument("--wheel", required=True)
    timer.add_argument("--version", required=True)
    timer.add_argument("--path", required=True)
    timer.add_argument("--warmup", required=True)
    timer.add_argument("--mode", choices=("warm", "cold"), required=True)
    args = parser.parse_args()
    if args.action == "capture":
        _capture(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
