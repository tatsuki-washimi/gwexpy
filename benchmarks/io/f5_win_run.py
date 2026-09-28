"""Independent, wheel-isolated B-F5 WIN correctness and decoder benchmark.

The controller imports no gwexpy code. Workers use an installed wheel under
``python -I`` and verify its package files before any evidence is accepted.
Correctness, structure, warm CPU, cold startup, and sampled memory are separate
runs. Evidence directories are create-only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

try:
    from . import f5_win_fixtures as fixtures
    from . import run as common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import f5_win_fixtures as fixtures
    import run as common

PRIMARY_CASE = "width-1-rate-4095"
ORDER = common.ABBA


def _digest() -> str:
    return common.harness_digest()


def _json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "win-fixtures.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["format"] != "B-F5 WIN baseline fixture manifest v1":
        raise ValueError("unexpected WIN fixture manifest format")
    for case in data["cases"]:
        if common.sha256(root / case["file"]) != case["sha256"]:
            raise ValueError(f"WIN fixture hash mismatch: {case['file']}")
    return data


def _warnings(items: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            "category": f"{item.category.__module__}.{item.category.__qualname__}",
            "message": str(item.message),
        }
        for item in items
    ]


def _trace_fingerprint(trace: Any) -> dict[str, Any]:
    data = trace.data
    return {
        "channel": str(trace.stats.channel),
        "sampling_rate": float(trace.stats.sampling_rate),
        "starttime": str(trace.stats.starttime),
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "values_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "values": data.tolist(),
    }


def _call(path: Path, public: bool) -> dict[str, Any]:
    from gwexpy.timeseries.io.win import _read_win_fixed, read_win_file

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = read_win_file(path) if public else _read_win_fixed(path)
            if public:
                order = list(result)
                series = next(iter(result.values()))
                values = series.value
                value = {
                    "keys": order,
                    "channel": str(series.channel),
                    "sample_rate": repr(series.sample_rate),
                    "sampling_rate_hz": float(series.sample_rate.value),
                    "t0": repr(series.t0),
                    "dtype": str(values.dtype),
                    "shape": list(values.shape),
                    "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "values": values.tolist(),
                }
            else:
                value = {
                    "trace_count": len(result),
                    "traces": [_trace_fingerprint(trace) for trace in result],
                }
            outcome: dict[str, Any] = {"outcome": "return", "result": value}
        except Exception as exc:
            outcome = {
                "outcome": "error",
                "error": {
                    "category": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                },
            }
    outcome["warnings"] = _warnings(caught)
    return outcome


def _correctness(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for case in manifest["cases"]:
        path = root / case["file"]
        cases[case["name"]] = {
            "fixture_sha256": case["sha256"],
            "decoder": _call(path, False),
            "public": _call(path, True),
        }
    return cases


def _validate_contract(cases: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reject evidence whose public or decoder behavior misses the wire recipe."""
    for case in manifest["cases"]:
        name = case["name"]
        expected = case["expected"]
        result = cases[name]
        for route in ("decoder", "public"):
            observed = result[route]
            if expected["error"] is None:
                if observed["outcome"] != "return":
                    raise RuntimeError(f"{name}/{route} unexpectedly failed")
                values = observed["result"]
                if route == "decoder":
                    if values["trace_count"] != 1:
                        raise RuntimeError(f"{name}/{route} trace count changed")
                    values = values["traces"][0]
                    if values["channel"] != expected["channel"]:
                        raise RuntimeError(f"{name}/{route} channel changed")
                    if values["sampling_rate"] != expected["sampling_rate"]:
                        raise RuntimeError(f"{name}/{route} rate changed")
                else:
                    if values["keys"] != [f"...{expected['channel']}"]:
                        raise RuntimeError(f"{name}/{route} key changed")
                    if values["sampling_rate_hz"] != expected["sampling_rate"]:
                        raise RuntimeError(f"{name}/{route} rate changed")
                if values["dtype"] != expected["dtype"] or (
                    values["values"] != expected["samples"]
                ):
                    raise RuntimeError(f"{name}/{route} values or dtype changed")
            elif observed["outcome"] != "error" or (
                observed["error"] != expected["error"]
            ):
                raise RuntimeError(f"{name}/{route} error changed")
            expected_warnings = [expected["warning"]] if route == "public" else []
            if observed["warnings"] != expected_warnings:
                raise RuntimeError(f"{name}/{route} warning changed")


def _structure(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from gwexpy.timeseries.io import win

    source_path = Path(inspect.getsourcefile(win._read_win_fixed) or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    append_sites: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "append" and isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"samples", "output"}:
                    append_sites[node.lineno] = ast.unparse(node)
    counts: dict[int, int] = {line: 0 for line in append_sites}

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "line" and frame.f_code.co_filename == str(source_path):
            if frame.f_lineno in counts:
                counts[frame.f_lineno] += 1
        return tracer

    sys.settrace(tracer)
    try:
        # Every width is exercised, including a 4095-sample 1-byte record.
        for case in manifest["cases"]:
            if case["expected"]["samples"] is not None:
                win._read_win_fixed(root / case["file"])
    finally:
        sys.settrace(None)
    return {
        "source_sha256": common.sha256(source_path),
        "source_path": str(source_path),
        "append_sites": {
            str(line): text for line, text in sorted(append_sites.items())
        },
        "append_line_hits": {
            str(line): count for line, count in sorted(counts.items())
        },
        "python_per_sample_accumulation_hits": sum(counts.values()),
    }


def _timed(path: Path) -> dict[str, Any]:
    from gwexpy.timeseries.io.win import _read_win_fixed

    start_wall = time.perf_counter_ns()
    start_cpu = time.process_time_ns()
    stream = _read_win_fixed(path)
    cpu_ns = time.process_time_ns() - start_cpu
    wall_ns = time.perf_counter_ns() - start_wall
    return {
        "outcome": "return",
        "cpu_ns": cpu_ns,
        "wall_ns": wall_ns,
        "sample_count": sum(len(trace.data) for trace in stream),
    }


def _worker(args: argparse.Namespace) -> None:
    audit = common._worker_audit(
        Path(args.wheel), args.version, full=args.mode == "audit"
    )
    if args.mode == "audit":
        print(json.dumps({"audit": audit}, sort_keys=True))
        return
    root = Path(args.fixtures)
    manifest = _manifest(root)
    if args.mode == "correctness":
        sample = {"cases": _correctness(root, manifest)}
    elif args.mode == "structure":
        sample = {"structure": _structure(root, manifest)}
    else:
        path = root / next(
            case["file"] for case in manifest["cases"] if case["name"] == PRIMARY_CASE
        )
        if args.mode == "service":
            fixtures.warm_cpu_call(
                path,
                __import__(
                    "gwexpy.timeseries.io.win", fromlist=["_read_win_fixed"]
                )._read_win_fixed,
            )
            print(json.dumps({"ready": True, "audit": audit}), flush=True)
            for line in sys.stdin:
                if line.strip() == "quit":
                    break
                if line.strip() != "sample":
                    raise ValueError("unknown service request")
                print(json.dumps({"audit": audit, "sample": _timed(path)}), flush=True)
            return
        if args.mode == "timing":
            # Cold route includes import and process startup at the controller.
            sample = _timed(path)
        elif args.mode == "memory":
            sample = _timed(path)
        else:
            raise ValueError(args.mode)
    print(json.dumps({"audit": audit, "sample": sample}, sort_keys=True))


def _command(args: argparse.Namespace, arm: str, mode: str) -> list[str]:
    suffix = arm.lower()
    return [
        str(getattr(args, f"python_{suffix}")),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        str(getattr(args, f"wheel_{suffix}")),
        "--version",
        getattr(args, f"version_{suffix}"),
        "--fixtures",
        str(Path(args.fixtures).resolve()),
        "--mode",
        mode,
    ]


def _invoke(args: argparse.Namespace, arm: str, mode: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        _command(args, arm, mode),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    trace: list[dict[str, Any]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            pids = sorted(common._descendants(process.pid))
            by_pid = {
                str(pid): pss
                for pid in pids
                if (pss := common._pss_kib(pid)) is not None
            }
            rss_by_pid = {
                str(pid): rss
                for pid in pids
                if (rss := common._rss_kib(pid)) is not None
            }
            trace.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "pss_kib_by_pid": by_pid,
                    "rss_kib_by_pid": rss_by_pid,
                    "tree_pss_kib": sum(by_pid.values()),
                    "tree_rss_kib": sum(rss_by_pid.values()),
                }
            )
            stop.wait(0.01)

    thread = None
    if mode == "memory":
        if sys.platform != "linux":
            process.kill()
            raise RuntimeError("PSS sampling requires Linux")
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    stdout, stderr = process.communicate()
    stop.set()
    if thread is not None:
        thread.join()
    if process.returncode:
        raise RuntimeError(f"F5 worker failed: {stderr[-4000:]}")
    result = json.loads(stdout)
    result["controller_elapsed_ns"] = time.perf_counter_ns() - started
    result["stderr"] = stderr
    if mode == "memory":
        result["memory"] = {
            "sampling_ms": 10,
            "trace": trace,
            "observed_pids": sorted(
                {pid for item in trace for pid in item["pss_kib_by_pid"]}
            ),
            "peak_tree_pss_kib": max((x["tree_pss_kib"] for x in trace), default=None),
            "peak_tree_rss_kib": max((x["tree_rss_kib"] for x in trace), default=None),
            "sampled_peak_limit": True,
        }
    return result


def _summary(
    records: list[dict[str, Any]], mode: str, temperature: str
) -> dict[str, Any]:
    keys = (
        ("peak_tree_pss_kib", "peak_tree_rss_kib")
        if mode == "memory"
        else ("wall_ns", "cpu_ns")
        if temperature == "warm"
        else ("controller_elapsed_ns", "cpu_ns")
    )
    result = {}
    for key in keys:
        values = [
            record["memory"][key]
            if mode == "memory"
            else record["controller_elapsed_ns"]
            if key == "controller_elapsed_ns"
            else record["sample"][key]
            for record in records
        ]
        median = statistics.median(values)
        result[key] = {
            "raw": values,
            "median": median,
            "mad": statistics.median(abs(value - median) for value in values),
        }
    return result


def _capture(args: argparse.Namespace) -> None:
    root = Path(args.fixtures).resolve()
    fixture = _manifest(root)
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audits = {arm: _invoke(args, arm, "audit")["audit"] for arm in ("A", "B")}
    if audits["A"]["python"] != audits["B"]["python"] or (
        audits["A"]["distributions"] != audits["B"]["distributions"]
    ):
        raise RuntimeError("Python or dependency versions differ between arms")
    for arm in ("A", "B"):
        audits[arm].update(
            {
                "source_sha": getattr(args, f"source_sha_{arm.lower()}"),
                "label": getattr(args, f"label_{arm.lower()}"),
                "install_mode": "wheel-no-deps",
            }
        )
    count = 1 if args.mode in ("correctness", "structure") else args.samples
    if count not in (1, 5, 9):
        raise ValueError("expected one preflight or five/nine metric samples")
    order = ("A", "B") if count == 1 else ORDER[: 2 * count]
    _json_new(
        destination / "manifest.json",
        {
            "schema": 1,
            "status": "UNBASELINED",
            "lane": "F5",
            "scenario": "all-win-cases"
            if args.mode in ("correctness", "structure")
            else PRIMARY_CASE,
            "mode": args.mode,
            "temperature": args.temperature,
            "samples_per_arm": count,
            "order": order,
            "harness_digest": _digest(),
            "fixture_manifest_sha256": common.sha256(root / "win-fixtures.json"),
            "fixture_generator_sha256": common.sha256(Path(fixtures.__file__)),
            "fixture_files": [
                {k: c[k] for k in ("name", "file", "sha256", "size_bytes")}
                for c in fixture["cases"]
            ],
            "host": {
                "platform": platform.platform(),
                "uname": list(platform.uname()),
                "cpu_count": os.cpu_count(),
            },
            "arms": audits,
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    services: dict[str, subprocess.Popen[str]] = {}
    try:
        if args.mode == "timing" and args.temperature == "warm":
            for arm in ("A", "B"):
                process = subprocess.Popen(
                    _command(args, arm, "service"),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                ready = json.loads(process.stdout.readline())
                if not ready.get("ready"):
                    raise RuntimeError(f"warm F5 worker failed: {ready}")
                services[arm] = process
        for index, arm in enumerate(order):
            if services:
                process = services[arm]
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write("sample\n")
                process.stdin.flush()
                record = json.loads(process.stdout.readline())
            else:
                record = _invoke(args, arm, args.mode)
            if any(record["audit"][key] != audits[arm][key] for key in record["audit"]):
                raise RuntimeError("wheel audit changed during run")
            records[arm].append(record)
            _json_new(destination / f"sample-{index:02d}-{arm}.json", record)
    finally:
        for process in services.values():
            if process.stdin is not None:
                process.stdin.write("quit\n")
                process.stdin.flush()
            process.communicate(timeout=20)
    for arm in ("A", "B"):
        _json_new(destination / f"raw-{arm}.json", records[arm])
    if args.mode == "correctness":
        before = records["A"][0]["sample"]["cases"]
        after = records["B"][0]["sample"]["cases"]
        _validate_contract(before, fixture)
        _validate_contract(after, fixture)
        if before != after:
            raise RuntimeError("B0/B1 WIN fingerprints differ")
        _json_new(
            destination / "fingerprint.json",
            {
                "A": before,
                "B": after,
                "equal": before == after,
            },
        )
    if args.mode in ("timing", "memory"):
        _json_new(
            destination / "summary.json",
            {
                arm: _summary(records[arm], args.mode, args.temperature)
                for arm in ("A", "B")
            },
        )
    print(destination)


def main() -> None:
    """Generate fixtures or capture isolated B0/B1 evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("fixtures")
    gen.add_argument("destination")
    capture = sub.add_parser("capture")
    capture.add_argument("--fixtures", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument(
        "--mode",
        choices=("correctness", "structure", "timing", "memory"),
        required=True,
    )
    capture.add_argument("--temperature", choices=("cold", "warm"), default="cold")
    capture.add_argument("--samples", type=int, default=5)
    for suffix in ("a", "b"):
        for name in ("python", "wheel", "version", "source-sha", "label"):
            capture.add_argument(f"--{name}-{suffix}", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--fixtures", required=True)
    worker.add_argument("--wheel", required=True)
    worker.add_argument("--version", required=True)
    worker.add_argument("--mode", required=True)
    args = parser.parse_args()
    if args.action == "fixtures":
        root = Path(args.destination)
        manifest = fixtures.materialize(root)
        _json_new(root / "win-fixtures.json", manifest)
    elif args.action == "_worker":
        _worker(args)
    else:
        _capture(args)


if __name__ == "__main__":
    main()
