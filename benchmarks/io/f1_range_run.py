"""Isolated B0/B1 public range-read baseline for HDF5, NDScope, NC, Zarr."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

try:
    from . import run
    from .f1_range_fixtures import write_fixture_set
except ImportError:  # Direct execution under python -I.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run
    from f1_range_fixtures import write_fixture_set


FORMATS = ("hdf5", "hdf.ndscope", "nc", "zarr")
ORDER = ("A", "B", "B", "A", "B", "A", "A", "B", "A", "B")


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "fixture-manifest.json").read_text())
    digest = hashlib.sha256()
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or resized F1 fixture {entry['path']}")
        actual = _sha(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Modified F1 fixture {entry['path']}")
        digest.update(entry["path"].encode() + b"\0" + bytes.fromhex(actual))
    if digest.hexdigest() != manifest["fixture_sha256"]:
        raise ValueError("F1 fixture-set digest mismatch")
    return manifest


def _case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in manifest["cases"] if case["id"] == case_id)


def _read(root: Path, case: dict[str, Any]) -> Any:
    from gwexpy.timeseries import TimeSeriesDict

    return TimeSeriesDict.read(root / case["source"], **case["kwargs"])


def _correctness(root: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for case in cases:
        with (
            warnings.catch_warnings(record=True) as caught,
            run._capture_route_logs(True) as logs,
        ):
            warnings.simplefilter("always")
            try:
                result = _read(root, case)
                payload: dict[str, Any] = {
                    "outcome": "return",
                    "fingerprint": run._fingerprint(result),
                }
            except Exception as exc:
                payload = {
                    "outcome": "error",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "error_message": str(exc),
                    "error_cause_type": (
                        f"{type(exc.__cause__).__module__}.{type(exc.__cause__).__qualname__}"
                        if exc.__cause__ is not None
                        else None
                    ),
                    "error_cause_message": str(exc.__cause__)
                    if exc.__cause__
                    else None,
                    "traceback": traceback.format_exc(),
                }
        payload["warnings"] = [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ]
        payload["logs"] = logs
        records[case["id"]] = payload
    return records


def _install_structure_probe(format_name: str, n_samples: int) -> dict[str, Any]:
    """Count full materializations at each reader's concrete array boundary."""
    import numpy as np

    counters: dict[str, Any] = {
        "probe_covered": True,
        "full_dataset_read_calls": 0,
        "payload_read_calls": 0,
        "payload_materialized_elements": 0,
        "probes": [],
    }

    def record(route: str, result: Any, key: Any) -> None:
        elements = int(np.asarray(result).size)
        counters["payload_read_calls"] += 1
        counters["payload_materialized_elements"] += elements
        counters["full_dataset_read_calls"] += int(elements == n_samples)
        counters["probes"].append(
            {"route": route, "index": repr(key), "elements": elements}
        )

    if format_name in ("hdf5", "hdf.ndscope"):
        import h5py

        original_get = h5py.Dataset.__getitem__
        original_array = h5py.Dataset.__array__

        def measured_get(self: Any, key: Any) -> Any:
            value = original_get(self, key)
            if self.name.endswith(("/signal", "/signal/raw")):
                record("h5py.Dataset.__getitem__", value, key)
            return value

        def measured_array(self: Any, *args: Any, **kwargs: Any) -> Any:
            value = original_array(self, *args, **kwargs)
            if self.name.endswith(("/signal", "/signal/raw")):
                record("h5py.Dataset.__array__", value, ())
            return value

        h5py.Dataset.__getitem__ = measured_get
        h5py.Dataset.__array__ = measured_array
    elif format_name == "nc":
        import xarray as xr

        original = xr.DataArray.values
        assert isinstance(original, property) and original.fget is not None

        def measured_values(self: Any) -> Any:
            value = original.fget(self)
            if self.name == "signal":
                record("xarray.DataArray.values", value, "all")
            return value

        xr.DataArray.values = property(measured_values, original.fset, original.fdel)
    else:
        import zarr

        original = zarr.Array.__getitem__

        def measured_get(self: Any, key: Any) -> Any:
            value = original(self, key)
            if self.name.endswith("/signal"):
                record("zarr.Array.__getitem__", value, key)
            return value

        zarr.Array.__getitem__ = measured_get
    return counters


def _worker(args: argparse.Namespace) -> None:
    audit = run._worker_audit(Path(args.wheel), args.version, full=args.mode == "audit")
    if args.mode == "audit":
        print(json.dumps({"audit": audit}, sort_keys=True))
        return
    root = Path(args.fixtures)
    manifest = json.loads((root / "fixture-manifest.json").read_text())
    if args.mode == "correctness":
        payload = _correctness(root, manifest["cases"])
        print(json.dumps({"audit": audit, "sample": payload}, sort_keys=True))
        return
    case = _case(manifest, f"{args.format}:short_intersecting")
    counters = (
        _install_structure_probe(args.format, manifest["n_samples"])
        if args.mode == "structure"
        else {}
    )
    if args.mode == "service":
        _read(root, case)
        print(json.dumps({"ready": True, "audit": audit}), flush=True)
        for command in sys.stdin:
            if command.strip() == "quit":
                break
            if command.strip() != "sample":
                raise ValueError(f"Unknown F1 service command {command!r}")
            start_wall = time.perf_counter_ns()
            start_cpu = time.process_time_ns()
            try:
                _read(root, case)
                payload = {
                    "outcome": "return",
                    "wall_ns": time.perf_counter_ns() - start_wall,
                    "cpu_ns": time.process_time_ns() - start_cpu,
                }
            except Exception as exc:
                payload = {
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            print(json.dumps({"audit": audit, "sample": payload}), flush=True)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            start_wall = time.perf_counter_ns()
            start_cpu = time.process_time_ns()
            result = _read(root, case)
            payload = {
                "outcome": "return",
                "wall_ns": time.perf_counter_ns() - start_wall,
                "cpu_ns": time.process_time_ns() - start_cpu,
                "counters": counters,
                "fingerprint": run._fingerprint(result)
                if args.mode == "structure"
                else None,
            }
        except Exception as exc:
            payload = {
                "outcome": "error",
                "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "counters": counters,
            }
    payload["warnings"] = [
        {
            "category": f"{w.category.__module__}.{w.category.__qualname__}",
            "message": str(w.message),
        }
        for w in caught
    ]
    print(json.dumps({"audit": audit, "sample": payload}, sort_keys=True))


def _command(
    args: argparse.Namespace, arm: str, mode: str, format_name: str
) -> list[str]:
    suffix = "a" if arm == "A" else "b"
    return [
        getattr(args, f"python_{suffix}"),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        getattr(args, f"wheel_{suffix}"),
        "--version",
        getattr(args, f"version_{suffix}"),
        "--fixtures",
        str(Path(args.fixtures).resolve()),
        "--mode",
        mode,
        "--format",
        format_name,
    ]


def _invoke(
    args: argparse.Namespace, arm: str, mode: str, format_name: str
) -> dict[str, Any]:
    command = _command(args, arm, mode, format_name)
    start = time.perf_counter_ns()
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    trace: list[dict[str, Any]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            pids = sorted(run._descendants(process.pid))
            pss = {
                str(pid): value
                for pid in pids
                if (value := run._pss_kib(pid)) is not None
            }
            rss = {
                str(pid): value
                for pid in pids
                if (value := run._rss_kib(pid)) is not None
            }
            trace.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "pss_kib_by_pid": pss,
                    "rss_kib_by_pid": rss,
                    "tree_pss_kib": sum(pss.values()),
                    "tree_rss_kib": sum(rss.values()),
                }
            )
            stop.wait(0.01)

    if mode == "memory" and sys.platform != "linux":
        process.kill()
        raise RuntimeError("Linux /proc is required for tree PSS")
    thread = threading.Thread(target=monitor, daemon=True) if mode == "memory" else None
    if thread is not None:
        thread.start()
    stdout, stderr = process.communicate()
    stop.set()
    if thread is not None:
        thread.join()
    if process.returncode:
        raise RuntimeError(f"F1 worker failed ({process.returncode}): {stderr[-4000:]}")
    result = json.loads(stdout)
    result["stderr"] = stderr
    result["controller_elapsed_ns"] = time.perf_counter_ns() - start
    if thread is not None:
        result["memory"] = {
            "sampling_ms": 10,
            "trace": trace,
            "peak_tree_pss_kib": max(
                (point["tree_pss_kib"] for point in trace), default=None
            ),
            "peak_tree_rss_kib": max(
                (point["tree_rss_kib"] for point in trace), default=None
            ),
            "sampled_peak_limit": True,
        }
    return result


def _warm_service(
    args: argparse.Namespace, arm: str, format_name: str
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        _command(args, arm, "service", format_name),
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
        raise RuntimeError(f"F1 warm worker failed: {process.stderr.read()[-4000:]}")
    return process


def _warm_sample(process: subprocess.Popen[str]) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    start = time.perf_counter_ns()
    process.stdin.write("sample\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("F1 warm worker exited before sample")
    result = json.loads(line)
    result["controller_elapsed_ns"] = time.perf_counter_ns() - start
    result["stderr"] = ""
    return result


def _summary(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    metrics = (
        ("peak_tree_pss_kib", "peak_tree_rss_kib")
        if mode == "memory"
        else ("wall_ns", "cpu_ns")
        if mode == "warm"
        else ("controller_elapsed_ns", "cpu_ns")
    )
    output = {}
    for metric in metrics:
        values = [
            record["memory"][metric]
            if mode == "memory"
            else record["sample"][metric]
            if metric != "controller_elapsed_ns"
            else record[metric]
            for record in records
        ]
        median = statistics.median(values)
        output[metric] = {
            "raw": values,
            "median": median,
            "mad": statistics.median(abs(value - median) for value in values),
        }
    return output


def _capture(args: argparse.Namespace) -> None:
    fixture_root = Path(args.fixtures).resolve()
    fixture = _fixture(fixture_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        arm: _invoke(args, arm, "audit", args.format)["audit"] for arm in ("A", "B")
    }
    if (
        audit["A"]["python"] != audit["B"]["python"]
        or audit["A"]["distributions"] != audit["B"]["distributions"]
    ):
        raise RuntimeError("F1 Python or dependency distributions differ")
    if args.mode == "correctness":
        order = ("A", "B")
    else:
        if args.samples != 5:
            raise ValueError("F1 baseline requires five samples per arm")
        order = ORDER
    _write_new(
        output / "manifest.json",
        {
            "schema": 1,
            "status": "UNBASELINED",
            "lane": "B-F1",
            "mode": args.mode,
            "format": args.format if args.mode != "correctness" else "all",
            "order": order,
            "samples_per_arm": 1 if args.mode == "correctness" else args.samples,
            "fixture_manifest_sha256": _sha(fixture_root / "fixture-manifest.json"),
            "fixture_sha256": fixture["fixture_sha256"],
            "harness_digest": run.harness_digest(),
            "host": {"platform": platform.platform(), "uname": list(platform.uname())},
            "arms": {
                arm: {
                    **audit[arm],
                    "source_sha": getattr(
                        args, f"source_sha_{'a' if arm == 'A' else 'b'}"
                    ),
                    "install_mode": "wheel-no-deps",
                }
                for arm in ("A", "B")
            },
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    warm = (
        {arm: _warm_service(args, arm, args.format) for arm in ("A", "B")}
        if args.mode == "warm"
        else {}
    )
    try:
        for index, arm in enumerate(order):
            sample = (
                _warm_sample(warm[arm])
                if warm
                else _invoke(args, arm, args.mode, args.format)
            )
            if any(sample["audit"][key] != audit[arm][key] for key in sample["audit"]):
                raise RuntimeError("F1 installed wheel audit changed during capture")
            if (
                args.mode != "correctness"
                and args.mode != "audit"
                and sample["sample"].get("outcome") != "return"
            ):
                raise RuntimeError(f"F1 benchmark route failed: {sample['sample']}")
            records[arm].append(sample)
            _write_new(output / f"sample-{index:02d}-{arm}.json", sample)
    finally:
        for process in warm.values():
            assert process.stdin is not None
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.communicate(timeout=20)
    for arm in ("A", "B"):
        _write_new(output / f"raw-{arm}.json", records[arm])
    if args.mode == "correctness":
        samples = {arm: records[arm][0]["sample"] for arm in ("A", "B")}
        comparable = {
            arm: {
                case: {key: value for key, value in item.items() if key != "traceback"}
                for case, item in cases.items()
            }
            for arm, cases in samples.items()
        }
        _write_new(
            output / "fingerprints.json",
            {
                "A": comparable["A"],
                "B": comparable["B"],
                "equal_cases": {
                    key: comparable["A"][key] == comparable["B"][key]
                    for key in comparable["A"]
                },
            },
        )
    elif args.mode in ("warm", "cold", "memory"):
        _write_new(
            output / "summary.json",
            {arm: _summary(records[arm], args.mode) for arm in ("A", "B")},
        )
    print(output)


def main() -> None:
    """Generate deterministic fixtures or collect a F1 B0/B1 baseline route."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fixtures = sub.add_parser("fixtures")
    fixtures.add_argument("destination")
    fixtures.add_argument("--samples", type=int, default=4096)
    fixtures.add_argument("--chunk", type=int, default=128)
    capture = sub.add_parser("capture")
    capture.add_argument("--fixtures", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument(
        "--mode",
        choices=("correctness", "structure", "warm", "cold", "memory"),
        required=True,
    )
    capture.add_argument("--format", choices=FORMATS, default="hdf5")
    capture.add_argument("--samples", type=int, default=5)
    for suffix in ("a", "b"):
        capture.add_argument(f"--python-{suffix}", required=True)
        capture.add_argument(f"--wheel-{suffix}", required=True)
        capture.add_argument(f"--version-{suffix}", required=True)
        capture.add_argument(f"--source-sha-{suffix}", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("--wheel", required=True)
    worker.add_argument("--version", required=True)
    worker.add_argument("--fixtures", required=True)
    worker.add_argument(
        "--mode",
        choices=(
            "audit",
            "correctness",
            "structure",
            "warm",
            "cold",
            "memory",
            "service",
        ),
        required=True,
    )
    worker.add_argument("--format", choices=FORMATS, required=True)
    args = parser.parse_args()
    if args.action == "fixtures":
        root = Path(args.destination)
        manifest = write_fixture_set(
            root, n_samples=args.samples, chunk_samples=args.chunk
        )
        _write_new(root / "fixture-manifest.json", manifest)
    elif args.action == "capture":
        _capture(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
