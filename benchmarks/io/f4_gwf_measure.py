"""Frozen-wheel F4 timing and Linux process-tree PSS capture.

This controller is independent of the GWexpy wheel being measured. Run each
mode/scenario into a new directory; never overwrite a baseline-v1 artifact.
The 256-frame stress files and small two-frame files come from the committed
``f4_gwf_fixtures.py`` helper and are verified before sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import select
import signal
import statistics
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

ORDER = "ABBA BAAB ABBA BAAB AB".replace(" ", "")
SCENARIOS = {
    "stress_serial": ("stress", False),
    "stress_parallel": ("stress", 2),
    "stress64_serial": ("pss", False),
    "stress64_parallel": ("pss", 2),
    "small_serial": ("small", False),
    "small_parallel": ("small", 2),
}
FROZEN_HELPER_SHA256 = {
    "f4_gwf_fixtures.py": "e01018185bae922cb582088bc30f62457d551b2951319dc4a4f37dbfc48c3dc7",
    "f4_gwf_stress_capture.py": "d8ef2672adfe2919025d73b64907b50ffc5df28bf352ffaf1df2c0c7dbb95daf",
}
PSS_HELPER_SHA256 = "e18d7a4e0c26d5a3491dcd135c2b382e90832c218c38c5b0465b566a7461ebbd"
FROZEN_RUNTIME_SHA256 = {
    "gwexpy/timeseries/_gwf_io.py": "f662a1f12e45fb06c0aed346317ad89e8709df59cde3b13ad6d03edb9c672207",
    "gwexpy/timeseries/io/gwf/__init__.py": "c5a3086525bc79182638956cebbcd8018f1d8c0613ad87361478b15b617a57b6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_spec(
    small: Path, stress: Path, pss: Path, scenario: str, *, verify_files: bool
) -> dict[str, Any]:
    kind, parallel = SCENARIOS[scenario]
    root = {"stress": stress, "pss": pss, "small": small}[kind]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_schema = (
        "gwexpy-v025-b-f4-stress-fixtures-v1"
        if kind == "stress"
        else (
            "gwexpy-v025-b-f4-pss-fixtures-v2"
            if kind == "pss"
            else "gwexpy-v025-b-f4-fixtures-v1"
        )
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"Unexpected F4 {kind} fixture schema")
    names = (
        manifest["source_order"]
        if kind in {"stress", "pss"}
        else manifest["temporal_cases"]["adjacent"]
    )
    expected = {item["name"]: item["sha256"] for item in manifest["files"]}
    if kind in {"stress", "pss"} and (
        len(names) != 256
        or manifest["parts"] != 256
        or manifest["samples_per_part"] != (8192 if kind == "stress" else 32768)
        or manifest["decoded_sample_bytes"]
        != (16 * 1024 * 1024 if kind == "stress" else 64 * 1024 * 1024)
    ):
        raise ValueError(f"F4 {kind} workload changed")
    if kind == "small" and len(names) != 2:
        raise ValueError("F4 small workload changed")
    if verify_files:
        for name in expected:
            if _sha256(root / name) != expected[name]:
                raise ValueError(f"F4 fixture changed: {name}")
    return {
        "root": str(root.resolve()),
        "names": names,
        "file_sha256": {name: expected[name] for name in names},
        "fixture_manifest_sha256": _sha256(root / "manifest.json"),
        "decoded_sample_bytes": manifest.get("decoded_sample_bytes"),
        "channel": manifest["channel"],
        "parallel": parallel,
    }


def _wheel_audit(wheel: Path, version: str) -> dict[str, Any]:
    import gwexpy

    imported = Path(gwexpy.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if not imported.is_relative_to(prefix):
        raise RuntimeError(f"GWexpy imported outside benchmark prefix: {imported}")
    actual_version = importlib.metadata.version("gwexpy")
    if actual_version != version:
        raise RuntimeError(f"Expected gwexpy {version}, got {actual_version}")
    site_packages = imported.parent.parent
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("gwexpy/") and not name.endswith("/")
        ]
        for name in members:
            installed = site_packages / name
            if not installed.is_file() or installed.read_bytes() != archive.read(name):
                raise RuntimeError(f"Installed wheel file differs: {name}")
    distributions = {
        (dist.metadata.get("Name") or "").lower(): dist.version
        for dist in importlib.metadata.distributions()
    }
    distributions.pop("gwexpy", None)
    return {
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(prefix),
        "python": platform.python_version(),
        "gwexpy_path": str(imported),
        "gwexpy_version": actual_version,
        "gwpy_version": importlib.metadata.version("gwpy"),
        "wheel_sha256": _sha256(wheel),
        "install_mode": "wheel-no-deps",
        "wheel_files_verified": len(members),
        "distributions": distributions,
    }


def _runtime_audit(version: str) -> dict[str, Any]:
    import gwexpy

    imported = Path(gwexpy.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    actual_version = importlib.metadata.version("gwexpy")
    if not imported.is_relative_to(prefix) or actual_version != version:
        raise RuntimeError("Sample imported unexpected GWexpy installation")
    return {
        "gwexpy_path": str(imported),
        "gwexpy_version": actual_version,
        "gwpy_version": importlib.metadata.version("gwpy"),
        "python": platform.python_version(),
    }


def _read_once(spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from gwexpy.timeseries import TimeSeriesDict

    sources = [Path(spec["root"]) / name for name in spec["names"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start_wall = time.perf_counter_ns()
        start_cpu = time.process_time_ns()
        result = TimeSeriesDict.read(
            sources,
            channels=[spec["channel"]],
            format="gwf",
            parallel=spec["parallel"],
        )
        cpu_ns = time.process_time_ns() - start_cpu
        wall_ns = time.perf_counter_ns() - start_wall
    channels = {}
    for key, series in result.items():
        values = np.ascontiguousarray(series.value)
        channels[str(key)] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "unit": str(series.unit),
            "t0_gps": float(series.t0.value),
            "dt_s": float(series.dt.value),
            "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return {
        "outcome": "return",
        "wall_ns": wall_ns,
        "cpu_ns_parent": cpu_ns,
        "fingerprint": {"channel_order": list(result), "channels": channels},
        "warnings": [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ],
    }


def _worker(args: argparse.Namespace) -> None:
    if args.mode == "audit":
        print(json.dumps({"audit": _wheel_audit(args.wheel, args.version)}), flush=True)
        return
    audit = _runtime_audit(args.version)
    spec = _fixture_spec(
        args.fixtures_small,
        args.fixtures_stress,
        args.fixtures_pss,
        args.scenario,
        verify_files=False,
    )
    if args.mode == "warm_service":
        _read_once(spec)
        print(json.dumps({"audit": audit, "ready": True}), flush=True)
        for command in sys.stdin:
            if command.strip() == "quit":
                return
            if command.strip() != "sample":
                raise ValueError(f"Unknown warm-service command: {command!r}")
            print(json.dumps({"audit": audit, "sample": _read_once(spec)}), flush=True)
        return
    if args.mode not in {"timing", "memory"}:
        raise ValueError(f"Unknown worker mode: {args.mode}")
    print(json.dumps({"audit": audit, "sample": _read_once(spec)}), flush=True)


def _process_tree(pid: int) -> set[int]:
    found = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            data = Path(f"/proc/{current}/task/{current}/children").read_text(
                encoding="ascii"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for child in map(int, data.split()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _proc_kib(pid: int, path: str, label: str) -> int | None:
    try:
        with open(f"/proc/{pid}/{path}", encoding="ascii") as stream:
            for line in stream:
                if line.startswith(label):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _proc_cmdline(pid: int) -> str | None:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _worker_command(
    python: Path, wheel: Path, version: str, args: argparse.Namespace, mode: str
) -> list[str]:
    return [
        str(python.absolute()),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        str(wheel.resolve()),
        "--version",
        version,
        "--fixtures-small",
        str(args.fixtures_small.resolve()),
        "--fixtures-stress",
        str(args.fixtures_stress.resolve()),
        "--fixtures-pss",
        str(args.fixtures_pss.resolve()),
        "--scenario",
        args.scenario,
        "--mode",
        mode,
    ]


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _readline_with_timeout(process: subprocess.Popen[str], seconds: int) -> str:
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], seconds)
    if not readable:
        raise TimeoutError(f"Worker {process.pid} did not respond in {seconds}s")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"Worker {process.pid} exited without a response")
    return line


def _invoke(
    python: Path,
    wheel: Path,
    version: str,
    args: argparse.Namespace,
    *,
    mode: str,
) -> dict[str, Any]:
    command = _worker_command(python, wheel, version, args, mode)
    started_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    trace: list[dict[str, Any]] = []
    cmdlines: dict[str, str | None] = {}
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            pids = sorted(_process_tree(process.pid))
            pss = {
                str(pid): value
                for pid in pids
                if (value := _proc_kib(pid, "smaps_rollup", "Pss:")) is not None
            }
            rss = {
                str(pid): value
                for pid in pids
                if (value := _proc_kib(pid, "status", "VmRSS:")) is not None
            }
            for pid in pids:
                if not cmdlines.get(str(pid)):
                    cmdlines[str(pid)] = _proc_cmdline(pid)
            trace.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "pss_kib_by_pid": pss,
                    "rss_kib_by_pid": rss,
                    "tree_pss_kib": sum(pss.values()),
                    "tree_rss_kib": sum(rss.values()),
                }
            )
            stop.wait(args.sample_ms / 1000)

    thread = None
    if mode == "memory":
        if sys.platform != "linux":
            process.kill()
            raise RuntimeError("Canonical F4 PSS evidence requires Linux")
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    try:
        stdout, stderr = process.communicate(timeout=180)
        elapsed_ns = time.perf_counter_ns() - started_ns
    except BaseException:
        _stop_process_group(process)
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join()
    if process.returncode:
        raise RuntimeError(f"Worker failed ({process.returncode}): {stderr[-4000:]}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid worker JSON: {stdout[-1000:]} {stderr[-1000:]}"
        ) from exc
    result["controller_elapsed_ns"] = elapsed_ns
    result["stderr"] = stderr
    if mode == "memory":
        spawn_pids = sorted(
            int(pid) for pid, line in cmdlines.items() if line and "spawn_main" in line
        )
        simultaneous_workers = max(
            (
                sum(pid in row["pss_kib_by_pid"] for pid in map(str, spawn_pids))
                for row in trace
            ),
            default=0,
        )
        result["memory"] = {
            "sampling_ms": args.sample_ms,
            "trace": trace,
            "cmdline_by_pid": cmdlines,
            "observed_spawn_worker_pids": spawn_pids,
            "observed_spawn_worker_count": len(spawn_pids),
            "max_simultaneous_spawn_workers": simultaneous_workers,
            "peak_tree_pss_kib": max(
                (row["tree_pss_kib"] for row in trace), default=None
            ),
            "peak_tree_rss_kib": max(
                (row["tree_rss_kib"] for row in trace), default=None
            ),
            "sampled_peak_limit": True,
        }
    return result


def _summary(
    records: list[dict[str, Any]], mode: str, temperature: str
) -> dict[str, Any]:
    if mode == "memory":
        metrics = ("peak_tree_pss_kib", "peak_tree_rss_kib")
    else:
        metrics = ("wall_ns", "cpu_ns_parent")

    def values(name: str) -> list[int]:
        if mode == "memory":
            return [record["memory"][name] for record in records]
        return [
            record["controller_elapsed_ns"]
            if name == "wall_ns" and temperature == "cold"
            else record["sample"][name]
            for record in records
        ]

    out = {}
    for name in metrics:
        numbers = values(name)
        median = statistics.median(numbers)
        out[name] = {
            "samples": numbers,
            "median": median,
            "mad": statistics.median(abs(value - median) for value in numbers),
        }
    return out


def _capture(args: argparse.Namespace) -> None:
    if args.samples not in (5, 9):
        raise ValueError("Use five baseline or nine release samples per arm")
    if args.mode == "memory" and args.temperature != "cold":
        raise ValueError("Memory capture uses fresh processes")
    if args.mode == "memory" and args.sample_ms not in (1, 10):
        raise ValueError("F4 Linux PSS uses 10 ms, or 1 ms only for worker retry")
    if args.mode == "memory" and args.scenario.startswith("stress_"):
        raise ValueError("F4 primary PSS uses the 64 MiB stress64 workload")
    if args.mode == "timing" and args.scenario.startswith("stress64_"):
        raise ValueError("F4 timing uses the 16 MiB stress workload")
    spec = _fixture_spec(
        args.fixtures_small,
        args.fixtures_stress,
        args.fixtures_pss,
        args.scenario,
        verify_files=True,
    )
    helper_hashes = {
        name: _sha256(Path(__file__).resolve().with_name(name))
        for name in FROZEN_HELPER_SHA256
    }
    if helper_hashes != FROZEN_HELPER_SHA256:
        raise RuntimeError("Frozen F4 baseline-v1 helper bytes changed")
    pss_helper_sha256 = _sha256(
        Path(__file__).resolve().with_name("f4_gwf_pss_fixtures.py")
    )
    if pss_helper_sha256 != PSS_HELPER_SHA256:
        raise RuntimeError("F4 baseline-v2 PSS helper bytes changed")
    repository = Path(__file__).resolve().parents[2]
    runtime_hashes = {
        name: _sha256(repository / name) for name in FROZEN_RUNTIME_SHA256
    }
    if runtime_hashes != FROZEN_RUNTIME_SHA256:
        raise RuntimeError("F4 old-R runtime bytes changed")
    arms = {
        "A": (args.python_a, args.wheel_a, args.version_a, args.source_sha_a, "B0"),
        "B": (args.python_b, args.wheel_b, args.version_b, args.source_sha_b, "B1"),
    }
    audits = {
        arm: _invoke(py, wheel, version, args, mode="audit")["audit"]
        for arm, (py, wheel, version, _, _) in arms.items()
    }
    if audits["A"]["python"] != audits["B"]["python"]:
        raise RuntimeError("B0/B1 Python versions differ")
    if audits["A"]["distributions"] != audits["B"]["distributions"]:
        raise RuntimeError("B0/B1 dependency versions differ")
    runtime_audits = {
        arm: {
            field: audit[field]
            for field in ("gwexpy_path", "gwexpy_version", "gwpy_version", "python")
        }
        for arm, audit in audits.items()
    }
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    order = ORDER[: 2 * args.samples]
    manifest = {
        "schema": "gwexpy-v025-b-f4-measurement-v1",
        "status": "UNBASELINED",
        "scenario": args.scenario,
        "mode": args.mode,
        "temperature": args.temperature,
        "samples_per_arm": args.samples,
        "order": list(order),
        "sample_ms": args.sample_ms if args.mode == "memory" else None,
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "baseline_v1_harness_sha256": helper_hashes,
        "pss_fixture_generator_sha256": pss_helper_sha256,
        "baseline_v1_unchanged": True,
        "frozen_old_r_runtime_sha256": runtime_hashes,
        "fixture_spec": spec,
        "arms": {
            arm: {
                **audits[arm],
                "source_sha": values[3],
                "label": values[4],
            }
            for arm, values in arms.items()
        },
        "host": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "cpu_count": os.cpu_count(),
        },
        "pss_definition": "max_t sum(Pss of parent and all live descendants at the same sample time); sampled lower bound",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    services: dict[str, subprocess.Popen[str]] = {}
    try:
        if args.mode == "timing" and args.temperature == "warm":
            for arm, (py, wheel, version, _, _) in arms.items():
                process = subprocess.Popen(
                    _worker_command(py, wheel, version, args, "warm_service"),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                services[arm] = process
                ready = json.loads(_readline_with_timeout(process, 180))
                if not ready.get("ready") or ready["audit"] != runtime_audits[arm]:
                    raise RuntimeError(f"Warm service failed for {arm}")
        for index, arm in enumerate(order):
            py, wheel, version, _, _ = arms[arm]
            if arm in services:
                process = services[arm]
                assert process.stdin is not None
                process.stdin.write("sample\n")
                process.stdin.flush()
                record = json.loads(_readline_with_timeout(process, 180))
            else:
                record = _invoke(py, wheel, version, args, mode=args.mode)
            if record["audit"] != runtime_audits[arm]:
                raise RuntimeError(f"Installed wheel changed during {arm} samples")
            records[arm].append(record)
            (destination / f"sample-{index:02d}-{arm}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        for process in services.values():
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.communicate(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                _stop_process_group(process)
    post_spec = _fixture_spec(
        args.fixtures_small,
        args.fixtures_stress,
        args.fixtures_pss,
        args.scenario,
        verify_files=True,
    )
    if post_spec != spec:
        raise RuntimeError("F4 fixture changed during measurement")
    for arm in ("A", "B"):
        (destination / f"raw-{arm}.json").write_text(
            json.dumps(records[arm], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (destination / "summary.json").write_text(
        json.dumps(
            {
                arm: _summary(records[arm], args.mode, args.temperature)
                for arm in ("A", "B")
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for field in ("fingerprint", "warnings"):
        for arm in ("A", "B"):
            observed = {
                json.dumps(record["sample"][field], sort_keys=True)
                for record in records[arm]
            }
            if len(observed) != 1:
                raise RuntimeError(f"{arm} public {field} changed across samples")
        if records["A"][0]["sample"][field] != records["B"][0]["sample"][field]:
            raise RuntimeError(f"B0/B1 public {field} differ")
    if args.mode == "memory" and spec["parallel"]:
        missed = [
            (arm, index)
            for arm, arm_records in records.items()
            for index, record in enumerate(arm_records)
            if record["memory"]["max_simultaneous_spawn_workers"] < 2
        ]
        if missed:
            raise RuntimeError(
                f"Parallel workers missed in {missed}; retry memory only at 1 ms"
            )
    manifest["status"] = "COMPLETE"
    manifest["public_fingerprint_and_warning_parity"] = True
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    (destination / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "_worker"):
        sub = subcommands.add_parser(command)
        sub.add_argument("--fixtures-small", type=Path, required=True)
        sub.add_argument("--fixtures-stress", type=Path, required=True)
        sub.add_argument("--fixtures-pss", type=Path, required=True)
        sub.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
        sub.add_argument(
            "--mode",
            choices=("audit", "timing", "memory", "warm_service"),
            required=True,
        )
        if command == "_worker":
            sub.add_argument("--wheel", type=Path, required=True)
            sub.add_argument("--version", required=True)
        else:
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--temperature", choices=("cold", "warm"), default="cold")
            sub.add_argument("--samples", type=int, default=5)
            sub.add_argument("--sample-ms", type=int, default=10)
            for arm in ("a", "b"):
                sub.add_argument(f"--python-{arm}", type=Path, required=True)
                sub.add_argument(f"--wheel-{arm}", type=Path, required=True)
                sub.add_argument(f"--version-{arm}", required=True)
                sub.add_argument(f"--source-sha-{arm}", required=True)
    return parser


if __name__ == "__main__":
    options = _parser().parse_args()
    if options.command == "capture":
        _capture(options)
    else:
        _worker(options)
