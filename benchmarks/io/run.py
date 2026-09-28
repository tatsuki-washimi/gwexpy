"""Isolated public-I/O evidence runner for the v0.2.5 performance campaign.

The controller imports no gwexpy code.  Each sample starts the nominated wheel
interpreter with ``-I`` and checks that its installed files match the wheel.
Use separate commands for correctness, structure, timing, and sampled memory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
import warnings
import zipfile
from pathlib import Path
from typing import Any

try:
    from .fixtures import make_fixtures, sha256, verify_fixtures
except ImportError:  # Direct execution with Python -I excludes the script directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixtures import make_fixtures, sha256, verify_fixtures


ROUTES = (
    "c1_many",
    "c1_merge_64",
    "c1_public_many",
    "c1_channel_order",
    "c1_integer_nan",
    "c1_unit_conversion",
    "c1_exact_gps_ns",
    "c1_first_provenance",
    "c1_reverse",
    "c1_gap_pad",
    "c1_gap_ignore",
    "c1_gap_raise",
    "c1_overlap",
    "c1_one",
    "c1_empty",
    "f2_1a",
    "f2_1b",
    "f2_2",
    "f2_3",
    "f2_writer",
)
FAULTS = (
    "selected_invalid",
    "unselected_invalid",
    "timestamp_invalid",
    "timestamp_irregular",
    "row_short",
    "row_extra",
    "early_unselected_late_selected",
    "early_selected_late_unselected",
)
ABBA = (
    "A",
    "B",
    "B",
    "A",
    "B",
    "A",
    "A",
    "B",
    "A",
    "B",
    "B",
    "A",
    "B",
    "A",
    "A",
    "B",
    "A",
    "B",
)


def harness_digest() -> str:
    """Digest the exact controller, scenario, and fixture generator bytes."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_fingerprint(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "selected": [
            repr(array.flat[index].item())
            for index in sorted({0, array.size // 2, array.size - 1})
        ]
        if array.size
        else [],
    }


def _series_fingerprint(series: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": f"{type(series).__module__}.{type(series).__qualname__}",
        "values": _array_fingerprint(series.value),
        "unit": str(series.unit),
        "name": str(series.name) if series.name is not None else None,
    }
    for name in ("times", "frequencies"):
        if hasattr(series, name):
            axis = getattr(series, name)
            result[name] = {
                "values": _array_fingerprint(axis.value),
                "unit": str(axis.unit),
            }
    for name in ("t0", "dt", "f0", "df", "epoch", "sample_rate"):
        if hasattr(series, name):
            attribute = getattr(series, name)
            result[name] = repr(attribute)
    for name in ("attrs", "_gwexpy_io"):
        attribute = getattr(series, name, None)
        if attribute:
            result[name] = repr(attribute)
    return result


def _fingerprint(result: Any) -> dict[str, Any]:
    if hasattr(result, "items"):
        return {
            "class": f"{type(result).__module__}.{type(result).__qualname__}",
            "order": [str(key) for key in result],
            "members": {
                str(key): _series_fingerprint(value) for key, value in result.items()
            },
            "attrs": repr(getattr(result, "attrs", None)),
        }
    if isinstance(result, Path):
        return {"path_sha256": sha256(result), "bytes": result.stat().st_size}
    return _series_fingerprint(result)


def _declared_counts(name: str, fixture: dict) -> dict[str, int]:
    """Return generator facts; these are not measured parser work counters."""
    if name in ("f2_1a", "f2_1b"):
        return {"input_rows": 4096, "input_tokens": 8192, "input_columns": 2}
    if name == "f2_2":
        return {"input_rows": 8192, "input_tokens": 8192 * 9, "input_columns": 9}
    if name == "f2_3":
        rows = fixture["large_rows"]
        return {"input_rows": rows, "input_tokens": rows * 17, "input_columns": 17}
    if name.startswith("f2_fault_"):
        return {"input_rows": 3, "input_columns": 3}
    return {}


def _scenario(
    name: str,
    paths: dict[str, Path],
    work_dir: Path,
    writer_series: Any = None,
    merge_parts: Any = None,
) -> Any:
    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.timeseries import TimeSeriesDict

    if name == "c1_merge_64":
        from gwexpy.timeseries.io._multi import read_multi_dict

        def read_part(index: int) -> Any:
            return merge_parts[index]

        return read_multi_dict(read_part, range(64), "synthetic")
    if name in (
        "c1_channel_order",
        "c1_integer_nan",
        "c1_unit_conversion",
        "c1_exact_gps_ns",
        "c1_first_provenance",
    ):
        return _synthetic_c1_case(name)

    segments = [paths[f"segment_{i:02d}"] for i in range(12)]
    if name.startswith("c1_") and name != "c1_public_many":
        from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

        reader = read_timeseriesdict_csv
    else:

        def reader(sources: Any, **kwargs: Any) -> Any:
            return TimeSeriesDict.read(sources, format="csv", **kwargs)

    if name == "c1_many" or name == "c1_public_many":
        return reader(segments)
    if name == "c1_reverse":
        return reader(segments[::-1])
    if name == "c1_gap_pad":
        return reader([segments[0], segments[2]])
    if name == "c1_gap_ignore":
        return reader([segments[0], segments[2]], gap="ignore")
    if name == "c1_gap_raise":
        return reader([segments[0], segments[2]], gap="raise")
    if name == "c1_overlap":
        return reader([segments[0], segments[0]])
    if name == "c1_one":
        return reader([segments[0]])
    if name == "c1_empty":
        return reader([])
    if name == "f2_1a":
        return FrequencySeries.read(paths["frequency"], format="csv")
    if name == "f2_1b":
        return FrequencySeries.read(paths["frequency"])
    if name == "f2_2":
        return TimeSeriesDict.read(paths["general"], format="csv")
    if name == "f2_3":
        return TimeSeriesDict.read(paths["large"], format="csv", channels=["ch1"])
    if name == "f2_writer":
        from gwexpy.timeseries.io.csv_enhanced import write_timeseries_csv

        target = work_dir / "written.csv"
        write_timeseries_csv(writer_series, target)
        return target
    if name.startswith("f2_fault_"):
        kind, selection = name.removeprefix("f2_fault_").rsplit("__", 1)
        return TimeSeriesDict.read(
            paths[f"fault_{kind}"],
            format="csv",
            channels=["ch1"] if selection == "selected" else None,
        )
    raise ValueError(f"unknown scenario {name}")


def _make_merge_parts() -> tuple[Any, ...]:
    """Allocate 64 regular segments before a merge-only stopwatch starts."""
    import numpy as np

    from gwexpy.timeseries import TimeSeries, TimeSeriesDict

    return tuple(
        TimeSeriesDict(
            {
                "merge": TimeSeries(
                    np.arange(4096, dtype=np.float64) + index,
                    t0=index * 1024,
                    dt=0.25,
                    unit="m",
                    name="merge",
                )
            }
        )
        for index in range(64)
    )


def _synthetic_c1_case(name: str) -> Any:
    """Exercise `_multi` cases that plain numeric CSV cannot represent."""
    import numpy as np

    from gwexpy.timeseries import TimeSeries, TimeSeriesDict
    from gwexpy.timeseries.io._multi import read_multi_dict

    if name == "c1_channel_order":
        first = TimeSeriesDict(
            {
                "z": TimeSeries([1, 2], t0=0, dt=1, unit="m"),
                "a": TimeSeries([3, 4], t0=0, dt=1, unit="m"),
            }
        )
        second = TimeSeriesDict(
            {
                "a": TimeSeries([5, 6], t0=2, dt=1, unit="m"),
                "n": TimeSeries([7, 8], t0=2, dt=1, unit="m"),
            }
        )
    elif name == "c1_integer_nan":
        first = TimeSeriesDict(
            {"int": TimeSeries(np.array([1, 2], dtype=np.int16), t0=0, dt=1)}
        )
        second = TimeSeriesDict(
            {"int": TimeSeries(np.array([3, 4], dtype=np.int16), t0=4, dt=1)}
        )
    elif name == "c1_unit_conversion":
        first = TimeSeriesDict({"unit": TimeSeries([1.0, 2.0], t0=0, dt=1, unit="m")})
        second = TimeSeriesDict(
            {"unit": TimeSeries([300.0, 400.0], t0=2, dt=1, unit="cm")}
        )
    elif name == "c1_exact_gps_ns":
        from ligotimegps import LIGOTimeGPS

        first = TimeSeriesDict(
            {"ns": TimeSeries([1.0, 2.0, 3.0, 4.0], t0=LIGOTimeGPS(1000, 1), dt=1e-9)}
        )
        second = TimeSeriesDict(
            {"ns": TimeSeries([5.0, 6.0, 7.0, 8.0], t0=LIGOTimeGPS(1000, 5), dt=1e-9)}
        )
    elif name == "c1_first_provenance":
        first = TimeSeriesDict({"p": TimeSeries([1.0, 2.0], t0=0, dt=1)})
        second = TimeSeriesDict({"p": TimeSeries([3.0, 4.0], t0=2, dt=1)})
        first.attrs = {"source": "first"}
        second.attrs = {"source": "second"}
    else:
        raise ValueError(name)
    sources = {0: first, 1: second}
    return read_multi_dict(lambda source: sources[source], [0, 1], "synthetic")


def _worker_audit(wheel: Path, expected_version: str, *, full: bool) -> dict:
    import gwexpy

    installed = Path(gwexpy.__file__).resolve().parent
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"gwexpy imported outside nominated interpreter: {installed}"
        )
    if gwexpy.__version__ != expected_version:
        raise RuntimeError(f"unexpected gwexpy version: {gwexpy.__version__}")
    result = {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).resolve()),
        "gwexpy_path": str(installed / "__init__.py"),
        "gwexpy_version": gwexpy.__version__,
    }
    if full:
        checked = 0
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if not member.startswith("gwexpy/") or member.endswith("/"):
                    continue
                installed_file = installed.parent / member
                if not installed_file.is_file():
                    raise RuntimeError(f"installed wheel file missing: {member}")
                if (
                    hashlib.sha256(archive.read(member)).digest()
                    != hashlib.sha256(installed_file.read_bytes()).digest()
                ):
                    raise RuntimeError(f"installed file differs from wheel: {member}")
                checked += 1
        if checked == 0:
            raise RuntimeError("wheel contains no gwexpy files")
        distributions = {
            distribution.metadata["Name"]
            .lower()
            .replace("_", "-"): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
        distributions.pop("gwexpy", None)
        result.update(
            {
                "wheel_sha256": sha256(wheel),
                "wheel_files_verified": checked,
                "distributions": distributions,
            }
        )
    return result


def _worker(args: argparse.Namespace) -> None:
    import tempfile

    audit = _worker_audit(Path(args.wheel), args.version, full=args.mode == "audit")
    if args.mode == "audit":
        print(json.dumps({"audit": audit}, sort_keys=True))
        return
    if args.mode == "service":
        _warm_service(args, audit)
        return
    counters: dict[str, int] = {}
    if args.mode == "structure" and args.scenario in ("c1_many", "c1_merge_64"):
        from gwexpy.timeseries import TimeSeries

        original = TimeSeries.append

        def measured_append(
            self: Any, other: Any, *positional: Any, **keyword: Any
        ) -> Any:
            if keyword.get("inplace") is False:
                counters["non_inplace_append_calls"] = (
                    counters.get("non_inplace_append_calls", 0) + 1
                )
            result = original(self, other, *positional, **keyword)
            counters["append_result_sample_bytes"] = (
                counters.get("append_result_sample_bytes", 0) + result.value.nbytes
            )
            return result

        TimeSeries.append = measured_append
    fixture = json.loads(
        (Path(args.fixtures) / "fixtures.json").read_text(encoding="utf-8")
    )
    paths = {
        key: Path(args.fixtures) / entry["name"]
        for key, entry in fixture["files"].items()
    }
    writer_series = None
    merge_parts = _make_merge_parts() if args.scenario == "c1_merge_64" else None
    if args.scenario == "f2_writer":
        import numpy as np

        from gwexpy.timeseries import TimeSeries

        writer_series = TimeSeries(
            np.arange(32768, dtype=np.float64) / 8,
            t0=1000,
            dt=0.25,
            name="writer",
            unit="m",
        )
    with tempfile.TemporaryDirectory(prefix="gwexpy-io-bench-") as directory:
        warning_context = (
            warnings.catch_warnings(record=True)
            if args.mode == "correctness"
            else contextlib.nullcontext([])
        )
        with warning_context as caught:
            if args.mode == "correctness":
                warnings.simplefilter("always")
            try:
                if args.mode == "timing" and args.temperature == "warm":
                    _scenario(
                        args.scenario,
                        paths,
                        Path(directory),
                        writer_series,
                        merge_parts,
                    )
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                result = _scenario(
                    args.scenario, paths, Path(directory), writer_series, merge_parts
                )
                cpu_ns = time.process_time_ns() - cpu_start
                wall_ns = time.perf_counter_ns() - wall_start
                payload: dict[str, Any] = {
                    "outcome": "return",
                    "fixture_counts": _declared_counts(args.scenario, fixture),
                    "counters": counters,
                    "wall_ns": wall_ns,
                    "cpu_ns": cpu_ns,
                }
                if args.mode == "correctness":
                    payload["fingerprint"] = _fingerprint(result)
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
                    "category": f"{item.category.__module__}.{item.category.__qualname__}",
                    "message": str(item.message),
                }
                for item in caught
            ]
            payload["warnings_recorded"] = args.mode == "correctness"
    print(json.dumps({"audit": audit, "sample": payload}, sort_keys=True))


def _warm_service(args: argparse.Namespace, audit: dict) -> None:
    """Keep one imported process per arm for interleaved warm repetitions."""
    import tempfile

    fixture = json.loads(
        (Path(args.fixtures) / "fixtures.json").read_text(encoding="utf-8")
    )
    paths = {
        key: Path(args.fixtures) / entry["name"]
        for key, entry in fixture["files"].items()
    }
    writer_series = None
    merge_parts = _make_merge_parts() if args.scenario == "c1_merge_64" else None
    if args.scenario == "f2_writer":
        import numpy as np

        from gwexpy.timeseries import TimeSeries

        writer_series = TimeSeries(
            np.arange(32768, dtype=np.float64) / 8,
            t0=1000,
            dt=0.25,
            name="writer",
            unit="m",
        )
    with tempfile.TemporaryDirectory(prefix="gwexpy-io-warm-") as directory:
        work_dir = Path(directory)
        _scenario(args.scenario, paths, work_dir, writer_series, merge_parts)
        print(json.dumps({"ready": True, "audit": audit}), flush=True)
        for command in sys.stdin:
            if command.strip() == "quit":
                break
            if command.strip() != "sample":
                raise ValueError(f"unknown warm worker command: {command!r}")
            try:
                started_wall = time.perf_counter_ns()
                started_cpu = time.process_time_ns()
                result = _scenario(
                    args.scenario, paths, work_dir, writer_series, merge_parts
                )
                cpu_ns = time.process_time_ns() - started_cpu
                wall_ns = time.perf_counter_ns() - started_wall
                del result
                payload = {
                    "outcome": "return",
                    "wall_ns": wall_ns,
                    "cpu_ns": cpu_ns,
                    "counters": {},
                    "warnings_recorded": False,
                }
            except Exception as exc:
                payload = {
                    "outcome": "error",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            print(json.dumps({"audit": audit, "sample": payload}), flush=True)


def _pss_kib(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _rss_kib(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _descendants(pid: int) -> set[int]:
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


def _invoke(
    python: Path,
    wheel: Path,
    version: str,
    fixtures: Path,
    scenario: str,
    mode: str,
    temperature: str,
    *,
    sample_ms: int = 10,
) -> dict:
    command = [
        str(python),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        str(wheel.resolve()),
        "--version",
        version,
        "--fixtures",
        str(fixtures.resolve()),
        "--scenario",
        scenario,
        "--mode",
        mode,
        "--temperature",
        temperature,
    ]
    started_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    trace: list[dict[str, Any]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            pids = sorted(_descendants(process.pid))
            by_pid = {
                str(pid): pss for pid in pids if (pss := _pss_kib(pid)) is not None
            }
            rss_by_pid = {
                str(pid): rss for pid in pids if (rss := _rss_kib(pid)) is not None
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
            stop.wait(sample_ms / 1000)

    thread = None
    if mode == "memory":
        if sys.platform != "linux":
            process.kill()
            raise RuntimeError("canonical PSS evidence requires Linux")
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    stdout, stderr = process.communicate()
    elapsed_ns = time.perf_counter_ns() - started_ns
    stop.set()
    if thread is not None:
        thread.join()
    if process.returncode:
        raise RuntimeError(f"worker failed ({process.returncode}): {stderr[-4000:]}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker emitted invalid JSON: {stdout[-1000:]} {stderr[-1000:]}"
        ) from exc
    result["controller_elapsed_ns"] = elapsed_ns
    result["stderr"] = stderr
    if mode == "memory":
        observed = sorted({pid for sample in trace for pid in sample["pss_kib_by_pid"]})
        result["memory"] = {
            "sampling_ms": sample_ms,
            "trace": trace,
            "observed_pids": observed,
            "observed_worker_count": len(observed) - int(str(process.pid) in observed),
            "peak_tree_pss_kib": max(
                (sample["tree_pss_kib"] for sample in trace), default=None
            ),
            "peak_tree_rss_kib": max(
                (sample["tree_rss_kib"] for sample in trace), default=None
            ),
            "sampled_peak_limit": True,
        }
    return result


def _start_warm_service(
    python: Path, wheel: Path, version: str, fixtures: Path, scenario: str
) -> subprocess.Popen[str]:
    command = [
        str(python),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--wheel",
        str(wheel.resolve()),
        "--version",
        version,
        "--fixtures",
        str(fixtures.resolve()),
        "--scenario",
        scenario,
        "--mode",
        "service",
        "--temperature",
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
    ready_line = process.stdout.readline()
    if not ready_line:
        assert process.stderr is not None
        raise RuntimeError(
            f"warm worker startup failed: {process.stderr.read()[-4000:]}"
        )
    ready = json.loads(ready_line)
    if not ready.get("ready"):
        raise RuntimeError(f"warm worker did not become ready: {ready}")
    return process


def _sample_warm_service(process: subprocess.Popen[str]) -> dict:
    assert process.stdin is not None and process.stdout is not None
    started_ns = time.perf_counter_ns()
    process.stdin.write("sample\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        assert process.stderr is not None
        raise RuntimeError(f"warm worker exited early: {process.stderr.read()[-4000:]}")
    result = json.loads(line)
    result["controller_elapsed_ns"] = time.perf_counter_ns() - started_ns
    result["stderr"] = ""
    return result


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _summary(samples: list[dict], mode: str, temperature: str) -> dict:
    metrics = (
        ("peak_tree_pss_kib", "peak_tree_rss_kib")
        if mode == "memory"
        else (
            ("controller_elapsed_ns", "cpu_ns")
            if temperature == "cold"
            else ("wall_ns", "cpu_ns")
        )
    )
    result = {}
    for metric in metrics:
        numbers = [
            sample["memory"][metric]
            if mode == "memory"
            else sample["controller_elapsed_ns"]
            if metric == "controller_elapsed_ns"
            else sample["sample"][metric]
            for sample in samples
        ]
        median = statistics.median(numbers)
        result[metric] = {
            "raw": numbers,
            "median": median,
            "mad": statistics.median(abs(number - median) for number in numbers),
        }
    return result


def _controller(args: argparse.Namespace) -> None:
    fixture_dir = Path(args.fixtures).resolve()
    fixture_manifest = verify_fixtures(fixture_dir)
    wheel_a, wheel_b = Path(args.wheel_a).resolve(), Path(args.wheel_b).resolve()
    # A venv's python binary can be a symlink; resolving it would escape the venv.
    python_a, python_b = Path(args.python_a).absolute(), Path(args.python_b).absolute()
    audit_a = _invoke(
        python_a, wheel_a, args.version_a, fixture_dir, args.scenario, "audit", "cold"
    )["audit"]
    audit_b = _invoke(
        python_b, wheel_b, args.version_b, fixture_dir, args.scenario, "audit", "cold"
    )["audit"]
    if (
        audit_a["python"] != audit_b["python"]
        or audit_a["distributions"] != audit_b["distributions"]
    ):
        raise RuntimeError(
            "Python or installed dependency versions differ between arms"
        )
    if args.mode == "timing" and args.temperature not in ("cold", "warm"):
        raise ValueError("timing requires --temperature cold or warm")
    if args.mode != "timing" and args.temperature != "cold":
        raise ValueError("structure, correctness, and memory use fresh processes")
    samples_per_arm = args.samples
    if args.mode == "correctness":
        samples_per_arm = 1
    elif samples_per_arm not in (5, 9):
        raise ValueError("use five baseline samples or nine release samples per arm")
    order = ABBA[: 2 * samples_per_arm]
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "status": "UNBASELINED",
        "lane": args.lane,
        "scenario": args.scenario,
        "mode": args.mode,
        "temperature": args.temperature,
        "samples_per_arm": samples_per_arm,
        "order": order,
        "harness_digest": harness_digest(),
        "fixture_manifest_sha256": sha256(fixture_dir / "fixtures.json"),
        "fixture_files": fixture_manifest["files"],
        "host": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "cpu_count": os.cpu_count(),
        },
        "arms": {
            "A": {
                **audit_a,
                "source_sha": args.source_sha_a,
                "label": args.label_a,
                "install_mode": args.install_mode,
            },
            "B": {
                **audit_b,
                "source_sha": args.source_sha_b,
                "label": args.label_b,
                "install_mode": args.install_mode,
            },
        },
    }
    _write_new(destination / "manifest.json", manifest)
    records: dict[str, list[dict]] = {"A": [], "B": []}
    warm_services: dict[str, subprocess.Popen[str]] = {}
    try:
        if args.mode == "timing" and args.temperature == "warm":
            warm_services = {
                "A": _start_warm_service(
                    python_a, wheel_a, args.version_a, fixture_dir, args.scenario
                ),
                "B": _start_warm_service(
                    python_b, wheel_b, args.version_b, fixture_dir, args.scenario
                ),
            }
        for index, arm in enumerate(order):
            if warm_services:
                result = _sample_warm_service(warm_services[arm])
            else:
                result = _invoke(
                    python_a if arm == "A" else python_b,
                    wheel_a if arm == "A" else wheel_b,
                    args.version_a if arm == "A" else args.version_b,
                    fixture_dir,
                    args.scenario,
                    args.mode,
                    args.temperature,
                    sample_ms=args.sample_ms,
                )
            expected_audit = audit_a if arm == "A" else audit_b
            if any(
                result["audit"][key] != expected_audit[key] for key in result["audit"]
            ):
                raise RuntimeError("installed wheel audit changed during run")
            records[arm].append(result)
            _write_new(destination / f"sample-{index:02d}-{arm}.json", result)
    finally:
        for process in warm_services.values():
            if process.stdin is not None:
                process.stdin.write("quit\n")
                process.stdin.flush()
            process.communicate(timeout=20)
    for arm in ("A", "B"):
        _write_new(destination / f"raw-{arm}.json", records[arm])
    if args.mode in ("timing", "memory"):
        _write_new(
            destination / "summary.json",
            {
                arm: _summary(records[arm], args.mode, args.temperature)
                for arm in ("A", "B")
            },
        )
    if args.mode == "correctness":
        before = records["A"][0]["sample"]
        after = records["B"][0]["sample"]
        for record in (before, after):
            record.pop("wall_ns", None)
            record.pop("cpu_ns", None)
            record.pop("traceback", None)
        _write_new(
            destination / "fingerprint.json",
            {
                "A": before,
                "B": after,
                "equal": before == after,
            },
        )
    print(destination)


def main() -> None:
    """Dispatch fixture generation, isolated workers, and evidence capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fixture_parser = sub.add_parser("fixtures")
    fixture_parser.add_argument("destination")
    fixture_parser.add_argument("--large-rows", type=int, default=65536)
    run = sub.add_parser("capture")
    run.add_argument("--fixtures", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--lane", required=True)
    run.add_argument(
        "--scenario",
        required=True,
        choices=ROUTES
        + tuple(
            f"f2_fault_{kind}__{selection}"
            for kind in FAULTS
            for selection in ("selected", "all")
        ),
    )
    run.add_argument(
        "--mode",
        required=True,
        choices=("correctness", "structure", "timing", "memory"),
    )
    run.add_argument("--temperature", choices=("cold", "warm"), default="cold")
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--sample-ms", type=int, choices=(1, 10), default=10)
    for suffix in ("a", "b"):
        run.add_argument(f"--python-{suffix}", required=True)
        run.add_argument(f"--wheel-{suffix}", required=True)
        run.add_argument(f"--version-{suffix}", required=True)
        run.add_argument(f"--source-sha-{suffix}", required=True)
        run.add_argument(f"--label-{suffix}", required=True)
    run.add_argument(
        "--install-mode", choices=("wheel-no-deps",), default="wheel-no-deps"
    )
    worker = sub.add_parser("_worker")
    worker.add_argument("--wheel", required=True)
    worker.add_argument("--version", required=True)
    worker.add_argument("--fixtures", required=True)
    worker.add_argument("--scenario", required=True)
    worker.add_argument("--mode", required=True)
    worker.add_argument("--temperature", required=True)
    args = parser.parse_args()
    if args.action == "fixtures":
        make_fixtures(Path(args.destination), large_rows=args.large_rows)
    elif args.action == "_worker":
        _worker(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
