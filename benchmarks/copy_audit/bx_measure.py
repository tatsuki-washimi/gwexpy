"""Standalone B-X/#586 installed-wheel public, copy, wall, and PSS harness.

Prepare fixtures with ``bx_fixtures.py``. Run captures only after the final
pre-X integrated source SHA is fixed and the B-X baseline slot is assigned.
Structure mode is instrumented; wall and PSS modes are separate, uninstrumented
worker runs. The controller never imports GWexpy from the candidate checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "gwexpy-v025-bx-capture-v1"
FIXTURE_SCHEMA = "gwexpy-v025-bx-fixtures-v1"
SCENARIOS = {
    "ats32": "ats_int32.ats",
    "ats64": "ats_int64.ats",
    "ats_overflow": "ats_overflow.ats",
    "matrix": "homogeneous_matrix.nc",
    "matrix_nan_inf": "matrix_nan_inf.nc",
    "matrix_int64_extrema": "matrix_int64_extrema.nc",
    "matrix_object_strings": "matrix_object_strings.nc",
}
PRIMARY_SCENARIOS = {"ats32", "ats64", "matrix"}
ORDER = tuple("ABBABAABABBABAABAB")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture(root: Path, scenario: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("B-X fixture schema differs from the frozen design")
    name = SCENARIOS[scenario]
    entries = [item for item in manifest["files"] if item["name"] == name]
    if len(entries) != 1 or _sha256(root / name) != entries[0]["sha256"]:
        raise ValueError("B-X fixture content differs from its manifest")
    return {
        "path": str((root / name).resolve()),
        "file_sha256": entries[0]["sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "manifest_entry": entries[0],
    }


def _wheel_audit(wheel: Path) -> dict[str, Any]:
    import gwexpy

    package = Path(gwexpy.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("GWexpy was imported outside the isolated wheel environment")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("gwexpy/") or member.endswith("/"):
                continue
            target = package.parent / member
            if not target.is_file() or target.read_bytes() != archive.read(member):
                raise RuntimeError(f"Installed wheel differs at {member}")
            checked += 1
    distributions = {}
    for name in ("numpy", "gwpy", "astropy", "xarray", "netCDF4"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "gwexpy_path": str(package),
        "gwexpy_version": importlib.metadata.version("gwexpy"),
        "python": sys.version.split()[0],
        "prefix": sys.prefix,
        "wheel_sha256": _sha256(wheel),
        "wheel_files_verified": checked,
        "distributions": distributions,
        "install_mode": "wheel-no-deps",
    }


def _public_read(
    scenario: str, path: Path, *, retain: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], Any]:
    import numpy as np

    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix

    result = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if scenario.startswith("matrix"):
                result = TimeSeriesMatrix.read(path, format="nc")
            else:
                result = TimeSeries.read(path, format="ats")
        except Exception as error:
            outcome = {
                "kind": "error",
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            }
        else:
            values = np.ascontiguousarray(result.value)
            if values.dtype.hasobject:
                value_bytes = json.dumps(values.tolist(), ensure_ascii=False).encode()
                values_encoding = "json"
            else:
                value_bytes = values.tobytes()
                values_encoding = "native_bytes"
            outcome = {
                "kind": "return",
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "values_sha256": hashlib.sha256(value_bytes).hexdigest(),
                "values_encoding": values_encoding,
                "first_values": values.flat[:4].tolist(),
                "last_values": values.flat[-4:].tolist(),
                "t0_float_hex": float(result.t0.value).hex(),
                "dt_float_hex": float(result.dt.value).hex(),
            }
            if scenario.startswith("matrix"):
                outcome.update(
                    row_keys=[repr(key) for key in result.row_keys()],
                    col_keys=[repr(key) for key in result.col_keys()],
                    units=[[str(item) for item in row] for row in result.units],
                )
            else:
                outcome.update(
                    unit=str(result.unit),
                    name=str(result.name),
                    channel=str(result.channel),
                    provenance=getattr(result, "_gwexpy_io", None),
                )
    public = {
        "outcome": outcome,
        "warnings": [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ],
    }
    return (public, result) if retain else public


def _structure_read(scenario: str, path: Path, samples: int) -> dict[str, Any]:
    """Count full-payload ``astype`` calls at the two source copy sites.

    NumPy's default ``ndarray.astype`` allocates a new buffer even for equal
    dtypes. C-call profiling counts the actual method calls only while the
    ATS scale function or NetCDF matrix ``lossless`` validator executes.
    This count is not a PSS estimate and is verified against a public read.
    """
    import numpy as np

    counts = {"full_payload_astype_calls": 0, "bytes_cast_from": 0}
    target_function = (
        "lossless" if scenario.startswith("matrix") else "_read_timeseries_ats_file"
    )
    target_file = "netcdf4_.py" if scenario.startswith("matrix") else "ats.py"

    def profile(frame: Any, event: str, function: Any) -> None:
        if event != "c_call" or getattr(function, "__name__", None) != "astype":
            return
        if (
            frame.f_code.co_name != target_function
            or not frame.f_code.co_filename.endswith(target_file)
        ):
            return
        array = getattr(function, "__self__", None)
        if isinstance(array, np.ndarray) and array.size == samples:
            counts["full_payload_astype_calls"] += 1
            counts["bytes_cast_from"] += array.nbytes

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        public = _public_read(scenario, path)
    finally:
        sys.setprofile(previous)
    return {"public": public, "structure": counts}


def _read_pss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


def _descendants(pid: int) -> set[int]:
    found = {pid}
    queue = [pid]
    while queue:
        current = queue.pop()
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for token in children.split():
            child = int(token)
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def _worker(args: argparse.Namespace) -> None:
    import numpy as np

    fixture = _fixture(args.fixtures, args.scenario)
    audit = _wheel_audit(args.wheel)
    if args.mode == "audit":
        print(json.dumps({"audit": audit}, sort_keys=True), flush=True)
        return
    path = Path(fixture["path"])
    old_error_mode = np.seterr(all=args.numpy_seterr)
    try:
        if args.mode == "structure":
            entry = fixture["manifest_entry"]
            samples = (
                entry["shape"][-1]
                if args.scenario.startswith("matrix")
                else entry["samples"]
            )
            result = _structure_read(args.scenario, path, samples)
        elif args.mode == "wall":
            _public_read(args.scenario, path)
            start_wall = time.perf_counter_ns()
            start_cpu = time.process_time_ns()
            public = _public_read(args.scenario, path)
            result = {
                "public": public,
                "wall_ns": time.perf_counter_ns() - start_wall,
                "cpu_ns_parent": time.process_time_ns() - start_cpu,
            }
        elif args.mode == "pss":
            public, held_result = _public_read(args.scenario, path, retain=True)
            result = {"public": public}
            time.sleep(0.05)  # Include retained output in the sampled tree peak.
            del held_result
        else:
            result = {"public": _public_read(args.scenario, path)}
    finally:
        np.seterr(**old_error_mode)
    print(
        json.dumps({"audit": audit, "fixture": fixture, **result}, sort_keys=True),
        flush=True,
    )


def _invoke(
    python: Path, wheel: Path, args: argparse.Namespace, mode: str
) -> dict[str, Any]:
    command = [
        str(python),
        "-I",
        str(Path(__file__).resolve()),
        "_worker",
        "--fixtures",
        str(args.fixtures.resolve()),
        "--scenario",
        args.scenario,
        "--mode",
        mode,
        "--numpy-seterr",
        args.numpy_seterr,
        "--wheel",
        str(wheel.resolve()),
    ]
    if mode != "pss":
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise RuntimeError(f"B-X worker failed: {process.stderr}")
        return json.loads(process.stdout)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    trace = []
    deadline = time.monotonic() + 180
    while process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            raise TimeoutError("B-X PSS worker exceeded 180 seconds")
        by_pid = {
            str(pid): value
            for pid in _descendants(process.pid)
            if (value := _read_pss_kib(pid)) is not None
        }
        trace.append(
            {
                "monotonic_ns": time.monotonic_ns(),
                "pss_kib_by_pid": by_pid,
                "tree_pss_kib": sum(by_pid.values()),
            }
        )
        time.sleep(args.sample_ms / 1000)
    stdout, stderr = process.communicate(timeout=10)
    if process.returncode:
        raise RuntimeError(f"B-X PSS worker failed: {stderr}")
    return {
        **json.loads(stdout),
        "peak_tree_pss_kib": max((row["tree_pss_kib"] for row in trace), default=0),
        "pss_trace": trace,
        "pss_sample_ms": args.sample_ms,
        "worker_stderr": stderr,
    }


def _summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [record[metric] for record in records]
    median = statistics.median(values)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "samples": values,
    }


def _capture(args: argparse.Namespace) -> None:
    if not args.pre_x_sha or len(args.pre_x_sha) != 40:
        raise ValueError("The final integrated pre-X source SHA must be fixed first")
    if len(args.source_a) != 40 or len(args.source_b) != 40:
        raise ValueError("Both measured source SHAs must be full 40-character IDs")
    if args.samples not in (5, 9):
        raise ValueError(
            "B-X uses five characterization or nine release samples per arm"
        )
    if args.mode in ("wall", "pss") and args.scenario not in PRIMARY_SCENARIOS:
        raise ValueError("B-X performance capture is limited to primary fixtures")
    if args.sample_ms not in (1, 10):
        raise ValueError("B-X PSS uses 10 ms, or 1 ms only for an explicit retry")
    if args.phase == "prex" and args.source_b != args.pre_x_sha:
        raise ValueError("The pre-X arm must match the integrated pre-X SHA")
    if args.phase == "candidate" and args.source_a != args.pre_x_sha:
        raise ValueError("The candidate comparison must use pre-X as arm A")
    fixture = _fixture(args.fixtures, args.scenario)
    arms = {
        "A": (args.python_a, args.wheel_a, args.source_a),
        "B": (args.python_b, args.wheel_b, args.source_b),
    }
    audits = {
        arm: _invoke(python, wheel, args, "audit")["audit"]
        for arm, (python, wheel, _) in arms.items()
    }
    if audits["A"]["python"] != audits["B"]["python"]:
        raise RuntimeError("B-X wheel arms use different Python versions")
    if audits["A"]["distributions"] != audits["B"]["distributions"]:
        raise RuntimeError("B-X wheel arms use different dependency versions")
    if (
        audits["A"]["wheel_sha256"] == audits["B"]["wheel_sha256"]
        and args.source_a != args.source_b
    ):
        raise RuntimeError("Different source SHAs were assigned the same wheel bytes")
    args.output.mkdir(parents=True, exist_ok=False)
    records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    order = ORDER[: 2 * args.samples]
    public_oracle = None
    for index, arm in enumerate(order):
        python, wheel, _ = arms[arm]
        sample = _invoke(python, wheel, args, args.mode)
        public = sample.get("public", sample)
        if args.mode != "audit":
            if public_oracle is None:
                public_oracle = public
            elif public != public_oracle:
                raise RuntimeError("B-X sample public fingerprint/warnings differ")
        records[arm].append(sample)
        (args.output / f"sample-{index:02d}-{arm}.json").write_text(
            json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metric = {"wall": "wall_ns", "pss": "peak_tree_pss_kib"}.get(args.mode)
    summary = (
        {arm: _summary(samples, metric) for arm, samples in records.items()}
        if metric
        else {}
    )
    cpu_summary = (
        {arm: _summary(samples, "cpu_ns_parent") for arm, samples in records.items()}
        if args.mode == "wall"
        else {}
    )
    manifest = {
        "schema": SCHEMA,
        "status": "RAW_UNQUALIFIED",
        "phase": args.phase,
        "mode": args.mode,
        "numpy_seterr": args.numpy_seterr,
        "scenario": args.scenario,
        "pre_x_integrated_sha": args.pre_x_sha,
        "fixture": fixture,
        "harness_sha256": _sha256(Path(__file__)),
        "fixture_generator_sha256": json.loads(
            (args.fixtures / "manifest.json").read_text()
        )["generator_sha256"],
        "order": order,
        "samples_per_arm": args.samples,
        "arms": {
            arm: {
                "source_sha": source,
                "wheel_sha256": audit["wheel_sha256"],
                "audit": audit,
            }
            for arm, (_, _, source) in arms.items()
            for audit in (audits[arm],)
        },
        "public_parity": True,
        "pss_definition": f"max_t sum(PSS of parent and all live descendants at the same {args.sample_ms} ms sample time); sampled lower bound"
        if args.mode == "pss"
        else None,
        "pss_sample_ms": args.sample_ms if args.mode == "pss" else None,
        "summary": summary,
        "cpu_summary": cpu_summary,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "_worker"):
        item = sub.add_parser(command)
        item.add_argument("--fixtures", type=Path, required=True)
        item.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
        item.add_argument(
            "--mode",
            choices=("audit", "public", "structure", "wall", "pss"),
            required=True,
        )
        item.add_argument(
            "--numpy-seterr", choices=("warn", "raise", "ignore"), default="warn"
        )
        if command == "_worker":
            item.add_argument("--wheel", type=Path, required=True)
        else:
            item.add_argument("--output", type=Path, required=True)
            item.add_argument(
                "--phase", choices=("historical", "prex", "candidate"), required=True
            )
            item.add_argument("--pre-x-sha", required=True)
            item.add_argument("--samples", type=int, default=9)
            item.add_argument("--sample-ms", type=int, default=10)
            for arm in ("a", "b"):
                item.add_argument(f"--python-{arm}", type=Path, required=True)
                item.add_argument(f"--wheel-{arm}", type=Path, required=True)
                item.add_argument(f"--source-{arm}", required=True)
    return parser


if __name__ == "__main__":
    options = _parser().parse_args()
    if options.command == "_worker":
        _worker(options)
    else:
        _capture(options)
