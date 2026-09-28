"""Deterministic GWF fixtures and B1 contract for the #588 benchmark.

Fixture bytes are written through GWpy so they do not depend on GWexpy read
results. ``write_fixtures`` also captures public-read behavior and therefore
must be run under old-R only; freeze and reuse its manifest for candidate
comparisons. The returned manifest is JSON serializable and hashes the actual
files written by the installed Frame backend.

Full capture runs 11 small public cases with a fresh spawn pool per parallel
case. Under a busy host, use resumable shards instead:

1. Run ``python benchmarks/io/f4_gwf_fixtures.py OUTPUT_DIR --prepare-only``
   once to write and hash the small fixture set without reading it.
2. Run one ``--capture-case CASE --route serial`` or ``--route parallel``
   invocation at a time. Each shard verifies all fixture hashes and the B1
   interpreter, package, source-commit, and helper-digest context before it
   updates ``manifest.json`` atomically.
3. For parallel characterization, wrap each invocation in an external
   ``timeout --kill-after=30s 180s``. A timeout is a failed capture, not a
   sample. Confirm the process group and workers have exited, then retry the
   identical case/route at most twice with the same interpreter, fixture
   hashes, worker count, and environment. Never change workload settings to
   make a shard complete. This timeout/retry policy applies to correctness
   characterization only; performance/PSS measurements follow the plan's
   separate rules.

Example parallel shard command (use the same interpreter and environment for
every shard):

``timeout --kill-after=30s 180s env PYTHONPATH=/path/to/old-R python benchmarks/io/f4_gwf_fixtures.py /path/to/frozen-fixtures --capture-case adjacent --route parallel``

If no usable GWF writer is installed, the manifest records the blocker and
fallback design; it never substitutes fake frame bytes for valid samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

CHANNEL = "K1:V025-F4-PRIMARY"
OTHER_CHANNEL = "K1:V025-F4-OTHER"
SAMPLE_RATE_HZ = 8.0
GPS_START = 1_000_000_000.0
SAMPLES_PER_PART = 8
STRESS_PARTS = 256
STRESS_SAMPLES_PER_PART = 8192

_READ_CASES: dict[str, tuple[list[str], list[str], dict[str, Any]]] = {
    "adjacent": (["part_00.gwf", "part_01.gwf"], [CHANNEL], {}),
    "unsorted_adjacent": (["part_01.gwf", "part_00.gwf"], [CHANNEL], {}),
    "gap_default": (["part_01.gwf", "part_02_gap.gwf"], [CHANNEL], {}),
    "gap_pad": (["part_01.gwf", "part_02_gap.gwf"], [CHANNEL], {"gap": "pad"}),
    "overlap_default": (
        ["part_02_gap.gwf", "part_03_overlap.gwf"],
        [CHANNEL],
        {},
    ),
    "selected_valid": (["part_00.gwf"], [CHANNEL], {}),
    "unselected_valid": (["unselected_valid.gwf"], [CHANNEL], {}),
    "out_of_range_valid": (
        ["part_05.gwf"],
        [CHANNEL],
        {"start": GPS_START, "end": GPS_START + 1.0},
    ),
    "selected_malformed": (
        ["part_00.gwf", "selected_malformed.gwf"],
        [CHANNEL],
        {},
    ),
    "unselected_malformed": (
        ["unselected_valid.gwf", "unselected_malformed.gwf"],
        [OTHER_CHANNEL],
        {},
    ),
    "out_of_range_malformed": (
        ["part_00.gwf", "out_of_range_malformed.gwf"],
        [CHANNEL],
        {"start": GPS_START, "end": GPS_START + 1.0},
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(values: np.ndarray, t0: float) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "t0_gps": t0,
        "dt_s": 1.0 / SAMPLE_RATE_HZ,
        "values_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "first_values": contiguous[:4].tolist(),
        "last_values": contiguous[-4:].tolist(),
    }


def _public_read_fingerprint(result: Any) -> dict[str, Any]:
    """Summarize a B1 public-read result without dropping axis or dtype data."""
    channels: dict[str, Any] = {}
    for name, series in result.items():
        values = np.ascontiguousarray(series.value)
        channels[str(name)] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "t0_gps": float(series.t0.value),
            "dt_s": float(series.dt.value),
            "span_gps": [float(value) for value in series.span],
            "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "first_values": values[:4].tolist(),
            "last_values": values[-4:].tolist(),
        }
    return {"channel_order": [str(name) for name in result], "channels": channels}


def _capture_public_read(
    sources: list[Path],
    *,
    channels: list[str],
    parallel: bool | int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Capture one public B1 result or exact exception and warning contract."""
    from gwexpy.timeseries import TimeSeriesDict

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = TimeSeriesDict.read(
                sources,
                channels=channels,
                format="gwf",
                parallel=parallel,
                **kwargs,
            )
        except Exception as exc:
            outcome = {
                "kind": "error",
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
        else:
            outcome = {
                "kind": "normal-return",
                "fingerprint": _public_read_fingerprint(result),
            }
        return {
            "parallel": parallel,
            "outcome": outcome,
            "warnings": [
                {
                    "category": f"{item.category.__module__}.{item.category.__qualname__}",
                    "message": str(item.message),
                }
                for item in captured
            ],
        }


class SharedLivePartCounter:
    """Shared exact counter for instrumented unique materialized part objects.

    Construct with a live ``multiprocessing.Manager``. Instrument each
    materialized part once at its creation/receipt and release it when its last
    owning collection drops it. The key includes PID and object identity, so
    aliases to the same object in a worker or parent are not double-counted,
    while a separately unpickled parent copy is a distinct live part. Keep the
    manager alive for the whole read and report ``peak`` as the structural gate.

    This is an instrumentation utility only; it does not monkeypatch or alter
    the public reader. Per-process peak counters must not be summed.
    """

    def __init__(self, manager: Any) -> None:
        self._active = manager.dict()
        self._peak = manager.Value("i", 0)
        self._lock = manager.RLock()

    @staticmethod
    def _key(part: Any) -> str:
        return f"{os.getpid()}:{id(part)}"

    def retain(self, part: Any) -> None:
        """Count *part* once while it is alive in the current process."""
        key = self._key(part)
        with self._lock:
            if key in self._active:
                return
            self._active[key] = True
            count = len(self._active)
            if count > self._peak.value:
                self._peak.value = count

    def release(self, part: Any) -> None:
        """Remove the current process's retained-part token for *part*."""
        key = self._key(part)
        with self._lock:
            self._active.pop(key, None)

    @property
    def current(self) -> int:
        """Return the number of currently retained unique part objects."""
        return len(self._active)

    @property
    def peak(self) -> int:
        """Return the maximum simultaneous retained-part count observed."""
        return int(self._peak.value)

    @contextmanager
    def hold(self, part: Any):
        """Retain *part* for the duration of a ``with`` block."""
        self.retain(part)
        try:
            yield part
        finally:
            self.release(part)


def _frame(path: Path, channel: str, values: np.ndarray, t0: float) -> None:
    from gwpy.timeseries import TimeSeries, TimeSeriesDict

    series = TimeSeries(
        values,
        sample_rate=SAMPLE_RATE_HZ,
        t0=t0,
        name=channel,
        channel=channel,
        unit="m",
    )
    TimeSeriesDict({channel: series}).write(path, format="gwf")


def write_fixtures(
    output_dir: str | Path, *, characterize_b1: bool = True
) -> dict[str, Any]:
    """Write deterministic normal, gap, overlap, unselected, and bad frames.

    The small behavior set contains adjacent, two-sample gap, overlapping, and
    reverse-ordered adjacent spans. The returned manifest records serial and
    parallel public reads for each route and the valid/malformed selection
    matrix. Run this only with old-R installed, then freeze and reuse the
    manifest when comparing a candidate.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        from gwpy.io.registry import default_registry
        from gwpy.timeseries import TimeSeriesDict

        default_registry.get_writer("gwf", TimeSeriesDict)
        default_registry.get_reader("gwf", TimeSeriesDict)
    except Exception as exc:
        blocker = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
        manifest = {
            "schema": "gwexpy-v025-b-f4-fixtures-v1",
            "generator": "benchmarks/io/f4_gwf_fixtures.py:write_fixtures",
            "backend": None,
            "capability_blocker": blocker,
            "expected_fingerprints": None,
            "files": [],
            "fallback_test_design": {
                "valid_fixture": "Use the repository's existing conformance GWF generator only after installing a supported reader and writer; do not fabricate frame bytes or sample expectations.",
                "fault_cases": [
                    "selected valid and malformed source",
                    "unselected valid and malformed source",
                    "out-of-range valid and malformed source",
                ],
                "temporal_cases": [
                    "adjacent",
                    "gap",
                    "overlap",
                    "unsorted source order",
                ],
                "required_measurement": "Maximum simultaneously retained part results across the process tree, sampled as a concurrent live count rather than a sum of per-process peaks.",
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    # Values encode source identity so ordering and selection mistakes are
    # visible without relying on random-number behavior.
    placements = [
        ("part_00.gwf", 0, GPS_START),
        ("part_01.gwf", 1, GPS_START + 1.0),
        ("part_02_gap.gwf", 2, GPS_START + 2.25),
        ("part_03_overlap.gwf", 3, GPS_START + 3.0),
        ("part_04.gwf", 4, GPS_START + 4.0),
        ("part_05.gwf", 5, GPS_START + 5.0),
    ]
    arrays: dict[str, np.ndarray] = {}
    for filename, part, t0 in placements:
        values = (part * 100 + np.arange(SAMPLES_PER_PART, dtype=np.int32)).astype(
            np.float64
        )
        path = output / filename
        _frame(path, CHANNEL, values, t0)
        arrays[filename] = values

    unselected = output / "unselected_valid.gwf"
    _frame(
        unselected,
        OTHER_CHANNEL,
        np.arange(SAMPLES_PER_PART, dtype=np.float64) + 9000,
        GPS_START + 100.0,
    )
    selected_bad = output / "selected_malformed.gwf"
    selected_bad.write_bytes(b"F4 deterministic malformed GWF fixture\n")
    unselected_bad = output / "unselected_malformed.gwf"
    unselected_bad.write_bytes(
        b"F4 deterministic malformed GWF fixture: unrelated channel request\n"
    )
    outside_bad = output / "out_of_range_malformed.gwf"
    outside_bad.write_bytes(b"F4 deterministic malformed GWF fixture: outside range\n")

    backend = "gwpy GWF public writer/reader"
    capability_blocker = None
    sources = {path.name: path for path in output.glob("*.gwf")}
    observed: dict[str, Any] = {}
    if characterize_b1:
        try:
            for case, (names, channels, read_kwargs) in _READ_CASES.items():
                observed[case] = {
                    "serial": _capture_public_read(
                        [sources[name] for name in names],
                        channels=channels,
                        parallel=False,
                        **read_kwargs,
                    ),
                    "parallel": _capture_public_read(
                        [sources[name] for name in names],
                        channels=channels,
                        parallel=2,
                        **read_kwargs,
                    ),
                }
        except Exception as exc:
            backend = None
            capability_blocker = (
                f"B1 public-read characterization stopped at {case}: "
                f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
            )
            observed.setdefault(case, {})["characterization_blocker"] = (
                capability_blocker
            )
    else:
        observed = {case: {"status": "pending"} for case in _READ_CASES}

    fixtures = sorted(path for path in output.glob("*.gwf") if path.is_file())
    manifest: dict[str, Any] = {
        "schema": "gwexpy-v025-b-f4-fixtures-v1",
        "generator": "benchmarks/io/f4_gwf_fixtures.py:write_fixtures",
        "backend": backend,
        "capability_blocker": capability_blocker,
        "channel": CHANNEL,
        "other_channel": OTHER_CHANNEL,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "samples_per_part": SAMPLES_PER_PART,
        "stress_fixture_recipe": {
            "parts": STRESS_PARTS,
            "samples_per_part": STRESS_SAMPLES_PER_PART,
            "decoded_sample_bytes": STRESS_PARTS * STRESS_SAMPLES_PER_PART * 8,
            "writer": "write_stress_fixtures(output_dir)",
            "purpose": "256 parts exceeds the maximum 8-worker queue by 32x while keeping decoded values fixed at 16 MiB.",
        },
        "source_order": [name for name, _, _ in placements],
        "temporal_cases": {
            "adjacent": ["part_00.gwf", "part_01.gwf"],
            "gap": ["part_01.gwf", "part_02_gap.gwf"],
            "overlap": ["part_02_gap.gwf", "part_03_overlap.gwf"],
            "deliberately_unsorted": [
                "part_04.gwf",
                "part_00.gwf",
                "part_03_overlap.gwf",
            ],
        },
        "fault_cases": {
            "selected_valid": "part_00.gwf",
            "selected_malformed": "selected_malformed.gwf",
            "out_of_range_valid": "part_05.gwf",
            "out_of_range_malformed": "out_of_range_malformed.gwf",
            "unselected_valid": "unselected_valid.gwf",
            "unselected_malformed": "unselected_malformed.gwf",
            "case_note": "Malformed GWF is source-level corruption. The unselected pair requests OTHER_CHANNEL from an OTHER_CHANNEL valid frame plus a malformed listed source; it tests whether B1 opens every source even when one cannot contribute selected samples.",
        },
        "b1_contract": {
            "comparison": "Each case stores serial and parallel=2 public B1 outcomes, including channel order and per-channel values hash, dtype, shape, t0, dt, span, warning category/message, or exact exception type/message.",
            "public_reads": observed,
            "runtime_context": _b1_runtime_context() if characterize_b1 else None,
            "gap_default": "B1 default merge mode is gap='raise'; discontiguous and overlapping spans should return captured errors. gap_pad separately captures padded gap output including NaN positions.",
            "structural_metric": "Instrument each unique materialized per-source TimeSeriesDict once at creation/receipt and release it when its last owning reference is discarded. SharedLivePartCounter reports max simultaneous object count across workers and parent. Do not count aliases (Future/completed_parts/parts references) twice or sum independent per-process peaks.",
            "structural_instrumentation": "For serial, retain after each read_one result and release after merge consumes it. For parallel, retain worker materialization and parent unpickle as separate physical objects, then release each after its last owner drops it. A shared manager counter spans pool startup through worker exit.",
        },
        "expected_fingerprints": {
            "individual_parts": {
                name: _fingerprint(values, t0)
                for name, (_, _, t0) in zip(arrays, placements, strict=True)
                for values in (arrays[name],)
            },
            "unselected_valid": _fingerprint(
                np.arange(SAMPLES_PER_PART, dtype=np.float64) + 9000,
                GPS_START + 100.0,
            ),
        },
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in fixtures
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _b1_runtime_context() -> dict[str, Any]:
    """Describe the Python and source context used for one B1 capture shard."""
    import gwexpy

    repo_root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package_root = Path(gwexpy.__file__).resolve().parent
    package_files = (
        Path("timeseries/_gwf_io.py"),
        Path("timeseries/collections.py"),
        Path("timeseries/timeseries.py"),
        Path("timeseries/io/gwf/__init__.py"),
    )
    tracked_environment = {
        name: os.environ.get(name)
        for name in (
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "gwexpy_file": str(Path(gwexpy.__file__).resolve()),
        "gwexpy_version": importlib.metadata.version("gwexpy"),
        "gwpy_version": importlib.metadata.version("gwpy"),
        "source_head": head,
        "loaded_package_source_hashes": {
            name.as_posix(): _sha256(package_root / name) for name in package_files
        },
        "runtime_environment_sha256": hashlib.sha256(
            json.dumps(tracked_environment, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "helper_sha256": _sha256(Path(__file__).resolve()),
    }


def characterize_b1_case(
    output_dir: str | Path,
    case: str,
    route: str,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Capture one resumable B1 case/route against already frozen fixtures.

    Use one CLI invocation per case and route under the same old-R interpreter.
    The function checks fixture hashes and an exact runtime-context record
    before updating the manifest atomically. A timed-out invocation leaves no
    record, so an identical later invocation can safely retry that shard.
    """
    if case not in _READ_CASES:
        raise ValueError(f"Unknown F4 case {case!r}; choose from {sorted(_READ_CASES)}")
    if route not in {"serial", "parallel"}:
        raise ValueError("route must be 'serial' or 'parallel'")
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "gwexpy-v025-b-f4-fixtures-v1":
        raise ValueError(
            "The output directory does not contain the F4 fixture manifest"
        )

    expected_files = {
        item["name"]: item["sha256"] for item in manifest.get("files", [])
    }
    names, channels, read_kwargs = _READ_CASES[case]
    for name in names:
        path = output / name
        if name not in expected_files or not path.is_file():
            raise ValueError(
                f"Frozen fixture {name!r} is missing from the manifest or disk"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_files[name]:
            raise ValueError(f"Frozen fixture hash changed for {name!r}: {actual_hash}")

    context = _b1_runtime_context()
    previous_context = manifest.get("b1_contract", {}).get("runtime_context")
    if previous_context is not None and previous_context != context:
        raise RuntimeError("B1 runtime context changed; refusing to mix capture shards")
    public_reads = manifest.setdefault("b1_contract", {}).setdefault("public_reads", {})
    case_results = public_reads.setdefault(case, {})
    case_results.pop("status", None)
    if route in case_results and not replace_existing:
        raise FileExistsError(f"Capture already exists for {case}/{route}")

    capture = _capture_public_read(
        [output / name for name in names],
        channels=channels,
        parallel=(False if route == "serial" else 2),
        **read_kwargs,
    )
    case_results[route] = capture
    manifest["b1_contract"]["runtime_context"] = context
    manifest["b1_contract"].setdefault("capture_status", {})[case] = {
        "serial": "captured" if "serial" in case_results else "pending",
        "parallel": "captured" if "parallel" in case_results else "pending",
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return capture


def write_stress_fixtures(output_dir: str | Path) -> dict[str, Any]:
    """Write the fixed many-parts workload for scalability/PSS runs.

    The recipe is intentionally separate from the small B1 behavior matrix.
    It creates 256 adjacent frames with 8,192 float64 samples each (16 MiB of
    decoded samples total), enough sources to expose source-count retention
    beyond an 8-worker queue. This routine is not called by ``write_fixtures``.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    first_hash = ""
    last_hash = ""
    for part in range(STRESS_PARTS):
        first_value = part * STRESS_SAMPLES_PER_PART
        values = np.arange(
            first_value,
            first_value + STRESS_SAMPLES_PER_PART,
            dtype=np.float64,
        )
        path = output / f"stress_{part:04d}.gwf"
        _frame(
            path,
            CHANNEL,
            values,
            GPS_START + part * STRESS_SAMPLES_PER_PART / SAMPLE_RATE_HZ,
        )
        digest = _sha256(path)
        files.append(
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest}
        )
        if part == 0:
            first_hash = _fingerprint(values, GPS_START)["values_sha256"]
        if part == STRESS_PARTS - 1:
            last_hash = _fingerprint(
                values,
                GPS_START + part * STRESS_SAMPLES_PER_PART / SAMPLE_RATE_HZ,
            )["values_sha256"]
    manifest = {
        "schema": "gwexpy-v025-b-f4-stress-fixtures-v1",
        "generator": "benchmarks/io/f4_gwf_fixtures.py:write_stress_fixtures",
        "channel": CHANNEL,
        "parts": STRESS_PARTS,
        "samples_per_part": STRESS_SAMPLES_PER_PART,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "decoded_sample_bytes": STRESS_PARTS * STRESS_SAMPLES_PER_PART * 8,
        "source_order": [f"stress_{part:04d}.gwf" for part in range(STRESS_PARTS)],
        "expected_payload_fingerprints": {
            "first_part_values_sha256": first_hash,
            "last_part_values_sha256": last_hash,
            "value_rule": "float64 arange(part_index * 8192, (part_index + 1) * 8192)",
            "time_rule": "t0 = 1000000000 + part_index * 8192 / 8 GPS seconds; all parts are adjacent",
        },
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--capture-case", choices=sorted(_READ_CASES))
    parser.add_argument("--route", choices=("serial", "parallel"))
    args = parser.parse_args()
    if args.capture_case:
        if args.route is None:
            parser.error("--capture-case requires --route")
        result = characterize_b1_case(args.output_dir, args.capture_case, args.route)
    elif args.prepare_only:
        if args.route is not None:
            parser.error("--route is only valid with --capture-case")
        result = write_fixtures(args.output_dir, characterize_b1=False)
    else:
        if args.route is not None:
            parser.error("--route is only valid with --capture-case")
        result = write_fixtures(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
