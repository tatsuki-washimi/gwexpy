"""Dependency-free, isolated B0/B1 benchmark infrastructure for #676.

The parent process owns protocol and evidence assembly.  Child processes import
``gwexpy`` only after their ``PYTHONPATH`` and working directory have been set
to the selected target tree; the imported package path is then verified before
any operation is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any, cast


@dataclass(frozen=True)
class BenchmarkProtocol:
    warmups: int = 3
    child_processes: int = 7
    minimum_measurement_seconds: float = 0.250
    max_attempts: int = 3
    stability_threshold: float = 0.05


B0_PROTOCOL = BenchmarkProtocol()


def _require_frozen_protocol(protocol: BenchmarkProtocol) -> None:
    """Reject every protocol other than the frozen #676 protocol."""
    if not isinstance(protocol, BenchmarkProtocol) or protocol != B0_PROTOCOL:
        raise ValueError(
            "frozen benchmark protocol requires warmups=3, exactly 7 raw "
            "samples, minimum batches=0.250s, max_attempts=3, and stability=5%"
        )


def _frozen_operation_payloads() -> list[dict[str, object]]:
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "shape": list(definition.shape),
        }
        for definition in OPERATIONS
    ]


@dataclass(frozen=True)
class BenchmarkSample:
    operation: str
    timings: tuple[float, ...]
    rss_bytes: int

    @property
    def median_seconds(self) -> float:
        if not self.timings:
            raise ValueError("a benchmark sample needs at least one timing")
        return float(median(self.timings))

    @property
    def mad_seconds(self) -> float:
        center = self.median_seconds
        return float(median(abs(value - center) for value in self.timings))


@dataclass(frozen=True)
class ComparisonDecision:
    passed: bool
    operation_deltas_us: dict[str, float]
    rss_deltas_bytes: dict[str, int]
    geometric_mean_ratio: float
    failed_operations: tuple[str, ...]
    stability_gate_passed: bool = True
    unstable_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationDefinition:
    name: str
    description: str
    shape: tuple[int, ...]


OPERATIONS: tuple[OperationDefinition, ...] = (
    OperationDefinition(
        "construct", "construct a bounded 2x2 TimeSeriesMatrix", (2, 2, 128)
    ),
    OperationDefinition("copy", "copy a prepared TimeSeriesMatrix", (2, 2, 128)),
    OperationDefinition(
        "slice", "slice the sample axis of a prepared matrix", (2, 2, 128)
    ),
    OperationDefinition(
        "asarray", "convert a prepared matrix with np.asarray", (2, 2, 128)
    ),
    OperationDefinition(
        "multiply", "multiply a prepared matrix by a Python scalar", (2, 2, 128)
    ),
    OperationDefinition(
        "quantity_left_multiply",
        "multiply a prepared matrix by a Quantity from the left",
        (2, 2, 128),
    ),
)


def minimum_iterations(single_call_seconds: float, minimum_seconds: float) -> int:
    """Return the smallest positive iteration count reaching a time target."""
    if single_call_seconds <= 0 or minimum_seconds <= 0:
        raise ValueError("timing and minimum duration must be positive")
    return max(1, math.ceil(minimum_seconds / single_call_seconds))


def next_iterations(
    current_iterations: int, measured_seconds: float, minimum_seconds: float
) -> int:
    """Increase a measured batch when runtime drift undershoots the target."""
    if current_iterations <= 0 or measured_seconds <= 0 or minimum_seconds <= 0:
        raise ValueError("iterations and durations must be positive")
    scaled = math.ceil(current_iterations * minimum_seconds / measured_seconds)
    return max(current_iterations + 1, scaled)


def is_stable(timings: Sequence[float], *, threshold: float) -> bool:
    """Use median absolute deviation / median as the variability measure."""
    sample = BenchmarkSample("stability", tuple(timings), 0)
    center = sample.median_seconds
    if center <= 0:
        raise ValueError("timings must have a positive median")
    return sample.mad_seconds / center <= threshold


def geometric_mean_ratio(ratios: Sequence[float]) -> float:
    """Return a positive geometric mean without introducing NumPy dependency."""
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must be non-empty and positive")
    return math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))


def compare_candidate(
    baseline: dict[str, BenchmarkSample],
    candidate: dict[str, BenchmarkSample],
) -> ComparisonDecision:
    """Apply the frozen timing, ratio, and RSS budgets to two result sets."""
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate operation sets must match")

    operation_deltas_us: dict[str, float] = {}
    rss_deltas_bytes: dict[str, int] = {}
    failed: list[str] = []
    ratios: list[float] = []
    for operation in sorted(baseline):
        base = baseline[operation]
        trial = candidate[operation]
        base_median = base.median_seconds
        trial_median = trial.median_seconds
        delta_seconds = trial_median - base_median
        operation_deltas_us[operation] = delta_seconds * 1_000_000
        rss_delta = trial.rss_bytes - base.rss_bytes
        rss_deltas_bytes[operation] = rss_delta
        ratios.append(trial_median / base_median)
        timing_limit = max(base_median * 0.20, 10e-6)
        rss_limit = max(int(base.rss_bytes * 0.10), 8 * 1024**2)
        timing_breach = delta_seconds > timing_limit
        if timing_breach or rss_delta > rss_limit:
            failed.append(operation)

    ratio = geometric_mean_ratio(ratios)
    if ratio > 1.10:
        failed.extend(
            operation for operation in sorted(baseline) if operation not in failed
        )
    return ComparisonDecision(
        passed=not failed,
        operation_deltas_us=operation_deltas_us,
        rss_deltas_bytes=rss_deltas_bytes,
        geometric_mean_ratio=ratio,
        failed_operations=tuple(failed),
    )


def verify_imported_source_paths(
    target_root: Path, imported_paths: Sequence[Path]
) -> tuple[str, ...]:
    """Verify existing imported source files and return target-relative paths."""
    root = target_root.resolve()
    relative: list[str] = []
    seen: set[str] = set()
    for imported in imported_paths:
        if ".." in imported.parts:
            raise ValueError(f"imported source path contains traversal: {imported}")
        resolved = imported.resolve()
        try:
            relative_path = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"imported source is outside target: {resolved}") from exc
        if not resolved.is_file():
            raise ValueError(f"imported source is not an existing file: {resolved}")
        relative_name = relative_path.as_posix()
        if relative_name in seen:
            raise ValueError(f"duplicate imported source path: {relative_name}")
        seen.add(relative_name)
        relative.append(relative_name)
    return tuple(sorted(relative))


def verify_imported_source_modules(
    target_root: Path, imported_modules: Sequence[Mapping[str, object]]
) -> tuple[dict[str, str], ...]:
    """Validate recorded modules, paths, and source hashes under ``target_root``."""
    if not imported_modules:
        raise ValueError("child must record at least one gwexpy source module")
    root = target_root.resolve()
    validated: list[dict[str, str]] = []
    names: set[str] = set()
    paths: set[str] = set()
    for record in imported_modules:
        if set(record) != {"name", "path", "sha256"}:
            raise TypeError("source module records must contain name, path, and sha256")
        name = record["name"]
        path = record["path"]
        sha256 = record["sha256"]
        if (
            not isinstance(name, str)
            or name != "gwexpy"
            and not name.startswith("gwexpy.")
            or any(not part.isidentifier() for part in name.split("."))
        ):
            raise ValueError(f"invalid gwexpy source module name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate gwexpy source module: {name}")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "." in Path(path).parts
            or path != Path(path).as_posix()
        ):
            raise ValueError(f"source module path must be relative: {path!r}")
        source = (root / Path(path)).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source module is outside target: {path!r}") from exc
        if not source.is_file():
            raise ValueError(f"source module is not an existing file: {path!r}")
        module_parts = name.split(".")[1:]
        module_base = Path("gwexpy").joinpath(*module_parts)
        package_path = root / module_base / "__init__.py"
        module_path = root / module_base.with_suffix(".py")
        if package_path.is_file():
            allowed_paths = {(module_base / "__init__.py").as_posix()}
        elif module_path.is_file():
            allowed_paths = {module_base.with_suffix(".py").as_posix()}
        else:
            raise ValueError(f"source module has no importable file: {name!r}")
        if path not in allowed_paths:
            raise ValueError(
                f"source module path does not match name {name!r}: {path!r}"
            )
        names.add(name)
        relative_name = source.relative_to(root).as_posix()
        if relative_name != path:
            raise ValueError(
                f"source module path resolves to a different file: {path!r}"
            )
        if relative_name in paths:
            raise ValueError(f"duplicate imported source path: {relative_name}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"source module hash is invalid: {name!r}")
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if sha256 != actual_sha256:
            raise ValueError(f"source module hash does not match: {name!r}")
        paths.add(relative_name)
        validated.append({"name": name, "path": relative_name, "sha256": actual_sha256})
    return tuple(sorted(validated, key=lambda item: item["name"]))


def _validated_runtime_file_list(
    relative_files: Sequence[str], *, require_nonempty: bool
) -> list[str]:
    if isinstance(relative_files, (str, bytes)) or not isinstance(
        relative_files, Sequence
    ):
        raise TypeError("runtime files must be a sequence of relative paths")
    files = list(relative_files)
    if require_nonempty and not files:
        raise ValueError("runtime file set must be non-empty")
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or "." in Path(path).parts
        or path != Path(path).as_posix()
        for path in files
    ):
        raise ValueError("benchmark runtime file paths must be relative")
    if len(files) != len(set(files)):
        raise ValueError(
            "benchmark runtime file paths must be unique; duplicate entries "
            "are not allowed"
        )
    return files


def runtime_file_sha256(target_root: Path, relative_files: Sequence[str]) -> str:
    """Hash a deterministic, target-relative frozen runtime file set."""
    root = target_root.resolve()
    digest = hashlib.sha256()
    files = _validated_runtime_file_list(relative_files, require_nonempty=True)
    for relative in sorted(files):
        relative_path = Path(relative)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"runtime file is outside target: {relative!r}") from exc
        if not path.is_file():
            raise ValueError(f"runtime file is not an existing file: {relative!r}")
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def candidate_runtime_file_set(
    target_root: Path, relative_files: Sequence[str]
) -> dict[str, object]:
    """Build the path-safe SHA-256 evidence block for a B1 candidate."""
    files = sorted(_validated_runtime_file_list(relative_files, require_nonempty=True))
    return {"files": files, "sha256": runtime_file_sha256(target_root, files)}


def _git_output(target_root: Path, *arguments: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=target_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("candidate root must be a usable Git worktree") from exc
    return [line for line in completed.stdout.splitlines() if line]


def _fixed_ref_candidates(fixed_origin_ref: str) -> tuple[str, ...]:
    if not fixed_origin_ref or fixed_origin_ref.startswith("-"):
        raise ValueError("fixed origin ref is invalid")
    if fixed_origin_ref.startswith("refs/"):
        return (fixed_origin_ref,)
    return tuple(
        f"refs/{prefix}/{fixed_origin_ref}" for prefix in ("heads", "remotes", "tags")
    )


def _validate_fixed_origin_ref(
    target_root: Path, fixed_origin_ref: str, fixed_sha: str
) -> str:
    """Require the declared local ref to resolve uniquely to ``fixed_sha``."""
    root = _validate_candidate_tree(target_root, fixed_sha)
    resolved: list[tuple[str, str]] = []
    for candidate in _fixed_ref_candidates(fixed_origin_ref):
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            resolved.append((candidate, completed.stdout.strip()))
    if not resolved:
        raise ValueError(f"fixed origin ref is missing: {fixed_origin_ref!r}")
    if len(resolved) != 1:
        raise ValueError(f"fixed origin ref is ambiguous: {fixed_origin_ref!r}")
    resolved_sha = resolved[0][1]
    if resolved_sha != fixed_sha:
        raise ValueError(
            "fixed_sha does not match the declared fixed origin ref: "
            f"{fixed_sha} != {resolved_sha}"
        )
    return resolved_sha


def _validate_candidate_tree(
    target_root: Path, fixed_sha: str, fixed_origin_ref: str | None = None
) -> Path:
    """Verify that ``target_root`` is the measured Git tree and base exists."""
    root = target_root.resolve()
    if not root.is_dir():
        raise ValueError("candidate root must be an existing directory")
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")[0]).resolve()
    if top_level != root:
        raise ValueError("candidate root must be the measured repository root")
    try:
        _git_output(root, "cat-file", "-e", f"{fixed_sha}^{{commit}}")
    except ValueError as exc:
        raise ValueError(
            "fixed_sha must identify the candidate tree's declared base"
        ) from exc
    if fixed_origin_ref is not None:
        _validate_fixed_origin_ref(root, fixed_origin_ref, fixed_sha)
    return root


def _runtime_source_path(path: str) -> bool:
    relative = Path(path)
    return (
        relative.parts[:1] == ("gwexpy",)
        and relative.suffix == ".py"
        and "__pycache__" not in relative.parts
    )


def authoritative_runtime_file_set(
    target_root: Path, fixed_sha: str
) -> dict[str, object]:
    """Derive all changed ``gwexpy`` runtime files from the declared base."""
    root = _validate_candidate_tree(target_root, fixed_sha)
    tracked = _git_output(
        root, "diff", "--name-only", "--diff-filter=ACMRTUXB", fixed_sha, "--"
    )
    untracked = _git_output(root, "ls-files", "--others", "--exclude-standard")
    ignored = _git_output(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--no-empty-directory",
    )
    changed = sorted(
        {
            path
            for path in (*tracked, *untracked, *ignored)
            if _runtime_source_path(path)
        }
    )
    files = _validated_runtime_file_list(changed, require_nonempty=True)
    return {
        "files": files,
        "sha256": runtime_file_sha256(root, files),
    }


def _module_path_from_name(name: str, available_paths: set[str]) -> str:
    module_parts = name.split(".")[1:]
    module_base = Path("gwexpy").joinpath(*module_parts)
    package_path = (module_base / "__init__.py").as_posix()
    module_path = module_base.with_suffix(".py").as_posix()
    if package_path in available_paths:
        return package_path
    if module_path in available_paths:
        return module_path
    raise ValueError(f"source module has no importable file: {name!r}")


def _git_blob_sha256(target_root: Path, fixed_sha: str, relative_path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "show", f"{fixed_sha}:{relative_path}"],
            cwd=target_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"fixed tree has no source module path: {relative_path!r}"
        ) from exc
    return hashlib.sha256(completed.stdout).hexdigest()


_FIXED_TREE_IMPORT_PROBE = """
import json
import sys
from pathlib import Path

import astropy
import gwpy
import numpy
import gwexpy

if (Path.cwd() / "gwexpy" / "timeseries" / "__init__.py").is_file():
    from gwexpy.timeseries import TimeSeriesMatrix

root = Path.cwd().resolve()
records = []
for name, module in sys.modules.items():
    if name != "gwexpy" and not name.startswith("gwexpy."):
        continue
    source = getattr(module, "__file__", None)
    if source is None:
        raise RuntimeError(f"loaded gwexpy module has no source file: {name}")
    records.append((name, Path(source).resolve().relative_to(root).as_posix()))
print(json.dumps(sorted(set(records))))
"""


@cache
def _authoritative_fixed_tree_source_modules(
    target_root: str, fixed_sha: str
) -> tuple[tuple[str, str], ...]:
    """Compute the expected gwexpy import closure from an immutable fixed tree."""
    root = _validate_candidate_tree(Path(target_root), fixed_sha)
    with isolated_worktree(root, fixed_sha) as fixed_root:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(fixed_root)
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _FIXED_TREE_IMPORT_PROBE],
                cwd=fixed_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            raw_records = json.loads(completed.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise ValueError(
                "fixed-SHA source tree import closure could not be computed"
            ) from exc
        if (
            not isinstance(raw_records, list)
            or not raw_records
            or not all(
                isinstance(record, list)
                and len(record) == 2
                and all(isinstance(value, str) and value for value in record)
                for record in raw_records
            )
        ):
            raise ValueError("fixed-SHA source tree import closure is invalid")
        records = [
            {
                "name": record[0],
                "path": record[1],
                "sha256": hashlib.sha256(
                    (fixed_root / record[1]).read_bytes()
                ).hexdigest(),
            }
            for record in raw_records
        ]
        validated = verify_imported_source_modules(fixed_root, records)
    return tuple((record["name"], record["path"]) for record in validated)


def _validate_fixed_tree_source_modules(
    target_root: Path,
    fixed_sha: str,
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], ...]:
    root = target_root.resolve()
    available_paths = set(
        _git_output(root, "ls-tree", "-r", "--name-only", fixed_sha, "gwexpy")
    )
    validated: list[dict[str, str]] = []
    names: set[str] = set()
    paths: set[str] = set()
    for record in records:
        name = record["name"]
        path = record["path"]
        if not isinstance(name, str) or name in names:
            raise ValueError("benchmark evidence source modules must be unique")
        if not isinstance(path, str) or path in paths:
            raise ValueError("benchmark evidence source modules must be unique")
        expected_path = _module_path_from_name(name, available_paths)
        if path != expected_path:
            raise ValueError(
                f"source module path does not match name {name!r}: {path!r}"
            )
        record_value: dict[str, str] = {"name": name, "path": path}
        if "sha256" in record:
            sha256 = record["sha256"]
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError(
                    f"benchmark evidence source module hash is invalid: {name!r}"
                )
            actual_sha256 = _git_blob_sha256(root, fixed_sha, path)
            if sha256 != actual_sha256:
                raise ValueError(
                    f"source module hash does not match fixed tree: {name!r}"
                )
            record_value["sha256"] = actual_sha256
        names.add(name)
        paths.add(path)
        validated.append(record_value)
    expected = set(_authoritative_fixed_tree_source_modules(str(root), fixed_sha))
    actual = {(record["name"], record["path"]) for record in validated}
    if actual != expected:
        raise ValueError(
            "B0 source modules must equal the authoritative fixed-SHA import closure"
        )
    return tuple(sorted(validated, key=lambda item: item["name"]))


def _module_record_key(record: Mapping[str, object]) -> tuple[str, ...]:
    name = record.get("name")
    path = record.get("path")
    if not isinstance(name, str) or not isinstance(path, str):
        raise TypeError("source module record name and path must be strings")
    sha256 = record.get("sha256")
    if sha256 is None:
        return (name, path)
    if not isinstance(sha256, str):
        raise TypeError("source module record hash must be a string")
    return (name, path, sha256)


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == "darwin" else 1024))


def _worker_matrix(operation: str):
    import numpy as np
    from astropy import units as u

    from gwexpy.timeseries import TimeSeriesMatrix

    data: Any = np.arange(2 * 2 * 128, dtype=float).reshape(2, 2, 128)

    def construct():
        return TimeSeriesMatrix(data, dt=0.01 * u.s, unit=u.V, name="benchmark")

    matrix = construct()
    actions = {
        "construct": construct,
        "copy": matrix.copy,
        "slice": lambda: matrix[:, :, 16:112],
        "asarray": lambda: np.asarray(matrix),
        "multiply": lambda: matrix * 2,
        "quantity_left_multiply": lambda: (2 * u.s) * matrix,
    }
    try:
        return actions[operation], np, u, matrix
    except KeyError as exc:
        raise ValueError(f"unknown benchmark operation: {operation!r}") from exc


def _loaded_gwexpy_source_modules(target_root: Path) -> tuple[dict[str, str], ...]:
    loaded: list[dict[str, object]] = []
    for name, module in sys.modules.items():
        if name != "gwexpy" and not name.startswith("gwexpy."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            raise RuntimeError(f"loaded gwexpy module has no source file: {name}")
        loaded.append({"name": name, "path": str(Path(source).resolve())})
    relative_records: list[dict[str, object]] = []
    for record in loaded:
        source_paths = verify_imported_source_paths(
            target_root, (Path(cast(str, record["path"])),)
        )
        relative_path = source_paths[0]
        relative_records.append(
            {
                "name": record["name"],
                "path": relative_path,
                "sha256": hashlib.sha256(
                    (target_root / relative_path).read_bytes()
                ).hexdigest(),
            }
        )
    return verify_imported_source_modules(target_root, tuple(relative_records))


def _worker_main(operation: str, protocol: BenchmarkProtocol) -> dict[str, object]:
    import astropy
    import gwpy
    import numpy as np

    import gwexpy

    action, _, _, _ = _worker_matrix(operation)
    for _ in range(protocol.warmups):
        action()

    calibration_start = time.perf_counter()
    calibration_calls = 0
    while (
        time.perf_counter() - calibration_start < protocol.minimum_measurement_seconds
    ):
        action()
        calibration_calls += 1
    calibration_seconds = time.perf_counter() - calibration_start
    iterations = minimum_iterations(
        calibration_seconds / max(calibration_calls, 1),
        protocol.minimum_measurement_seconds,
    )

    rss_before = _rss_bytes()
    while True:
        started = time.perf_counter()
        for _ in range(iterations):
            action()
        elapsed = time.perf_counter() - started
        if elapsed >= protocol.minimum_measurement_seconds:
            break
        iterations = next_iterations(
            iterations, elapsed, protocol.minimum_measurement_seconds
        )
    rss_after = _rss_bytes()
    target_root = Path.cwd()
    source_modules = _loaded_gwexpy_source_modules(target_root)
    return {
        "operation": operation,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "per_operation_seconds": elapsed / iterations,
        "rss_bytes": max(0, rss_after - rss_before),
        "source_modules": list(source_modules),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(aliased=True),
            "numpy": np.__version__,
            "astropy": astropy.__version__,
            "gwpy": gwpy.__version__,
            "gwexpy": getattr(gwexpy, "__version__", "unknown"),
        },
    }


def _child_environment(target_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target_root.resolve())
    return environment


def _run_child(
    target_root: Path, operation: str, protocol: BenchmarkProtocol
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        operation,
    ]
    completed = subprocess.run(
        command,
        cwd=target_root,
        env=_child_environment(target_root),
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        raise RuntimeError(f"benchmark child wrote stderr: {completed.stderr}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark child did not emit JSON") from exc
    return validate_child_payload(payload, target_root, operation, protocol)


def validate_child_payload(
    payload: object,
    target_root: Path,
    expected_operation: str,
    protocol: BenchmarkProtocol,
) -> dict[str, object]:
    """Schema-validate and parent-validate one untrusted child payload."""
    if not isinstance(payload, dict):
        raise TypeError("benchmark child payload must be an object")
    required = {
        "operation",
        "iterations",
        "elapsed_seconds",
        "per_operation_seconds",
        "rss_bytes",
        "source_modules",
        "environment",
    }
    if set(payload) != required:
        raise ValueError("benchmark child payload schema mismatch")
    if payload["operation"] != expected_operation:
        raise ValueError("benchmark child operation mismatch")
    if (
        not isinstance(payload["iterations"], int)
        or isinstance(payload["iterations"], bool)
        or payload["iterations"] <= 0
    ):
        raise TypeError("benchmark iterations must be a positive integer")
    for key in ("elapsed_seconds", "per_operation_seconds"):
        value = payload[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise TypeError(f"benchmark {key} must be a positive finite number")
    elapsed = float(cast(float, payload["elapsed_seconds"]))
    per_operation = float(cast(float, payload["per_operation_seconds"]))
    if elapsed < protocol.minimum_measurement_seconds:
        raise ValueError(
            "benchmark elapsed_seconds is below the minimum measurement duration"
        )
    iterations = int(cast(int, payload["iterations"]))
    representation_tolerance = 4 * max(
        math.ulp(elapsed),
        math.ulp(per_operation),
        math.ulp(elapsed / iterations),
    )
    expected_per_operation = elapsed / iterations
    if abs(per_operation - expected_per_operation) > representation_tolerance:
        raise ValueError(
            "benchmark per-operation timing is inconsistent with elapsed_seconds and iterations"
        )
    if (
        not isinstance(payload["rss_bytes"], int)
        or isinstance(payload["rss_bytes"], bool)
        or payload["rss_bytes"] < 0
    ):
        raise TypeError("benchmark RSS must be a non-negative integer")
    modules = payload["source_modules"]
    if not isinstance(modules, list) or not all(
        isinstance(record, dict) for record in modules
    ):
        raise TypeError("benchmark source_modules must be a list of objects")
    validated_modules = verify_imported_source_modules(
        target_root, cast(Sequence[Mapping[str, object]], modules)
    )
    environment = payload["environment"]
    expected_environment = {"python", "platform", "numpy", "astropy", "gwpy", "gwexpy"}
    if not isinstance(environment, dict) or set(environment) != expected_environment:
        raise ValueError("benchmark environment schema mismatch")
    if not all(isinstance(value, str) and value for value in environment.values()):
        raise TypeError("benchmark environment values must be non-empty strings")
    validated = dict(payload)
    validated["source_modules"] = list(validated_modules)
    return validated


def _run_attempt(
    target_root: Path,
    operation: str,
    protocol: BenchmarkProtocol,
    attempt: int,
) -> dict[str, object]:
    raw_samples = [
        _run_child(target_root, operation, protocol)
        for _ in range(protocol.child_processes)
    ]
    raw_samples = [
        validate_child_payload(sample, target_root, operation, protocol)
        for sample in raw_samples
    ]
    timings = tuple(
        float(cast(float, sample["per_operation_seconds"])) for sample in raw_samples
    )
    stable = is_stable(timings, threshold=protocol.stability_threshold)
    selected = BenchmarkSample(
        operation=operation,
        timings=timings,
        rss_bytes=max(int(cast(int, sample["rss_bytes"])) for sample in raw_samples),
    )
    return {
        "attempt": attempt,
        "stable": stable,
        "median_seconds": selected.median_seconds,
        "mad_seconds": selected.mad_seconds,
        "raw_samples": raw_samples,
    }


def _evidence_results(evidence: Mapping[str, object]) -> Mapping[str, object]:
    results = evidence.get("results", evidence)
    if not isinstance(results, Mapping):
        raise TypeError("benchmark evidence results must be an object")
    return results


def _recomputed_attempt_statistics(
    raw_attempt: Mapping[str, object], protocol: BenchmarkProtocol
) -> tuple[float, float, bool]:
    """Recompute summary fields solely from an attempt's raw timing samples."""
    raw_samples = raw_attempt.get("raw_samples")
    if (
        not isinstance(raw_samples, Sequence)
        or isinstance(raw_samples, (str, bytes))
        or len(raw_samples) != protocol.child_processes
    ):
        raise ValueError(
            f"benchmark evidence requires exactly {protocol.child_processes} raw samples"
        )
    timings: list[float] = []
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            raise TypeError("benchmark raw sample evidence must be an object")
        timing = sample.get("per_operation_seconds")
        if (
            not isinstance(timing, (int, float))
            or isinstance(timing, bool)
            or not math.isfinite(float(timing))
            or float(timing) <= 0
        ):
            raise TypeError("benchmark evidence timing must be positive and finite")
        timings.append(float(timing))
    sample = BenchmarkSample("validated", tuple(timings), 0)
    median_seconds = sample.median_seconds
    mad_seconds = sample.mad_seconds
    return (
        median_seconds,
        mad_seconds,
        is_stable(timings, threshold=protocol.stability_threshold),
    )


def _summary_float_matches(stored: object, expected: float) -> bool:
    if (
        not isinstance(stored, (int, float))
        or isinstance(stored, bool)
        or not math.isfinite(float(stored))
        or float(stored) < 0
    ):
        return False
    actual = float(stored)
    if expected == 0.0:
        return actual == 0.0
    return abs(actual - expected) <= 4 * max(math.ulp(actual), math.ulp(expected))


def _evidence_stability_gate(
    evidence: Mapping[str, object],
    *,
    protocol: BenchmarkProtocol = B0_PROTOCOL,
) -> tuple[str, ...]:
    unstable: list[str] = []
    for operation, raw_result in _evidence_results(evidence).items():
        if not isinstance(operation, str) or not isinstance(raw_result, Mapping):
            raise TypeError("benchmark operation evidence must be an object")
        attempts = raw_result.get("attempts")
        if (
            not isinstance(attempts, Sequence)
            or isinstance(attempts, (str, bytes))
            or not attempts
        ):
            raise ValueError("benchmark operation evidence has no attempts")
        final = attempts[-1]
        if not isinstance(final, Mapping):
            raise TypeError("benchmark attempt evidence must be an object")
        _, _, final_stable = _recomputed_attempt_statistics(final, protocol)
        if raw_result.get("stability") != ("stable" if final_stable else "unstable"):
            raise ValueError("benchmark aggregate stability does not match raw samples")
        if not final_stable:
            unstable.append(operation)
    return tuple(sorted(unstable))


def _validate_evidence_source_modules(
    value: object,
    *,
    phase: str,
    fixed_sha: str,
    target_root: Path | None = None,
) -> tuple[dict[str, str], ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(record, Mapping) for record in value)
    ):
        raise ValueError("benchmark evidence source modules must be non-empty")
    names: set[str] = set()
    paths: set[str] = set()
    records: list[Mapping[str, object]] = []
    for record in value:
        allowed_keys = {"name", "path"}
        if phase == "B1":
            allowed_keys.add("sha256")
        if set(record) != allowed_keys:
            raise ValueError("benchmark evidence source module schema mismatch")
        name = record["name"]
        path = record["path"]
        if (
            not isinstance(name, str)
            or (name != "gwexpy" and not name.startswith("gwexpy."))
            or any(not part.isidentifier() for part in name.split("."))
        ):
            raise ValueError("benchmark evidence source module name is invalid")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "." in Path(path).parts
            or path != Path(path).as_posix()
        ):
            raise ValueError("benchmark evidence source module path is invalid")
        if phase == "B1":
            sha256 = record["sha256"]
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError("benchmark evidence source module hash is invalid")
        if name in names or path in paths:
            raise ValueError("benchmark evidence source modules must be unique")
        names.add(name)
        paths.add(path)
        records.append(record)
    if target_root is not None:
        if phase == "B0":
            return _validate_fixed_tree_source_modules(target_root, fixed_sha, records)
        return verify_imported_source_modules(target_root, records)
    return tuple(
        {key: cast(str, record[key]) for key in record}
        for record in sorted(records, key=lambda item: cast(str, item["name"]))
    )


def _validate_evidence_sample(
    sample: object,
    operation: str,
    protocol: BenchmarkProtocol,
    *,
    phase: str,
    fixed_sha: str,
    target_root: Path | None = None,
) -> None:
    if not isinstance(sample, Mapping):
        raise TypeError("benchmark raw sample evidence must be an object")
    required = {
        "operation",
        "iterations",
        "elapsed_seconds",
        "per_operation_seconds",
        "rss_bytes",
        "source_modules",
        "environment",
    }
    if set(sample) != required:
        raise ValueError("benchmark raw sample schema mismatch")
    if sample["operation"] != operation:
        raise ValueError("benchmark raw sample operation mismatch")
    iterations = sample["iterations"]
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        raise TypeError("benchmark evidence iterations must be positive integers")
    elapsed = sample["elapsed_seconds"]
    per_operation = sample["per_operation_seconds"]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        for value in (elapsed, per_operation)
    ):
        raise TypeError("benchmark evidence timings must be positive and finite")
    elapsed_value = float(elapsed)
    per_operation_value = float(per_operation)
    if elapsed_value < protocol.minimum_measurement_seconds:
        raise ValueError("benchmark evidence batch is below 0.250 seconds")
    tolerance = 4 * max(
        math.ulp(elapsed_value),
        math.ulp(per_operation_value),
        math.ulp(elapsed_value / iterations),
    )
    if abs(per_operation_value - elapsed_value / iterations) > tolerance:
        raise ValueError("benchmark evidence timing is inconsistent")
    rss = sample["rss_bytes"]
    if not isinstance(rss, int) or isinstance(rss, bool) or rss < 0:
        raise TypeError("benchmark evidence RSS must be a non-negative integer")
    _validate_evidence_source_modules(
        sample["source_modules"],
        phase=phase,
        fixed_sha=fixed_sha,
        target_root=target_root,
    )
    environment = sample["environment"]
    expected_environment = {"python", "platform", "numpy", "astropy", "gwpy", "gwexpy"}
    if (
        not isinstance(environment, Mapping)
        or set(environment) != expected_environment
        or not all(isinstance(value, str) and value for value in environment.values())
    ):
        raise ValueError("benchmark evidence environment schema mismatch")


def _validate_runtime_file_set(
    value: object, *, phase: str, target_root: Path | None = None
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"files", "sha256", "status"}:
        raise ValueError("benchmark runtime file set schema mismatch")
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("benchmark runtime file paths must be a frozen list")
    _validated_runtime_file_list(files, require_nonempty=phase == "B1")
    if files != sorted(files):
        raise ValueError("benchmark runtime file paths must be sorted and frozen")
    sha256 = value["sha256"]
    if sha256 is not None and (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("benchmark runtime file set SHA-256 is invalid")
    if phase == "B0":
        if sha256 is not None or files:
            raise ValueError("B0 cannot contain candidate runtime file evidence")
    elif sha256 is None:
        raise ValueError("B1 requires candidate runtime file evidence")
    elif target_root is not None and sha256 != runtime_file_sha256(target_root, files):
        raise ValueError("benchmark runtime file set SHA-256 does not match files")


def _validate_evidence(
    evidence: Mapping[str, object],
    *,
    expected_phase: str,
    target_root: Path | None = None,
) -> None:
    if expected_phase == "B1" and target_root is None:
        raise ValueError("B1 evidence validation requires a candidate root")
    required = {
        "schema",
        "phase",
        "fixed_origin_ref",
        "fixed_sha",
        "environment",
        "protocol",
        "operations",
        "imported_gwexpy_source_modules",
        "results",
        "stability_gate",
        "candidate_evidence",
    }
    if set(evidence) != required:
        raise ValueError("benchmark evidence schema mismatch")
    if evidence["schema"] != "gwexpy.series_matrix_benchmark.v1":
        raise ValueError("unsupported benchmark evidence schema")
    if evidence["phase"] != expected_phase:
        raise ValueError(f"benchmark evidence phase must be {expected_phase}")
    fixed_origin_ref = evidence["fixed_origin_ref"]
    if not isinstance(fixed_origin_ref, str) or not fixed_origin_ref:
        raise ValueError("benchmark evidence fixed origin ref is invalid")
    fixed_sha = evidence["fixed_sha"]
    if (
        not isinstance(fixed_sha, str)
        or len(fixed_sha) != 40
        or any(character not in "0123456789abcdef" for character in fixed_sha)
    ):
        raise ValueError("benchmark evidence fixed_sha must be a 40-character SHA")
    protocol = evidence["protocol"]
    if protocol != asdict(B0_PROTOCOL):
        raise ValueError("frozen benchmark protocol is required in evidence")
    operations = evidence["operations"]
    if operations != _frozen_operation_payloads():
        raise ValueError("benchmark evidence operation definitions are not frozen")
    environment = evidence["environment"]
    expected_environment = {"python", "platform", "numpy", "astropy", "gwpy", "gwexpy"}
    if (
        not isinstance(environment, Mapping)
        or set(environment) != expected_environment
        or not all(isinstance(value, str) and value for value in environment.values())
    ):
        raise ValueError("benchmark evidence environment schema mismatch")
    if target_root is not None:
        _validate_fixed_origin_ref(target_root, fixed_origin_ref, fixed_sha)
    top_level_modules = _validate_evidence_source_modules(
        evidence["imported_gwexpy_source_modules"],
        phase=expected_phase,
        fixed_sha=fixed_sha,
        target_root=target_root,
    )
    top_level_module_keys = {_module_record_key(record) for record in top_level_modules}
    results = evidence["results"]
    if not isinstance(results, Mapping) or set(results) != {
        definition.name for definition in OPERATIONS
    }:
        raise ValueError("benchmark evidence operation results are incomplete")
    for operation, raw_result in results.items():
        if not isinstance(raw_result, Mapping) or set(raw_result) != {
            "attempts",
            "stability",
        }:
            raise ValueError("benchmark operation evidence schema mismatch")
        attempts = raw_result["attempts"]
        if (
            not isinstance(attempts, Sequence)
            or isinstance(attempts, (str, bytes))
            or not attempts
            or len(attempts) > B0_PROTOCOL.max_attempts
        ):
            raise ValueError("benchmark attempts must be between one and three")
        for expected_attempt, raw_attempt in enumerate(attempts, start=1):
            if not isinstance(raw_attempt, Mapping) or set(raw_attempt) != {
                "attempt",
                "stable",
                "median_seconds",
                "mad_seconds",
                "raw_samples",
            }:
                raise ValueError("benchmark attempt evidence schema mismatch")
            if raw_attempt["attempt"] != expected_attempt:
                raise ValueError("benchmark attempts must be sequential")
            if not isinstance(raw_attempt["stable"], bool):
                raise TypeError("benchmark attempt stability must be boolean")
            for key in ("median_seconds", "mad_seconds"):
                value = raw_attempt[key]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise TypeError("benchmark attempt statistics must be finite")
            raw_samples = raw_attempt["raw_samples"]
            if (
                not isinstance(raw_samples, Sequence)
                or isinstance(raw_samples, (str, bytes))
                or len(raw_samples) != B0_PROTOCOL.child_processes
            ):
                raise ValueError("benchmark evidence requires exactly 7 raw samples")
            for sample in raw_samples:
                _validate_evidence_sample(
                    sample,
                    operation,
                    B0_PROTOCOL,
                    phase=expected_phase,
                    fixed_sha=fixed_sha,
                    target_root=target_root,
                )
                assert isinstance(sample, Mapping)
                sample_module_keys = {
                    _module_record_key(record)
                    for record in cast(
                        Sequence[Mapping[str, object]], sample["source_modules"]
                    )
                }
                if sample_module_keys != top_level_module_keys:
                    raise ValueError(
                        "benchmark top-level source modules must equal every raw-sample module"
                    )
            median_seconds, mad_seconds, stable = _recomputed_attempt_statistics(
                raw_attempt, B0_PROTOCOL
            )
            if not _summary_float_matches(
                raw_attempt["median_seconds"], median_seconds
            ):
                raise ValueError("benchmark attempt median does not match raw samples")
            if not _summary_float_matches(raw_attempt["mad_seconds"], mad_seconds):
                raise ValueError("benchmark attempt MAD does not match raw samples")
            if raw_attempt["stable"] is not stable:
                raise ValueError(
                    "benchmark attempt stability does not match raw samples"
                )
        stability = raw_result["stability"]
        if stability not in {"stable", "unstable"}:
            raise ValueError("benchmark stability must be stable or unstable")
        _, _, final_stable = _recomputed_attempt_statistics(
            cast(Mapping[str, object], attempts[-1]), B0_PROTOCOL
        )
        if final_stable != (stability == "stable"):
            raise ValueError("benchmark stability does not match the final attempt")
    candidate_evidence = evidence["candidate_evidence"]
    if not isinstance(candidate_evidence, Mapping) or set(candidate_evidence) != {
        "decision",
        "issue_637",
        "runtime_file_set",
    }:
        raise ValueError("benchmark candidate evidence schema mismatch")
    if candidate_evidence["issue_637"] != f"not evaluated in {expected_phase}":
        raise ValueError("benchmark evidence must not claim issue #637")
    _validate_runtime_file_set(
        candidate_evidence["runtime_file_set"],
        phase=expected_phase,
        target_root=target_root,
    )
    if expected_phase == "B1":
        authoritative = authoritative_runtime_file_set(
            cast(Path, target_root), fixed_sha
        )
        declared = cast(Mapping[str, object], candidate_evidence["runtime_file_set"])
        if (
            declared["files"] != authoritative["files"]
            or declared["sha256"] != authoritative["sha256"]
        ):
            raise ValueError(
                "benchmark runtime file set does not equal the authoritative candidate changes"
            )
    stability_gate = evidence["stability_gate"]
    if not isinstance(stability_gate, Mapping) or set(stability_gate) != {
        "adoptable",
        "unstable_operations",
        "rule",
    }:
        raise ValueError("benchmark stability gate schema mismatch")
    unstable = list(_evidence_stability_gate(evidence))
    if stability_gate["unstable_operations"] != unstable:
        raise ValueError("benchmark stability gate is inconsistent with results")
    if stability_gate["adoptable"] != (not unstable):
        raise ValueError("benchmark stability gate adoptability is inconsistent")


def _samples_from_evidence(
    evidence: Mapping[str, object],
    *,
    protocol: BenchmarkProtocol = B0_PROTOCOL,
) -> dict[str, BenchmarkSample]:
    samples: dict[str, BenchmarkSample] = {}
    for operation, raw_result in _evidence_results(evidence).items():
        if not isinstance(operation, str) or not isinstance(raw_result, Mapping):
            raise TypeError("benchmark operation evidence must be an object")
        attempts = raw_result.get("attempts")
        if (
            not isinstance(attempts, Sequence)
            or isinstance(attempts, (str, bytes))
            or not attempts
        ):
            raise ValueError(f"benchmark evidence has no attempts for {operation!r}")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise TypeError("benchmark attempt evidence must be an object")
            median_seconds, mad_seconds, stable = _recomputed_attempt_statistics(
                attempt, protocol
            )
            if not _summary_float_matches(
                attempt.get("median_seconds"), median_seconds
            ):
                raise ValueError("benchmark attempt median does not match raw samples")
            if not _summary_float_matches(attempt.get("mad_seconds"), mad_seconds):
                raise ValueError("benchmark attempt MAD does not match raw samples")
            if attempt.get("stable") is not stable:
                raise ValueError(
                    "benchmark attempt stability does not match raw samples"
                )
        final = attempts[-1]
        if not isinstance(final, Mapping):
            raise TypeError("benchmark attempt evidence must be an object")
        _, _, final_stable = _recomputed_attempt_statistics(final, protocol)
        if raw_result.get("stability") != ("stable" if final_stable else "unstable"):
            raise ValueError("benchmark aggregate stability does not match raw samples")
        raw_samples = final.get("raw_samples")
        if (
            not isinstance(raw_samples, Sequence)
            or isinstance(raw_samples, (str, bytes))
            or len(raw_samples) != protocol.child_processes
        ):
            raise ValueError(
                f"benchmark evidence requires exactly {protocol.child_processes} "
                f"raw samples for {operation!r}"
            )
        timings: list[float] = []
        rss_values: list[int] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, Mapping):
                raise TypeError("benchmark raw sample evidence must be an object")
            timing = raw_sample.get("per_operation_seconds")
            rss = raw_sample.get("rss_bytes")
            if (
                not isinstance(timing, (int, float))
                or isinstance(timing, bool)
                or not math.isfinite(float(timing))
                or float(timing) <= 0
            ):
                raise TypeError("benchmark evidence timing must be positive and finite")
            if not isinstance(rss, int) or isinstance(rss, bool) or rss < 0:
                raise TypeError("benchmark evidence RSS must be a non-negative integer")
            timings.append(float(timing))
            rss_values.append(rss)
        samples[operation] = BenchmarkSample(
            operation=operation,
            timings=tuple(timings),
            rss_bytes=max(rss_values),
        )
    return samples


def adopt_candidate(
    baseline_evidence: Mapping[str, object],
    candidate_evidence: Mapping[str, object],
    *,
    candidate_root: Path,
) -> ComparisonDecision:
    """Apply the stability gate before any numeric adoption comparison."""
    _validate_evidence(
        baseline_evidence,
        expected_phase="B0",
        target_root=candidate_root,
    )
    _validate_evidence(
        candidate_evidence,
        expected_phase="B1",
        target_root=candidate_root,
    )
    if baseline_evidence["fixed_sha"] != candidate_evidence["fixed_sha"]:
        raise ValueError("baseline and candidate must use the same fixed SHA")
    unstable = tuple(
        sorted(
            set(_evidence_stability_gate(baseline_evidence))
            | set(_evidence_stability_gate(candidate_evidence))
        )
    )
    if unstable:
        return ComparisonDecision(
            passed=False,
            operation_deltas_us={},
            rss_deltas_bytes={},
            geometric_mean_ratio=math.nan,
            failed_operations=unstable,
            stability_gate_passed=False,
            unstable_operations=unstable,
        )
    decision = compare_candidate(
        _samples_from_evidence(baseline_evidence, protocol=B0_PROTOCOL),
        _samples_from_evidence(candidate_evidence, protocol=B0_PROTOCOL),
    )
    return replace(decision, stability_gate_passed=True, unstable_operations=())


def capture_target(
    target_root: Path,
    *,
    protocol: BenchmarkProtocol = B0_PROTOCOL,
) -> dict[str, dict[str, object]]:
    """Capture all operations for an already-selected, isolated target tree."""
    results: dict[str, dict[str, object]] = {}
    for definition in OPERATIONS:
        attempts: list[dict[str, object]] = []
        for attempt in range(1, protocol.max_attempts + 1):
            result = _run_attempt(target_root, definition.name, protocol, attempt)
            attempts.append(result)
            if bool(result["stable"]):
                break
        results[definition.name] = {
            "attempts": attempts,
            "stability": "stable" if attempts[-1]["stable"] else "unstable",
        }
    return results


@contextmanager
def isolated_worktree(repo_root: Path, fixed_sha: str) -> Iterator[Path]:
    """Yield a temporary detached worktree and clean up only its exact path."""
    temporary_path = Path(
        tempfile.mkdtemp(prefix=".gwexpy-b0-", dir=tempfile.gettempdir())
    )
    added = False
    try:
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                "--quiet",
                str(temporary_path),
                fixed_sha,
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        added = True
        yield temporary_path
    finally:
        if added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(temporary_path)],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        if temporary_path.exists():
            shutil.rmtree(temporary_path)


def _require_exact_capture_samples(
    results: Mapping[str, object], protocol: BenchmarkProtocol
) -> None:
    """Require every captured attempt to contain the frozen raw sample count."""
    if not isinstance(results, Mapping):
        raise TypeError("benchmark capture results must be an object")
    for operation, raw_result in results.items():
        if not isinstance(raw_result, Mapping):
            raise TypeError("benchmark capture operation result must be an object")
        attempts = raw_result.get("attempts")
        if (
            not isinstance(attempts, Sequence)
            or isinstance(attempts, (str, bytes))
            or not attempts
            or len(attempts) > protocol.max_attempts
        ):
            raise ValueError("benchmark capture attempts exceed the frozen maximum")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise TypeError("benchmark capture attempt must be an object")
            raw_samples = attempt.get("raw_samples")
            if (
                not isinstance(raw_samples, Sequence)
                or isinstance(raw_samples, (str, bytes))
                or len(raw_samples) != protocol.child_processes
            ):
                raise ValueError(
                    f"benchmark capture requires exactly {protocol.child_processes} "
                    "independent raw samples per attempt"
                )


def _hashless_source_modules(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    return [
        {"name": cast(str, record["name"]), "path": cast(str, record["path"])}
        for record in records
    ]


def capture_b0(
    repo_root: Path,
    *,
    origin_ref: str = "origin/main",
    protocol: BenchmarkProtocol = B0_PROTOCOL,
    runtime_files: Sequence[str] = (),
) -> dict[str, object]:
    """Capture B0 from a clean detached checkout of a fixed origin SHA."""
    _require_frozen_protocol(protocol)
    fixed_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{origin_ref}^{{commit}}"],
        cwd=repo_root,
        text=True,
    ).strip()
    with isolated_worktree(repo_root, fixed_sha) as target_root:
        results = capture_target(target_root, protocol=protocol)
        _require_exact_capture_samples(results, protocol)
        raw_samples: list[dict[str, object]] = []
        for operation_name, operation in results.items():
            attempts = cast(list[dict[str, object]], operation["attempts"])
            for attempt in attempts:
                samples = cast(list[dict[str, object]], attempt["raw_samples"])
                validated_samples = [
                    validate_child_payload(
                        sample, target_root, operation_name, protocol
                    )
                    for sample in samples
                ]
                for sample in validated_samples:
                    sample["source_modules"] = _hashless_source_modules(
                        cast(Sequence[Mapping[str, object]], sample["source_modules"])
                    )
                attempt["raw_samples"] = validated_samples
                raw_samples.extend(validated_samples)
        source_modules = sorted(
            {
                (record["name"], record["path"])
                for sample in raw_samples
                for record in cast(list[dict[str, str]], sample["source_modules"])
            }
        )
        environment = next(
            cast(dict[str, object], sample["environment"]) for sample in raw_samples
        )
        evidence: dict[str, object] = {
            "schema": "gwexpy.series_matrix_benchmark.v1",
            "phase": "B0",
            "fixed_origin_ref": origin_ref,
            "fixed_sha": fixed_sha,
            "environment": environment,
            "protocol": asdict(protocol),
            "operations": _frozen_operation_payloads(),
            "imported_gwexpy_source_modules": [
                {"name": name, "path": path} for name, path in source_modules
            ],
            "results": results,
            "stability_gate": {
                "adoptable": not bool(_evidence_stability_gate({"results": results})),
                "unstable_operations": list(
                    _evidence_stability_gate({"results": results})
                ),
                "rule": "all baseline and candidate operations must be stable before numeric comparison",
            },
            "candidate_evidence": {
                "decision": "pending",
                "issue_637": "not evaluated in B0",
                "runtime_file_set": {
                    "files": list(sorted(runtime_files)),
                    "sha256": None,
                    "status": "candidate-only; no B1 candidate supplied",
                },
            },
        }
        _validate_evidence(evidence, expected_phase="B0", target_root=target_root)
        return evidence


def capture_candidate(
    repo_root: Path,
    *,
    origin_ref: str = "origin/main",
    protocol: BenchmarkProtocol = B0_PROTOCOL,
    runtime_files: Sequence[str] = (),
) -> dict[str, object]:
    """Capture B1 in the selected candidate tree using the frozen protocol."""
    _require_frozen_protocol(protocol)
    fixed_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{origin_ref}^{{commit}}"],
        cwd=repo_root,
        text=True,
    ).strip()
    target_root = repo_root.resolve()
    _validate_fixed_origin_ref(target_root, origin_ref, fixed_sha)
    results = capture_target(target_root, protocol=protocol)
    _require_exact_capture_samples(results, protocol)
    raw_samples: list[dict[str, object]] = []
    for operation_name, operation in results.items():
        attempts = cast(list[dict[str, object]], operation["attempts"])
        for attempt in attempts:
            samples = cast(list[dict[str, object]], attempt["raw_samples"])
            raw_samples.extend(
                validate_child_payload(sample, target_root, operation_name, protocol)
                for sample in samples
            )
    source_modules = sorted(
        {
            (record["name"], record["path"], record["sha256"])
            for sample in raw_samples
            for record in cast(list[dict[str, str]], sample["source_modules"])
        }
    )
    environment = next(
        cast(dict[str, object], sample["environment"]) for sample in raw_samples
    )
    runtime_file_set = candidate_runtime_file_set(target_root, runtime_files)
    runtime_file_set["status"] = "candidate-only; frozen runtime files supplied"
    _validate_runtime_file_set(runtime_file_set, phase="B1", target_root=target_root)
    authoritative = authoritative_runtime_file_set(target_root, fixed_sha)
    if (
        runtime_file_set["files"] != authoritative["files"]
        or runtime_file_set["sha256"] != authoritative["sha256"]
    ):
        raise ValueError(
            "candidate runtime files must equal the authoritative changes from fixed_sha"
        )
    return {
        "schema": "gwexpy.series_matrix_benchmark.v1",
        "phase": "B1",
        "fixed_origin_ref": origin_ref,
        "fixed_sha": fixed_sha,
        "environment": environment,
        "protocol": asdict(protocol),
        "operations": _frozen_operation_payloads(),
        "imported_gwexpy_source_modules": [
            {"name": name, "path": path, "sha256": sha256}
            for name, path, sha256 in source_modules
        ],
        "results": results,
        "stability_gate": {
            "adoptable": not bool(_evidence_stability_gate({"results": results})),
            "unstable_operations": list(_evidence_stability_gate({"results": results})),
            "rule": "all baseline and candidate operations must be stable before numeric comparison",
        },
        "candidate_evidence": {
            "decision": "pending",
            "issue_637": "not evaluated in B1",
            "runtime_file_set": runtime_file_set,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--capture-b0", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--origin-ref", default="origin/main")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker_main(args.worker, B0_PROTOCOL), sort_keys=True))
        return 0
    if args.capture_b0:
        if args.output is None:
            parser.error("--capture-b0 requires --output")
        write_json(
            args.output,
            capture_b0(args.repo_root.resolve(), origin_ref=args.origin_ref),
        )
        return 0
    parser.error("choose --worker or --capture-b0")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
