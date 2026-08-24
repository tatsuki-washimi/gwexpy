"""Focused tests for the isolated #676 benchmark protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import scripts.benchmarks.series_matrix_benchmark as benchmark
from scripts.benchmarks.series_matrix_benchmark import (
    B0_PROTOCOL,
    BenchmarkProtocol,
    BenchmarkSample,
    ComparisonDecision,
    adopt_candidate,
    authoritative_runtime_file_set,
    candidate_runtime_file_set,
    compare_candidate,
    geometric_mean_ratio,
    is_stable,
    minimum_iterations,
    next_iterations,
    runtime_file_sha256,
    validate_child_payload,
    verify_imported_source_modules,
    verify_imported_source_paths,
)


def _child_payload(target: Path, *, source_path: str = "gwexpy/__init__.py") -> dict:
    source = target / source_path
    return {
        "operation": "copy",
        "iterations": 1,
        "elapsed_seconds": 0.250,
        "per_operation_seconds": 0.250,
        "rss_bytes": 0,
        "source_modules": [
            {
                "name": "gwexpy",
                "path": source_path,
                "sha256": (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    if source.is_file()
                    else "0" * 64
                ),
            }
        ],
        "environment": {
            "python": "3.12.0",
            "platform": "test",
            "numpy": "2.0.0",
            "astropy": "7.0.0",
            "gwpy": "4.0.0",
            "gwexpy": "0.2.0",
        },
    }


def test_b0_protocol_is_frozen() -> None:
    assert B0_PROTOCOL.warmups == 3
    assert B0_PROTOCOL.child_processes == 7
    assert B0_PROTOCOL.minimum_measurement_seconds == 0.250
    assert B0_PROTOCOL.max_attempts == 3
    assert B0_PROTOCOL.stability_threshold == 0.05


def test_calibration_reaches_the_minimum_duration() -> None:
    assert minimum_iterations(0.050, 0.250) == 5
    assert minimum_iterations(0.250, 0.250) == 1
    with pytest.raises(ValueError):
        minimum_iterations(0.0, 0.250)


def test_under_target_measurement_increases_the_next_batch() -> None:
    assert next_iterations(1, 0.160, 0.250) == 2
    assert next_iterations(10, 0.249, 0.250) == 11
    with pytest.raises(ValueError):
        next_iterations(1, 0.0, 0.250)


def test_stability_uses_mad_relative_to_median() -> None:
    stable = [0.100, 0.101, 0.099, 0.100, 0.100, 0.101, 0.099]
    unstable = [0.050, 0.100, 0.200, 0.100, 0.050, 0.200, 0.100]
    assert is_stable(stable, threshold=0.05)
    assert not is_stable(unstable, threshold=0.05)


def test_comparison_enforces_all_three_candidate_budgets() -> None:
    baseline = {
        "construct": BenchmarkSample(
            operation="construct", timings=(0.100,), rss_bytes=100 * 1024**2
        ),
        "copy": BenchmarkSample(
            operation="copy", timings=(0.100,), rss_bytes=100 * 1024**2
        ),
    }
    candidate = {
        "construct": BenchmarkSample(
            operation="construct", timings=(0.119,), rss_bytes=108 * 1024**2
        ),
        "copy": BenchmarkSample(
            operation="copy", timings=(0.101,), rss_bytes=100 * 1024**2
        ),
    }
    decision = compare_candidate(baseline, candidate)
    assert isinstance(decision, ComparisonDecision)
    assert decision.operation_deltas_us["construct"] == pytest.approx(19_000)
    assert decision.passed is True
    assert decision.geometric_mean_ratio == pytest.approx((1.19 * 1.01) ** 0.5)


def test_comparison_rejects_timing_and_rss_budget_breaches() -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(0.100,), rss_bytes=100 * 1024**2)
    }
    slow = {
        "op": BenchmarkSample(operation="op", timings=(0.121,), rss_bytes=111 * 1024**2)
    }
    decision = compare_candidate(baseline, slow)
    assert not decision.passed
    assert decision.failed_operations == ("op",)


def test_rss_only_failure_exceeds_the_10_percent_budget() -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(0.100,), rss_bytes=100 * 1024**2)
    }
    candidate = {
        "op": BenchmarkSample(
            operation="op", timings=(0.100,), rss_bytes=110 * 1024**2 + 1
        )
    }
    decision = compare_candidate(baseline, candidate)
    assert decision.geometric_mean_ratio == pytest.approx(1.0)
    assert decision.operation_deltas_us["op"] == pytest.approx(0.0)
    assert decision.failed_operations == ("op",)


def test_geometric_mean_only_failure_is_independent_of_per_operation_budget() -> None:
    baseline = {
        "a": BenchmarkSample(operation="a", timings=(1.0,), rss_bytes=1),
        "b": BenchmarkSample(operation="b", timings=(1.0,), rss_bytes=1),
    }
    candidate = {
        "a": BenchmarkSample(operation="a", timings=(1.11,), rss_bytes=1),
        "b": BenchmarkSample(operation="b", timings=(1.11,), rss_bytes=1),
    }
    decision = compare_candidate(baseline, candidate)
    assert decision.geometric_mean_ratio == pytest.approx(1.11)
    assert decision.operation_deltas_us == {
        "a": pytest.approx(110_000),
        "b": pytest.approx(110_000),
    }
    assert decision.failed_operations == ("a", "b")


@pytest.mark.parametrize(
    ("baseline_seconds", "candidate_seconds", "compensating_ratio"),
    [(0.000010, 0.000020, 0.604999), (0.000100, 0.000120, 1.1 / 1.2 - 1e-9)],
)
def test_exact_timing_budget_boundary_is_accepted(
    baseline_seconds: float, candidate_seconds: float, compensating_ratio: float
) -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(baseline_seconds,), rss_bytes=1),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(1.0,), rss_bytes=1
        ),
    }
    candidate = {
        "op": BenchmarkSample(
            operation="op", timings=(candidate_seconds,), rss_bytes=1
        ),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(compensating_ratio,), rss_bytes=1
        ),
    }
    decision = compare_candidate(baseline, candidate)
    assert decision.operation_deltas_us["op"] == pytest.approx(
        max(baseline_seconds * 0.20, 10e-6) * 1_000_000
    )
    assert decision.passed


def test_exact_eight_mib_rss_boundary_is_accepted() -> None:
    baseline = {"op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=1)}
    candidate = {
        "op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=1 + 8 * 1024**2)
    }
    assert compare_candidate(baseline, candidate).passed


def test_timing_budget_equality_is_accepted_but_nextafter_is_not() -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(0.000001,), rss_bytes=1),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(1.0,), rss_bytes=1
        ),
    }
    equal = {
        "op": BenchmarkSample(operation="op", timings=(0.000011,), rss_bytes=1),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(0.1,), rss_bytes=1
        ),
    }
    just_over = {
        "op": BenchmarkSample(
            operation="op",
            timings=(
                math.nextafter(
                    math.nextafter(0.000011, float("inf")),
                    float("inf"),
                ),
            ),
            rss_bytes=1,
        ),
        "compensating": equal["compensating"],
    }
    assert compare_candidate(baseline, equal).passed
    assert not compare_candidate(baseline, just_over).passed


def test_twenty_percent_timing_equality_is_accepted_but_nextafter_is_not() -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(0.001,), rss_bytes=1),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(1.0,), rss_bytes=1
        ),
    }
    equal = {
        "op": BenchmarkSample(operation="op", timings=(0.0012,), rss_bytes=1),
        "compensating": BenchmarkSample(
            operation="compensating", timings=(1 / 1.2,), rss_bytes=1
        ),
    }
    just_over = {
        "op": BenchmarkSample(
            operation="op",
            timings=(math.nextafter(0.0012, float("inf")),),
            rss_bytes=1,
        ),
        "compensating": equal["compensating"],
    }
    assert compare_candidate(baseline, equal).passed
    assert not compare_candidate(baseline, just_over).passed


def test_geometric_mean_equality_is_accepted_but_nextafter_is_not() -> None:
    baseline = {"op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=1)}
    equal = {"op": BenchmarkSample(operation="op", timings=(1.1,), rss_bytes=1)}
    just_over = {
        "op": BenchmarkSample(
            operation="op",
            timings=(__import__("math").nextafter(1.1, float("inf")),),
            rss_bytes=1,
        )
    }
    assert compare_candidate(baseline, equal).passed
    assert not compare_candidate(baseline, just_over).passed


def test_rss_equality_is_accepted_but_nextafter_is_not() -> None:
    baseline = {"op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=1)}
    equal = {
        "op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=1 + 8 * 1024**2)
    }
    just_over = {
        "op": BenchmarkSample(
            operation="op", timings=(1.0,), rss_bytes=1 + 8 * 1024**2 + 1
        )
    }
    assert compare_candidate(baseline, equal).passed
    assert not compare_candidate(baseline, just_over).passed


def test_rss_ten_percent_equality_is_accepted_but_nextafter_is_not() -> None:
    baseline = {
        "op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=100 * 1024**2)
    }
    equal = {
        "op": BenchmarkSample(operation="op", timings=(1.0,), rss_bytes=110 * 1024**2)
    }
    just_over = {
        "op": BenchmarkSample(
            operation="op", timings=(1.0,), rss_bytes=110 * 1024**2 + 1
        )
    }
    assert compare_candidate(baseline, equal).passed
    assert not compare_candidate(baseline, just_over).passed


def test_parent_payload_requires_protocol_duration_and_consistent_rate(
    tmp_path: Path,
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    too_short = _child_payload(tmp_path)
    too_short["elapsed_seconds"] = 0.249
    too_short["per_operation_seconds"] = 0.249
    with pytest.raises(ValueError, match="minimum measurement"):
        validate_child_payload(too_short, tmp_path, "copy", B0_PROTOCOL)
    inconsistent = _child_payload(tmp_path)
    inconsistent["iterations"] = 2
    with pytest.raises(ValueError, match="per-operation"):
        validate_child_payload(inconsistent, tmp_path, "copy", B0_PROTOCOL)


def test_module_provenance_requires_name_to_path_mapping(tmp_path: Path) -> None:
    package = tmp_path / "gwexpy"
    (package / "a").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "a" / "__init__.py").write_text("")
    (package / "a" / "b.py").write_text("")
    valid = verify_imported_source_modules(
        tmp_path,
        (
            {
                "name": "gwexpy",
                "path": "gwexpy/__init__.py",
                "sha256": hashlib.sha256(
                    (package / "__init__.py").read_bytes()
                ).hexdigest(),
            },
            {
                "name": "gwexpy.a.b",
                "path": "gwexpy/a/b.py",
                "sha256": hashlib.sha256(
                    (package / "a" / "b.py").read_bytes()
                ).hexdigest(),
            },
        ),
    )
    assert len(valid) == 2
    invalid = (
        {
            "name": "gwexpy",
            "path": "gwexpy/a/b.py",
            "sha256": "0" * 64,
        },
        {
            "name": "gwexpy.a.b",
            "path": "gwexpy/a/__init__.py",
            "sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        verify_imported_source_modules(tmp_path, invalid)


def test_module_provenance_rejects_duplicate_names_and_paths(tmp_path: Path) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="duplicate"):
        verify_imported_source_modules(
            tmp_path,
            (
                {
                    "name": "gwexpy",
                    "path": "gwexpy/__init__.py",
                    "sha256": hashlib.sha256(
                        (package / "__init__.py").read_bytes()
                    ).hexdigest(),
                },
                {
                    "name": "gwexpy",
                    "path": "gwexpy/__init__.py",
                    "sha256": hashlib.sha256(
                        (package / "__init__.py").read_bytes()
                    ).hexdigest(),
                },
            ),
        )


def test_tracked_b0_summary_is_compact_and_reproducible() -> None:
    repository_root = Path(__file__).parents[2]
    evidence_root = repository_root / "docs" / "plans" / "evidence" / "v0.2.0-b0"
    summary = (evidence_root / "series_matrix_b0_summary.md").read_text(
        encoding="utf-8"
    )

    assert "fixed SHA" in summary
    assert "3 warm-ups" in summary
    assert "7 independent child processes" in summary
    assert "250 ms" in summary
    assert "SHA-256" in summary
    assert "reproduction" in summary
    assert not (evidence_root / "series_matrix_b0.json").exists()


def test_fixed_sha_must_equal_the_declared_origin_ref(tmp_path: Path) -> None:
    candidate_root = _candidate_root(tmp_path)
    fixed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=candidate_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    benchmark._validate_fixed_origin_ref(candidate_root, "origin/main", fixed_sha)

    forged_sha = subprocess.run(
        ["git", "commit-tree", "HEAD^{tree}", "-m", "forged"],
        cwd=candidate_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="fixed_sha|origin ref"):
        benchmark._validate_fixed_origin_ref(candidate_root, "origin/main", forged_sha)


def test_isolated_worktree_uses_system_temporary_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: dict[str, object] = {}
    target = tmp_path / "isolated"

    def fake_mkdtemp(*, prefix: str, dir: str) -> str:
        requested.update(prefix=prefix, dir=dir)
        target.mkdir()
        return str(target)

    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    with benchmark.isolated_worktree(tmp_path, "0" * 40) as isolated:
        assert isolated == target

    assert requested["prefix"] == ".gwexpy-b0-"
    assert requested["dir"] == tempfile.gettempdir()


def test_fixed_origin_ref_must_exist_and_be_unambiguous(tmp_path: Path) -> None:
    candidate_root = _candidate_root(tmp_path)
    fixed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=candidate_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="missing|origin ref"):
        benchmark._validate_fixed_origin_ref(candidate_root, "missing/ref", fixed_sha)

    subprocess.run(
        ["git", "update-ref", "refs/heads/ambiguous", fixed_sha],
        cwd=candidate_root,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/ambiguous", fixed_sha],
        cwd=candidate_root,
        check=True,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        benchmark._validate_fixed_origin_ref(candidate_root, "ambiguous", fixed_sha)


def test_authoritative_runtime_scope_includes_ignored_python_but_not_caches(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    (candidate_root / ".gitignore").write_text(
        "gwexpy/evil.py\ngwexpy/__pycache__/\n*.pyc\n"
    )
    (candidate_root / "gwexpy" / "evil.py").write_text("evil = True\n")
    cache = candidate_root / "gwexpy" / "__pycache__"
    cache.mkdir()
    (cache / "evil.cpython-312.pyc").write_bytes(b"not source")

    fixed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=candidate_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime = authoritative_runtime_file_set(candidate_root, fixed_sha)

    assert runtime["files"] == ["gwexpy/__init__.py", "gwexpy/evil.py"]
    assert all(not path.endswith((".pyc", ".pyo")) for path in runtime["files"])
    assert "gwexpy/__pycache__/evil.cpython-312.pyc" not in runtime["files"]


def test_b1_requires_each_raw_sample_to_match_top_level_module_set(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    extra = candidate_root / "gwexpy" / "extra.py"
    extra.write_text("extra = True\n")
    extra_record: dict[str, object] = {
        "name": "gwexpy.extra",
        "path": "gwexpy/extra.py",
        "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
    }
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    top_level_modules = cast(
        list[dict[str, object]], evidence["imported_gwexpy_source_modules"]
    )
    top_level_modules.append(deepcopy(extra_record))
    results = cast(dict[str, object], evidence["results"])
    for raw_result in results.values():
        attempts = cast(
            list[dict[str, object]], cast(dict[str, object], raw_result)["attempts"]
        )
        for attempt in attempts:
            for sample in cast(list[dict[str, object]], attempt["raw_samples"]):
                sample_modules = cast(list[dict[str, object]], sample["source_modules"])
                sample_modules.append(deepcopy(extra_record))
    runtime_files = candidate_runtime_file_set(
        candidate_root, ("gwexpy/__init__.py", "gwexpy/extra.py")
    )
    runtime_files["status"] = "candidate-only"
    cast(dict[str, object], evidence["candidate_evidence"])["runtime_file_set"] = (
        runtime_files
    )

    first_result = cast(dict[str, object], results["copy"])
    first_attempt = cast(list[dict[str, object]], first_result["attempts"])[0]
    first_sample = cast(list[dict[str, object]], first_attempt["raw_samples"])[0]
    first_sample_modules = cast(list[dict[str, object]], first_sample["source_modules"])
    first_sample_modules.pop()

    with pytest.raises(ValueError, match="every raw-sample module|module set"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


@pytest.mark.parametrize("mutation", ["added", "forged", "duplicated", "mismatched"])
def test_b1_rejects_any_per_sample_provenance_deviation(
    tmp_path: Path, mutation: str
) -> None:
    candidate_root = _candidate_root(tmp_path)
    extra = candidate_root / "gwexpy" / "extra.py"
    extra.write_text("extra = True\n")
    extra_record: dict[str, object] = {
        "name": "gwexpy.extra",
        "path": "gwexpy/extra.py",
        "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
    }
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    results = cast(dict[str, object], evidence["results"])
    copy_result = cast(dict[str, object], results["copy"])
    attempts = cast(list[dict[str, object]], copy_result["attempts"])
    first_attempt = attempts[0]
    raw_samples = cast(list[dict[str, object]], first_attempt["raw_samples"])
    sample = raw_samples[0]
    modules = cast(list[dict[str, object]], sample["source_modules"])
    if mutation == "added":
        modules.append(extra_record)
    elif mutation == "forged":
        modules[0]["sha256"] = "0" * 64
    elif mutation == "duplicated":
        modules.append(deepcopy(modules[0]))
    else:
        modules[0]["path"] = "gwexpy/extra.py"

    with pytest.raises(ValueError, match="source|unique|module"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


def test_runtime_file_set_rejects_duplicate_entries(tmp_path: Path) -> None:
    path = tmp_path / "gwexpy" / "matrix.py"
    path.parent.mkdir()
    path.write_text("")
    with pytest.raises(ValueError, match="duplicate"):
        runtime_file_sha256(tmp_path, ("gwexpy/matrix.py", "gwexpy/matrix.py"))


def test_b1_runtime_file_schema_rejects_empty_files_with_forged_sha() -> None:
    runtime_file_set = {
        "files": [],
        "sha256": "0" * 64,
        "status": "candidate-only; frozen runtime files supplied",
    }
    with pytest.raises(ValueError, match="non-empty"):
        benchmark._validate_runtime_file_set(runtime_file_set, phase="B1")


def test_candidate_runtime_file_set_requires_a_nonempty_unique_frozen_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gwexpy" / "matrix.py"
    path.parent.mkdir()
    path.write_text("class Matrix: pass\n")

    with pytest.raises(ValueError, match="non-empty"):
        candidate_runtime_file_set(tmp_path, ())
    with pytest.raises(ValueError, match="unique"):
        candidate_runtime_file_set(tmp_path, ("gwexpy/matrix.py", "gwexpy/matrix.py"))


def test_b1_runtime_file_schema_rejects_sha_not_matching_target_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gwexpy" / "matrix.py"
    path.parent.mkdir()
    path.write_text("class Matrix: pass\n")
    runtime_file_set = candidate_runtime_file_set(tmp_path, ("gwexpy/matrix.py",))
    runtime_file_set["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not match"):
        benchmark._validate_runtime_file_set(
            {**runtime_file_set, "status": "candidate-only"},
            phase="B1",
            target_root=tmp_path,
        )


def _evidence_with_stability(
    stability: str,
    *,
    phase: str = "B1",
    sample_count: int = 7,
    protocol: dict[str, object] | None = None,
    candidate_root: Path | None = None,
) -> dict[str, object]:
    if protocol is None:
        protocol = {
            "warmups": 3,
            "child_processes": 7,
            "minimum_measurement_seconds": 0.250,
            "max_attempts": 3,
            "stability_threshold": 0.05,
        }
    source_record = {"name": "gwexpy", "path": "gwexpy/__init__.py"}
    if phase == "B1":
        source_record["sha256"] = (
            hashlib.sha256(
                (candidate_root / "gwexpy" / "__init__.py").read_bytes()
            ).hexdigest()
            if candidate_root is not None
            else "0" * 64
        )
    results = {}
    for definition in benchmark.OPERATIONS:
        operation_stability = (
            "unstable"
            if stability == "unstable" and definition.name == "slice"
            else "stable"
        )
        timings = (
            [0.25, 0.25, 1.0, 1.0, 1.0, 10.0, 10.0]
            if operation_stability == "unstable"
            else [1.0] * sample_count
        )
        samples = [
            {
                "operation": definition.name,
                "iterations": 1,
                "elapsed_seconds": timing,
                "per_operation_seconds": timing,
                "rss_bytes": 1,
                "source_modules": [deepcopy(source_record)],
                "environment": {
                    "python": "3.12.0",
                    "platform": "test",
                    "numpy": "2.0.0",
                    "astropy": "7.0.0",
                    "gwpy": "4.0.0",
                    "gwexpy": "0.2.0",
                },
            }
            for timing in (timings if sample_count == 7 else [1.0] * sample_count)
        ]
        actual_sample = BenchmarkSample(
            definition.name,
            tuple(timings if sample_count == 7 else [1.0] * sample_count),
            1,
        )
        results[definition.name] = {
            "stability": operation_stability,
            "attempts": [
                {
                    "attempt": 1,
                    "stable": is_stable(
                        actual_sample.timings,
                        threshold=B0_PROTOCOL.stability_threshold,
                    ),
                    "median_seconds": actual_sample.median_seconds,
                    "mad_seconds": actual_sample.mad_seconds,
                    "raw_samples": samples,
                }
            ],
        }
    files: list[str] = [] if phase == "B0" else ["gwexpy/__init__.py"]
    runtime_file_set: dict[str, object] = {
        "files": files,
        "sha256": None if phase == "B0" else "0" * 64,
        "status": "candidate-only",
    }
    if phase == "B1" and candidate_root is not None:
        runtime_file_set["sha256"] = runtime_file_sha256(candidate_root, files)
    return {
        "schema": "gwexpy.series_matrix_benchmark.v1",
        "phase": phase,
        "fixed_origin_ref": "origin/main",
        "fixed_sha": (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=candidate_root, text=True
            ).strip()
            if candidate_root is not None
            else "0" * 40
        ),
        "protocol": protocol,
        "operations": benchmark._frozen_operation_payloads(),
        "environment": {
            "python": "3.12.0",
            "platform": "test",
            "numpy": "2.0.0",
            "astropy": "7.0.0",
            "gwpy": "4.0.0",
            "gwexpy": "0.2.0",
        },
        "imported_gwexpy_source_modules": [deepcopy(source_record)],
        "results": results,
        "stability_gate": {
            "adoptable": stability == "stable",
            "unstable_operations": [] if stability == "stable" else ["slice"],
            "rule": "all baseline and candidate operations must be stable before numeric comparison",
        },
        "candidate_evidence": {
            "decision": "pending",
            "issue_637": f"not evaluated in {phase}",
            "runtime_file_set": runtime_file_set,
        },
    }


def _candidate_root(tmp_path: Path) -> Path:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("candidate = False\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "gwexpy/__init__.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=tmp_path,
        check=True,
    )
    (package / "__init__.py").write_text("candidate = True\n")
    return tmp_path


def test_evidence_recomputes_every_attempt_summary_from_raw_samples(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    operation = cast(dict[str, object], evidence["results"])["copy"]
    assert isinstance(operation, dict)
    attempt = cast(list[dict[str, object]], operation["attempts"])[0]
    timings = [0.25, 0.25, 1.0, 1.0, 1.0, 10.0, 10.0]
    samples = cast(list[dict[str, object]], attempt["raw_samples"])
    for sample, timing in zip(samples, timings):
        sample["elapsed_seconds"] = timing
        sample["per_operation_seconds"] = timing
    attempt["stable"] = True
    attempt["median_seconds"] = 0.25
    attempt["mad_seconds"] = 0.0
    operation["stability"] = "stable"

    with pytest.raises(ValueError, match="median|MAD|stability"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


def test_evidence_recomputes_aggregate_stability_from_all_attempts(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    operation = cast(dict[str, object], evidence["results"])["copy"]
    assert isinstance(operation, dict)
    first = cast(list[dict[str, object]], operation["attempts"])[0]
    second = deepcopy(first)
    second["attempt"] = 2
    for sample, timing in zip(
        cast(list[dict[str, object]], second["raw_samples"]),
        [0.25, 0.25, 1.0, 1.0, 1.0, 10.0, 10.0],
    ):
        sample["elapsed_seconds"] = timing
        sample["per_operation_seconds"] = timing
    second["stable"] = False
    second["median_seconds"] = 1.0
    second["mad_seconds"] = 0.75
    operation["attempts"] = [first, second]
    operation["stability"] = "stable"

    with pytest.raises(ValueError, match="stability"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


def test_b1_runtime_files_must_equal_all_tracked_and_untracked_runtime_changes(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    untracked = candidate_root / "gwexpy" / "untracked.py"
    untracked.write_text("untracked = True\n")
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )

    with pytest.raises(ValueError, match="runtime file set|authoritative"):
        adopt_candidate(
            _evidence_with_stability(
                "stable", phase="B0", candidate_root=candidate_root
            ),
            evidence,
            candidate_root=candidate_root,
        )


def test_b1_fixed_sha_must_be_a_commit_in_the_measured_candidate_tree(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    evidence["fixed_sha"] = "0" * 40

    with pytest.raises(ValueError, match="fixed_sha|base"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


def test_b1_provenance_rejects_forged_hashes_in_top_level_and_raw_samples(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    top_level_modules = cast(
        list[dict[str, object]], evidence["imported_gwexpy_source_modules"]
    )
    top_level_modules[0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash|source"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )

    evidence = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    results = cast(dict[str, object], evidence["results"])
    copy_result = cast(dict[str, object], results["copy"])
    attempts = cast(list[dict[str, object]], copy_result["attempts"])
    raw_samples = cast(list[dict[str, object]], attempts[0]["raw_samples"])
    sample = cast(dict[str, object], raw_samples[0])
    sample_modules = cast(list[dict[str, object]], sample["source_modules"])
    sample_modules[0]["path"] = "gwexpy/stale.py"
    with pytest.raises(ValueError, match="source|existing|path"):
        benchmark._validate_evidence(
            evidence, expected_phase="B1", target_root=candidate_root
        )


def test_unstable_operation_blocks_adoption_before_numeric_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_numeric_comparison_is_reached(*args, **kwargs):
        raise AssertionError("numeric comparison must not run")

    monkeypatch.setattr(
        benchmark, "compare_candidate", fail_if_numeric_comparison_is_reached
    )
    candidate_root = _candidate_root(tmp_path)
    decision = adopt_candidate(
        _evidence_with_stability("stable", phase="B0", candidate_root=candidate_root),
        _evidence_with_stability("unstable", phase="B1", candidate_root=candidate_root),
        candidate_root=candidate_root,
    )
    assert not decision.passed
    assert decision.stability_gate_passed is False
    assert decision.unstable_operations == ("slice",)


def test_adoption_rejects_b1_empty_runtime_file_set_even_with_forged_sha(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    candidate = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    candidate_evidence = cast(dict[str, object], candidate["candidate_evidence"])
    candidate_evidence["runtime_file_set"] = {
        "files": [],
        "sha256": "f" * 64,
        "status": "candidate-only; frozen runtime files supplied",
    }
    with pytest.raises(ValueError, match="non-empty"):
        adopt_candidate(
            _evidence_with_stability(
                "stable", phase="B0", candidate_root=candidate_root
            ),
            candidate,
            candidate_root=candidate_root,
        )


def test_adoption_requires_an_authoritative_candidate_root() -> None:
    with pytest.raises(TypeError, match="candidate_root"):
        adopt_candidate(
            _evidence_with_stability("stable", phase="B0"),
            _evidence_with_stability("stable", phase="B1"),
        )  # type: ignore[call-arg]


def test_adoption_recomputes_candidate_sha_before_stability_or_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_root = _candidate_root(tmp_path)
    candidate = _evidence_with_stability(
        "stable", phase="B1", candidate_root=candidate_root
    )
    candidate_evidence = cast(dict[str, object], candidate["candidate_evidence"])
    runtime_file_set = cast(dict[str, object], candidate_evidence["runtime_file_set"])
    runtime_file_set["sha256"] = "0" * 64

    original_stability_gate = benchmark._evidence_stability_gate
    stability_calls = 0

    def fail_if_candidate_stability_is_checked(evidence):
        nonlocal stability_calls
        stability_calls += 1
        if stability_calls > 1:
            raise AssertionError("stability must not run before SHA verification")
        return original_stability_gate(evidence)

    monkeypatch.setattr(
        benchmark,
        "_evidence_stability_gate",
        fail_if_candidate_stability_is_checked,
    )
    with pytest.raises(ValueError, match="does not match"):
        adopt_candidate(
            _evidence_with_stability(
                "stable", phase="B0", candidate_root=candidate_root
            ),
            candidate,
            candidate_root=candidate_root,
        )


def test_stable_evidence_reaches_numeric_comparison(tmp_path: Path) -> None:
    candidate_root = _candidate_root(tmp_path)
    decision = adopt_candidate(
        _evidence_with_stability("stable", phase="B0", candidate_root=candidate_root),
        _evidence_with_stability("stable", phase="B1", candidate_root=candidate_root),
        candidate_root=candidate_root,
    )
    assert decision.passed
    assert decision.stability_gate_passed is True


@pytest.mark.parametrize("sample_count", [1, 6, 8])
def test_adoption_rejects_evidence_without_exactly_seven_samples(
    tmp_path: Path,
    sample_count: int,
) -> None:
    candidate_root = _candidate_root(tmp_path)
    with pytest.raises(ValueError, match="exactly 7"):
        adopt_candidate(
            _evidence_with_stability(
                "stable", phase="B0", candidate_root=candidate_root
            ),
            _evidence_with_stability(
                "stable",
                phase="B1",
                sample_count=sample_count,
                candidate_root=candidate_root,
            ),
            candidate_root=candidate_root,
        )


def test_adoption_rejects_custom_protocol(tmp_path: Path) -> None:
    custom_protocol: dict[str, object] = {
        "warmups": 2,
        "child_processes": 7,
        "minimum_measurement_seconds": 0.250,
        "max_attempts": 3,
        "stability_threshold": 0.05,
    }
    candidate_root = _candidate_root(tmp_path)
    with pytest.raises(ValueError, match="frozen benchmark protocol"):
        adopt_candidate(
            _evidence_with_stability(
                "stable", phase="B0", candidate_root=candidate_root
            ),
            _evidence_with_stability(
                "stable",
                phase="B1",
                protocol=custom_protocol,
                candidate_root=candidate_root,
            ),
            candidate_root=candidate_root,
        )


def test_capture_b0_rejects_custom_protocol(tmp_path: Path) -> None:
    custom_protocol = BenchmarkProtocol(
        warmups=2,
        child_processes=7,
        minimum_measurement_seconds=0.250,
        max_attempts=3,
        stability_threshold=0.05,
    )
    with pytest.raises(ValueError, match="frozen benchmark protocol"):
        benchmark.capture_b0(tmp_path, protocol=custom_protocol)


def test_capture_candidate_freezes_protocol_and_runtime_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _candidate_root(tmp_path)
    fixed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *args, **kwargs: fixed_sha,
    )
    monkeypatch.setattr(
        benchmark,
        "capture_target",
        lambda target, protocol: _capture_results(
            target, source_path="gwexpy/__init__.py"
        ),
    )

    evidence = benchmark.capture_candidate(
        tmp_path, runtime_files=("gwexpy/__init__.py",)
    )

    assert evidence["phase"] == "B1"
    assert isinstance(evidence["fixed_sha"], str)
    assert evidence["protocol"] == benchmark.asdict(benchmark.B0_PROTOCOL)
    candidate_evidence = cast(dict[str, object], evidence["candidate_evidence"])
    assert candidate_evidence["issue_637"] == "not evaluated in B1"
    assert cast(dict[str, object], candidate_evidence["runtime_file_set"])["files"] == [
        "gwexpy/__init__.py"
    ]


def test_capture_candidate_rejects_custom_protocol(tmp_path: Path) -> None:
    custom_protocol = BenchmarkProtocol(
        warmups=2,
        child_processes=7,
        minimum_measurement_seconds=0.250,
        max_attempts=3,
        stability_threshold=0.05,
    )
    with pytest.raises(ValueError, match="frozen benchmark protocol"):
        benchmark.capture_candidate(tmp_path, protocol=custom_protocol)


def test_child_payload_is_schema_validated_and_target_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    completed = SimpleNamespace(
        stderr="", stdout=json.dumps(_child_payload(tmp_path)), returncode=0
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: completed)

    result = benchmark._run_child(tmp_path, "copy", B0_PROTOCOL)

    assert result["source_modules"] == [
        {
            "name": "gwexpy",
            "path": "gwexpy/__init__.py",
            "sha256": hashlib.sha256(
                (tmp_path / "gwexpy" / "__init__.py").read_bytes()
            ).hexdigest(),
        }
    ]


@pytest.mark.parametrize(
    "source_path", ["/etc/passwd", "gwexpy/missing.py", "../outside.py"]
)
def test_child_payload_rejects_forged_nonexistent_or_nonisolated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_path: str
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    completed = SimpleNamespace(
        stderr="",
        stdout=json.dumps(_child_payload(tmp_path, source_path=source_path)),
        returncode=0,
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises((TypeError, ValueError)):
        benchmark._run_child(tmp_path, "copy", B0_PROTOCOL)


def test_capture_target_schema_validates_every_child_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(
        benchmark,
        "OPERATIONS",
        (benchmark.OperationDefinition("copy", "copy", (1,)),),
    )
    monkeypatch.setattr(
        benchmark,
        "_run_child",
        lambda target, operation, protocol: _child_payload(target),
    )
    protocol = BenchmarkProtocol(
        warmups=0,
        child_processes=2,
        minimum_measurement_seconds=0.001,
        max_attempts=1,
        stability_threshold=0.05,
    )

    result = benchmark.capture_target(tmp_path, protocol=protocol)

    assert result["copy"]["stability"] == "stable"


def test_capture_target_rejects_a_forged_child_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(
        benchmark,
        "OPERATIONS",
        (benchmark.OperationDefinition("copy", "copy", (1,)),),
    )
    monkeypatch.setattr(
        benchmark,
        "_run_child",
        lambda target, operation, protocol: _child_payload(
            target, source_path="gwexpy/missing.py"
        ),
    )
    protocol = BenchmarkProtocol(
        warmups=0,
        child_processes=1,
        minimum_measurement_seconds=0.001,
        max_attempts=1,
        stability_threshold=0.05,
    )

    with pytest.raises(ValueError):
        benchmark.capture_target(tmp_path, protocol=protocol)


def _capture_results(target: Path, *, source_path: str) -> dict[str, dict[str, object]]:
    samples = [_child_payload(target, source_path=source_path) for _ in range(7)]
    return {
        "copy": {
            "attempts": [
                {
                    "attempt": 1,
                    "stable": True,
                    "median_seconds": 0.250,
                    "mad_seconds": 0.0,
                    "raw_samples": samples,
                }
            ],
            "stability": "stable",
        }
    }


def test_capture_b0_validates_source_modules_and_emits_relative_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _candidate_root(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "OPERATIONS",
        (benchmark.OperationDefinition("copy", "copy", (1,)),),
    )
    monkeypatch.setattr(
        benchmark,
        "capture_target",
        lambda target, protocol: _capture_results(
            target, source_path="gwexpy/__init__.py"
        ),
    )

    evidence = benchmark.capture_b0(tmp_path)

    assert isinstance(evidence["fixed_sha"], str)
    assert evidence["imported_gwexpy_source_modules"] == [
        {
            "name": "gwexpy",
            "path": "gwexpy/__init__.py",
        }
    ]
    assert all(
        "sha256" not in record for record in evidence["imported_gwexpy_source_modules"]
    )
    results = cast(dict[str, object], evidence["results"])
    for raw_result in results.values():
        attempts = cast(
            list[dict[str, object]], cast(dict[str, object], raw_result)["attempts"]
        )
        for attempt in attempts:
            for sample in cast(list[dict[str, object]], attempt["raw_samples"]):
                sample_modules = cast(list[dict[str, object]], sample["source_modules"])
                assert all("sha256" not in record for record in sample_modules)

    output_path = tmp_path.parent / "captured-b0.json"
    benchmark.write_json(output_path, evidence)
    persisted = json.loads(output_path.read_text())
    assert persisted == evidence
    benchmark._validate_evidence(persisted, expected_phase="B0", target_root=tmp_path)
    assert str(tmp_path) not in json.dumps(evidence)


def test_capture_b0_rejects_forged_source_module_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "gwexpy"
    package.mkdir()
    (package / "__init__.py").write_text("")

    @contextmanager
    def selected_target(*args, **kwargs):
        yield tmp_path

    monkeypatch.setattr(benchmark, "isolated_worktree", selected_target)
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *args, **kwargs: "fixed-sha\n",
    )
    monkeypatch.setattr(
        benchmark,
        "capture_target",
        lambda target, protocol: _capture_results(target, source_path="/etc/passwd"),
    )

    with pytest.raises(ValueError):
        benchmark.capture_b0(tmp_path)


def test_geometric_mean_ratio_is_deterministic() -> None:
    assert geometric_mean_ratio((1.0, 1.21)) == pytest.approx(1.1)
    with pytest.raises(ValueError):
        geometric_mean_ratio(())


def test_imported_source_paths_must_be_exclusive_to_target_tree(tmp_path: Path) -> None:
    target = tmp_path / "target"
    package = target / "gwexpy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    assert verify_imported_source_paths(target, (package / "__init__.py",)) == (
        "gwexpy/__init__.py",
    )
    with pytest.raises(ValueError, match="outside target"):
        verify_imported_source_paths(target, (Path("/tmp/not-target/gwexpy.py"),))


def test_runtime_file_hash_is_order_independent_and_path_safe(tmp_path: Path) -> None:
    first = tmp_path / "gwexpy" / "a.py"
    second = tmp_path / "gwexpy" / "b.py"
    first.parent.mkdir()
    first.write_text("a = 1\n")
    second.write_text("b = 2\n")
    one = runtime_file_sha256(tmp_path, ("gwexpy/a.py", "gwexpy/b.py"))
    two = runtime_file_sha256(tmp_path, ("gwexpy/b.py", "gwexpy/a.py"))
    assert one == two
    assert str(tmp_path) not in json.dumps({"sha256": one})


def test_candidate_runtime_file_set_records_relative_files_and_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gwexpy" / "matrix.py"
    path.parent.mkdir()
    path.write_text("class Matrix: pass\n")
    evidence = candidate_runtime_file_set(tmp_path, ("gwexpy/matrix.py",))
    assert evidence == {
        "files": ["gwexpy/matrix.py"],
        "sha256": runtime_file_sha256(tmp_path, ("gwexpy/matrix.py",)),
    }


def test_b1_decision_defers_composition_without_a_target_version() -> None:
    repository_root = Path(__file__).parents[2]
    decision = (
        repository_root
        / "docs"
        / "plans"
        / "evidence"
        / "v0.2.0-b1"
        / "series_matrix_b1_decision.md"
    ).read_text(encoding="utf-8")

    assert "D21 selects B0 for v0.2.0" in decision
    assert "adopted: false" in decision
    assert "No target version" in decision
    assert "does not implement general direct NumPy ufunc composition" in decision
