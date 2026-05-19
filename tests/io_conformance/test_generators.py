"""Tests for deterministic IO conformance generators."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.io_conformance import conftest as guard_conftest
from tests.io_conformance.generators import GENERATOR_SPECS, iter_generator_specs

_FROZEN_GENERATOR_IDS = ("csv_txt", "audio", "hdf5", "hdf_ndscope", "gwf")


def _file_tree(base_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path.relative_to(base_dir) for path in base_dir.rglob("*") if path.is_file())
    )


def _read_tree(base_dir: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(base_dir): path.read_bytes()
        for path in sorted(base_dir.rglob("*"))
        if path.is_file()
    }


def test_generator_registry_is_frozen_and_stable() -> None:
    specs = iter_generator_specs()
    assert specs == GENERATOR_SPECS
    assert tuple(spec.name for spec in specs) == _FROZEN_GENERATOR_IDS
    assert tuple(spec.entrypoint for spec in specs) == ("generate",) * len(specs)
    assert tuple(spec.module_name for spec in specs) == tuple(
        f"tests.io_conformance.generators.{generator_id}"
        for generator_id in _FROZEN_GENERATOR_IDS
    )


@pytest.mark.parametrize("spec", iter_generator_specs(), ids=lambda spec: spec.name)
def test_generators_are_deterministic_and_confined(tmp_path: Path, spec) -> None:
    module = importlib.import_module(spec.module_name)
    entrypoint = getattr(module, spec.entrypoint)

    run_a = tmp_path / f"{spec.name}_a"
    run_b = tmp_path / f"{spec.name}_b"
    result_a = entrypoint(run_a)
    result_b = entrypoint(run_b)

    assert isinstance(result_a, dict)
    assert isinstance(result_b, dict)
    assert set(result_a) == set(result_b)

    for output in (run_a, run_b):
        assert output.exists()
        for artifact in output.rglob("*"):
            if artifact.is_file():
                assert artifact.is_relative_to(output)

    for result, output in ((result_a, run_a), (result_b, run_b)):
        for value in result.values():
            artifact_path = Path(value)
            assert artifact_path.is_absolute()
            assert artifact_path.is_relative_to(output)

    assert _file_tree(run_a) == _file_tree(run_b)
    if spec.name == "gwf":
        assert (run_a / "manifest.json").read_bytes() == (
            run_b / "manifest.json"
        ).read_bytes()
        return
    assert _read_tree(run_a) == _read_tree(run_b)


@pytest.mark.parametrize("spec", iter_generator_specs(), ids=lambda spec: spec.name)
def test_generator_source_passes_import_guard(spec) -> None:
    guard_conftest._guard_generator_source(spec)
