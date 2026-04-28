from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_release_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_release_artifacts", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")


def _write_sdist(path: Path, members: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            payload = b""
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info)


def test_release_artifact_hygiene_accepts_clean_pair(tmp_path: Path):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ["gwexpy/__init__.py", "gwexpy-0.1.1.dist-info/METADATA"],
    )
    _write_sdist(
        dist / "gwexpy-0.1.1.tar.gz",
        ["gwexpy-0.1.1/pyproject.toml", "gwexpy-0.1.1/gwexpy/__init__.py"],
    )

    assert module.check_artifacts(dist) == []


def test_release_artifact_hygiene_rejects_internal_trees_and_bytecode(
    tmp_path: Path,
):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ["gwexpy/__pycache__/__init__.cpython-311.pyc"],
    )
    _write_sdist(
        dist / "gwexpy-0.1.1.tar.gz",
        ["gwexpy-0.1.1/docs_internal/notes.md", "gwexpy-0.1.1/.agent/tmp/log"],
    )

    problems = module.check_artifacts(dist)

    assert any("__pycache__" in problem for problem in problems)
    assert any("docs_internal" in problem for problem in problems)
    assert any(".agent" in problem for problem in problems)


def test_release_artifact_hygiene_requires_single_wheel_and_sdist(tmp_path: Path):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()

    problems = module.check_artifacts(dist)

    assert "Expected exactly one wheel" in problems[0]
    assert "Expected exactly one sdist" in problems[1]
