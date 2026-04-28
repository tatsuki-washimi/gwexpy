from __future__ import annotations

import importlib.util
import stat
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


def _write_wheel_symlink(path: Path, member: str) -> None:
    info = zipfile.ZipInfo(member)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "target")


def _write_sdist(path: Path, members: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            payload = b""
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info)


def _write_sdist_symlink(path: Path, member: str) -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
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


def test_release_artifact_hygiene_rejects_package_internal_tests(
    tmp_path: Path,
):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ["gwexpy/histogram/tests/test_histogram_statistical.py"],
    )
    _write_sdist(
        dist / "gwexpy-0.1.1.tar.gz",
        ["gwexpy-0.1.1/gwexpy/histogram/tests/test_histogram_statistical.py"],
    )

    problems = module.check_artifacts(dist)

    assert any("gwexpy/histogram/tests" in problem for problem in problems)


def test_release_artifact_hygiene_rejects_package_internal_agent_docs(
    tmp_path: Path,
):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ["gwexpy/gui/AGENTS.md"],
    )
    _write_sdist(
        dist / "gwexpy-0.1.1.tar.gz",
        ["gwexpy-0.1.1/gwexpy/gui/CLAUDE.md"],
    )

    problems = module.check_artifacts(dist)

    assert any("AGENTS.md" in problem for problem in problems)
    assert any("CLAUDE.md" in problem for problem in problems)


def test_release_artifact_hygiene_rejects_symlink_members(tmp_path: Path):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel_symlink(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        "gwexpy/gui/reference-dtt",
    )
    _write_sdist_symlink(
        dist / "gwexpy-0.1.1.tar.gz",
        "gwexpy-0.1.1/gwexpy/gui/reference_ndscope",
    )

    problems = module.check_artifacts(dist)

    assert any("reference-dtt is a symlink" in problem for problem in problems)
    assert any("reference_ndscope is a symlink" in problem for problem in problems)


def test_release_artifact_hygiene_reports_forbidden_symlink_once(tmp_path: Path):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel_symlink(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ".agent/tmp/reference",
    )
    _write_sdist_symlink(
        dist / "gwexpy-0.1.1.tar.gz",
        "gwexpy-0.1.1/.agent/tmp/reference",
    )

    problems = module.check_artifacts(dist)

    assert problems == [
        "gwexpy-0.1.1-py3-none-any.whl:.agent/tmp/reference is a symlink",
        "gwexpy-0.1.1.tar.gz:gwexpy-0.1.1/.agent/tmp/reference is a symlink",
    ]


def test_release_artifact_hygiene_rejects_generated_artifacts(tmp_path: Path):
    module = load_script_module()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "gwexpy-0.1.1-py3-none-any.whl",
        ["gwexpy/.ipynb_checkpoints/notebook.ipynb"],
    )
    _write_sdist(
        dist / "gwexpy-0.1.1.tar.gz",
        ["gwexpy-0.1.1/htmlcov/index.html", "gwexpy-0.1.1/.DS_Store"],
    )

    problems = module.check_artifacts(dist)

    assert any(".ipynb_checkpoints" in problem for problem in problems)
    assert any("htmlcov" in problem for problem in problems)
    assert any(".DS_Store" in problem for problem in problems)
