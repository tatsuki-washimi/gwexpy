"""Contracts for versioned, fail-closed release-control configuration."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "scripts" / "ci" / "release_contracts.json"
LOADER_PATH = ROOT / "scripts" / "ci" / "release_contract.py"
V023_IMPLEMENTATION_BASE = "a8085b71446d3ef3417a7e5b5ac8efb156368eac"
V023_PLAN = (
    ROOT
    / "docs"
    / "developers"
    / "plans"
    / "20260902_v0.2.3_gwpy_behavioral_compatibility.md"
)
V0114_MANIFEST = (
    ROOT
    / "docs"
    / "developers"
    / "plans"
    / "manifests"
    / "audit-manifest-v0.1.14-release-readiness.yaml"
)


def load_contract_module():
    spec = importlib.util.spec_from_file_location("release_contract", LOADER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_commit_available(revision: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _candidate_revision() -> str:
    """Select the PR head instead of GitHub's synthetic merge commit when present."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        try:
            event = json.loads(Path(event_path).read_text(encoding="utf-8"))
            revision = event["pull_request"]["head"]["sha"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            pass
        else:
            if (
                isinstance(revision, str)
                and len(revision) == 40
                and all(character in "0123456789abcdef" for character in revision)
                and _git_commit_available(revision)
            ):
                return revision
    return "HEAD"


def _review_scope_covers(path: str, scope: set[str]) -> bool:
    return any(path == entry or path.startswith(f"{entry}/") for entry in scope)


def _changed_paths_between(repo: Path, base: str, candidate: str) -> set[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            f"{base}..{candidate}",
            "--",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    raw_paths = result.stdout
    if not isinstance(raw_paths, bytes) or (
        raw_paths and not raw_paths.endswith(b"\0")
    ):
        raise AssertionError("git diff did not return canonical NUL-delimited bytes")
    try:
        paths = [
            raw_path.decode("utf-8", errors="strict")
            for raw_path in raw_paths.split(b"\0")
        ]
    except UnicodeDecodeError as exc:
        raise AssertionError("git diff returned a non-UTF-8 path") from exc
    if paths[-1] != "" or any(not path for path in paths[:-1]):
        raise AssertionError("git diff returned a malformed path list")
    return set(paths[:-1])


def test_release_contracts_cover_frozen_releases_and_v023_lane() -> None:
    assert CONTRACTS_PATH.is_file()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))

    assert data["schema"] == "gwexpy-release-contracts-v1"
    assert set(data["releases"]) == {
        "v0.1.13",
        "v0.1.14",
        "v0.2.0",
        "v0.2.2",
        "v0.2.3",
    }

    v0113 = data["releases"]["v0.1.13"]
    assert v0113["plan_path"] == (
        "docs/plans/2026-08-06-v0.1.13-sol-no-go-followup-plan.md"
    )
    assert v0113["review_evidence_path"].endswith(
        "audit-manifest-v0.1.13-sol-followup.yaml"
    )
    assert v0113["review_evidence_schema"] == ("gwexpy-v0113-review-evidence-v1")
    assert set(v0113["review_lanes"]) == {"A", "B"}
    assert v0113["artifact_prefix"] == "v0113-integration-evidence"
    assert v0113["protected_refs"] == ["main", "maint/0.1"]

    v0114 = data["releases"]["v0.1.14"]
    assert v0114["plan_path"] == "docs/plans/2026-08-08-v0114-release-plan.md"
    assert v0114["review_evidence_path"].endswith(
        "audit-manifest-v0.1.14-release-readiness.yaml"
    )
    assert v0114["review_evidence_schema"] == ("gwexpy-v0114-review-evidence-v1")
    assert set(v0114["review_lanes"]) == {
        "scientific",
        "release-security",
        "completion",
    }
    assert v0114["s_to_r_allowed_paths"] == [
        "docs/developers/plans/manifests/audit-manifest-v0.1.14-release-readiness.yaml",
        "docs/plans/2026-08-08-v0114-release-plan.md",
    ]
    assert v0114["artifact_prefix"] == "v0114-integration-evidence"
    assert v0114["protected_refs"] == ["main", "maint/0.1"]
    assert (
        "tests/docs/test_docs_redesign_public_content.py"
        in v0114["review_lanes"]["scientific"]
    )

    scope_union = {path for paths in v0114["review_lanes"].values() for path in paths}
    assert {
        "docs/web/en/user_guide/io_formats.md",
        "docs/web/ja/user_guide/io_formats.md",
        "docs_redesign/about/changelog.md",
        "docs_redesign/how-to/io_formats.md",
        "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po",
        "docs_redesign/locales/ja/LC_MESSAGES/how-to/io_formats.po",
        "gwexpy/frequencyseries/io/dttxml.py",
        "gwexpy/gui/loaders/loaders.py",
        "requirements-dev.txt",
        "tests/docs/test_docs_redesign_public_content.py",
        "tests/docs/test_docs_redesign_release_facts.py",
        "tests/test_release_gate_workflow_contract.py",
        "tests/test_run_gate_junit_contract.py",
        "tests/timeseries",
    } <= scope_union

    v020 = data["releases"]["v0.2.0"]
    assert v020["protected_refs"] == ["main", "maint/0.2"]

    v022 = data["releases"]["v0.2.2"]
    assert v022["plan_path"].endswith(
        "20260901_v0.2.2_gwpy_behavioral_compatibility.md"
    )
    assert v022["review_evidence_path"].endswith(
        "audit-manifest-v0.2.2-release-readiness.yaml"
    )
    assert set(v022["review_lanes"]) == {
        "scientific-compatibility",
        "performance",
        "release-security",
    }
    assert v022["artifact_prefix"] == "v022-integration-evidence"
    assert v022["protected_refs"] == ["main", "maint/0.2"]

    v023 = data["releases"]["v0.2.3"]
    assert v023["plan_path"] == (
        "docs/developers/plans/20260902_v0.2.3_gwpy_behavioral_compatibility.md"
    )
    assert v023["review_evidence_path"] == (
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-release-readiness.yaml"
    )
    assert v023["review_evidence_schema"] == "gwexpy-v023-review-evidence-v1"
    assert v023["payload_schema"] == "gwexpy-v023-release-payload-v1"
    assert v023["integration_evidence_schema"] == (
        "gwexpy-v023-integration-evidence-v1"
    )
    assert set(v023["review_lanes"]) == {
        "data-model",
        "release-security",
        "scientific-compatibility",
    }
    assert {
        ".github/workflows/publish-release.yml",
        "scripts/ci",
        "tests/test_publish_release_workflow.py",
        "tests/test_qualification_skip_evidence.py",
        "tests/test_release_contracts.py",
    } <= set(v023["review_lanes"]["release-security"])
    assert {"gwexpy/fields", "gwexpy/types", "tests/fields", "tests/types"} <= set(
        v023["review_lanes"]["data-model"]
    )
    assert {
        ".github/workflows/test-compat-gwpy.yml",
        "docs/developers/contracts/public_io_contract.json",
        "docs/developers/contracts/public_io_contract.md",
        "gwexpy/analysis/coupling_result.py",
        "gwexpy/frequencyseries",
        "gwexpy/spectrogram",
        "gwexpy/time",
        "gwexpy/timeseries",
        "scripts/audit_gwpy_overrides.py",
        "tests/io/test_csv_txt_contract.py",
        "tests/io/test_gwpy_csv_phase4_compat.py",
        "tests/io/test_gwpy_hdf5_compat.py",
        "tests/io/test_gwpy_override_terminal_io.py",
        "tests/io/test_hdf5_timeseries_family.py",
        "tests/io/test_io_contract.py",
        "tests/io/test_io_docs_contract_sync.py",
        "tests/io/test_reader_start_end_contract.py",
        "tests/io_conformance/test_contract_schema_v3.py",
        "tests/io_conformance/test_read_conformance.py",
        "tests/test_compatibility_fixes.py",
        "tests/test_gwpy_constructor_terminal_compat.py",
        "tests/test_gwpy_override_inventory.py",
        "tests/test_import_order.py",
    } <= set(v023["review_lanes"]["scientific-compatibility"])
    assert v023["s_to_r_allowed_paths"] == [
        "docs/developers/plans/20260902_v0.2.3_gwpy_behavioral_compatibility.md",
        "docs/developers/plans/manifests/audit-manifest-v0.2.3-release-readiness.yaml",
    ]
    assert v023["artifact_prefix"] == "v023-integration-evidence"
    assert v023["protected_refs"] == ["main", "maint/0.2"]
    assert "qualification_profile" not in v023


def test_v023_plan_limits_hdf5_private_augmentation_to_named_markers() -> None:
    plan = V023_PLAN.read_text(encoding="utf-8")

    for marker in (
        "`_gwexpy_sidecar_json_v1`",
        "`_gwexpy_sidecar_json_v2`",
        "dataset の `epoch` attribute",
    ):
        assert marker in plan
    assert (
        "これら以外の dataset/group topology、logical path、native attributes "
        "は、GWpy oracle と一致させる。"
    ) in plan


def test_v023_review_lanes_cover_every_fixed_base_candidate_change() -> None:
    if not _git_commit_available(V023_IMPLEMENTATION_BASE):
        pytest.fail(
            "fixed v0.2.3 implementation base is unavailable; "
            "the authoritative review-scope gate requires fetch-depth: 0"
        )
    candidate = _candidate_revision()
    changed_paths = _changed_paths_between(ROOT, V023_IMPLEMENTATION_BASE, candidate)
    contracts = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    lanes = contracts["releases"]["v0.2.3"]["review_lanes"]
    scope = {path for paths in lanes.values() for path in paths}

    uncovered = sorted(
        path for path in changed_paths if not _review_scope_covers(path, scope)
    )
    assert uncovered == []


def test_v023_signoff_report_is_in_all_same_candidate_review_lanes() -> None:
    contracts = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    lanes = contracts["releases"]["v0.2.3"]["review_lanes"]
    signoff_report = (
        "docs/developers/reports/"
        "report_v0.2.3_human_scientific_data_model_signoff_20260903.md"
    )

    assert {lane for lane, paths in lanes.items() if signoff_report in paths} == {
        "data-model",
        "release-security",
        "scientific-compatibility",
    }


def test_review_scope_diff_preserves_both_paths_of_a_rename(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Qualification Test")
    git("config", "user.email", "qualification@example.invalid")
    old_path = repo / "outside-scope.txt"
    old_path.write_text("same content\n", encoding="utf-8")
    git("add", "--all")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")

    new_path = repo / "covered" / "inside-scope.txt"
    new_path.parent.mkdir()
    old_path.rename(new_path)
    git("add", "--all")
    git("commit", "-q", "-m", "rename")

    changed_paths = _changed_paths_between(repo, base, "HEAD")

    assert changed_paths == {"outside-scope.txt", "covered/inside-scope.txt"}
    assert sorted(
        path for path in changed_paths if not _review_scope_covers(path, {"covered"})
    ) == ["outside-scope.txt"]


def test_review_scope_diff_preserves_newlines_in_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Qualification Test")
    git("config", "user.email", "qualification@example.invalid")
    git("commit", "-q", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")

    changed_path = "covered/line\nbreak.txt"
    path = repo / changed_path
    path.parent.mkdir()
    path.write_text("content\n", encoding="utf-8")
    git("add", "--all")
    git("commit", "-q", "-m", "newline path")

    assert _changed_paths_between(repo, base, "HEAD") == {changed_path}


def test_release_contract_loader_rejects_unknown_tags() -> None:
    contracts = load_contract_module()

    with pytest.raises(contracts.ReleaseContractError, match="unsupported release tag"):
        contracts.release_contract("v0.1.15")


def test_release_contract_loader_rejects_unknown_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = load_contract_module()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    data["schema"] = "gwexpy-release-contracts-v2"
    modified = tmp_path / "release_contracts.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(contracts, "CONTRACT_PATH", modified)

    with pytest.raises(
        contracts.ReleaseContractError, match="invalid release contracts"
    ):
        contracts.release_contract("v0.1.14")


def test_release_contract_loader_returns_defensive_values() -> None:
    contracts = load_contract_module()
    first = contracts.release_contract("v0.1.13")
    first["s_to_r_allowed_paths"].append("unexpected")

    second = contracts.release_contract("v0.1.13")
    assert "unexpected" not in second["s_to_r_allowed_paths"]


def test_release_contract_loader_rejects_unsafe_artifact_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = load_contract_module()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    data["releases"]["v0.1.14"]["artifact_prefix"] = "safe\nartifact_prefix=untrusted"
    modified = tmp_path / "release_contracts.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(contracts, "CONTRACT_PATH", modified)

    with pytest.raises(contracts.ReleaseContractError, match="artifact_prefix"):
        contracts.release_contract("v0.1.14")


@pytest.mark.parametrize(
    "protected_refs",
    [
        ["main"],
        ["main", "main"],
        ["main", "maint/0.2", "maint/0.1"],
        ["main", "../maint/0.2"],
        ["main", "refs/heads/maint/0.2"],
        ["main", "main//evil"],
        ["main", ".hidden"],
        ["main", "main/.hidden"],
        ["main", "main/foo.lock"],
        ["main", "maint/\x00evil"],
        ["main", 2],
    ],
)
def test_release_contract_loader_rejects_invalid_protected_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protected_refs: list[object],
) -> None:
    contracts = load_contract_module()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    data["releases"]["v0.1.14"]["protected_refs"] = protected_refs
    modified = tmp_path / "release_contracts.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(contracts, "CONTRACT_PATH", modified)

    with pytest.raises(contracts.ReleaseContractError, match="protected refs"):
        contracts.release_contract("v0.1.14")


def test_release_contract_loader_rejects_branch_checkout_shorthand(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "checkout-history"
    repo.mkdir()
    for args in (
        ("init", "-b", "main"),
        ("config", "user.name", "Release Test"),
        ("config", "user.email", "release-test@example.invalid"),
        ("commit", "--allow-empty", "-m", "initial"),
        ("branch", "previous"),
        ("checkout", "previous"),
        ("checkout", "main"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    contracts = load_contract_module()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    data["releases"]["v0.1.14"]["protected_refs"] = ["@{-1}", "main"]
    modified = tmp_path / "release_contracts.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(contracts, "CONTRACT_PATH", modified)
    monkeypatch.chdir(repo)

    with pytest.raises(contracts.ReleaseContractError, match="protected refs"):
        contracts.release_contract("v0.1.14")


def test_release_contract_cli_emits_exact_tag_protected_refs() -> None:
    contracts = load_contract_module()

    assert contracts.protected_refs("v0.2.0") == ["main", "maint/0.2"]
    assert contracts.protected_refs("v0.2.2") == ["main", "maint/0.2"]
    assert contracts.protected_refs("v0.2.3") == ["main", "maint/0.2"]


def test_v0114_manifest_is_a_sanitized_review_evidence_container() -> None:
    assert V0114_MANIFEST.is_file()
    source = V0114_MANIFEST.read_text(encoding="utf-8")

    assert source.count("review_evidence_json:") == 1
    assert source.startswith("review_evidence_json: |\n")
    assert '"schema": "gwexpy-v0114-review-evidence-v1"' in source
