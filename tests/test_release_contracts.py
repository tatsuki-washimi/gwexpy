"""Contracts for versioned, fail-closed release-control configuration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "scripts" / "ci" / "release_contracts.json"
LOADER_PATH = ROOT / "scripts" / "ci" / "release_contract.py"
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


def test_release_contracts_cover_v0113_regression_and_v0114_lane() -> None:
    assert CONTRACTS_PATH.is_file()
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))

    assert data["schema"] == "gwexpy-release-contracts-v1"
    assert set(data["releases"]) == {"v0.1.13", "v0.1.14"}

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


def test_release_contract_loader_rejects_unknown_tags() -> None:
    contracts = load_contract_module()

    with pytest.raises(contracts.ReleaseContractError, match="unsupported release tag"):
        contracts.release_contract("v0.1.15")


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


def test_v0114_manifest_is_a_fail_closed_review_evidence_placeholder() -> None:
    assert V0114_MANIFEST.is_file()
    source = V0114_MANIFEST.read_text(encoding="utf-8")

    assert source.count("review_evidence_json:") == 1
    assert '"schema": "gwexpy-v0114-review-evidence-v1"' in source
    assert '"entries": []' in source
