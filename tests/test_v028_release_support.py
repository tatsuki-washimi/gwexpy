"""Fail-closed release validation for v0.2.8."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_release.py"
CONTRACTS_PATH = ROOT / "scripts" / "ci" / "release_contracts.json"


def v028_contract() -> dict[str, object]:
    document = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    return document["releases"]["v0.2.8"]


def test_v028_contract_has_frozen_review_base_and_three_lanes() -> None:
    contract = v028_contract()
    assert contract["review_base_sha"] == "bcf6e0c98b01249b742124676330d8f98a3adab2"
    assert contract["review_evidence_schema"] == "gwexpy-v028-review-evidence-v2"
    assert contract["protected_refs"] == ["main"]
    assert set(contract["review_lanes"]) == {
        "documentation",
        "release-security",
        "scientific-data-model",
    }
    assert contract["s_to_r_allowed_paths"] == [
        contract["plan_path"],
        contract["review_evidence_path"],
    ]


def test_v028_requires_review_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "validate_release_v028", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    contract = v028_contract()
    if "issue_number" not in contract["promotion"]["release_go"]:
        # Exercise this gate while the dedicated release tracking issue is pending.
        monkeypatch.setattr(validator, "_release_contract", lambda tag: contract)

    with pytest.raises(
        validator.ReleaseValidationError,
        match="v0.2.8 requires release review evidence",
    ):
        validator.validate_release(ROOT, "0" * 40, "v0.2.8")
