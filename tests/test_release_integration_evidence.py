from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "integrate_release_evidence.py"


def test_v027_integration_uses_v2_review_size_profile(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "v027_integrate_release_evidence", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evidence_path = "docs/review.yaml"
    contract = {
        "review_evidence_path": evidence_path,
        "review_evidence_schema": "gwexpy-v027-review-evidence-v2",
    }
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text("{}\n", encoding="utf-8")
    source = tmp_path / evidence_path
    source.parent.mkdir(parents=True)
    source.write_text("review\n", encoding="utf-8")
    captured = {}
    promotion = SimpleNamespace(
        configured_contract=lambda _version: contract,
        candidate_profile=lambda _contract: {
            "evidence_schemas": {
                "gwexpy-integration-evidence-v1": "gwexpy-integration-evidence-v1"
            }
        },
        PromotionManifestError=ValueError,
    )

    def load_review_document(_path, *, allow_v026_size):
        captured["allow_v026_size"] = allow_v026_size
        return SimpleNamespace(
            data={"human_approval": {"comment_id": 123}}, raw_sha256="f" * 64
        )

    review_validator = SimpleNamespace(
        load_review_document=load_review_document,
        ReleaseReviewEvidenceError=ValueError,
    )
    contract_loader = SimpleNamespace(
        review_evidence_profile=lambda _schema: "v2",
        ReleaseContractError=ValueError,
    )
    monkeypatch.setattr(
        module,
        "_load_sibling",
        lambda name: {
            "release_promotion": promotion,
            "validate_release_review_evidence": review_validator,
            "release_contract": contract_loader,
        }[name],
    )

    module.integrate_review_evidence("0.2.7", tmp_path, aggregate)

    assert captured["allow_v026_size"] is True
    result = json.loads(aggregate.read_text(encoding="utf-8"))
    assert result["review_evidence"]["human_approval"]["comment_id"] == 123
