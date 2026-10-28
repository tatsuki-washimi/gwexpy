#!/usr/bin/env python3
"""Bind the configured review approval to the same-run integration aggregate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


class IntegrationEvidenceError(ValueError):
    """Raised when review evidence cannot be bound to the aggregate."""


def _load_sibling(name: str) -> Any:
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrationEvidenceError(f"{name} is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrationEvidenceError(f"duplicate aggregate JSON key: {key}")
        result[key] = value
    return result


def integrate_review_evidence(
    version: str, source_root: Path, aggregate_path: Path
) -> None:
    """Attach the parsed approval and original file digest to the aggregate."""
    promotion = _load_sibling("release_promotion")
    review_validator = _load_sibling("validate_release_review_evidence")
    contract_loader = _load_sibling("release_contract")
    try:
        contract = promotion.configured_contract(version)
        profile = promotion.candidate_profile(contract)
    except promotion.PromotionManifestError as exc:
        raise IntegrationEvidenceError(str(exc)) from exc
    root = source_root.resolve()
    review_path = (root / contract["review_evidence_path"]).resolve()
    try:
        review_path.relative_to(root)
    except ValueError as exc:
        raise IntegrationEvidenceError(
            "review evidence must remain inside the source root"
        ) from exc
    try:
        document = review_validator.load_review_document(
            review_path,
            allow_v026_size=(
                contract_loader.review_evidence_profile(
                    contract["review_evidence_schema"]
                )
                == "v2"
            ),
        )
    except (
        review_validator.ReleaseReviewEvidenceError,
        contract_loader.ReleaseContractError,
    ) as exc:
        raise IntegrationEvidenceError(f"invalid review evidence: {exc}") from exc
    human_approval = document.data.get("human_approval")
    if not isinstance(human_approval, dict):
        raise IntegrationEvidenceError("review evidence has no human approval object")

    try:
        aggregate = json.loads(
            aggregate_path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IntegrationEvidenceError("invalid integration aggregate JSON") from exc
    if not isinstance(aggregate, dict):
        raise IntegrationEvidenceError("integration aggregate must be a JSON object")
    aggregate["schema"] = profile["evidence_schemas"]["gwexpy-integration-evidence-v1"]
    aggregate["review_evidence"] = {
        "path": contract["review_evidence_path"],
        "sha256": document.raw_sha256,
        "human_approval": human_approval,
    }
    aggregate_path.write_text(
        json.dumps(aggregate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        integrate_review_evidence(args.version, args.source_root, args.aggregate)
    except IntegrationEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
