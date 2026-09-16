"""Tests for workflow_notebooks.json manifest integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"
MANIFEST_PATH = DOCS / "workflow_notebooks.json"


def _load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_schema_and_uniqueness() -> None:
    manifest = _load_manifest()
    assert manifest.get("schema_version") == 1
    assert isinstance(manifest.get("notebooks"), list)

    seen_ids = set()
    seen_public = set()
    valid_groups = {"core", "fitting", "control"}

    for entry in manifest["notebooks"]:
        for key in (
            "id",
            "public",
            "canonical",
            "group",
            "cell_timeout_seconds",
            "required_outputs",
        ):
            assert key in entry, f"Entry missing key {key}: {entry}"

        assert entry["id"] not in seen_ids, f"Duplicate id: {entry['id']}"
        seen_ids.add(entry["id"])

        assert entry["public"] not in seen_public, (
            f"Duplicate public: {entry['public']}"
        )
        seen_public.add(entry["public"])

        assert entry["group"] in valid_groups, f"Invalid group: {entry['group']}"
        assert isinstance(entry["cell_timeout_seconds"], int)
        assert entry["cell_timeout_seconds"] > 0
        assert isinstance(entry["required_outputs"], list)


def test_manifest_active_notebooks_exist_and_match_canonical() -> None:
    manifest = _load_manifest()
    for entry in manifest["notebooks"]:
        public_path = DOCS / entry["public"]
        canonical_path = ROOT / entry["canonical"]

        assert public_path.exists(), f"Public notebook not found: {public_path}"
        assert canonical_path.exists(), (
            f"Canonical notebook not found: {canonical_path}"
        )

        pub_data = json.loads(public_path.read_text(encoding="utf-8"))
        can_data = json.loads(canonical_path.read_text(encoding="utf-8"))

        pub_code = [
            c for c in pub_data.get("cells", []) if c.get("cell_type") == "code"
        ]
        can_code = [
            c for c in can_data.get("cells", []) if c.get("cell_type") == "code"
        ]

        assert len(pub_code) == len(can_code), (
            f"Code cell count mismatch for {entry['id']}: "
            f"public={len(pub_code)}, canonical={len(can_code)}"
        )

        for i, (pc, cc) in enumerate(zip(pub_code, can_code)):
            p_src = "".join(pc.get("source", []))
            c_src = "".join(cc.get("source", []))
            assert p_src == c_src, f"Code mismatch in cell {i} for {entry['id']}"

        # Check clean source
        for cell in pub_data.get("cells", []):
            assert cell.get("id"), f"Missing cell ID in {public_path}"
            if cell.get("cell_type") == "code":
                assert cell.get("outputs") == [], f"Unstripped outputs in {public_path}"
                assert cell.get("execution_count") is None, (
                    f"Execution count present in {public_path}"
                )
