"""Preflight wrapper tests (existing validators only)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "release_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("release_preflight", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preparation_runs_release_contract_only():
    mod = load_module()
    payload = mod.run_preflight(ROOT, "v0.2.8", "preparation")
    names = [item["name"] for item in payload["results"]]
    assert names == ["release_contract_review_evidence_path"]
    assert payload["results"][0]["exit_code"] == 0


def test_cli_json_preparation():
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--repo-root",
            str(ROOT),
            "--release-tag",
            "v0.2.8",
            "--phase",
            "preparation",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["phase"] == "preparation"
    assert payload["results"][0]["exit_code"] == 0
