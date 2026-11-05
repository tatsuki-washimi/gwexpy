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
    inputs = mod.PreflightInputs(repo_root=ROOT, release_tag="v0.2.8")
    payload = mod.run_preflight(inputs, "preparation")
    assert payload["summary"] == {"executed": 1, "not_evaluated": 0, "executed_failures": 0}
    step = payload["results"][0]
    assert step["disposition"] == "executed"
    assert step["exit_code"] == 0


def test_source_without_inputs_is_not_evaluated_not_success():
    mod = load_module()
    inputs = mod.PreflightInputs(repo_root=ROOT, release_tag="v0.2.8")
    payload = mod.run_preflight(inputs, "source")
    assert payload["summary"]["not_evaluated"] >= 1
    assert payload["summary"]["executed_failures"] == 0
    assert payload["results"][0]["disposition"] == "not_evaluated"
    assert mod.exit_code_for_payload(payload) == 0


def test_prepublish_without_inputs_does_not_run_help():
    mod = load_module()
    inputs = mod.PreflightInputs(repo_root=ROOT, release_tag="v0.2.8")
    payload = mod.run_preflight(inputs, "prepublish")
    for step in payload["results"]:
        assert step["disposition"] == "not_evaluated"
        argv = step.get("argv")
        assert argv is None


def test_source_phase_uses_source_review_validator_phase(monkeypatch):
    mod = load_module()
    captured: list[list[str]] = []

    def fake_run(argv, *, cwd):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "source_review_status=READY\n", "")

    monkeypatch.setattr(mod, "_run", fake_run)
    inputs = mod.PreflightInputs(
        repo_root=ROOT,
        release_tag="v0.2.6",
        review_evidence=ROOT / "docs/developers/plans/manifests/audit-manifest-v0.2.6-release-readiness.yaml",
        reviewed_commit="a" * 40,
    )
    payload = mod.run_preflight(inputs, "source")
    evidence_argv = payload["results"][0]["argv"]
    assert evidence_argv is not None
    phase_index = evidence_argv.index("--phase")
    assert evidence_argv[phase_index + 1] == "source-review"
    assert "--phase" in captured[0]


def test_source_review_phase_accepts_empty_human_approval_fixture(tmp_path):
    """Regression: source-review must validate pre-approval evidence, not candidate rules."""
    from tests.test_v026_review_evidence_v2 import _fixture, _validate

    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    result = _validate(validator, repo, contract, source_sha, path, phase="source-review")
    assert data["human_approval"]["verdict"] == ""
    assert result["source_sha"] == source_sha


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
    assert payload["schema"] == "gwexpy-release-preflight-v2"
    assert payload["results"][0]["disposition"] == "executed"
