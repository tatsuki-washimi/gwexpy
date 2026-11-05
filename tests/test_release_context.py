"""Read-only release context helper tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "release_context.py"


def load_module():
    spec = importlib.util.spec_from_file_location("release_context", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_contract_presence_for_v028():
    mod = load_module()
    ctx = mod.build_context(ROOT, release_tag="v0.2.8", tracking_issue=None, post_release_pr=None)
    assert ctx["release_contracts"]["state"] == "verified"
    assert ctx["release_contracts"]["value"] is True
    assert ctx["workflows"]["publish_release_yml"]["value"] is True


def test_v029_registration_observed_without_judging_readiness():
    mod = load_module()
    ctx = mod.build_context(ROOT, release_tag="v0.2.9", tracking_issue=None, post_release_pr=None)
    registered = ctx["release_contracts"]["value"]
    assert ctx["release_contracts"]["state"] == "verified"
    assert isinstance(registered, bool)


def test_open_post_release_pr_is_verified_not_unknown():
    """Regression: completed post-release work must not be reported as unknown when gh shows OPEN."""
    mod = load_module()
    gh_payload = json.dumps(
        {"number": 865, "state": "OPEN", "title": "post-release", "url": "https://example/pr/865"}
    )

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "pr"]:
            return subprocess.CompletedProcess(cmd, 0, gh_payload, "")
        return subprocess.CompletedProcess(cmd, 1, "", "not found")

    with mock.patch("subprocess.run", side_effect=fake_run):
        ctx = mod.build_context(
            ROOT, release_tag="v0.2.8", tracking_issue=None, post_release_pr=865
        )
    pr = ctx["github"]["post_release_pr"]
    assert pr["state"] == "verified"
    assert pr["value"]["state"] == "OPEN"


def test_cli_emits_json(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--repo-root", str(repo)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "gwexpy-release-context-v1"
    assert payload["git"]["head"]["state"] == "verified"
