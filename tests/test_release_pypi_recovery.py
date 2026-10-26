"""Tests for Owner-bound authorization of a partial PyPI continuation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "verify_release_pypi_recovery.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_release_pypi_recovery_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_owner_recovery_verifier_exists():
    assert SCRIPT.is_file()


def body(
    *,
    version="0.2.6",
    tag="v0.2.6",
    source="a" * 40,
    run_id=12345,
    manifest="b" * 64,
    files=None,
    decision="GO",
):
    files = files or {
        "gwexpy-0.2.6-py3-none-any.whl": "c" * 64,
        "gwexpy-0.2.6.tar.gz": "d" * 64,
    }
    file_record = ",".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return "\n".join(
        [
            "GWEXPY-PYPI-RECOVERY-v1",
            f"version={version}",
            f"tag={tag}",
            f"source_sha={source}",
            f"candidate_run_id={run_id}",
            f"promotion_manifest_sha256={manifest}",
            f"approved_files={file_record}",
            f"decision={decision}",
        ]
    )


def expected():
    return {
        "version": "0.2.6",
        "tag": "v0.2.6",
        "source_sha": "a" * 40,
        "candidate_run_id": 12345,
        "promotion_manifest_sha256": "b" * 64,
        "candidate_files": {
            "gwexpy-0.2.6-py3-none-any.whl": "c" * 64,
            "gwexpy-0.2.6.tar.gz": "d" * 64,
        },
        "missing_files": {"gwexpy-0.2.6.tar.gz": "d" * 64},
    }


def comment(comment_id: int, text: str, author="tatsuki-washimi"):
    return {
        "id": comment_id,
        "user": {"login": author},
        "body": text,
        "created_at": f"2026-10-26T00:00:{comment_id % 60:02d}Z",
    }


def test_exact_release_owner_decision_authorizes_only_candidate_missing_file():
    module = load_module()
    decision = comment(100, body())
    module.validate_recovery_decision(
        [decision], owner_login="tatsuki-washimi", expected=expected()
    )


@pytest.mark.parametrize(
    "comments",
    [
        [],
        [comment(100, body(), author="another-user")],
        [comment(100, body(decision="HOLD"))],
        [comment(100, body(source="e" * 40))],
        [comment(100, body(files={"gwexpy-0.2.6.tar.gz": "0" * 64}))],
    ],
)
def test_missing_or_mismatched_owner_decision_fails_closed(comments):
    module = load_module()
    with pytest.raises(module.RecoveryApprovalError):
        module.validate_recovery_decision(
            comments, owner_login="tatsuki-washimi", expected=expected()
        )


def test_latest_bound_owner_decision_supersedes_an_earlier_hold():
    module = load_module()
    module.validate_recovery_decision(
        [comment(100, body(decision="HOLD")), comment(101, body())],
        owner_login="tatsuki-washimi",
        expected=expected(),
    )


def test_approved_superset_can_resume_after_one_of_two_missing_files_uploaded():
    module = load_module()
    approved = {
        "gwexpy-0.2.6-py3-none-any.whl": "c" * 64,
        "gwexpy-0.2.6.tar.gz": "d" * 64,
    }
    current = expected()
    current["missing_files"] = {"gwexpy-0.2.6.tar.gz": "d" * 64}
    module.validate_recovery_decision(
        [comment(101, body(files=approved))],
        owner_login="tatsuki-washimi",
        expected=current,
    )
