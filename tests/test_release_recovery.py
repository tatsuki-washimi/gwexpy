"""Release Owner recovery authorization must bind the exact published bytes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "verify_release_recovery.py"
)


def load_recovery():
    spec = importlib.util.spec_from_file_location("verify_release_recovery", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record() -> tuple[object, dict[str, str]]:
    recovery = load_recovery()
    expected, owner = recovery.expected_authorization(
        tag="v0.2.6",
        source_sha="a1ff8e2f18ef4e599727d6df6fdff61fce7019bd",
        tag_object_sha="e9f30e6b410d6c599880f741f188f48db4067869",
        recovery_pr=812,
        candidate_run_id=40161240805,
        manifest_sha256="8b517b336d503ebedc1f6ac92070dc73bf9c144b091d01a3b680a80e70f64a8d",
        sdist_sha256="b67f0335e7fa60fa9af8305912dec136ad1cd6d3cea6db0adb8c438837f9a783",
        wheel_sha256="3c1552de056ded5f3f934a30f3454133dcf13374e10b0dac0a7ff8ac3d0c2003",
        zenodo_date_disposition="correct-after-archive",
    )
    return recovery, {"owner": owner, **expected}


def comment(fields: dict[str, str], *, comment_id: int = 1) -> dict[str, object]:
    recovery = load_recovery()
    values = {key: value for key, value in fields.items() if key != "owner"}
    body = (
        recovery.MARKER
        + "\n"
        + "".join(f"{key}={value}\n" for key, value in values.items()).rstrip("\n")
    )
    return {
        "id": comment_id,
        "body": body,
        "created_at": "2026-10-27T00:00:00Z",
        "updated_at": "2026-10-27T00:00:00Z",
        "user": {"login": fields["owner"]},
    }


def test_exact_release_owner_recovery_authorization_is_accepted():
    recovery, fields = record()

    accepted = recovery.validate_recovery_authorization(
        [comment(fields)],
        owner_login=fields["owner"],
        expected={key: value for key, value in fields.items() if key != "owner"},
    )

    assert accepted["publication_decision"] == "GO"
    assert accepted["wheel_sha256"] == fields["wheel_sha256"]


def test_latest_owner_hold_supersedes_an_earlier_recovery_go():
    recovery, fields = record()
    earlier = comment(fields, comment_id=10)
    hold_fields = dict(fields, publication_decision="HOLD")
    hold = comment(hold_fields, comment_id=11)

    with pytest.raises(recovery.RecoveryAuthorizationError, match="is HOLD"):
        recovery.validate_recovery_authorization(
            [earlier, hold],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )

    crlf_hold = comment(hold_fields, comment_id=12)
    crlf_hold["body"] = str(crlf_hold["body"]).replace("\n", "\r\n")
    with pytest.raises(recovery.RecoveryAuthorizationError, match="is HOLD"):
        recovery.validate_recovery_authorization(
            [earlier, crlf_hold],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )

    later_wrong_pr = comment(dict(fields, recovery_pr="813"), comment_id=12)
    with pytest.raises(
        recovery.RecoveryAuthorizationError, match="recovery_pr mismatch"
    ):
        recovery.validate_recovery_authorization(
            [earlier, later_wrong_pr],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )


def test_newer_malformed_or_edited_owner_record_cannot_fall_back_to_an_older_go():
    recovery, fields = record()
    expected = {key: value for key, value in fields.items() if key != "owner"}
    earlier = comment(fields, comment_id=10)

    malformed = comment(fields, comment_id=11)
    malformed["body"] = recovery.MARKER + " trailing text\nversion=0.2.6"
    with pytest.raises(recovery.RecoveryAuthorizationError, match="later.*malformed"):
        recovery.validate_recovery_authorization(
            [earlier, malformed], owner_login=fields["owner"], expected=expected
        )

    edited_without_marker = comment(fields, comment_id=12)
    edited_without_marker["body"] = "withdrawn"
    edited_without_marker["updated_at"] = "2026-10-27T01:00:00Z"
    with pytest.raises(recovery.RecoveryAuthorizationError, match="later.*edited"):
        recovery.validate_recovery_authorization(
            [earlier, edited_without_marker],
            owner_login=fields["owner"],
            expected=expected,
        )


def test_a_new_unedited_go_supersedes_an_older_edited_record():
    recovery, fields = record()
    edited = comment(fields, comment_id=10)
    edited["updated_at"] = "2026-10-27T01:00:00Z"
    latest = comment(fields, comment_id=11)

    accepted = recovery.validate_recovery_authorization(
        [edited, latest],
        owner_login=fields["owner"],
        expected={key: value for key, value in fields.items() if key != "owner"},
    )

    assert accepted["publication_decision"] == "GO"


def test_recovery_authorization_rejects_a_different_distribution_hash():
    recovery, fields = record()
    wrong = dict(fields, wheel_sha256="0" * 64)

    with pytest.raises(
        recovery.RecoveryAuthorizationError, match="wheel_sha256 mismatch"
    ):
        recovery.validate_recovery_authorization(
            [comment(wrong)],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )


def test_recovery_authorization_rejects_wrong_owner_control_code_or_edited_comment():
    recovery, fields = record()
    bad_author = comment(fields)
    bad_author["user"] = {"login": "someone-else"}
    with pytest.raises(recovery.RecoveryAuthorizationError, match="no matching"):
        recovery.validate_recovery_authorization(
            [bad_author],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )

    bad_control = dict(fields, control_ref="refs/tags/v0.2.6-other-control")
    with pytest.raises(
        recovery.RecoveryAuthorizationError, match="control_ref mismatch"
    ):
        recovery.validate_recovery_authorization(
            [comment(bad_control)],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )

    bad_tag_object = dict(fields, tag_object_sha="3" * 40)
    with pytest.raises(recovery.RecoveryAuthorizationError, match="no matching"):
        recovery.validate_recovery_authorization(
            [comment(bad_tag_object)],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )

    edited = comment(fields)
    edited["updated_at"] = "2026-10-27T01:00:00Z"
    with pytest.raises(recovery.RecoveryAuthorizationError, match="was edited"):
        recovery.validate_recovery_authorization(
            [edited],
            owner_login=fields["owner"],
            expected={key: value for key, value in fields.items() if key != "owner"},
        )


def test_recovery_authorization_rejects_noncanonical_record():
    recovery = load_recovery()
    with pytest.raises(recovery.RecoveryAuthorizationError, match="canonical"):
        recovery.parse_recovery_authorization(
            recovery.MARKER + "\nversion=0.2.6\n\nextra"
        )


def test_recovery_record_requires_an_explicit_valid_zenodo_disposition():
    recovery, fields = record()
    invalid = dict(fields, zenodo_date_disposition="guess-the-publication-date")
    with pytest.raises(recovery.RecoveryAuthorizationError, match="invalid Zenodo"):
        recovery.parse_recovery_authorization(
            recovery.MARKER
            + "\n"
            + "".join(
                f"{key}={value}\n" for key, value in invalid.items() if key != "owner"
            ).rstrip("\n")
        )


def test_recovery_control_tag_must_be_the_approved_main_pr_merge():
    recovery = load_recovery()
    pull = {
        "number": 812,
        "merged": True,
        "merged_at": "2026-10-27T01:00:00Z",
        "merge_commit_sha": "4" * 40,
        "base": {
            "ref": "main",
            "repo": {"full_name": "tatsuki-washimi/gwexpy"},
        },
    }

    recovery.validate_recovery_pull_request(
        pull,
        files=[{"filename": path} for path in recovery.REQUIRED_RECOVERY_PR_PATHS],
        repository="tatsuki-washimi/gwexpy",
        number=812,
        control_sha="4" * 40,
    )
    with pytest.raises(
        recovery.RecoveryAuthorizationError, match="approved main merge"
    ):
        recovery.validate_recovery_pull_request(
            {**pull, "merge_commit_sha": "5" * 40},
            files=[{"filename": path} for path in recovery.REQUIRED_RECOVERY_PR_PATHS],
            repository="tatsuki-washimi/gwexpy",
            number=812,
            control_sha="4" * 40,
        )
    with pytest.raises(
        recovery.RecoveryAuthorizationError, match="approved main merge"
    ):
        recovery.validate_recovery_pull_request(
            {**pull, "base": {**pull["base"], "ref": "release"}},
            files=[{"filename": path} for path in recovery.REQUIRED_RECOVERY_PR_PATHS],
            repository="tatsuki-washimi/gwexpy",
            number=812,
            control_sha="4" * 40,
        )


def test_pypi_recovery_control_pr_may_contain_the_scoped_repair_paths():
    recovery = load_recovery()
    pull = {
        "number": 836,
        "merged": True,
        "merged_at": "2026-10-27T02:00:00Z",
        "merge_commit_sha": "6" * 40,
        "base": {
            "ref": "main",
            "repo": {"full_name": "tatsuki-washimi/gwexpy"},
        },
    }
    repair_paths = {
        ".github/workflows/publish-release.yml",
        "RELEASING.md",
        "docs/developers/plans/manifests/audit-manifest-v0.2.6-release-recovery.yaml",
        "scripts/ci/release_contract.py",
        "scripts/ci/release_contracts.json",
        "scripts/ci/verify_release_recovery.py",
        "tests/test_publish_release_workflow.py",
        "tests/test_release_contracts.py",
        "tests/test_release_recovery.py",
    }
    assert recovery.REQUIRED_RECOVERY_PYPI_REPAIR_PATHS <= repair_paths

    recovery.validate_recovery_pull_request(
        pull,
        files=[{"filename": path} for path in repair_paths],
        repository="tatsuki-washimi/gwexpy",
        number=836,
        control_sha="6" * 40,
    )


def test_pypi_recovery_control_pr_requires_the_scoped_guard_and_workflow_paths():
    recovery = load_recovery()
    pull = {
        "number": 836,
        "merged": True,
        "merged_at": "2026-10-27T02:00:00Z",
        "merge_commit_sha": "6" * 40,
        "base": {
            "ref": "main",
            "repo": {"full_name": "tatsuki-washimi/gwexpy"},
        },
    }

    with pytest.raises(recovery.RecoveryAuthorizationError, match="required"):
        recovery.validate_recovery_pull_request(
            pull,
            files=[{"filename": ".github/workflows/publish-release.yml"}],
            repository="tatsuki-washimi/gwexpy",
            number=836,
            control_sha="6" * 40,
        )

    with pytest.raises(recovery.RecoveryAuthorizationError, match="omits required"):
        recovery.validate_recovery_pull_request(
            pull,
            files=[{"filename": "README.md"}],
            repository="tatsuki-washimi/gwexpy",
            number=836,
            control_sha="6" * 40,
        )


def test_release_readback_recovery_pr_may_contain_its_scoped_repair_paths():
    recovery = load_recovery()
    pull = {
        "number": 837,
        "merged": True,
        "merged_at": "2026-10-27T02:30:00Z",
        "merge_commit_sha": "7" * 40,
        "base": {
            "ref": "main",
            "repo": {"full_name": "tatsuki-washimi/gwexpy"},
        },
    }
    repair_paths = {
        ".github/workflows/publish-release.yml",
        "RELEASING.md",
        "docs/developers/plans/manifests/audit-manifest-v0.2.6-release-recovery.yaml",
        "scripts/ci/release_contract.py",
        "scripts/ci/release_contracts.json",
        "scripts/ci/verify_release_recovery.py",
        "tests/test_publish_release_workflow.py",
        "tests/test_release_contracts.py",
        "tests/test_release_recovery.py",
    }
    assert recovery.REQUIRED_RECOVERY_RELEASE_READBACK_REPAIR_PATHS <= repair_paths

    recovery.validate_recovery_pull_request(
        pull,
        files=[{"filename": path} for path in repair_paths],
        repository="tatsuki-washimi/gwexpy",
        number=837,
        control_sha="7" * 40,
    )
