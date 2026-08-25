"""Regression contract for Wave 3 HDF5 transaction audit ancestry."""

import subprocess
from pathlib import Path

import pytest
import yaml

_MANIFEST = (
    Path(__file__).parents[2]
    / "docs/developers/plans/manifests/audit-manifest-wave3-hdf5-transaction.yaml"
)
_REPOSITORY = _MANIFEST.parents[4]


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY,
        capture_output=True,
        check=False,
        text=True,
    )
    output = result.stdout.strip()
    assert result.returncode == 0, (
        f"git {' '.join(arguments)} failed in {_REPOSITORY}: "
        f"stdout={output!r}, stderr={result.stderr.strip()!r}"
    )
    assert output, f"git {' '.join(arguments)} returned no output"
    return output


def _assert_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=_REPOSITORY,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 1:
        raise AssertionError(
            f"{ancestor} is not an ancestor of {descendant} in {_REPOSITORY}"
        )
    assert result.returncode == 0, (
        f"git merge-base --is-ancestor {ancestor} {descendant} failed in {_REPOSITORY}: "
        f"stdout={result.stdout.strip()!r}, stderr={result.stderr.strip()!r}"
    )


def test_wave3_hdf5_manifest_has_non_self_referential_evidence_ancestry() -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))

    assert (
        manifest["remediation_base_head"] == "604de8f3b1efb1b910f7bbc484006ccf0570cbd0"
    )
    assert manifest["evidence_test_head"] == "035b3934af238fea119a1d67b8b2c176057cf387"
    assert "current_head" not in manifest
    assert manifest["manifest_revision_resolution"] == (
        "The manifest-containing commit is resolved with git rev-parse HEAD at review time; "
        "it is not embedded as a self-referential field."
    )
    repository_head = _git("rev-parse", "HEAD")
    assert repository_head == _git("rev-parse", "--verify", "HEAD^{commit}")
    assert len(repository_head) == 40 and all(
        character in "0123456789abcdef" for character in repository_head
    )
    _assert_ancestor(manifest["remediation_base_head"], manifest["evidence_test_head"])
    _assert_ancestor(manifest["evidence_test_head"], repository_head)


def test_git_ancestry_helper_reports_a_non_ancestor_commit() -> None:
    with pytest.raises(AssertionError, match="not an ancestor"):
        _assert_ancestor(
            "035b3934af238fea119a1d67b8b2c176057cf387",
            "604de8f3b1efb1b910f7bbc484006ccf0570cbd0",
        )
