"""Guard DOC-2: local absolute paths are limited to the reviewed exception ledger.

The ledger pins each excepted file by SHA-256, so an excepted audit original cannot be
edited silently, and any new local path in a tracked file fails this test.
"""

import csv
import hashlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    REPO_ROOT / "docs/developers/plans/manifests/v026-doc2-local-path-exceptions.tsv"
)
# Split so that this file does not itself contain the searched string.
NEEDLE = "/home" + "/washimi"


def _ledger_rows():
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _tracked_hits():
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("requires a git checkout")
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "grep",
            "-l",
            "-z",
            "-F",
            NEEDLE,
            "--",
            ".",
            ":!*.zip",
        ],
        capture_output=True,
    )
    assert proc.returncode in (0, 1), proc.stderr.decode("utf-8", "replace")
    return {path for path in proc.stdout.decode("utf-8").split("\0") if path}


def test_local_path_hits_are_exactly_the_reviewed_exceptions():
    excepted = {row["path"] for row in _ledger_rows() if row["category"] != "E-ZIP"}
    assert _tracked_hits() == excepted


def test_reviewed_exceptions_match_pinned_hashes():
    mismatches = []
    for row in _ledger_rows():
        digest = hashlib.sha256((REPO_ROOT / row["path"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            mismatches.append(row["path"])
    assert mismatches == []
