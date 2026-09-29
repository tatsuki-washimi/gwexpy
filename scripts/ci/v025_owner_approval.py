"""Canonical owner approval scope for the v0.2.5 source freeze."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

DISPOSITION_PATH = (
    "docs/developers/reports/v0.2.5-s3-prequalification/disposition-proposal.md"
)
EXCEPTIONS = (
    "#585-SDB-CONCURRENT-WAL-SNAPSHOT",
    "#589-NATIVE-DTTXML-UNSELECTED-PAYLOAD",
)
DISPOSITIONS = (
    ("ATS-TRUNC-001", "RETAINED_PARTIAL_SALVAGE"),
    ("GBD-COUNT-001", "RETAINED_COUNT_TRUNCATION"),
    ("GBD-LEGACY-001", "OUTSIDE_GL500_SCOPE"),
    (
        "HDF5-DISCOVERY-TS-DICT-GROUP-001",
        "INAPPLICABLE_PLAIN_GROUP_CONTROL",
    ),
    ("HDF5-HIST-DATASET-001", "INAPPLICABLE_PHYSICAL_DATASET_LAYOUT"),
    ("NC-MATRIX-008", "INAPPLICABLE_FIXTURE"),
    ("NC-MATRIX-009", "INAPPLICABLE_INDEPENDENT_AXIS"),
    ("OBSPY-DUP-BASE-001", "EXPECTED_MISSING_DEPENDENCY"),
    ("TDMS-TIME-007", "RETAINED_ROOT_DATETIME_FALLBACK"),
    ("TDMS-TIME-008", "RETAINED_RELATIVE_ZERO"),
    ("TDMS-UNIT-001", "RETAINED_UNIT_PROVENANCE_LIMIT"),
    ("ZARR-DTYPE-003", "RETAINED_DTYPE_WIDENING"),
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def disposition_document_sha256(repo_root: Path, reviewed_commit: str) -> str:
    """Hash the exact disposition document in reviewed source S."""
    if SHA40.fullmatch(reviewed_commit) is None:
        raise ValueError("invalid reviewed source SHA")
    entry = subprocess.run(
        ["git", "ls-tree", "-z", reviewed_commit, "--", DISPOSITION_PATH],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if (
        entry.returncode
        or not entry.stdout.startswith(b"100644 blob ")
        or entry.stdout.count(b"\0") != 1
        or not entry.stdout.endswith(b"\t" + DISPOSITION_PATH.encode() + b"\0")
    ):
        raise ValueError("reviewed disposition document is not a regular file")
    result = subprocess.run(
        ["git", "show", f"{reviewed_commit}:{DISPOSITION_PATH}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("reviewed disposition document is unavailable")
    return hashlib.sha256(result.stdout).hexdigest()


def canonical_comment_lines(
    reviewed_commit: str,
    scientific_scope_digest: str,
    disposition_digest: str,
) -> list[str]:
    """Bind each exception and disposition to one immutable owner comment."""
    if (
        SHA40.fullmatch(reviewed_commit) is None
        or SHA256.fullmatch(scientific_scope_digest) is None
        or SHA256.fullmatch(disposition_digest) is None
    ):
        raise ValueError("invalid approval binding")
    return [
        "GWEXPY-RELEASE-APPROVAL v0.2.5",
        f"S: {reviewed_commit}",
        f"SCOPE: {scientific_scope_digest}",
        f"DISPOSITION-DOCUMENT-SHA256: {disposition_digest}",
        *(f"EXCEPTION: {name} APPROVED" for name in EXCEPTIONS),
        *(
            f"DISPOSITION: {finding_id} {decision}"
            for finding_id, decision in DISPOSITIONS
        ),
        "VERDICT: APPROVED",
    ]
