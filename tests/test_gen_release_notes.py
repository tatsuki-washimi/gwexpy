"""Contract tests for tools/gen_release_notes.py (9 items).

TDD: these tests are written BEFORE the single-version CLI is implemented.
They define the exact contract the new --version flag must satisfy.

Test list
---------
1. v0.1.11 golden  — byte-identical with existing release_notes/v0.1.11.md
2. only-target     — only the target version file changes; others unchanged
3. unreleased-rejected — specifying "Unreleased" as version fails (exit != 0)
4. missing-version     — requesting a non-existent version fails
5. duplicate-heading   — CHANGELOG with duplicate version heading fails
6. boundary-accuracy   — body ends exactly at the next heading boundary
7. crlf-normalization  — CRLF input produces LF output
8. trailing-newline    — output always ends with exactly one LF
9. idempotent          — running twice produces no diff
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "gen_release_notes.py"
RELEASE_NOTES_DIR = Path(__file__).resolve().parents[1] / "release_notes"


def _run(
    args: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def _load_module():
    """Import gen_release_notes as a module (for unit-level function tests)."""
    spec = importlib.util.spec_from_file_location("gen_release_notes", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_changelog(
    versions: list[tuple[str, str]], *, include_unreleased: bool = False
) -> str:
    """Build a minimal CHANGELOG.md text for testing.

    Each entry in *versions* is (version_str, body_text).
    The sections are emitted newest-first as Keep-a-Changelog requires.
    """
    parts = ["# Changelog\n\n"]
    if include_unreleased:
        parts.append("## Unreleased\n\nSome unreleased change.\n\n")
    for version, body in versions:
        # Generate a fake date to satisfy SECTION_RE
        parts.append(f"## [{version}] - 2026-01-01\n\n{body}\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Test 1: v0.1.11 golden — byte-identical with tracked file
# ---------------------------------------------------------------------------


def test_golden_v0_1_11(tmp_path: Path) -> None:
    """--version 0.1.11 must produce a file byte-identical to the tracked golden."""
    golden = RELEASE_NOTES_DIR / "v0.1.11.md"
    assert golden.exists(), f"Golden file missing: {golden}"

    out_dir = tmp_path / "release_notes"
    out_dir.mkdir()
    # Copy all existing files so the script can overwrite in-place safely
    for f in RELEASE_NOTES_DIR.glob("*.md"):
        (out_dir / f.name).write_bytes(f.read_bytes())

    # Copy CHANGELOG into tmp_path so the script resolves ROOT correctly
    changelog_src = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    (tmp_path / "CHANGELOG.md").write_bytes(changelog_src.read_bytes())

    result = _run(["--version", "0.1.11", "--root", str(tmp_path)])
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    produced = (out_dir / "v0.1.11.md").read_bytes()
    expected = golden.read_bytes()
    assert produced == expected, (
        "Output is not byte-identical with the golden file.\n"
        f"Produced length: {len(produced)}, expected: {len(expected)}"
    )


# ---------------------------------------------------------------------------
# Test 2: only the target version file is changed; others are untouched
# ---------------------------------------------------------------------------


def test_only_target_version_changed(tmp_path: Path) -> None:
    """Running --version X.Y.Z must not alter any other release_notes/*.md file."""
    changelog_text = _minimal_changelog(
        [
            ("0.1.2", "Second release body."),
            ("0.1.1", "First release body."),
            ("0.1.0", "Initial release body."),
        ]
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    out_dir = tmp_path / "release_notes"
    out_dir.mkdir()

    sentinel = b"DO NOT TOUCH"
    (out_dir / "v0.1.0.md").write_bytes(sentinel)
    (out_dir / "v0.1.2.md").write_bytes(sentinel)

    result = _run(["--version", "0.1.1", "--root", str(tmp_path)])
    assert result.returncode == 0, result.stderr

    assert (out_dir / "v0.1.0.md").read_bytes() == sentinel, "v0.1.0.md was modified"
    assert (out_dir / "v0.1.2.md").read_bytes() == sentinel, "v0.1.2.md was modified"
    assert (out_dir / "v0.1.1.md").exists(), "v0.1.1.md was not created"


# ---------------------------------------------------------------------------
# Test 3: "Unreleased" must be rejected (fail-closed)
# ---------------------------------------------------------------------------


def test_unreleased_rejected(tmp_path: Path) -> None:
    """Specifying 'Unreleased' as the version must exit non-zero."""
    changelog_text = _minimal_changelog(
        [("0.1.0", "Body.")],
        include_unreleased=True,
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "Unreleased", "--root", str(tmp_path)])
    assert result.returncode != 0, "Expected non-zero exit for 'Unreleased' version"


# ---------------------------------------------------------------------------
# Test 4: non-existent version must be rejected
# ---------------------------------------------------------------------------


def test_missing_version_rejected(tmp_path: Path) -> None:
    """Requesting a version absent from CHANGELOG must exit non-zero."""
    changelog_text = _minimal_changelog([("0.1.0", "Body.")])
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "9.9.9", "--root", str(tmp_path)])
    assert result.returncode != 0, "Expected non-zero exit for missing version"


# ---------------------------------------------------------------------------
# Test 5: duplicate version heading in CHANGELOG must be rejected
# ---------------------------------------------------------------------------


def test_duplicate_heading_rejected(tmp_path: Path) -> None:
    """A CHANGELOG with two identical version headings must exit non-zero."""
    changelog_text = textwrap.dedent("""\
        # Changelog

        ## [0.1.0] - 2026-01-01

        First occurrence.

        ## [0.1.0] - 2026-01-02

        Duplicate occurrence.
    """)
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "0.1.0", "--root", str(tmp_path)])
    assert result.returncode != 0, "Expected non-zero exit for duplicate heading"


# ---------------------------------------------------------------------------
# Test 6: boundary accuracy — body ends exactly at the next heading
# ---------------------------------------------------------------------------


def test_boundary_accuracy(tmp_path: Path) -> None:
    """The extracted body must not bleed into the next section."""
    changelog_text = _minimal_changelog(
        [
            ("0.1.1", "Body for 0.1.1 only."),
            ("0.1.0", "Body for 0.1.0 only."),
        ]
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "0.1.1", "--root", str(tmp_path)])
    assert result.returncode == 0, result.stderr

    content = (tmp_path / "release_notes" / "v0.1.1.md").read_text(encoding="utf-8")
    assert "Body for 0.1.1 only." in content
    assert "Body for 0.1.0 only." not in content
    assert "## [0.1.0]" not in content


# ---------------------------------------------------------------------------
# Test 7: CRLF input → LF output
# ---------------------------------------------------------------------------


def test_crlf_input_produces_lf_output(tmp_path: Path) -> None:
    """CHANGELOG with CRLF line endings must produce LF-only output."""
    lf_changelog = _minimal_changelog([("0.1.0", "Some body.")])
    crlf_changelog = lf_changelog.replace("\n", "\r\n")
    (tmp_path / "CHANGELOG.md").write_bytes(crlf_changelog.encode("utf-8"))
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "0.1.0", "--root", str(tmp_path)])
    assert result.returncode == 0, result.stderr

    raw = (tmp_path / "release_notes" / "v0.1.0.md").read_bytes()
    assert b"\r\n" not in raw, "Output contains CRLF line endings"
    assert b"\n" in raw, "Output contains no LF at all"


# ---------------------------------------------------------------------------
# Test 8: trailing newline — exactly one LF at end of file
# ---------------------------------------------------------------------------


def test_exactly_one_trailing_newline(tmp_path: Path) -> None:
    """Output must end with exactly one LF byte regardless of CHANGELOG format."""
    # Build a CHANGELOG whose raw section body has zero trailing newlines
    raw = "# Changelog\n\n## [0.1.0] - 2026-01-01\n\nSome body without trailing newline"
    (tmp_path / "CHANGELOG.md").write_text(raw, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--version", "0.1.0", "--root", str(tmp_path)])
    assert result.returncode == 0, result.stderr

    data = (tmp_path / "release_notes" / "v0.1.0.md").read_bytes()
    assert data.endswith(b"\n"), "File does not end with LF"
    assert not data.endswith(b"\n\n"), "File ends with more than one LF"


# ---------------------------------------------------------------------------
# Test 9: idempotent — running twice produces identical bytes
# ---------------------------------------------------------------------------


def test_idempotent(tmp_path: Path) -> None:
    """Running the script twice on the same input must produce identical output."""
    changelog_text = _minimal_changelog([("0.1.0", "Stable body.")])
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    r1 = _run(["--version", "0.1.0", "--root", str(tmp_path)])
    assert r1.returncode == 0, r1.stderr
    first_run = (tmp_path / "release_notes" / "v0.1.0.md").read_bytes()

    r2 = _run(["--version", "0.1.0", "--root", str(tmp_path)])
    assert r2.returncode == 0, r2.stderr
    second_run = (tmp_path / "release_notes" / "v0.1.0.md").read_bytes()

    assert first_run == second_run, (
        "Output differs between two consecutive runs (not idempotent)"
    )


# ---------------------------------------------------------------------------
# Test 10: bulk mode generates present legacy versions without erroring
# ---------------------------------------------------------------------------


def test_bulk_mode_generates_present_versions(tmp_path: Path) -> None:
    """Bulk mode (no --version) must succeed for present versions even if 0.1.12 is not yet in CHANGELOG."""
    changelog_text = _minimal_changelog(
        [
            ("0.1.11", "v0.1.11 body."),
            ("0.1.10", "v0.1.10 body."),
        ]
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog_text, encoding="utf-8")
    (tmp_path / "release_notes").mkdir()

    result = _run(["--root", str(tmp_path)])
    assert result.returncode == 0, f"Bulk mode failed: {result.stderr}"
    assert (tmp_path / "release_notes" / "v0.1.11.md").exists()
    assert (tmp_path / "release_notes" / "v0.1.10.md").exists()
    assert not (tmp_path / "release_notes" / "v0.1.12.md").exists()
