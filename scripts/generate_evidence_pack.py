#!/usr/bin/env python3
"""Generate an Evidence Pack / Audit Manifest for a PR or task.

Reads git diff information and produces a filled-in Audit Manifest (markdown)
suitable for pasting into a PR description or saving to a file.

Example usage::

    python scripts/generate_evidence_pack.py \\
        --base main \\
        --task "Add evidence-pack generator" \\
        --tests "pytest tests/test_generate_evidence_pack_script.py PASS" \\
        --tests "ruff check PASS" \\
        --skills "setup_plan,verify_physics" \\
        | tee /tmp/evidence.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Label rules — kept in sync with .harness/hooks/hooks.json risk-label hook.
#
# IMPORTANT: If you update the grep patterns in hooks.json's risk-label hook,
# update LABEL_RULES below to match.  The sync test in
# tests/test_generate_evidence_pack_script.py will catch divergence.
# ---------------------------------------------------------------------------
LABEL_RULES: list[tuple[str, str]] = [
    (r"^gwexpy/(fields|signal|spectrogram)/", "needs-physics-review"),
    (
        r"^(pyproject\.toml|CITATION\.cff|gwexpy/_version\.py|CHANGELOG\.md)$",
        "needs-release-check",
    ),
    (r"^(pyproject\.toml|gwexpy/interop/|gwexpy/gui/)", "needs-optional-deps-check"),
    (
        r"^(README\.md|docs/|pyproject\.toml|gwexpy/(fields|signal|spectrogram|timeseries|frequencyseries)/)",
        "needs-docs-sync",
    ),
]


def _run(cmd: list[str], *, timeout: int = 30) -> str:
    """Run a command and return its stdout; return empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(
            f"[generate_evidence_pack] warning: command failed: {exc}", file=sys.stderr
        )
        return ""


def _git_root() -> Path:
    """Return the repository root, or cwd if not inside a git repo."""
    output = _run(["git", "rev-parse", "--show-toplevel"])
    root = output.strip()
    return Path(root) if root else Path.cwd()


def get_changed_files(base: str) -> list[str]:
    """Return deduplicated list of changed/untracked file paths relative to repo root.

    Strategy:
    1. ``git diff --name-only <base>...HEAD``  (three-dot merge-base diff)
    2. Fallback to ``git diff --name-only <base>``  (simple two-dot diff)
    3. ``git diff --name-only``  (working-tree unstaged changes)
    4. ``git ls-files --others --exclude-standard``  (untracked files)
    """
    seen: set[str] = set()
    files: list[str] = []

    def _add(output: str) -> None:
        for line in output.splitlines():
            path = line.strip()
            if path and path not in seen:
                seen.add(path)
                files.append(path)

    # 1. Three-dot diff (most informative for PR context)
    three_dot = _run(["git", "diff", "--name-only", f"{base}...HEAD"])
    if three_dot.strip():
        _add(three_dot)
    else:
        # 2. Simple two-dot fallback
        _add(_run(["git", "diff", "--name-only", base]))

    # 3. Unstaged working-tree changes
    _add(_run(["git", "diff", "--name-only"]))

    # 4. Untracked files
    _add(_run(["git", "ls-files", "--others", "--exclude-standard"]))

    return files


def get_diff_stat(base: str) -> str:
    """Return ``git diff --stat <base>`` summary (two-dot, for readability)."""
    return _run(["git", "diff", "--stat", base], timeout=30).strip()


def suggest_labels(files: list[str]) -> list[str]:
    """Apply LABEL_RULES to *files* and return sorted list of matching labels."""
    matched: set[str] = set()
    for pattern, label in LABEL_RULES:
        rx = re.compile(pattern)
        if any(rx.search(f) for f in files):
            matched.add(label)
    return sorted(matched)


def build_manifest(
    *,
    task: str,
    files: list[str],
    diff_stat: str,
    tests: list[str],
    skills: list[str],
    labels: list[str],
) -> str:
    """Render the Audit Manifest as a markdown string."""
    today = date.today().isoformat()

    # --- Files Modified section ---
    if files:
        files_list = "\n".join(f"  - {f}" for f in files)
    else:
        files_list = "  <!-- TODO: no changed files detected -->"

    if diff_stat:
        stat_block = f"\n\n  <details><summary>git diff --stat</summary>\n\n  ```\n  {diff_stat}\n  ```\n  </details>"
    else:
        stat_block = ""

    # --- Verification section ---
    if tests:
        verification_items = "\n".join(f"  - [x] {t}" for t in tests)
    else:
        verification_items = (
            "  - [ ] pytest PASS  <!-- TODO: add test results -->\n"
            "  - [ ] ruff/mypy clean  <!-- TODO: confirm -->"
        )

    # --- Skills Used section ---
    if skills:
        skills_str = ", ".join(skills)
    else:
        skills_str = "<!-- TODO: list skills used (e.g. setup_plan, verify_physics) -->"

    # --- Labels section ---
    if labels:
        labels_str = ", ".join(f"`{lb}`" for lb in labels)
    else:
        labels_str = "<!-- TODO: no labels suggested; check manually -->"

    manifest = f"""\
## Audit Manifest

- **Task**: {task}
- **Date**: {today}
- **Status**: <!-- TODO: Completed / Blocked -->
- **Files Modified**:
{files_list}{stat_block}
- **Verification**:
{verification_items}
- **Skills Used**: {skills_str}
- **Recommended Labels**: {labels_str}
- **Known Gaps**: <!-- TODO: describe any known gaps or deferred work -->
- **Physics Review**: <!-- TODO: N/A or describe verify_physics outcome -->
"""
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base ref for git diff comparison (default: main)",
    )
    parser.add_argument(
        "--task",
        default="<!-- TODO: describe the task or issue number -->",
        help="Task description or issue number",
    )
    parser.add_argument(
        "--tests",
        action="append",
        default=[],
        metavar="RESULT",
        help="Test result string, e.g. 'pytest tests/io PASS'. May be repeated.",
    )
    parser.add_argument(
        "--skills",
        default="",
        help="Comma-separated list of skills used, e.g. 'setup_plan,verify_physics'",
    )
    args = parser.parse_args(argv)

    skills: list[str] = (
        [s.strip() for s in args.skills.split(",") if s.strip()] if args.skills else []
    )

    files = get_changed_files(args.base)
    diff_stat = get_diff_stat(args.base)
    labels = suggest_labels(files)

    manifest = build_manifest(
        task=args.task,
        files=files,
        diff_stat=diff_stat,
        tests=args.tests,
        skills=skills,
        labels=labels,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
