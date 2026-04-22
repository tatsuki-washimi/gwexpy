#!/usr/bin/env python3
"""Guard repository hygiene for changed files, including notebook bloat."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PREFIXES = (
    "docs/.doctrees/",
    "docs/_build/",
    "scratch/.venv_docs/",
    ".venv-ci/",
    ".conda-envs/",
    ".conda-pkgs/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
)

DISPLAY_ONLY_TAG = "display-only"
FORBIDDEN_CELL_METADATA_KEYS = (
    "trusted",
    "collapsed",
    "scrolled",
    "ExecuteTime",
)
FORBIDDEN_NOTEBOOK_METADATA_KEYS = (
    "widgets",
    "varInspector",
    "toc",
    "livereveal",
    "rise",
    "vscode",
)
DEFAULT_MAX_OUTPUT_JSON_BYTES = 200_000
DEFAULT_MAX_TOTAL_OUTPUT_JSON_BYTES = 500_000
NOTEBOOK_HYGIENE_PREFIXES = ("docs/web/", "examples/")


class Violation(NamedTuple):
    """A single repository hygiene violation."""

    path: str
    rule: str
    message: str


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _run_git_command(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git command failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_staged_files() -> list[str]:
    """Return staged paths that should be checked for hygiene regressions."""
    return _run_git_command(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def list_changed_files(base: str, head: str) -> list[str]:
    """Return changed paths between two git refs."""
    return _run_git_command(
        ["diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}"]
    )


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_tags(notebook: dict) -> set[str]:
    if not notebook.get("cells"):
        return set()
    metadata = notebook["cells"][0].get("metadata", {})
    tags = metadata.get("tags", [])
    if isinstance(tags, list):
        return {str(tag) for tag in tags}
    return set()


def _serialized_json_bytes(payload: object) -> int:
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _should_check_notebook_hygiene(normalized_path: str) -> bool:
    return normalized_path.startswith(NOTEBOOK_HYGIENE_PREFIXES)


def _check_notebook(
    path: Path,
    *,
    normalized_path: str,
    max_output_json_bytes: int,
    max_total_output_json_bytes: int,
) -> list[Violation]:
    notebook = _load_notebook(path)
    violations: list[Violation] = []
    tags = _notebook_tags(notebook)
    allow_committed_outputs = DISPLAY_ONLY_TAG in tags

    forbidden_root_keys = sorted(
        key
        for key in notebook.get("metadata", {})
        if key in FORBIDDEN_NOTEBOOK_METADATA_KEYS
    )
    if forbidden_root_keys:
        violations.append(
            Violation(
                normalized_path,
                "notebook-forbidden-notebook-metadata",
                "Forbidden notebook metadata keys present: "
                + ", ".join(forbidden_root_keys),
            )
        )

    total_output_json_bytes = 0
    saw_outputs = False
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if (
            cell.get("cell_type") == "code"
            and cell.get("execution_count") is not None
            and not allow_committed_outputs
        ):
            violations.append(
                Violation(
                    normalized_path,
                    "notebook-execution-count-present",
                    "cell[{cell_index}] retains "
                    f"execution_count={cell.get('execution_count')}. "
                    "Clean source notebooks should not keep execution counts.",
                )
            )

        metadata = cell.get("metadata", {})
        forbidden_cell_keys = sorted(
            key for key in metadata if key in FORBIDDEN_CELL_METADATA_KEYS
        )
        if forbidden_cell_keys:
            violations.append(
                Violation(
                    normalized_path,
                    "notebook-forbidden-cell-metadata",
                    f"cell[{cell_index}] contains forbidden metadata keys: "
                    + ", ".join(forbidden_cell_keys),
                )
            )

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        saw_outputs = True
        for output_index, output in enumerate(outputs):
            output_json_bytes = _serialized_json_bytes(output)
            total_output_json_bytes += output_json_bytes
            if output_json_bytes > max_output_json_bytes:
                violations.append(
                    Violation(
                        normalized_path,
                        "notebook-output-too-large",
                        f"cell[{cell_index}] output[{output_index}] serializes to "
                        f"{output_json_bytes} bytes; limit is {max_output_json_bytes}.",
                    )
                )

    if total_output_json_bytes > max_total_output_json_bytes:
        violations.append(
            Violation(
                normalized_path,
                "notebook-total-output-too-large",
                "Notebook outputs serialize to "
                f"{total_output_json_bytes} bytes in total; "
                f"limit is {max_total_output_json_bytes}.",
            )
        )

    if saw_outputs and not allow_committed_outputs:
        violations.append(
            Violation(
                normalized_path,
                "notebook-outputs-present",
                "Tracked notebooks under 'docs/web/' and 'examples/' must be committed "
                "clean. Build docs from executed temp trees or generated artifacts, "
                "then strip notebook outputs before committing source files. Use the "
                f"'{DISPLAY_ONLY_TAG}' tag only for intentional checked-in outputs.",
            )
        )

    return violations


def check_paths(
    paths: list[str],
    *,
    repo_root: Path = REPO_ROOT,
    max_output_json_bytes: int = DEFAULT_MAX_OUTPUT_JSON_BYTES,
    max_total_output_json_bytes: int = DEFAULT_MAX_TOTAL_OUTPUT_JSON_BYTES,
) -> list[Violation]:
    """Check paths for generated artifacts and notebook hygiene violations."""
    violations: list[Violation] = []
    seen_paths: set[str] = set()

    for original_path in paths:
        normalized_path = _normalize(original_path)
        if not normalized_path or normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)

        path = repo_root / normalized_path
        if not path.exists():
            continue

        matched_prefix = next(
            (
                prefix
                for prefix in FORBIDDEN_PREFIXES
                if normalized_path.startswith(prefix)
            ),
            None,
        )
        if matched_prefix:
            violations.append(
                Violation(
                    normalized_path,
                    "forbidden-artifact-path",
                    "Path is under forbidden generated-artifact prefix "
                    f"'{matched_prefix}'.",
                )
            )
            continue

        if path.suffix == ".ipynb" and _should_check_notebook_hygiene(normalized_path):
            violations.extend(
                _check_notebook(
                    path,
                    normalized_path=normalized_path,
                    max_output_json_bytes=max_output_json_bytes,
                    max_total_output_json_bytes=max_total_output_json_bytes,
                )
            )

    return violations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hygiene check CLI."""
    parser = argparse.ArgumentParser(
        description="Check changed files for repo-hygiene regressions."
    )
    parser.add_argument("paths", nargs="*", help="Explicit paths to check.")
    parser.add_argument("--base", help="Base ref for git diff.")
    parser.add_argument("--head", default="HEAD", help="Head ref for git diff.")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check staged files from git diff --cached.",
    )
    parser.add_argument(
        "--max-output-json-bytes",
        type=int,
        default=DEFAULT_MAX_OUTPUT_JSON_BYTES,
        help="Maximum serialized JSON size for a single notebook output.",
    )
    parser.add_argument(
        "--max-total-output-json-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_OUTPUT_JSON_BYTES,
        help="Maximum serialized JSON size for all outputs in one notebook.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return list(args.paths)
    if args.base:
        return list_changed_files(args.base, args.head)
    return list_staged_files()


def main() -> int:
    """Run the hygiene check CLI."""
    args = parse_args()
    paths = _resolve_paths(args)
    if not paths:
        print("No files to check.")
        return 0

    violations = check_paths(
        paths,
        max_output_json_bytes=args.max_output_json_bytes,
        max_total_output_json_bytes=args.max_total_output_json_bytes,
    )
    if not violations:
        print("Success: no hygiene violations detected.")
        return 0

    print("Repository hygiene violations detected:", file=sys.stderr)
    for violation in violations:
        print(
            f"  - {violation.path}: [{violation.rule}] {violation.message}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
