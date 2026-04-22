from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPLAY_ONLY_TAG = "display-only"
HEAVY_TAG = "ci-heavy"
DEFAULT_OUTPUT_DIR = Path("/tmp/gwexpy_changed_notebooks")
CI_SKIP_BOOTSTRAP_COMMENT = (
    "# Skipped in CI: Colab/bootstrap dependency install cell.\n"
)
BOOTSTRAP_PATTERNS = (
    re.compile(r"(^|\n)\s*%pip\s+install\b"),
    re.compile(r"(^|\n)\s*!pip\s+install\b"),
)
BOOTSTRAP_HINTS = ("gwexpy[all]", "colab", "scipy<", "numpy<", "astropy<", "gwpy<")


def run_command(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a subprocess from the repository root."""
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def resolve_default_base() -> str:
    """Resolve the default comparison base used for PR-style diffs."""
    for candidate in ("origin/main", "origin/master", "main", "master"):
        result = run_command(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            capture_output=True,
        )
        if result.returncode == 0:
            return candidate
    raise SystemExit("Could not resolve a default base branch. Pass --base explicitly.")


def list_changed_notebooks(base: str, head: str) -> list[str]:
    """Return changed notebook paths from git diff."""
    result = run_command(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}", "--", "*.ipynb"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git diff failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def filter_changed_notebooks(paths: Iterable[str], repo_root: Path = REPO_ROOT) -> list[Path]:
    """Keep only existing notebooks that should participate in notebook checks."""
    notebooks: list[Path] = []
    for rel_path in paths:
        if rel_path.endswith(".ipynb") is False:
            continue
        if "_build" in rel_path or ".ipynb_checkpoints" in rel_path:
            continue
        abs_path = repo_root / rel_path
        if abs_path.is_file():
            notebooks.append(abs_path)
    return sorted(notebooks)


def classify_notebook(path: Path) -> str:
    """Classify a notebook using the same first-cell tags as CI."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised through CLI failure
        raise RuntimeError(f"Failed to read notebook metadata: {path}") from exc

    tags = notebook.get("cells", [{}])[0].get("metadata", {}).get("tags", [])
    if DISPLAY_ONLY_TAG in tags:
        return "display-only"
    if HEAVY_TAG in tags:
        return "heavy"
    return "light"


def _is_bootstrap_install_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    source = "".join(cell.get("source", []))
    normalized = source.lower()
    return any(pattern.search(source) for pattern in BOOTSTRAP_PATTERNS) and any(
        hint in normalized for hint in BOOTSTRAP_HINTS
    )


def _sanitize_notebook_for_ci(path: Path, destination: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if not _is_bootstrap_install_cell(cell):
            continue
        cell["source"] = [CI_SKIP_BOOTSTRAP_COMMENT]
        cell["outputs"] = []
        cell["execution_count"] = None
        changed = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return changed


def summarize_groups(groups: dict[str, list[Path]]) -> None:
    """Print a compact summary of notebook groups."""
    print(
        "Changed notebooks:"
        f" {len(groups['light'])} light,"
        f" {len(groups['heavy'])} heavy,"
        f" {len(groups['display-only'])} display-only"
    )
    for category in ("light", "heavy", "display-only"):
        if not groups[category]:
            continue
        print(f"\n[{category}]")
        for path in groups[category]:
            print(path.relative_to(REPO_ROOT))


def run_light_notebooks(paths: list[Path], output_dir: Path, kernel: str) -> None:
    """Execute light notebooks with papermill."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = output_dir / "_inputs"
    for path in paths:
        rel_path = path.relative_to(REPO_ROOT)
        input_path = input_root / rel_path
        output_path = output_dir / rel_path
        sanitized = _sanitize_notebook_for_ci(path, input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "papermill",
            str(input_path),
            str(output_path),
            "--kernel",
            kernel,
        ]
        print(f"\n[light] Executing {rel_path}")
        if sanitized:
            print(f"[light] Sanitized bootstrap install cell(s) in {rel_path}")
        result = run_command(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def run_heavy_notebooks(paths: list[Path]) -> None:
    """Check heavy notebooks with nbval-lax."""
    rel_paths = [str(path.relative_to(REPO_ROOT)) for path in paths]
    cmd = [sys.executable, "-m", "pytest", "--nbval-lax", "-v", "-q", *rel_paths]
    print("\n[heavy] Running nbval syntax checks")
    result = run_command(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CI-style checks for notebooks changed in the current PR."
    )
    parser.add_argument("--base", default=None, help="Base ref for git diff. Defaults to origin/main or origin/master.")
    parser.add_argument("--head", default="HEAD", help="Head ref for git diff. Defaults to HEAD.")
    parser.add_argument("--kernel", default="python3", help="Kernel name for papermill. Defaults to python3.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for papermill outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show notebook classification without executing anything.",
    )
    return parser.parse_args()


def main() -> int:
    """Run PR-scoped notebook checks."""
    args = parse_args()
    base = args.base or resolve_default_base()
    output_dir = Path(args.output_dir)

    changed_paths = list_changed_notebooks(base, args.head)
    notebooks = filter_changed_notebooks(changed_paths)
    if not notebooks:
        print(f"No changed notebooks found in {base}...{args.head}.")
        return 0

    groups = {"light": [], "heavy": [], "display-only": []}
    for notebook in notebooks:
        category = classify_notebook(notebook)
        groups[category].append(notebook)

    summarize_groups(groups)

    if args.list_only:
        return 0

    if groups["light"]:
        print(f"\nPapermill outputs will be written under {output_dir}")
        run_light_notebooks(groups["light"], output_dir, args.kernel)
    else:
        print("\nNo light notebooks to execute.")

    if groups["heavy"]:
        run_heavy_notebooks(groups["heavy"])
    else:
        print("\nNo heavy notebooks to check.")

    if groups["display-only"]:
        print("\nDisplay-only notebooks were skipped.")

    print("\nAll changed notebook checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
