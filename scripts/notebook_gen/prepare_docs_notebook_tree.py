#!/usr/bin/env python3
"""Build a temporary docs source tree with executed notebook pages."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPLAY_ONLY_TAG = "display-only"
DOCS_NOTEBOOK_PREFIX = "docs/web/"
CI_SKIP_BOOTSTRAP_COMMENT = (
    "# Skipped in CI: Colab/bootstrap dependency install cell.\n"
)
BOOTSTRAP_PATTERNS = (
    re.compile(r"(^|\n)\s*%pip\s+install\b"),
    re.compile(r"(^|\n)\s*!pip\s+install\b"),
)
BOOTSTRAP_HINTS = ("gwexpy[all]", "colab", "scipy<", "numpy<", "astropy<", "gwpy<")
CANONICAL_NOTEBOOK_LOCALE = "en"
FALLBACK_NOTEBOOK_LOCALES = ("ja",)
NOTEBOOK_LANGUAGE_TAGS = {"en": "lang-en", "ja": "lang-ja"}
DERIVED_NOTEBOOK_SOURCE_MAP = "_derived_notebook_sources.json"


class ExecutionResult(NamedTuple):
    """Execution outcome for one notebook in the temp docs tree."""

    path: str
    returncode: int
    stdout: str
    stderr: str


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _docname_from_rel_path(rel_path: str) -> str:
    rel = Path(rel_path)
    docs_rel = rel.relative_to("docs")
    return docs_rel.with_suffix("").as_posix()


def _run_git(repo_root: Path, args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git command failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resolve_default_base(repo_root: Path) -> str:
    for candidate in ("origin/main", "origin/master", "main", "master"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    raise SystemExit("Could not resolve a default base branch. Pass --base explicitly.")


def _list_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode().strip() or "git ls-files failed")
    return [
        _normalize(raw_path.decode())
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def _is_display_only(path: Path) -> bool:
    notebook = _load_notebook(path)
    if not notebook.get("cells"):
        return False
    metadata = notebook["cells"][0].get("metadata", {})
    tags = metadata.get("tags", [])
    return isinstance(tags, list) and DISPLAY_ONLY_TAG in tags


def _locale_from_rel_path(rel_path: str) -> str | None:
    rel = Path(rel_path)
    if rel.parts[:2] != ("docs", "web") or len(rel.parts) < 4:
        return None
    locale = rel.parts[2]
    return locale if locale in NOTEBOOK_LANGUAGE_TAGS else None


def _is_bootstrap_install_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    source = "".join(cell.get("source", []))
    normalized = source.lower()
    return any(pattern.search(source) for pattern in BOOTSTRAP_PATTERNS) and any(
        hint in normalized for hint in BOOTSTRAP_HINTS
    )


def _sanitize_notebook_for_ci(path: Path) -> bool:
    notebook = _load_notebook(path)
    changed = False
    for cell in notebook.get("cells", []):
        if not _is_bootstrap_install_cell(cell):
            continue
        cell["source"] = [CI_SKIP_BOOTSTRAP_COMMENT]
        cell["outputs"] = []
        cell["execution_count"] = None
        changed = True
    if changed:
        _save_notebook(path, notebook)
    return changed


def _iter_docs_notebooks(repo_root: Path) -> list[str]:
    notebooks: list[str] = []
    for path in sorted((repo_root / "docs" / "web").rglob("*.ipynb")):
        rel_path = path.relative_to(repo_root).as_posix()
        if ".ipynb_checkpoints" in rel_path or "_build" in rel_path:
            continue
        if _is_display_only(path):
            continue
        notebooks.append(rel_path)
    return notebooks


def _list_changed_docs_notebooks(repo_root: Path, base: str, head: str) -> list[str]:
    changed_paths = _run_git(
        repo_root,
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base}...{head}",
            "--",
            "*.ipynb",
        ],
    )
    notebooks: list[str] = []
    for rel_path in changed_paths:
        normalized = _normalize(rel_path)
        if not normalized.startswith(DOCS_NOTEBOOK_PREFIX):
            continue
        path = repo_root / normalized
        if not path.exists() or _is_display_only(path):
            continue
        notebooks.append(normalized)
    return sorted(notebooks)


def _copy_repo_tree(repo_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for rel_path in _list_tracked_files(repo_root):
        source = repo_root / rel_path
        if not source.exists():
            continue
        destination = output_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
            continue
        shutil.copy2(source, destination)


def _sanitize_docs_notebooks_for_ci(output_root: Path, notebook_paths: list[str]) -> None:
    sanitized = 0
    for rel_path in notebook_paths:
        if _sanitize_notebook_for_ci(output_root / rel_path):
            sanitized += 1
    if sanitized:
        print(f"Sanitized {sanitized} notebook(s) to skip Colab bootstrap cells in CI.")


def _localized_notebook_path(rel_path: str, locale: str) -> str | None:
    rel = Path(rel_path)
    if rel.parts[:2] != ("docs", "web") or len(rel.parts) < 4:
        return None
    return Path(*rel.parts[:2], locale, *rel.parts[3:]).as_posix()


def _localized_doc_exists(output_root: Path, rel_path: str, locale: str) -> bool:
    localized_rel = _localized_notebook_path(rel_path, locale)
    if localized_rel is None:
        return False
    localized_path = output_root / localized_rel
    stem_path = localized_path.with_suffix("")
    for suffix in (".ipynb", ".md", ".rst"):
        if stem_path.with_suffix(suffix).exists():
            return True
    return False


def _materialize_missing_locale_notebooks(output_root: Path) -> dict[str, list[str]]:
    mirror_map: dict[str, list[str]] = {}
    canonical_root = output_root / "docs" / "web" / CANONICAL_NOTEBOOK_LOCALE
    if not canonical_root.exists():
        return mirror_map

    created = 0
    for source_path in sorted(canonical_root.rglob("*.ipynb")):
        rel_path = source_path.relative_to(output_root).as_posix()
        for locale in FALLBACK_NOTEBOOK_LOCALES:
            if _localized_doc_exists(output_root, rel_path, locale):
                continue
            fallback_rel = _localized_notebook_path(rel_path, locale)
            fallback_path = output_root / fallback_rel
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, fallback_path)
            mirror_map.setdefault(rel_path, []).append(fallback_rel)
            created += 1

    if created:
        print(
            "Materialized"
            f" {created} missing localized notebook fallback(s) from"
            f" {CANONICAL_NOTEBOOK_LOCALE.upper()} source."
        )
    return mirror_map


def _mirror_executed_notebook_fallbacks(
    output_root: Path,
    rel_path: str,
    mirror_map: dict[str, list[str]],
) -> None:
    source_path = output_root / rel_path
    for fallback_rel in mirror_map.get(rel_path, []):
        fallback_path = output_root / fallback_rel
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, fallback_path)


def _filter_notebook_markdown_for_locale(path: Path, locale: str) -> bool:
    notebook = _load_notebook(path)
    locale_tag = NOTEBOOK_LANGUAGE_TAGS[locale]
    all_lang_tags = set(NOTEBOOK_LANGUAGE_TAGS.values())
    filtered_cells: list[dict] = []
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            filtered_cells.append(cell)
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if not isinstance(tags, list):
            filtered_cells.append(cell)
            continue
        lang_tags = [tag for tag in tags if tag in all_lang_tags]
        if not lang_tags:
            filtered_cells.append(cell)
            continue
        if locale_tag in lang_tags:
            filtered_cells.append(cell)
            continue
        changed = True

    if changed:
        notebook["cells"] = filtered_cells
        _save_notebook(path, notebook)
    return changed


def _localize_docs_notebooks(output_root: Path) -> None:
    localized = 0
    for path in sorted((output_root / "docs" / "web").rglob("*.ipynb")):
        rel_path = path.relative_to(output_root).as_posix()
        locale = _locale_from_rel_path(rel_path)
        if locale is None:
            continue
        if _filter_notebook_markdown_for_locale(path, locale):
            localized += 1
    if localized:
        print(f"Localized markdown cells in {localized} notebook(s) for docs build.")


def _write_derived_notebook_source_map(
    output_root: Path,
    mirror_map: dict[str, list[str]],
) -> None:
    if not mirror_map:
        return
    derived_map: dict[str, str] = {}
    for source_rel, fallback_rels in mirror_map.items():
        for fallback_rel in fallback_rels:
            derived_map[_docname_from_rel_path(fallback_rel)] = source_rel
    target = output_root / "docs" / DERIVED_NOTEBOOK_SOURCE_MAP
    target.write_text(
        json.dumps(derived_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _execute_notebook(output_root: Path, rel_path: str, kernel: str) -> ExecutionResult:
    notebook_path = output_root / rel_path
    executed_path = notebook_path.with_name(f"{notebook_path.stem}.executed.ipynb")
    if executed_path.exists():
        executed_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "papermill",
        rel_path,
        executed_path.relative_to(output_root).as_posix(),
        "--kernel",
        kernel,
    ]
    result = subprocess.run(
        cmd,
        cwd=output_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        executed_path.replace(notebook_path)
    elif executed_path.exists():
        executed_path.unlink()
    return ExecutionResult(rel_path, result.returncode, result.stdout, result.stderr)


def _execute_notebooks(
    output_root: Path,
    notebook_paths: list[str],
    *,
    kernel: str,
    jobs: int,
    mirror_map: dict[str, list[str]],
) -> None:
    if not notebook_paths:
        print("No docs notebooks selected for execution.")
        return

    print(f"Executing {len(notebook_paths)} docs notebooks with {jobs} worker(s).")
    failures: list[ExecutionResult] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {
            executor.submit(_execute_notebook, output_root, rel_path, kernel): rel_path
            for rel_path in notebook_paths
        }
        for future in as_completed(futures):
            result = future.result()
            status = "ok" if result.returncode == 0 else "failed"
            print(f"[{status}] {result.path}")
            if result.stdout.strip():
                print(result.stdout.rstrip())
            if result.stderr.strip():
                print(result.stderr.rstrip(), file=sys.stderr)
            if result.returncode != 0:
                failures.append(result)
                continue
            _mirror_executed_notebook_fallbacks(output_root, result.path, mirror_map)

    if failures:
        print("\nNotebook execution failures:", file=sys.stderr)
        for result in failures:
            print(f"  - {result.path}", file=sys.stderr)
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for docs tree preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare a temporary docs source tree with executed notebook pages."
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help=f"Repository root to copy from. Defaults to {REPO_ROOT}.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination directory for the temporary copied repository tree.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "changed"),
        default="full",
        help="Whether to execute all docs notebooks or only changed docs notebooks.",
    )
    parser.add_argument("--base", default=None, help="Base ref for changed mode.")
    parser.add_argument("--head", default="HEAD", help="Head ref for changed mode.")
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Kernel name for papermill.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel notebook workers.")
    return parser.parse_args()


def main() -> int:
    """Build the temp docs tree and execute the selected notebooks."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.mode == "full":
        notebook_paths = _iter_docs_notebooks(repo_root)
    else:
        base = args.base or _resolve_default_base(repo_root)
        notebook_paths = _list_changed_docs_notebooks(repo_root, base, args.head)

    _copy_repo_tree(repo_root, output_root)
    mirror_map = _materialize_missing_locale_notebooks(output_root)
    _sanitize_docs_notebooks_for_ci(output_root, notebook_paths)
    _execute_notebooks(
        output_root,
        notebook_paths,
        kernel=args.kernel,
        jobs=args.jobs,
        mirror_map=mirror_map,
    )
    _localize_docs_notebooks(output_root)
    _write_derived_notebook_source_map(output_root, mirror_map)
    print(f"Prepared docs source tree at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
