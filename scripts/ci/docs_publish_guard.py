"""Validate the read-only Docs artifact and fail closed on stale publications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "gwexpy-docs-pages-artifact-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_PREVIEW_SUBJECT_RE = re.compile(
    r"^deploy\(preview\): docs_redesign ([0-9a-f]{40})$"
)


class DocsPublishError(RuntimeError):
    """Raised when a Docs artifact or publication target cannot be trusted."""


def _revision(value: object, label: str) -> str:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        raise DocsPublishError(f"{label} must be a full lowercase Git commit SHA")
    return value


def _build_info(path: Path, expected_revision: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DocsPublishError(f"missing build-info or non-regular file: {path}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocsPublishError(f"invalid build-info JSON: {path}") from exc
    if not isinstance(info, dict):
        raise DocsPublishError(f"invalid build-info object: {path}")
    source_revision = _revision(info.get("source_revision"), f"{path} source_revision")
    if expected_revision is not None and source_revision != expected_revision:
        raise DocsPublishError(
            f"{path} is bound to {source_revision}, expected {expected_revision}"
        )
    if info.get("dirty") is not False:
        raise DocsPublishError(f"{path} reports a dirty Docs source")
    return info


def _site_files(site_dir: Path) -> dict[str, str]:
    if site_dir.is_symlink() or not site_dir.is_dir():
        raise DocsPublishError(f"site root is missing or is a symlink: {site_dir}")

    files: dict[str, str] = {}
    for path in sorted(site_dir.rglob("*")):
        if path.is_symlink():
            raise DocsPublishError(f"symlink is forbidden in Docs artifact: {path}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise DocsPublishError(
                f"non-regular file is forbidden in Docs artifact: {path}"
            )
        relative = path.relative_to(site_dir).as_posix()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
            raise DocsPublishError(f"unsafe Docs artifact path: {relative}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    required = {"index.html", "build-info.json", "ja/index.html", "ja/build-info.json"}
    if missing := required - files.keys():
        raise DocsPublishError(
            f"Docs artifact is missing required files: {sorted(missing)}"
        )
    return files


def _validate_bilingual_site(site_dir: Path, expected_revision: str) -> None:
    english = _build_info(site_dir / "build-info.json")
    japanese = _build_info(site_dir / "ja" / "build-info.json")
    if english["source_revision"] != japanese["source_revision"]:
        raise DocsPublishError("EN and JA builds must bind the same source revision")
    if english["source_revision"] != expected_revision:
        raise DocsPublishError(
            f"Docs build is bound to {english['source_revision']}, "
            f"expected {expected_revision}"
        )


def create_artifact_manifest(
    site_dir: Path, manifest_path: Path, expected_revision: str
) -> dict[str, Any]:
    """Write a stable per-file manifest for the complete EN/JA HTML tree."""
    expected_revision = _revision(expected_revision, "expected source revision")
    site_dir = Path(os.path.abspath(site_dir))
    manifest_path = Path(os.path.abspath(manifest_path))
    if manifest_path == site_dir or site_dir in manifest_path.parents:
        raise DocsPublishError("manifest must be outside the site tree")

    _validate_bilingual_site(site_dir, expected_revision)
    manifest = {
        "schema": SCHEMA,
        "source_revision": expected_revision,
        "files": _site_files(site_dir),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_artifact(artifact_root: Path, expected_revision: str) -> dict[str, Any]:
    """Verify the downloaded artifact's shape, source, EN/JA identity and hashes."""
    expected_revision = _revision(expected_revision, "expected source revision")
    artifact_root = Path(os.path.abspath(artifact_root))
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise DocsPublishError(
            f"artifact root is missing or is a symlink: {artifact_root}"
        )

    children = {path.name for path in artifact_root.iterdir()}
    if children != {"manifest.json", "site"}:
        raise DocsPublishError(
            f"artifact root must contain only manifest.json and site/: {sorted(children)}"
        )

    manifest_path = artifact_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DocsPublishError("artifact manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocsPublishError("artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "source_revision",
        "files",
    }:
        raise DocsPublishError("artifact manifest has an unsupported structure")
    if manifest.get("schema") != SCHEMA:
        raise DocsPublishError("artifact manifest has an unsupported schema")
    if manifest.get("source_revision") != expected_revision:
        raise DocsPublishError(
            "artifact manifest is bound to a different source revision"
        )
    if not isinstance(manifest.get("files"), dict):
        raise DocsPublishError("artifact file inventory must be an object")

    site_dir = artifact_root / "site"
    _validate_bilingual_site(site_dir, expected_revision)
    actual_files = _site_files(site_dir)
    declared_files = manifest["files"]
    if declared_files != actual_files:
        expected_paths = (
            set(declared_files)
            if all(isinstance(path, str) for path in declared_files)
            else set()
        )
        actual_paths = set(actual_files)
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        if missing or extra:
            raise DocsPublishError(
                f"artifact file inventory differs (missing={missing}, extra={extra})"
            )
        raise DocsPublishError("artifact file hash mismatch")
    return manifest


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise DocsPublishError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _deployed_revision(site_dir: Path) -> str:
    if site_dir.is_symlink() or not site_dir.is_dir():
        raise DocsPublishError(f"published site is not a regular directory: {site_dir}")
    build_info = site_dir / "build-info.json"
    if build_info.is_symlink():
        raise DocsPublishError(f"published build-info is a symlink: {build_info}")
    if build_info.exists():
        info = _build_info(build_info)
        return str(info["source_revision"])

    # Older preview output predates build-info.json. Its existing publisher
    # bound each gh-pages commit to the source SHA in the exact subject below.
    # Accept only that known path and format; unknown legacy state fails closed.
    if site_dir.name != "preview-docs-redesign":
        raise DocsPublishError("published Docs site is missing build-info.json")
    try:
        gh_pages_root = Path(_git(site_dir, "rev-parse", "--show-toplevel"))
        relative_site = site_dir.resolve().relative_to(gh_pages_root.resolve())
    except (DocsPublishError, ValueError) as exc:
        raise DocsPublishError(
            "legacy preview source binding requires a gh-pages checkout"
        ) from exc
    if relative_site.as_posix() != "preview-docs-redesign":
        raise DocsPublishError("published Docs site is missing build-info.json")
    subject = _git(
        gh_pages_root,
        "log",
        "-1",
        "--format=%s",
        "--",
        relative_site.as_posix(),
    )
    match = LEGACY_PREVIEW_SUBJECT_RE.fullmatch(subject)
    if match is None:
        raise DocsPublishError("legacy preview source binding is not trusted")
    return _revision(match.group(1), "legacy preview source revision")


def check_publication_order(
    repository: Path,
    candidate_revision: str,
    deployed_site: Path,
    main_ref: str = "HEAD",
) -> str | None:
    """Reject stale runs before `rsync --delete`; accept only main's current tip."""
    candidate_revision = _revision(candidate_revision, "candidate source revision")
    repository = repository.resolve()
    deployed_site = Path(os.path.abspath(deployed_site))
    current_main_revision = _git(repository, "rev-parse", main_ref)
    if current_main_revision != candidate_revision:
        raise DocsPublishError(
            "candidate is not the current main commit; refusing a stale publication"
        )
    _git(repository, "cat-file", "-e", f"{candidate_revision}^{{commit}}")

    if deployed_site.is_symlink():
        raise DocsPublishError(f"published site is a symlink: {deployed_site}")
    if not deployed_site.exists():
        return None

    deployed_revision = _deployed_revision(deployed_site)
    _git(repository, "cat-file", "-e", f"{deployed_revision}^{{commit}}")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            deployed_revision,
            candidate_revision,
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if ancestry.returncode != 0:
        raise DocsPublishError(
            "currently published source is not an ancestor of the candidate; "
            "refusing a stale publication or untracked rollback"
        )
    return deployed_revision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest_parser = commands.add_parser("create-manifest")
    manifest_parser.add_argument("--site-dir", type=Path, required=True)
    manifest_parser.add_argument("--manifest", type=Path, required=True)
    manifest_parser.add_argument("--expected-revision", required=True)

    verify_parser = commands.add_parser("verify-artifact")
    verify_parser.add_argument("--artifact-root", type=Path, required=True)
    verify_parser.add_argument("--expected-revision", required=True)

    order_parser = commands.add_parser("check-order")
    order_parser.add_argument("--repository", type=Path, required=True)
    order_parser.add_argument("--candidate-revision", required=True)
    order_parser.add_argument("--deployed-site", type=Path, required=True)
    order_parser.add_argument("--main-ref", default="HEAD")

    args = parser.parse_args(argv)
    try:
        if args.command == "create-manifest":
            manifest = create_artifact_manifest(
                args.site_dir, args.manifest, args.expected_revision
            )
            print(
                f"Created manifest for {len(manifest['files'])} files at "
                f"{manifest['source_revision']}"
            )
        elif args.command == "verify-artifact":
            manifest = verify_artifact(args.artifact_root, args.expected_revision)
            print(
                f"Verified {len(manifest['files'])} Docs files for "
                f"{manifest['source_revision']}"
            )
        else:
            deployed = check_publication_order(
                args.repository,
                args.candidate_revision,
                args.deployed_site,
                args.main_ref,
            )
            print(
                "Publication order verified: "
                f"deployed={deployed or 'none'} candidate={args.candidate_revision}"
            )
    except DocsPublishError as exc:
        print(f"Docs publication rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
