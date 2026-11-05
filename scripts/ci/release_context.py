#!/usr/bin/env python3
"""Read-only release context from git, workflows, and optional GitHub CLI.

Does not evaluate release pass/fail or call Requirement Registry (ai-harness).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

Observation = Literal["verified", "not_observed", "unknown", "conflict"]
RELEASE_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _obs(value: Any, state: Observation, *, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": state, "value": value}
    if detail:
        payload["detail"] = detail
    return payload


def _git_head(repo_root: Path) -> dict[str, Any]:
    proc = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if proc.returncode != 0:
        return _obs(None, "unknown", detail=(proc.stderr or proc.stdout).strip())
    sha = proc.stdout.strip()
    if not SHA40.fullmatch(sha):
        return _obs(sha, "conflict", detail="HEAD is not a 40-char SHA")
    return _obs(sha, "verified")


def _git_tag(repo_root: Path, release_tag: str | None) -> dict[str, Any]:
    if release_tag is None:
        return _obs(None, "not_observed", detail="--release-tag not provided")
    if not RELEASE_TAG.fullmatch(release_tag):
        return _obs(release_tag, "conflict", detail="invalid release tag format")
    proc = _run(["git", "rev-parse", f"refs/tags/{release_tag}^{{commit}}"], cwd=repo_root)
    if proc.returncode != 0:
        return _obs(None, "verified", detail=f"tag {release_tag} not present locally")
    sha = proc.stdout.strip()
    return _obs(sha, "verified")


def _workflow_files(repo_root: Path) -> dict[str, Any]:
    publish = repo_root / ".github" / "workflows" / "publish-release.yml"
    post = repo_root / ".github" / "workflows" / "post-release-qualification.yml"
    legacy = repo_root / ".github" / "workflows" / "release.yml"
    return {
        "publish_release_yml": _obs(
            publish.is_file(),
            "verified" if publish.is_file() else "not_observed",
        ),
        "post_release_qualification_yml": _obs(
            post.is_file(),
            "verified" if post.is_file() else "not_observed",
        ),
        "legacy_release_yml": _obs(
            legacy.is_file(),
            "verified" if legacy.is_file() else "not_observed",
            detail="legacy release.yml must not be used for current promotion flow"
            if legacy.is_file()
            else None,
        ),
    }


def _release_contract_entry(repo_root: Path, release_tag: str | None) -> dict[str, Any]:
    path = repo_root / "scripts" / "ci" / "release_contracts.json"
    if not path.is_file():
        return _obs(None, "not_observed", detail=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _obs(None, "conflict", detail=str(exc))
    releases = data.get("releases")
    if not isinstance(releases, dict):
        return _obs(None, "conflict", detail="releases mapping missing")
    if release_tag is None:
        return _obs(sorted(releases.keys()), "verified", detail="tag list only")
    return _obs(release_tag in releases, "verified")


def _github_issue(repo_root: Path, issue_number: int | None) -> dict[str, Any]:
    if issue_number is None:
        return _obs(None, "not_observed", detail="--tracking-issue not provided")
    if shutil.which("gh") is None:
        return _obs(None, "not_observed", detail="gh CLI unavailable")
    proc = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--json",
            "number,state,title,url",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return _obs(None, "unknown", detail=(proc.stderr or proc.stdout).strip())
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _obs(None, "conflict", detail="gh issue JSON parse failed")
    return _obs(payload, "verified")


def _github_pr(repo_root: Path, pr_number: int | None) -> dict[str, Any]:
    if pr_number is None:
        return _obs(None, "not_observed", detail="--post-release-pr not provided")
    if shutil.which("gh") is None:
        return _obs(None, "not_observed", detail="gh CLI unavailable")
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,state,title,url",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return _obs(None, "unknown", detail=(proc.stderr or proc.stdout).strip())
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _obs(None, "conflict", detail="gh PR JSON parse failed")
    return _obs(payload, "verified")


def build_context(
    repo_root: Path,
    *,
    release_tag: str | None,
    tracking_issue: int | None,
    post_release_pr: int | None,
) -> dict[str, Any]:
    return {
        "schema": "gwexpy-release-context-v1",
        "repo_root": str(repo_root.resolve()),
        "release_tag": release_tag,
        "git": {"head": _git_head(repo_root), "tag_commit": _git_tag(repo_root, release_tag)},
        "workflows": _workflow_files(repo_root),
        "release_contracts": _release_contract_entry(repo_root, release_tag),
        "github": {
            "tracking_issue": _github_issue(repo_root, tracking_issue),
            "post_release_pr": _github_pr(repo_root, post_release_pr),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-tag")
    parser.add_argument("--tracking-issue", type=int)
    parser.add_argument("--post-release-pr", type=int)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if _run(["git", "rev-parse"], cwd=repo_root).returncode != 0:
        print("release_context: not a git repository", file=sys.stderr)
        return 2
    context = build_context(
        repo_root,
        release_tag=args.release_tag,
        tracking_issue=args.tracking_issue,
        post_release_pr=args.post_release_pr,
    )
    print(json.dumps(context, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
