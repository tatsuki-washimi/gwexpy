"""Read existing publication evidence and persist one review PR per version.

JSON records are valid YAML. Resolver and persistence run in separate jobs; no
command publishes packages, merges PRs, deletes refs, or executes release code.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from validate_pypi_readback import _expected_files, read_pypi_with_retry

WORKFLOW = ".github/workflows/publish-release.yml"
READBACK_JOB = "Read back exact published PyPI files"
SCHEMA = "gwexpy-post-release-v1"
STATUSES = (
    "blocked",
    "retryable_failure",
    "pending_review",
    "pending_external",
    "complete",
)
TAG_PATTERN = r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
REQUIRED_ITEMS = {
    "publication_identity",
    "github",
    "pypi",
    "zenodo",
    "conda",
    "docs",
    "pages",
    "remote_cleanup",
    "local_cleanup",
}
RELEASE_IDENTITY = (
    "tag",
    "version",
    "source_sha",
    "tag_object_sha",
    "promotion_manifest_sha256",
    "distributions",
)
PUBLICATION_IDENTITY = (
    "repository",
    "run_id",
    "run_attempt",
    "workflow_path",
    "workflow_sha",
    "event",
    "candidate_run_id",
    "candidate_run_attempt",
    "readback_job",
)


def require(condition: Any, message: str) -> None:
    """Reject unverifiable evidence before creating accepted state."""
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    """Hash exact publication bytes."""
    return hashlib.sha256(data).hexdigest()


def make_handoff(
    manifest: dict[str, Any],
    raw: bytes,
    *,
    run_id: int,
    run_attempt: int,
    workflow_sha: str,
    event: str,
    tag_object_sha: str,
) -> dict[str, Any]:
    """Bind verified payload identity to its publication controller attempt."""
    version, files = _expected_files(manifest)
    require(re.fullmatch(r"[0-9a-f]{40}", tag_object_sha), "annotated tag SHA required")
    require(re.fullmatch(r"[0-9a-f]{40}", workflow_sha), "controller SHA required")
    return {
        "schema": "gwexpy-post-release-handoff-v1",
        "release": dict(
            tag=manifest["tag"],
            version=version,
            source_sha=manifest["source_sha"],
            tag_object_sha=tag_object_sha,
            promotion_manifest_sha256=sha256(raw),
            distributions=[
                dict(name=name, sha256=digest, size_in_bytes=size)
                for name, (digest, size) in sorted(files.items())
            ],
        ),
        "publication": dict(
            repository=manifest["repository"],
            run_id=run_id,
            run_attempt=run_attempt,
            workflow_path=WORKFLOW,
            workflow_sha=workflow_sha,
            event=event,
            candidate_run_id=manifest["run_id"],
            candidate_run_attempt=manifest["run_attempt"],
        ),
    }


def validate_identity(
    *,
    repository: str,
    tag: str,
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    manifest_raw: bytes,
    tag_object_sha: str,
    source_sha: str,
    handoff: dict[str, Any] | None,
    allow_backfill: bool = False,
    job_log: str = "",
) -> dict[str, Any]:
    """Require the exact attempt's readback, manifest, tag and handoff identity.

    Older runs may instead prove tag/hash using their exact successful job log.
    Missing artifacts AND missing logs cannot establish accepted provenance.
    """
    require(re.fullmatch(TAG_PATTERN, tag), "invalid release tag")
    require(
        run.get("repository", {}).get("full_name") == repository, "wrong repository"
    )
    require(run.get("path") == WORKFLOW, "wrong publication workflow")
    require(
        run.get("status") == "completed" and run.get("conclusion") == "success",
        "publication run did not succeed",
    )
    require(
        run.get("event") in ("push", "workflow_dispatch"), "invalid publication event"
    )
    require(isinstance(run.get("id"), int) and run["id"] > 0, "invalid run ID")
    require(
        isinstance(run.get("run_attempt"), int) and run["run_attempt"] > 0,
        "invalid attempt",
    )
    matches = [job for job in jobs if job.get("name") == READBACK_JOB]
    require(len(matches) == 1, "exact pypi_readback job missing or ambiguous")
    job = matches[0]
    require(
        job.get("status") == "completed"
        and job.get("conclusion") == "success"
        and job.get("run_id") == run["id"]
        and job.get("run_attempt", run["run_attempt"]) == run["run_attempt"],
        "exact pypi_readback job did not succeed for selected attempt",
    )
    manifest = json.loads(manifest_raw)
    require(
        manifest.get("repository") == repository
        and manifest.get("tag") == tag
        and manifest.get("version") == tag[1:]
        and manifest.get("source_sha") == source_sha
        and manifest.get("release_sha") == source_sha
        and manifest.get("workflow_path") == WORKFLOW,
        "promotion manifest identity mismatch",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", source_sha), "invalid source SHA")
    expected = make_handoff(
        manifest,
        manifest_raw,
        run_id=run["id"],
        run_attempt=run["run_attempt"],
        workflow_sha=run["head_sha"],
        event=run["event"],
        tag_object_sha=tag_object_sha,
    )
    if handoff is None:
        require(allow_backfill, "automatic publication requires handoff artifact")

        def values(key: str) -> set[str]:
            return set(
                re.findall(
                    rf"(?m)^(?:\d{{4}}-\S+\s+)?\s*{key}: ([^\r\n]+)\r?$", job_log
                )
            )

        require(
            values("EXPECTED_TAG") == {tag}
            and values("MANIFEST_SHA256") == {sha256(manifest_raw)},
            "backfill job log does not bind exact tag and manifest hash",
        )
    else:
        require(
            handoff == expected, "handoff identity differs from publication or manifest"
        )
    result = dict(release=expected["release"], publication=expected["publication"])
    result["publication"]["readback_job"] = dict(
        id=job["id"], url=job["html_url"], conclusion="success"
    )
    return result


def initialize_manifest(identity: dict[str, Any], observed: str) -> dict[str, Any]:
    """Initialize independent objectives without asserting later completion."""
    definitions = {
        "publication_identity": (
            "complete",
            "GitHub Actions",
            "Verify publication identity and Release/PyPI bytes",
            "",
            "",
        ),
        "github": (
            "pending_external",
            "GitHub Actions",
            "Observe Release notes, frozen notes/review-evidence hashes, assets and publication time",
            "Frozen notes/review-evidence hashes and independent GitHub observation not checked",
            "Run GitHub observer",
        ),
        "pypi": (
            "pending_external",
            "GitHub Actions",
            "Record PyPI files and upload timestamps",
            "Independent PyPI observation missing",
            "Run PyPI observer",
        ),
        "zenodo": (
            "pending_external",
            "GitHub Actions",
            "Observe version DOI, archive identity and approved date disposition",
            "Zenodo not observed",
            "Run Zenodo observer",
        ),
        "conda": (
            "pending_external",
            "GitHub Actions",
            "Observe feedstock and actual channel package",
            "Conda not observed",
            "Run conda observer; docs remain independently eligible",
        ),
        "docs": (
            "pending_review",
            "Release Owner",
            "Prepare and review EN/JA release/history updates",
            "Docs preparation and human review outstanding",
            "Record GitHub/PyPI/Zenodo observations, prepare docs, then human review",
        ),
        "pages": (
            "pending_external",
            "GitHub Actions",
            "Verify live EN/JA revision and artifact manifest hash",
            "Deployment not observed",
            "After docs merge, run isolated Pages and revision/hash readback",
        ),
        "remote_cleanup": (
            "pending_external",
            "GitHub Actions",
            "Assess remote branches using ancestry and open PRs",
            "Assessment missing",
            "Run read-only remote assessment",
        ),
        "local_cleanup": (
            "pending_external",
            "maintainer local host",
            "Assess worktrees, dirty files and active sessions",
            "Local evidence missing",
            "Supply maintainer local-host assessment",
        ),
    }
    items: dict[str, Any] = {
        key: dict(
            status=status,
            actor=actor,
            objective=objective,
            evidence=[],
            pending_reason=reason,
            resume_condition=resume,
        )
        for key, (status, actor, objective, reason, resume) in definitions.items()
    }
    items["publication_identity"]["evidence"] = [
        dict(
            url=identity["publication"]["readback_job"]["url"],
            sha256=identity["release"]["promotion_manifest_sha256"],
            observed_at_utc=observed,
            observation="Exact pypi_readback attempt succeeded; tag, manifest and Release/PyPI bytes reverified",
        )
    ]
    return dict(
        schema=SCHEMA,
        version=identity["release"]["version"],
        recorded_at_utc=observed,
        post_release_status="pending_review",
        **copy.deepcopy(identity),
        items=items,
    )


def validate_record(record: dict[str, Any]) -> None:
    """Require evidence or explicit pending reason and resumption condition."""
    require(record.get("schema") == SCHEMA, "unsupported post-release schema")
    require(
        record.get("version") == record["release"]["version"], "record version mismatch"
    )
    require(REQUIRED_ITEMS <= record["items"].keys(), "required objectives are missing")
    for item in record["items"].values():
        require(item.get("status") in STATUSES, "invalid objective status")
        for field in (
            "actor",
            "objective",
            "evidence",
            "pending_reason",
            "resume_condition",
        ):
            require(field in item, f"objective missing {field}")
        require(isinstance(item["evidence"], list), "invalid evidence list")
        for entry in item["evidence"]:
            require(
                all(
                    key in entry
                    for key in ("url", "sha256", "observed_at_utc", "observation")
                ),
                "incomplete evidence",
            )
            require(
                entry["url"].startswith("https://")
                and entry["observed_at_utc"].endswith("Z")
                and entry["observation"],
                "invalid evidence URL/time/observation",
            )
            require(
                entry["sha256"] is None
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
                "invalid evidence hash",
            )
        if item["status"] == "complete":
            require(item["evidence"], "complete objective requires evidence")
        else:
            require(
                item["pending_reason"] and item["resume_condition"],
                "pending objective needs resumption",
            )


def merge_manifest(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Resume same identity without resetting human decisions or losing evidence."""
    validate_record(new)
    if old is None:
        return copy.deepcopy(new)
    validate_record(old)
    require(
        all(
            old["release"].get(key) == new["release"].get(key)
            for key in RELEASE_IDENTITY
        )
        and all(
            old["publication"].get(key) == new["publication"].get(key)
            for key in PUBLICATION_IDENTITY
        ),
        "stored publication identity differs; owner review required",
    )
    result = copy.deepcopy(old)
    for name, item in new["items"].items():
        if name not in result["items"]:
            result["items"][name] = copy.deepcopy(item)
        else:
            for key, value in item.items():
                result["items"][name].setdefault(key, copy.deepcopy(value))
            evidence = result["items"][name]["evidence"]
            for entry in item["evidence"]:
                if not any(
                    all(
                        previous.get(key) == entry.get(key)
                        for key in ("url", "sha256", "observation")
                    )
                    for previous in evidence
                ):
                    evidence.append(copy.deepcopy(entry))
    result["post_release_status"] = next(
        status
        for status in STATUSES
        if any(item["status"] == status for item in result["items"].values())
    )
    validate_record(result)
    if result != old:
        result["recorded_at_utc"] = new["recorded_at_utc"]
    return result


def api(
    repository: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str = "GET",
    raw: bool = False,
) -> Any:
    """Call authenticated gh without exposing token values in arguments."""
    command = [
        "gh",
        "api",
        "-X",
        method,
        f"repos/{repository}" + (f"/{path}" if path else ""),
    ]
    if raw:
        command += ["--allow-escape-sequences"]
        if path.startswith("releases/assets/"):
            command += ["-H", "Accept: application/octet-stream"]
    if payload is not None:
        command += ["--input", "-"]
    completed = subprocess.run(
        command,
        input=json.dumps(payload).encode() if payload is not None else None,
        capture_output=True,
        check=True,
    )
    return completed.stdout if raw else json.loads(completed.stdout)


def pages(repository: str, path: str, key: str | None = None) -> list[Any]:
    """Read complete result sets including closed PRs; bound broken pagination."""
    result = []
    for page in range(1, 1001):
        separator = "&" if "?" in path else "?"
        data = api(repository, f"{path}{separator}per_page=100&page={page}")
        entries = data[key] if key else data
        result.extend(entries)
        if len(entries) < 100:
            return result
    raise ValueError("API pagination limit exceeded")


def read_handoff(repository: str, run: dict[str, Any]) -> dict[str, Any] | None:
    """Read exact-attempt handoff without extracting untrusted archive paths."""
    name = f"post-release-handoff-{run['id']}-{run['run_attempt']}"
    artifacts = [
        entry
        for entry in pages(
            repository, f"actions/runs/{run['id']}/artifacts", "artifacts"
        )
        if entry["name"] == name and not entry["expired"]
    ]
    require(len(artifacts) <= 1, "ambiguous handoff artifacts")
    if not artifacts:
        return None
    raw = api(repository, f"actions/artifacts/{artifacts[0]['id']}/zip", raw=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(
            archive.namelist() == ["post-release-handoff.json"],
            "unsafe handoff archive",
        )
        require(
            archive.getinfo("post-release-handoff.json").file_size < 1024 * 1024,
            "oversized handoff",
        )
        return json.loads(archive.read("post-release-handoff.json"))


def validate_release_asset_names(manifest: dict[str, Any], names: set[str]) -> None:
    """Require exactly the five immutable assets selected by the promotion."""
    _, files = _expected_files(manifest)
    sidecars = [
        entry["name"]
        for artifact in manifest["artifacts"]
        if artifact["role"].startswith("sidecar:")
        for entry in artifact["files"]
    ]
    require(
        len(sidecars) == 2
        and set(sidecars) == {"distribution-sha256.json", "LICENSE.sha256"},
        "promotion sidecar asset set is invalid",
    )
    require(
        names == {"promotion-manifest.json", *files, *sidecars},
        "Release asset set differs from promotion",
    )


def resolve(
    repository: str, run_id: int, tag: str | None, *, automatic: bool
) -> dict[str, Any]:
    """Recheck existing Release and PyPI bytes without building or uploading."""
    require(run_id > 0, "positive publication run ID required")
    run = api(repository, f"actions/runs/{run_id}")
    require(
        run.get("path") == WORKFLOW
        and run.get("repository", {}).get("full_name") == repository,
        "wrong workflow or repository",
    )
    require(
        run.get("status") == "completed" and run.get("conclusion") == "success",
        "publication failed",
    )
    handoff = read_handoff(repository, run)
    if tag is None:
        require(handoff is not None, "automatic publication requires handoff")
        assert handoff is not None
        tag = handoff["release"]["tag"]
    require(re.fullmatch(TAG_PATTERN, tag), "invalid release tag")
    jobs = pages(
        repository, f"actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs", "jobs"
    )
    matches = [
        job
        for job in jobs
        if job["name"] == READBACK_JOB and job["conclusion"] == "success"
    ]
    require(len(matches) == 1, "successful exact pypi_readback job required")
    log = ""
    if handoff is None and not automatic:
        log = api(repository, f"actions/jobs/{matches[0]['id']}/logs", raw=True).decode(
            "utf-8"
        )
    release = api(repository, f"releases/tags/{tag}")
    require(
        not release["draft"]
        and not release["prerelease"]
        and release["tag_name"] == tag,
        "Release is not a published final release",
    )
    assets = {entry["name"]: entry for entry in release["assets"]}
    require(
        len(assets) == len(release["assets"]) and "promotion-manifest.json" in assets,
        "missing or duplicate Release manifest",
    )

    def asset_bytes(name: str) -> bytes:
        require(name in assets, f"missing Release asset: {name}")
        return api(repository, f"releases/assets/{assets[name]['id']}", raw=True)

    raw = asset_bytes("promotion-manifest.json")
    manifest = json.loads(raw)
    ref = api(repository, f"git/ref/tags/{tag}")["object"]
    require(ref["type"] == "tag", "annotated tag required")
    obj = api(repository, f"git/tags/{ref['sha']}")
    require(
        obj["tag"] == tag and obj["object"]["type"] == "commit", "tag identity mismatch"
    )
    identity = validate_identity(
        repository=repository,
        tag=tag,
        run=run,
        jobs=jobs,
        manifest_raw=raw,
        tag_object_sha=ref["sha"],
        source_sha=obj["object"]["sha"],
        handoff=handoff,
        allow_backfill=not automatic,
        job_log=log,
    )
    _, expected = _expected_files(manifest)
    validate_release_asset_names(manifest, set(assets))
    files = {name: asset_bytes(name) for name in expected}
    for artifact in manifest["artifacts"]:
        if artifact["role"].startswith("sidecar:"):
            for entry in artifact["files"]:
                data = asset_bytes(entry["name"])
                require(
                    len(data) == entry["size_in_bytes"]
                    and sha256(data) == entry["sha256"],
                    "Release sidecar hash mismatch",
                )
    read_pypi_with_retry(
        manifest=manifest, payload_files=files, github_release_files=files
    )
    return initialize_manifest(
        identity, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def persist(repository: str, record: dict[str, Any]) -> str:
    """CAS-update stable review branch, preserving all other files and edits.

    Non-force ref updates reject racing commits. Retry reads the winning state.
    A merged PR permits one successor only for changed evidence; identical
    observations are no-ops. A closed unmerged PR requires owner intervention.
    """
    validate_record(record)
    tag = record["release"]["tag"]
    require(
        tag == "v" + record["version"] and re.fullmatch(TAG_PATTERN, tag),
        "invalid branch version",
    )
    branch = f"post-release/{tag}"
    path = f"docs/developers/plans/manifests/audit-manifest-{tag}-post-release.yaml"
    default = api(repository, "")["default_branch"]
    owner = repository.split("/")[0]
    pull_path = f"pulls?state=all&head={owner}:{branch}&base={default}"
    pulls = pages(repository, pull_path)
    active = [pull for pull in pulls if pull["state"] == "open"]
    require(
        len(active) <= 1, "multiple active version PRs require owner reconciliation"
    )
    if pulls and not active:
        require(
            pulls[0].get("merged_at"),
            "existing PR closed unmerged; owner must reconcile",
        )
    merged_refresh = bool(pulls and not active)
    reviewed_head = pulls[0].get("head", {}).get("sha") if merged_refresh else None
    branches = {entry["name"] for entry in pages(repository, "branches")}
    if branch not in branches:
        base = api(repository, f"git/ref/heads/{default}")["object"]["sha"]
        try:
            api(
                repository,
                "git/refs",
                dict(ref=f"refs/heads/{branch}", sha=base),
                method="POST",
            )
        except subprocess.CalledProcessError:
            api(repository, f"git/ref/heads/{branch}")
    for _attempt in range(5):
        head = api(repository, f"git/ref/heads/{branch}")["object"]["sha"]
        # A successor starts from the current reviewed default-branch tree.
        # Its extra parent retains the stable branch's history without force.
        base_head = (
            api(repository, f"git/ref/heads/{default}")["object"]["sha"]
            if merged_refresh
            else head
        )
        commit = api(repository, f"git/commits/{base_head}")
        tree = api(repository, f"git/trees/{commit['tree']['sha']}?recursive=1")
        require(
            not tree.get("truncated"), "truncated tree cannot safely merge manifest"
        )
        if merged_refresh and head != reviewed_head and head != base_head:
            require(
                reviewed_head,
                "merged PR lacks reviewed head SHA; owner must reconcile",
            )
            stable_commit = api(repository, f"git/commits/{head}")
            stable_tree = api(
                repository,
                f"git/trees/{stable_commit['tree']['sha']}?recursive=1",
            )
            reviewed_commit = api(repository, f"git/commits/{reviewed_head}")
            reviewed_tree = api(
                repository,
                f"git/trees/{reviewed_commit['tree']['sha']}?recursive=1",
            )
            require(
                not stable_tree.get("truncated") and not reviewed_tree.get("truncated"),
                "truncated stable/reviewed tree cannot establish safe successor",
            )

            def files_without_manifest(
                tree_result: dict[str, Any],
            ) -> dict[str, tuple[str, str, str]]:
                return {
                    entry["path"]: (entry["mode"], entry["type"], entry["sha"])
                    for entry in tree_result["tree"]
                    if entry["type"] != "tree" and entry["path"] != path
                }

            stable_files = files_without_manifest(stable_tree)
            trusted_base_files = [files_without_manifest(reviewed_tree)]
            head_comparison = api(repository, f"compare/{head}...{base_head}")
            if head_comparison.get("behind_by") == 0 and head_comparison.get(
                "status"
            ) in ("ahead", "identical"):
                # A recreated or retained branch may point at a later default
                # commit than the reviewed PR head. It is safe when that exact
                # tree is already on the trusted default-branch history.
                trusted_base_files.append(stable_files)
            stable_commit_parents = stable_commit.get("parents", [])
            for parent in stable_commit_parents:
                parent_sha = parent.get("sha")
                if not parent_sha:
                    continue
                if parent_sha == base_head:
                    trusted_base_files.append(files_without_manifest(tree))
                    continue
                comparison = api(repository, f"compare/{parent_sha}...{base_head}")
                if comparison.get("behind_by") != 0 or comparison.get("status") not in (
                    "ahead",
                    "identical",
                ):
                    continue
                parent_commit = api(repository, f"git/commits/{parent_sha}")
                parent_tree = api(
                    repository,
                    f"git/trees/{parent_commit['tree']['sha']}?recursive=1",
                )
                require(
                    not parent_tree.get("truncated"),
                    "truncated parent tree cannot establish safe successor",
                )
                trusted_base_files.append(files_without_manifest(parent_tree))
            require(
                any(stable_files == approved for approved in trusted_base_files),
                "unreviewed stable-branch file changes; owner must reconcile",
            )
        matches = [entry for entry in tree["tree"] if entry["path"] == path]
        old = None
        stable_manifest_sha = None
        if matches:
            require(
                matches[0]["type"] == "blob" and matches[0]["mode"] == "100644",
                "unsafe manifest tree entry",
            )
            blob = api(repository, f"git/blobs/{matches[0]['sha']}")
            old = json.loads(base64.b64decode(blob["content"]))
        merged = merge_manifest(old, record)
        if merged_refresh and base_head != head:
            # A preceding attempt may have committed evidence but failed before
            # opening its successor PR. Do not discard that durable evidence.
            previous_commit = api(repository, f"git/commits/{head}")
            previous_tree = api(
                repository, f"git/trees/{previous_commit['tree']['sha']}?recursive=1"
            )
            require(not previous_tree.get("truncated"), "truncated stable branch tree")
            previous = [
                entry for entry in previous_tree["tree"] if entry["path"] == path
            ]
            if previous:
                require(
                    previous[0]["type"] == "blob" and previous[0]["mode"] == "100644",
                    "unsafe stable manifest",
                )
                previous_blob = api(repository, f"git/blobs/{previous[0]['sha']}")
                stable_manifest_sha = previous[0]["sha"]
                merged = merge_manifest(
                    merged, json.loads(base64.b64decode(previous_blob["content"]))
                )
        if merged == old:
            if pulls:
                return (active or pulls)[0]["html_url"]
            # A branch commit can succeed before PR creation fails. On retry,
            # the record is already present, but review still has to start.
            # Compare only the dedicated manifest path so an already-merged
            # identical record does not create an empty PR.
            default_head = api(repository, f"git/ref/heads/{default}")["object"]["sha"]
            default_commit = api(repository, f"git/commits/{default_head}")
            default_tree = api(
                repository,
                f"git/trees/{default_commit['tree']['sha']}?recursive=1",
            )
            require(
                not default_tree.get("truncated"),
                "truncated default tree cannot establish review diff",
            )
            default_matches = [
                entry for entry in default_tree["tree"] if entry["path"] == path
            ]
            require(
                len(default_matches) <= 1,
                "duplicate manifest entries on default branch",
            )
            if (
                default_matches
                and matches
                and (default_matches[0]["sha"] == matches[0]["sha"])
            ):
                return f"https://github.com/{repository}/blob/{default}/{path}"
            # Do not add another manifest commit. Continue below to recreate
            # the missing review PR from the already-persisted stable branch.
            break
        content = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
        blob = api(
            repository,
            "git/blobs",
            dict(content=content, encoding="utf-8"),
            method="POST",
        )
        if matches and matches[0]["sha"] == blob["sha"]:
            break
        if stable_manifest_sha == blob["sha"]:
            # A prior attempt already committed this merged manifest. Reuse
            # that branch head and retry only the missing review PR.
            break
        tree = api(
            repository,
            "git/trees",
            dict(
                base_tree=commit["tree"]["sha"],
                tree=[dict(path=path, mode="100644", type="blob", sha=blob["sha"])],
            ),
            method="POST",
        )
        new_commit = api(
            repository,
            "git/commits",
            dict(
                message=f"docs: reconcile {tag} post-release evidence",
                tree=tree["sha"],
                parents=list(dict.fromkeys([head, base_head])),
            ),
            method="POST",
        )
        try:
            api(
                repository,
                f"git/refs/heads/{branch}",
                dict(sha=new_commit["sha"], force=False),
                method="PATCH",
            )
            break
        except subprocess.CalledProcessError:
            continue
    else:
        raise ValueError(
            "concurrent persistence exceeded five retries; resume same version"
        )
    pulls = pages(repository, pull_path)
    active = [pull for pull in pulls if pull["state"] == "open"]
    require(
        len(active) <= 1, "multiple active version PRs require owner reconciliation"
    )
    if not active and pulls:
        require(
            pulls[0].get("merged_at"),
            "existing PR closed unmerged; owner must reconcile",
        )
    if not active:
        try:
            pull = api(
                repository,
                "pulls",
                dict(
                    title=f"[AGENT:post-release] {tag} post-release reconciliation",
                    head=branch,
                    base=default,
                    body=f"Persist verified publication identity and pending objectives for {tag}.\n\n"
                    f"Audit record: `{path}`. Human review and merge are required. "
                    "GITHUB_TOKEN-created PR checks may require explicit maintainer action or approval. "
                    "The controller does not merge PRs or delete branches.",
                ),
                method="POST",
            )
            return pull["html_url"]
        except subprocess.CalledProcessError:
            active = [
                pull for pull in pages(repository, pull_path) if pull["state"] == "open"
            ]
            require(len(active) == 1, "PR creation failed; resume same version")
    return active[0]["html_url"]


def main() -> int:
    """Expose separate read-only resolver and write-only persistence commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    handoff = sub.add_parser("handoff")
    handoff.add_argument("--manifest", type=Path, required=True)
    handoff.add_argument("--output", type=Path, required=True)
    resolver = sub.add_parser("resolve")
    resolver.add_argument("--release-run-id", type=int, required=True)
    resolver.add_argument("--release-tag")
    resolver.add_argument("--automatic", action="store_true")
    resolver.add_argument("--output", type=Path, required=True)
    writer = sub.add_parser("persist")
    writer.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "handoff":
        raw = args.manifest.read_bytes()
        record = make_handoff(
            json.loads(raw),
            raw,
            run_id=int(os.environ["GITHUB_RUN_ID"]),
            run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
            workflow_sha=os.environ["GITHUB_WORKFLOW_SHA"],
            event=os.environ["GITHUB_EVENT_NAME"],
            tag_object_sha=os.environ["TAG_OBJECT_SHA"],
        )
    elif args.command == "resolve":
        record = resolve(
            os.environ["GITHUB_REPOSITORY"],
            args.release_run_id,
            args.release_tag,
            automatic=args.automatic,
        )
    else:
        print(
            persist(
                os.environ["GITHUB_REPOSITORY"], json.loads(args.record.read_text())
            )
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
