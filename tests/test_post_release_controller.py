"""Post-release identity, monotonic persistence, and workflow boundaries."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def controller():
    path = ROOT / "scripts/ci/post_release.py"
    assert path.exists(), "shared post-release controller is missing"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("post_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def identity_inputs():
    manifest = json.loads(
        (ROOT / "tests/fixtures/release/promotion-manifest-v1.json").read_text()
    )
    raw = json.dumps(manifest).encode()
    run = dict(
        id=200,
        run_attempt=1,
        path=".github/workflows/publish-release.yml",
        repository={"full_name": "example/gwexpy"},
        status="completed",
        conclusion="success",
        event="push",
        head_branch="v0.2.5",
        head_sha=manifest["source_sha"],
    )
    jobs = [
        dict(
            name="Read back exact published PyPI files",
            conclusion="success",
            status="completed",
            id=300,
            run_id=200,
            run_attempt=1,
            html_url="https://github.com/example/gwexpy/actions/runs/200/job/300",
        )
    ]
    return manifest, raw, run, jobs


def validated(module, inputs):
    manifest, raw, run, jobs = inputs
    handoff = module.make_handoff(
        manifest,
        raw,
        run_id=run["id"],
        run_attempt=1,
        workflow_sha=run["head_sha"],
        event="push",
        tag_object_sha="f" * 40,
    )
    return module.validate_identity(
        repository="example/gwexpy",
        tag="v0.2.5",
        run=run,
        jobs=jobs,
        manifest_raw=raw,
        tag_object_sha="f" * 40,
        source_sha=manifest["source_sha"],
        handoff=handoff,
    )


def test_handoff_binds_publication_separately_from_candidate(identity_inputs):
    m = controller()
    identity = validated(m, identity_inputs)
    assert identity["publication"]["run_id"] == 200
    assert identity["publication"]["candidate_run_id"] == 987654
    assert (
        identity["release"]["promotion_manifest_sha256"]
        == hashlib.sha256(identity_inputs[1]).hexdigest()
    )
    assert len(identity["release"]["distributions"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "workflow",
        "repository",
        "failed",
        "job_failed",
        "job_missing",
        "candidate",
        "tag",
        "handoff_hash",
        "handoff_attempt",
    ],
)
def test_rejects_unverified_publication(identity_inputs, mutation):
    m = controller()
    manifest, raw, run, jobs = identity_inputs
    handoff = m.make_handoff(
        manifest,
        raw,
        run_id=200,
        run_attempt=1,
        workflow_sha=run["head_sha"],
        event="push",
        tag_object_sha="f" * 40,
    )
    tag = "v0.2.5"
    if mutation == "workflow":
        run["path"] = ".github/workflows/other.yml"
    if mutation == "repository":
        run["repository"]["full_name"] = "attacker/fork"
    if mutation == "failed":
        run["conclusion"] = "failure"
    if mutation == "job_failed":
        jobs[0]["conclusion"] = "failure"
    if mutation == "job_missing":
        jobs.clear()
    if mutation == "candidate":
        jobs[0]["name"] = "Qualify candidate"
    if mutation == "tag":
        tag = "v0.2.6"
    if mutation == "handoff_hash":
        handoff["release"]["promotion_manifest_sha256"] = "0" * 64
    if mutation == "handoff_attempt":
        handoff["publication"]["run_attempt"] = 2
    with pytest.raises(ValueError):
        m.validate_identity(
            repository="example/gwexpy",
            tag=tag,
            run=run,
            jobs=jobs,
            manifest_raw=raw,
            tag_object_sha="f" * 40,
            source_sha=manifest["source_sha"],
            handoff=handoff,
        )


def test_missing_handoff_requires_exact_job_log_identity(identity_inputs):
    m = controller()
    manifest, raw, run, jobs = identity_inputs
    with pytest.raises(ValueError, match="backfill"):
        m.validate_identity(
            repository="example/gwexpy",
            tag="v0.2.5",
            run=run,
            jobs=jobs,
            manifest_raw=raw,
            tag_object_sha="f" * 40,
            source_sha=manifest["source_sha"],
            handoff=None,
            allow_backfill=True,
        )
    identity = m.validate_identity(
        repository="example/gwexpy",
        tag="v0.2.5",
        run=run,
        jobs=jobs,
        manifest_raw=raw,
        tag_object_sha="f" * 40,
        source_sha=manifest["source_sha"],
        handoff=None,
        allow_backfill=True,
        job_log=(
            "2026-10-27T00:00:00Z   EXPECTED_TAG: v0.2.5\n"
            f"2026-10-27T00:00:00Z   MANIFEST_SHA256: {hashlib.sha256(raw).hexdigest()}\n"
        ),
    )
    assert identity["publication"]["run_id"] == 200


def test_manifest_refresh_preserves_prior_human_state_and_evidence(identity_inputs):
    m = controller()
    initial = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    assert initial["schema"] == "gwexpy-post-release-v1"
    assert initial["items"]["docs"]["status"] == "pending_review"
    assert initial["items"]["pages"]["status"] == "pending_external"
    old = copy.deepcopy(initial)
    old["items"]["zenodo"]["evidence"] = [
        {
            "url": "https://zenodo.org/records/1",
            "sha256": None,
            "observed_at_utc": "2026-10-27T01:01:00Z",
            "observation": "record pending",
        }
    ]
    old["items"]["docs"]["human_note"] = "retain this edit"
    refreshed = m.merge_manifest(old, initial)
    assert (
        refreshed["items"]["zenodo"]["evidence"] == old["items"]["zenodo"]["evidence"]
    )
    assert refreshed["items"]["docs"]["human_note"] == "retain this edit"
    changed = copy.deepcopy(initial)
    changed["release"]["source_sha"] = "0" * 40
    with pytest.raises(ValueError, match="identity"):
        m.merge_manifest(old, changed)
    broken = copy.deepcopy(initial)
    broken["items"]["zenodo"]["status"] = "complete"
    with pytest.raises(ValueError, match="evidence"):
        m.merge_manifest(broken, initial)


def test_workflow_has_trusted_read_only_resolver_and_serialized_writer():
    path = ROOT / ".github/workflows/post-release.yml"
    assert path.exists(), "post-release workflow is missing"
    text = path.read_text()
    assert "workflow_run:" in text and "workflows: [Publish release]" in text
    assert "release_run_id:" in text and "release_tag:" in text
    assert "github.ref == format" in text
    assert "cancel-in-progress: false" in text
    read, write = text.split("\n  persist:", 1)
    assert "contents: write" not in read and "pull-requests: write" not in read
    assert "contents: write" in write and "pull-requests: write" in write
    assert "post_release.py resolve" in read and "post_release.py persist" in write
    publish = (ROOT / ".github/workflows/publish-release.yml").read_text()
    readback = publish.split("\n  pypi_readback:", 1)[1]
    assert readback.index("validate_pypi_readback.py") < readback.index(
        "post_release.py handoff"
    )
    assert "post-release-handoff-" in readback


def test_retry_merge_accepts_service_timestamps_without_changing_publication_identity(
    identity_inputs,
):
    m = controller()
    initial = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    old = copy.deepcopy(initial)
    old["publication"]["github_published_at_utc"] = "2026-10-26T01:00:00Z"
    assert m.merge_manifest(old, initial)["publication"][
        "github_published_at_utc"
    ].endswith("Z")


def test_same_version_merge_is_idempotent(identity_inputs):
    m = controller()
    record = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    assert m.merge_manifest(record, record) == record


def test_complete_record_cannot_omit_required_objectives(identity_inputs):
    m = controller()
    record = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    record["items"] = {"publication_identity": record["items"]["publication_identity"]}
    with pytest.raises(ValueError, match="objectives"):
        m.validate_record(record)


def test_repeat_readback_at_later_time_is_semantic_noop(identity_inputs):
    m = controller()
    identity = validated(m, identity_inputs)
    old = m.initialize_manifest(identity, "2026-10-27T01:00:00Z")
    new = m.initialize_manifest(identity, "2026-10-27T02:00:00Z")
    assert m.merge_manifest(old, new) == old


class GitHubStore:
    """In-memory Git Data API boundary; production merge/persistence stays real."""

    def __init__(self, record, pull_state=None):
        import base64

        self.base64 = base64
        self.path = (
            "docs/developers/plans/manifests/audit-manifest-v0.2.5-post-release.yaml"
        )
        content = json.dumps(record, indent=2) + "\n"
        digest = hashlib.sha256(content.encode()).hexdigest()
        self.blobs = {digest: content}
        self.trees = {
            "t0": [{"path": self.path, "mode": "100644", "type": "blob", "sha": digest}]
        }
        self.commits = {"h0": {"tree": {"sha": "t0"}}}
        self.refs = {"main": "h0", "post-release/v0.2.5": "h0"}
        self.pulls = (
            []
            if pull_state is None
            else [
                dict(
                    state=pull_state,
                    merged_at="2026-10-27T01:00:00Z"
                    if pull_state == "closed"
                    else None,
                    html_url="https://github.com/example/gwexpy/pull/1",
                    number=1,
                    head={"sha": "h0"},
                )
            ]
        )
        self.writes = []

    def api(self, repository, path, payload=None, *, method="GET", raw=False):
        route = path.split("?", 1)[0]
        if method != "GET":
            self.writes.append((route, payload))
        if route == "":
            return {"default_branch": "main"}
        if route == "branches":
            return [{"name": name} for name in self.refs]
        if route == "pulls":
            if method == "POST":
                pull = dict(
                    state="open",
                    html_url=f"https://github.com/example/gwexpy/pull/{len(self.pulls) + 1}",
                    merged_at=None,
                )
                self.pulls.insert(0, pull)
                return pull
            return self.pulls
        if route.startswith("compare/"):
            base, head = route.removeprefix("compare/").split("...", 1)
            if base == head:
                return {"status": "identical", "behind_by": 0}
            pending = [head]
            seen = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                if current == base:
                    return {"status": "ahead", "behind_by": 0}
                commit = self.commits.get(current, {})
                pending.extend(
                    parent.get("sha") if isinstance(parent, dict) else parent
                    for parent in commit.get("parents", [])
                )
            return {"status": "diverged", "behind_by": 1}
        if route.startswith("git/ref/heads/"):
            return {"object": {"sha": self.refs[route.removeprefix("git/ref/heads/")]}}
        if route.startswith("git/refs/heads/"):
            assert payload["force"] is False
            self.refs[route.removeprefix("git/refs/heads/")] = payload["sha"]
            return {}
        if route == "git/blobs":
            digest = hashlib.sha256(payload["content"].encode()).hexdigest()
            self.blobs[digest] = payload["content"]
            return {"sha": digest}
        if route.startswith("git/blobs/"):
            return {
                "content": self.base64.b64encode(
                    self.blobs[route.rsplit("/", 1)[1]].encode()
                ).decode()
            }
        if route == "git/trees":
            sha = f"t{len(self.trees)}"
            entries = (
                copy.deepcopy(self.trees[payload["base_tree"]])
                if payload.get("base_tree")
                else []
            )
            for update in payload["tree"]:
                entries = [
                    entry for entry in entries if entry["path"] != update["path"]
                ]
                entries.append(copy.deepcopy(update))
            self.trees[sha] = entries
            return {"sha": sha}
        if route.startswith("git/trees/"):
            return {"tree": self.trees[route.rsplit("/", 1)[1]], "truncated": False}
        if route == "git/commits":
            sha = f"h{len(self.commits)}"
            self.commits[sha] = {
                "tree": {"sha": payload["tree"]},
                "parents": [{"sha": parent} for parent in payload["parents"]],
            }
            return {"sha": sha}
        if route.startswith("git/commits/"):
            return self.commits[route.rsplit("/", 1)[1]]
        raise AssertionError((route, method))


def test_merged_pr_identical_resume_has_no_writes_or_duplicate(
    identity_inputs, monkeypatch
):
    m = controller()
    record = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    store = GitHubStore(record, "closed")
    monkeypatch.setattr(m, "api", store.api)
    assert m.persist("example/gwexpy", record).endswith("/1")
    assert store.writes == []


def test_retry_after_manifest_commit_but_before_pr_creation_opens_pr(
    identity_inputs, monkeypatch
):
    m = controller()
    record = m.initialize_manifest(
        validated(m, identity_inputs), "2026-10-27T01:00:00Z"
    )
    store = GitHubStore(record)
    store.trees["tmain"] = []
    store.commits["hmain"] = {"tree": {"sha": "tmain"}}
    store.refs["main"] = "hmain"
    monkeypatch.setattr(m, "api", store.api)

    url = m.persist("example/gwexpy", record)

    assert url.endswith("/1")
    assert len([pull for pull in store.pulls if pull["state"] == "open"]) == 1
    assert [route for route, _ in store.writes] == ["pulls"]


def test_closed_unmerged_pr_during_persistence_blocks_replacement(
    identity_inputs, monkeypatch
):
    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="Upload timestamps recorded",
        )
    ]
    store = GitHubStore(old, "open")

    def api(repository, path, payload=None, *, method="GET", raw=False):
        result = store.api(repository, path, payload, method=method, raw=raw)
        if path == "git/refs/heads/post-release/v0.2.5" and method == "PATCH":
            store.pulls[0]["state"] = "closed"
        return result

    monkeypatch.setattr(m, "api", api)

    with pytest.raises(ValueError, match="closed unmerged"):
        m.persist("example/gwexpy", new)

    assert [pull for pull in store.pulls if pull["state"] == "open"] == []
    assert [route for route, _ in store.writes].count("pulls") == 0


def test_successor_race_with_unreviewed_file_blocks_without_dropping_edit(
    identity_inputs, monkeypatch
):
    import subprocess

    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="Upload timestamps recorded",
        )
    ]
    store = GitHubStore(old, "closed")
    raced = False

    def api(repository, path, payload=None, **kwargs):
        nonlocal raced
        if path == "git/refs/heads/post-release/v0.2.5" and not raced:
            raced = True
            content = "keep this concurrent edit\n"
            digest = hashlib.sha256(content.encode()).hexdigest()
            store.blobs[digest] = content
            store.trees["t2"] = [
                *copy.deepcopy(store.trees["t0"]),
                dict(path="notes.txt", mode="100644", type="blob", sha=digest),
            ]
            store.commits["h2"] = {
                "tree": {"sha": "t2"},
                "parents": [{"sha": "h0"}],
            }
            store.refs["post-release/v0.2.5"] = "h2"
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return store.api(repository, path, payload, **kwargs)

    monkeypatch.setattr(m, "api", api)

    with pytest.raises(ValueError, match="unreviewed"):
        m.persist("example/gwexpy", new)

    assert raced
    assert store.refs["post-release/v0.2.5"] == "h2"
    retained = store.trees["t2"]
    notes = next(entry for entry in retained if entry["path"] == "notes.txt")
    assert store.blobs[notes["sha"]] == "keep this concurrent edit\n"
    assert len(store.pulls) == 1


def test_successor_retry_recovers_after_pr_failure_with_new_default_content(
    identity_inputs, monkeypatch
):
    import subprocess

    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="Upload timestamps recorded",
        )
    ]
    store = GitHubStore(old, "closed")
    content = "reviewed default-branch update\n"
    digest = hashlib.sha256(content.encode()).hexdigest()
    store.blobs[digest] = content
    store.trees["tmain"] = [
        *copy.deepcopy(store.trees["t0"]),
        dict(path="docs/maintainer-note.txt", mode="100644", type="blob", sha=digest),
    ]
    store.commits["hmain"] = {"tree": {"sha": "tmain"}, "parents": ["h0"]}
    store.refs["main"] = "hmain"
    fail_pr_once = True

    def api(repository, path, payload=None, **kwargs):
        nonlocal fail_pr_once
        if path == "pulls" and kwargs.get("method", "GET") == "POST" and fail_pr_once:
            fail_pr_once = False
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return store.api(repository, path, payload, **kwargs)

    monkeypatch.setattr(m, "api", api)
    with pytest.raises(ValueError, match="PR creation failed"):
        m.persist("example/gwexpy", new)
    committed_head = store.refs["post-release/v0.2.5"]
    assert committed_head != "h0"

    url = m.persist("example/gwexpy", new)

    assert url.endswith("/2")
    assert store.refs["post-release/v0.2.5"] == committed_head
    assert len([pull for pull in store.pulls if pull["state"] == "open"]) == 1
    tree_sha = store.commits[committed_head]["tree"]["sha"]
    paths = {entry["path"] for entry in store.trees[tree_sha]}
    assert "docs/maintainer-note.txt" in paths


def test_default_advance_after_branch_read_uses_trusted_stable_head(
    identity_inputs, monkeypatch
):
    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="Upload timestamps recorded",
        )
    ]
    store = GitHubStore(old, "closed")
    content_a = "reviewed main change A\n"
    digest_a = hashlib.sha256(content_a.encode()).hexdigest()
    store.blobs[digest_a] = content_a
    store.trees["tmain-old"] = [
        *copy.deepcopy(store.trees["t0"]),
        dict(path="docs/reviewed-a.txt", mode="100644", type="blob", sha=digest_a),
    ]
    store.commits["hmain-old"] = {
        "tree": {"sha": "tmain-old"},
        "parents": [{"sha": "h0"}],
    }
    content_b = "reviewed main change B\n"
    digest_b = hashlib.sha256(content_b.encode()).hexdigest()
    store.blobs[digest_b] = content_b
    store.trees["tmain-new"] = [
        *copy.deepcopy(store.trees["tmain-old"]),
        dict(path="docs/reviewed-b.txt", mode="100644", type="blob", sha=digest_b),
    ]
    store.commits["hmain-new"] = {
        "tree": {"sha": "tmain-new"},
        "parents": [{"sha": "hmain-old"}],
    }
    store.refs["main"] = "hmain-old"
    store.refs["post-release/v0.2.5"] = "hmain-old"
    advance_default_once = True

    def api(repository, path, payload=None, **kwargs):
        nonlocal advance_default_once
        if path == "git/ref/heads/main" and advance_default_once:
            advance_default_once = False
            store.refs["main"] = "hmain-new"
        return store.api(repository, path, payload, **kwargs)

    monkeypatch.setattr(m, "api", api)

    url = m.persist("example/gwexpy", new)

    assert url.endswith("/2")
    head = store.refs["post-release/v0.2.5"]
    tree_sha = store.commits[head]["tree"]["sha"]
    paths = {entry["path"] for entry in store.trees[tree_sha]}
    assert "docs/reviewed-a.txt" in paths
    assert "docs/reviewed-b.txt" in paths


def test_merged_pr_new_evidence_creates_one_successor_on_stable_branch(
    identity_inputs, monkeypatch
):
    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="Upload timestamps recorded",
        )
    ]
    store = GitHubStore(old, "closed")
    monkeypatch.setattr(m, "api", store.api)
    url = m.persist("example/gwexpy", new)
    assert url.endswith("/2")
    assert len([p for p in store.pulls if p["state"] == "open"]) == 1
    assert m.persist("example/gwexpy", new) == url
    assert len(store.pulls) == 2
    assert set(store.refs) == {"main", "post-release/v0.2.5"}


def test_concurrent_commit_retry_preserves_winning_human_edit(
    identity_inputs, monkeypatch
):
    import subprocess

    m = controller()
    old = m.initialize_manifest(validated(m, identity_inputs), "2026-10-27T01:00:00Z")
    new = copy.deepcopy(old)
    new["items"]["pypi"]["evidence"] = [
        dict(
            url="https://pypi.org/project/gwexpy/0.2.5/",
            sha256=None,
            observed_at_utc="2026-10-27T02:00:00Z",
            observation="New observation",
        )
    ]
    store = GitHubStore(old, "open")
    raced = False

    def api(repository, path, payload=None, **kwargs):
        nonlocal raced
        if path.startswith("git/refs/heads/") and not raced:
            raced = True
            winner = copy.deepcopy(old)
            winner["items"]["docs"]["human_note"] = "Keep reviewed human edit"
            blob = store.api(
                repository, "git/blobs", dict(content=json.dumps(winner)), method="POST"
            )
            tree = store.api(
                repository,
                "git/trees",
                dict(
                    tree=[
                        dict(
                            path=store.path, mode="100644", type="blob", sha=blob["sha"]
                        )
                    ]
                ),
                method="POST",
            )
            commit = store.api(
                repository,
                "git/commits",
                dict(tree=tree["sha"], parents=["h0"]),
                method="POST",
            )
            store.refs["post-release/v0.2.5"] = commit["sha"]
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return store.api(repository, path, payload, **kwargs)

    monkeypatch.setattr(m, "api", api)
    m.persist("example/gwexpy", new)
    final = store.commits[store.refs["post-release/v0.2.5"]]
    blob = store.trees[final["tree"]["sha"]][0]["sha"]
    record = json.loads(store.blobs[blob])
    assert raced
    assert record["items"]["docs"]["human_note"] == "Keep reviewed human edit"
    assert record["items"]["pypi"]["evidence"] == new["items"]["pypi"]["evidence"]


def test_release_asset_set_is_exactly_manifest_payload_and_sidecars(identity_inputs):
    m = controller()
    manifest = identity_inputs[0]
    names = {
        "promotion-manifest.json",
        "distribution-sha256.json",
        "LICENSE.sha256",
        "gwexpy-0.2.5-py3-none-any.whl",
        "gwexpy-0.2.5.tar.gz",
    }
    assert m.validate_release_asset_names(manifest, names) is None
    with pytest.raises(ValueError, match="asset set"):
        m.validate_release_asset_names(manifest, names | {"unexpected.txt"})
    with pytest.raises(ValueError, match="asset set"):
        m.validate_release_asset_names(manifest, names - {"LICENSE.sha256"})
