"""Independent service readbacks and conservative owned transitions."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/ci"))
import post_release as controller


def record(version="0.2.7"):
    return controller.initialize_manifest(
        dict(
            release=dict(
                version=version,
                tag="v" + version,
                source_sha="a" * 40,
                tag_object_sha="b" * 40,
                promotion_manifest_sha256="c" * 64,
                distributions=[
                    dict(
                        name="gwexpy-" + version + ".tar.gz",
                        sha256="d" * 64,
                        size_in_bytes=12,
                    ),
                    dict(
                        name="gwexpy-" + version + "-py3-none-any.whl",
                        sha256="e" * 64,
                        size_in_bytes=14,
                    ),
                ],
            ),
            publication=dict(
                repository="example/gwexpy",
                readback_job=dict(url="https://example.org/job"),
            ),
        ),
        "2026-10-27T00:00:00Z",
    )


def test_observer_contract_missing():
    assert callable(getattr(controller, "docs_eligible", None))


def test_pypi_exact_files_and_service_times():
    import post_release_observers as observer

    r = record()
    data = {
        "info": {"version": r["version"]},
        "urls": [
            dict(
                filename=f["name"],
                digests={"sha256": f["sha256"]},
                size=f["size_in_bytes"],
                url="https://files.pythonhosted.org/" + f["name"],
                upload_time_iso_8601="2026-10-27T02:47:29.588259Z",
            )
            for f in r["release"]["distributions"]
        ],
    }
    result = observer.pypi(r, data)
    assert result["files"][0]["upload_time_iso_8601"].endswith("588259Z")
    data["urls"][0]["size"] += 1
    with pytest.raises(ValueError, match="identity"):
        observer.pypi(r, data)


def test_owned_updates_preserve_blocks_decisions_and_deduplicate():
    import post_release_observers as observer

    r = record()
    observer.update(
        r,
        "pypi",
        "complete",
        "https://pypi.org/pypi/gwexpy/0.2.7/json",
        {"files": []},
        "now",
    )
    first = copy.deepcopy(r)
    observer.update(
        r,
        "pypi",
        "complete",
        "https://pypi.org/pypi/gwexpy/0.2.7/json",
        {"files": []},
        "later",
    )
    assert r == first
    r["items"]["pypi"]["status"] = "blocked"
    observer.update(
        r, "pypi", "complete", "https://pypi.org/pypi/gwexpy/0.2.7/json", {}, "later"
    )
    assert r["items"]["pypi"]["status"] == "blocked"
    r["items"]["pypi"]["human_decision"] = {"status": "pending_review"}
    observer.update(r, "pypi", "retryable_failure", "https://pypi.org", {}, "later")
    assert r["items"]["pypi"]["human_decision"] == {"status": "pending_review"}


def test_zenodo_date_review_and_conda_do_not_gate_docs():
    import post_release_observers as observer

    r = record("0.2.6")
    payload = dict(
        id=23680359,
        doi="10.5281/zenodo.23680359",
        conceptdoi="10.5281/zenodo.19059422",
        metadata=dict(version="v0.2.6", publication_date="2026-10-27"),
        files=[
            dict(
                key="example/gwexpy-v0.2.6.zip",
                checksum="md5:e587201a22f3acc9be92d4915c6d08f6",
                size=36592642,
            )
        ],
        created="2026-10-27T03:00:00Z",
        updated="2026-10-27T03:00:00Z",
    )
    result = observer.zenodo(r, [payload])
    assert result["record_id"] == 23680359
    for service in ("github", "pypi"):
        observer.update(
            r,
            service,
            "complete",
            "https://example.org/" + service,
            {"readback_succeeded": True},
            "2026-10-27T04:00:00Z",
        )
    observer.update(
        r,
        "zenodo",
        "pending_review",
        "https://zenodo.org/records/23680359",
        result,
        "2026-10-27T04:00:00Z",
    )
    assert controller.docs_eligible(r)
    assert r["items"]["conda"]["status"] == "pending_external"
    assert (
        observer.conda_package(r, {"files": [{"version": "0.2.4"}]})["status"]
        == "pending_external"
    )
    payload["metadata"]["version"] = "v0.2.7"
    with pytest.raises(ValueError):
        observer.zenodo(r, [payload])


def test_merge_preserves_human_review_and_blocked():
    import post_release_observers as observer

    r = record()
    r["items"]["pypi"]["status"] = "pending_review"
    r["items"]["pypi"]["pending_reason"] = "Human requested inspection"
    new = record()
    observer.update(
        new,
        "pypi",
        "complete",
        "https://pypi.org",
        {"readback_succeeded": True},
        "2026-10-27T00:00:00Z",
    )
    merged = controller.merge_manifest(r, new)
    assert merged["items"]["pypi"]["status"] == "pending_review"
    assert merged["items"]["pypi"]["pending_reason"] == "Human requested inspection"


def test_transport_failure_does_not_stop_other_services(monkeypatch):
    import post_release_observers as observer

    r = record()
    monkeypatch.setattr(
        observer, "github", lambda _: (_ for _ in ()).throw(OSError("offline"))
    )

    def get(url):
        if "pypi.org" in url:
            return {"info": {"version": r["version"]}, "urls": []}
        if "zenodo.org" in url:
            return {"hits": {"hits": []}}
        return {"files": []}

    monkeypatch.setattr(observer, "public_json", get)
    result = observer.observe(r)
    assert result["items"]["github"]["status"] == "retryable_failure"
    assert result["items"]["pypi"]["status"] == "blocked"
    assert result["items"]["zenodo"]["status"] == "pending_external"
    assert result["items"]["conda_package"]["status"] == "pending_external"
    assert not controller.docs_eligible(result)


def github_fixture(monkeypatch, r):
    import base64
    import hashlib
    import json

    import post_release_observers as observer

    content = {"notes": b"Release text\n", "review": b"Approved review\n"}
    files = {
        f["name"]: (b"x" if ".whl" in f["name"] else b"yy")
        for f in r["release"]["distributions"]
    }
    files.update({"distribution-sha256.json": b"{}", "LICENSE.sha256": b"hash"})

    def frozen(path, data):
        return dict(path=path, sha256=hashlib.sha256(data).hexdigest())

    manifest = dict(
        source_sha=r["release"]["source_sha"],
        tag=r["release"]["tag"],
        release_notes=frozen(
            "release_notes/" + r["release"]["tag"] + ".md", content["notes"]
        ),
        review_evidence=frozen("review.yaml", content["review"]),
        artifacts=[],
    )
    for name, data in files.items():
        manifest["artifacts"].append(
            dict(
                role="sidecar:" + name
                if name in ("distribution-sha256.json", "LICENSE.sha256")
                else "payload",
                files=[
                    dict(
                        name=name,
                        sha256=hashlib.sha256(data).hexdigest(),
                        size_in_bytes=len(data),
                    )
                ],
            )
        )
    # The common promotion parser requires one payload artifact holding both files.
    payloads = [a for a in manifest["artifacts"] if a["role"] == "payload"]
    manifest["artifacts"] = [
        dict(role="payload", files=[f for a in payloads for f in a["files"]])
    ] + [a for a in manifest["artifacts"] if a["role"] != "payload"]
    manifest.update(version=r["version"], schema="gwexpy-release-promotion-manifest-v1")
    raw = json.dumps(manifest).encode()
    r["release"]["distributions"] = copy.deepcopy(manifest["artifacts"][0]["files"])
    r["release"]["promotion_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    files["promotion-manifest.json"] = raw
    assets = [dict(name=n, id=i, size=len(d)) for i, (n, d) in enumerate(files.items())]

    def api(repo, path, **kwargs):
        if path.startswith("git/ref/tags/"):
            return {"object": {"type": "tag", "sha": r["release"]["tag_object_sha"]}}
        if path.startswith("git/tags/"):
            return dict(
                object={"type": "commit", "sha": r["release"]["source_sha"]},
                tagger={"date": "2026-10-26T00:00:00Z"},
            )
        if path.startswith("releases/tags/"):
            return dict(
                draft=False,
                prerelease=False,
                tag_name=r["release"]["tag"],
                assets=assets,
                body=content["notes"].decode(),
                published_at="2026-10-27T01:38:07Z",
                created_at="2026-10-26T00:00:00Z",
            )
        if path.startswith("releases/assets/"):
            return files[assets[int(path.split("/")[-1])]["name"]]
        if path.startswith("contents/"):
            return {
                "content": base64.b64encode(
                    content["notes"] if "release_notes/" in path else content["review"]
                ).decode()
            }
        if path.startswith("commits/"):
            return {"commit": {"committer": {"date": "2026-10-24T00:00:00Z"}}}
        raise AssertionError(path)

    monkeypatch.setattr(observer.c, "api", api)
    return files


def test_github_rebinds_five_assets_frozen_source_and_timestamps(monkeypatch):
    import post_release_observers as observer

    r = record()
    files = github_fixture(monkeypatch, r)
    result = observer.github(r)
    assert len(result["assets"]) == 5
    assert result["published_at"] != result["tagger_date"]
    files[next(n for n in files if n.endswith(".whl"))] = b"changed"
    with pytest.raises(ValueError, match="asset identity"):
        observer.github(r)


def test_zenodo_previously_observed_archive_cannot_change():
    import post_release_observers as observer

    r = record()
    payload = dict(
        id=999,
        doi="10.5281/zenodo.999",
        conceptdoi="10.5281/zenodo.19059422",
        metadata=dict(version="v0.2.7", publication_date="2026-11-01"),
        files=[
            dict(key="example/gwexpy-v0.2.7.zip", checksum="md5:" + "a" * 32, size=123)
        ],
    )
    first = observer.zenodo(r, [payload])
    observer.update(
        r,
        "zenodo",
        "pending_review",
        "https://zenodo.org/records/999",
        first,
        "2026-11-01T00:00:00Z",
    )
    payload["files"][0]["checksum"] = "md5:" + "b" * 32
    with pytest.raises(ValueError, match="changed"):
        observer.zenodo(r, [payload])


def test_daily_resume_discovers_data_on_default_and_active_version_ref(monkeypatch):
    import base64
    import json

    import post_release_observers as observer

    default_record, branch_record = record("0.2.6"), record("0.2.7")
    refs = []

    def api(repo, path, **kwargs):
        assert not kwargs, "Resume discovery must use GET only"
        if not path:
            return {"default_branch": "main"}
        if path.startswith("git/ref/heads/"):
            ref = path.removeprefix("git/ref/heads/")
            refs.append(ref)
            return {"object": {"sha": ref}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": path.removeprefix("git/commits/")}}
        if path.startswith("git/trees/"):
            is_main = path.startswith("git/trees/main?")
            r = default_record if is_main else branch_record
            return {
                "tree": [
                    dict(
                        path="docs/developers/plans/manifests/audit-manifest-"
                        + r["release"]["tag"]
                        + "-post-release.yaml",
                        type="blob",
                        mode="100644",
                        sha=r["version"],
                    )
                ]
            }
        if path.startswith("git/blobs/"):
            r = default_record if path.endswith("0.2.6") else branch_record
            return {"content": base64.b64encode(json.dumps(r).encode()).decode()}
        raise AssertionError(path)

    monkeypatch.setattr(observer.c, "api", api)
    monkeypatch.setattr(
        observer.c,
        "pages",
        lambda *_: [
            dict(
                head=dict(
                    ref="post-release/v0.2.7", repo={"full_name": "example/gwexpy"}
                ),
                base={"ref": "main"},
            ),
            dict(
                head=dict(ref="evil", repo={"full_name": "other/repo"}),
                base={"ref": "main"},
            ),
        ],
    )
    seen, written = [], []

    def observe(r):
        seen.append(r["version"])
        return r

    monkeypatch.setattr(observer, "observe", observe)
    monkeypatch.setattr(
        observer.c, "persist", lambda repo, r: written.append(r["version"])
    )
    observer.resume("example/gwexpy")
    assert refs == ["main", "post-release/v0.2.7"]
    assert seen == written == ["0.2.6", "0.2.7"]


def test_workflow_has_read_only_observation_and_daily_trusted_resume():
    import yaml

    workflow = yaml.load(
        (
            Path(__file__).resolve().parents[1] / ".github/workflows/post-release.yml"
        ).read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"]["schedule"]
    assert workflow["jobs"]["resume"]["steps"][-1]["run"].endswith(
        "post_release.py resume"
    )
    resolve = workflow["jobs"]["resolve"]["steps"]
    assert any("post_release.py observe" in step.get("run", "") for step in resolve)
    assert workflow["jobs"]["resolve"]["permissions"]["contents"] == "read"
    assert (
        "github.workflow_sha" in workflow["jobs"]["resume"]["steps"][0]["with"]["ref"]
    )


def test_github_manifest_must_bind_record_distribution_identity(monkeypatch):
    import post_release_observers as observer

    r = record()
    github_fixture(monkeypatch, r)
    r["release"]["distributions"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="distribution"):
        observer.github(r)


def test_zenodo_search_uses_anonymous_limit_and_filters_versions(monkeypatch):
    import post_release_observers as observer

    calls = []

    def get(url):
        calls.append(url)
        return {
            "hits": {
                "hits": [
                    dict(id=1, metadata={"version": "0.2.6"}),
                    dict(id=2, metadata={"version": "v0.2.7"}),
                ]
            }
        }

    monkeypatch.setattr(observer, "public_json", get)
    assert observer.zenodo_records(
        record(),
        "https://zenodo.org/api/records?q=conceptrecid%3A19059422&size=25&all_versions=true",
    ) == [dict(id=2, metadata={"version": "v0.2.7"})]
    assert "size=25" in calls[0]
