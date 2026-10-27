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
    monkeypatch.setattr(
        observer,
        "feedstock",
        lambda _: {
            "status": "pending_external",
            "reason": "Exact-version feedstock PR absent",
        },
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
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
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


def test_redispatch_loads_durable_zenodo_identity_before_observation(monkeypatch):
    import post_release_observers as observer

    r = record()
    payload = dict(
        id=999,
        doi="10.5281/zenodo.999",
        conceptdoi="10.5281/zenodo.19059422",
        metadata=dict(version="0.2.7", publication_date="2026-11-01"),
        files=[
            dict(key="example/gwexpy-v0.2.7.zip", checksum="md5:" + "a" * 32, size=123)
        ],
    )
    observer.update(
        r,
        "zenodo",
        "pending_review",
        "https://zenodo.org/records/999",
        observer.zenodo(r, [payload]),
        "2026-11-01T00:00:00Z",
    )
    monkeypatch.setattr(
        observer, "discover_records", lambda repository, tag=None: ({"v0.2.7": r}, [])
    )
    monkeypatch.setattr(observer, "github", lambda _: {"readback_succeeded": True})
    monkeypatch.setattr(observer, "zenodo_records", lambda *_: [payload])
    monkeypatch.setattr(observer, "public_json", lambda _: {"files": []})
    monkeypatch.setattr(
        observer,
        "feedstock",
        lambda _: {"status": "pending_external", "reason": "absent"},
    )
    payload["files"][0]["checksum"] = "md5:" + "b" * 32
    observed = observer.observe_with_durable(record())
    assert observed["items"]["zenodo"]["status"] == "blocked"
    successful = [
        e
        for e in observed["items"]["zenodo"]["evidence"]
        if '"readback_succeeded": true' in e["observation"]
    ]
    assert len(successful) == 1
    persisted = controller.merge_manifest(r, observed)
    assert persisted["items"]["zenodo"]["status"] == "blocked"


def test_zenodo_v026_uses_generic_first_readback_identity():
    import post_release_observers as observer

    data = dict(
        id=123,
        doi="10.5281/zenodo.123",
        conceptdoi="10.5281/zenodo.19059422",
        metadata=dict(version="0.2.6", publication_date="2026-10-28"),
        files=[
            dict(key="example/gwexpy-v0.2.6.zip", checksum="md5:" + "a" * 32, size=100)
        ],
    )
    assert observer.zenodo(record("0.2.6"), [data])["record_id"] == 123


def inventory():
    return {
        "files": [
            dict(
                version="0.2.7",
                basename="noarch/gwexpy-0.2.7-py_0.conda",
                download_url="//api.anaconda.org/download/conda-forge/gwexpy/0.2.7/noarch/gwexpy-0.2.7-py_0.conda",
                labels=["main", "broken"],
                attrs=dict(subdir="noarch", build="py_0", build_number=0),
                sha256="a" * 64,
                size=123,
            )
        ]
    }


def test_broken_inventory_package_absent_from_final_repodata_is_pending(monkeypatch):
    import post_release_observers as observer

    monkeypatch.setattr(
        observer, "public_json", lambda _: {"packages": {}, "packages.conda": {}}
    )
    assert observer.conda_package(record(), inventory())["status"] == "pending_external"


def test_conda_requires_exact_build_and_hash_in_final_repodata(monkeypatch):
    import post_release_observers as observer

    info = dict(
        name="gwexpy",
        version="0.2.7",
        build="py_0",
        build_number=0,
        sha256="a" * 64,
        size=123,
        subdir="noarch",
    )
    monkeypatch.setattr(
        observer,
        "public_json",
        lambda _: {"packages.conda": {"gwexpy-0.2.7-py_0.conda": info}},
    )
    assert observer.conda_package(record(), inventory())["status"] == "complete"
    info["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="repodata"):
        observer.conda_package(record(), inventory())


def feedstock_pull(number=20):
    return dict(
        number=number,
        title="gwexpy v0.2.7",
        state="open",
        merged_at=None,
        html_url=f"https://github.com/conda-forge/gwexpy-feedstock/pull/{number}",
        updated_at="2026-11-01T00:00:00Z",
        user={"login": "regro-cf-autotick-bot"},
        head=dict(
            ref="0.2.7_h123",
            sha="f" * 40,
            repo={"full_name": "regro-cf-autotick-bot/gwexpy-feedstock"},
        ),
        base={"repo": {"full_name": "conda-forge/gwexpy-feedstock"}},
    )


def test_next_version_feedstock_discovery_and_ambiguity(monkeypatch):
    import post_release_observers as observer

    pulls = [feedstock_pull()]
    monkeypatch.setattr(
        observer.c,
        "pages",
        lambda repo, path: pulls if path.startswith("pulls?") else [],
    )
    monkeypatch.setattr(
        observer,
        "feedstock_recipe",
        lambda pull, r: {"version": "0.2.7", "source_sha256": "d" * 64},
    )
    result = observer.feedstock(record())
    assert result["status"] == "pending_review"
    assert result["head_sha"] == "f" * 40
    assert result["recipe"]["source_sha256"] == "d" * 64
    pulls.append(feedstock_pull(21))
    with pytest.raises(ValueError, match="ambiguous"):
        observer.feedstock(record())


def test_daily_resume_isolates_record_failure(monkeypatch):
    import post_release_observers as observer

    monkeypatch.setattr(
        observer,
        "discover_records",
        lambda *_: (
            {"v0.2.6": record("0.2.6"), "v0.2.7": record()},
            [{"tag": "v0.2.8", "status": "blocked", "reason": "Malformed record"}],
        ),
    )

    def observe(r):
        if r["version"] == "0.2.6":
            raise OSError("offline")
        return r

    written = []
    monkeypatch.setattr(observer, "observe", observe)
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
    monkeypatch.setattr(
        observer.c, "persist", lambda repo, r: written.append(r["version"])
    )
    results = observer.resume("example/gwexpy")
    assert written == ["0.2.7"]
    assert {r["tag"]: r["status"] for r in results} == {
        "v0.2.8": "blocked",
        "v0.2.6": "retryable_failure",
        "v0.2.7": "pending_review",
    }


@pytest.mark.parametrize(
    "malformed",
    [
        [],
        {
            "schema": controller.SCHEMA,
            "version": "0.2.6",
            "release": {"version": "0.2.6"},
            "items": [],
        },
    ],
)
def test_discovery_malformed_record_does_not_hide_other_version(monkeypatch, malformed):
    import base64
    import json

    import post_release_observers as observer

    def api(repo, path, **kwargs):
        if path == "":
            return {"default_branch": "main"}
        if path.startswith("git/ref/"):
            return {"object": {"sha": "head"}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": "tree"}}
        if path.startswith("git/trees/"):
            return {
                "tree": [
                    dict(
                        path=f"docs/developers/plans/manifests/audit-manifest-v{v}-post-release.yaml",
                        type="blob",
                        mode="100644",
                        sha=v,
                    )
                    for v in ("0.2.6", "0.2.7")
                ]
            }
        content = malformed if path.endswith("0.2.6") else record()
        return {"content": base64.b64encode(json.dumps(content).encode()).decode()}

    monkeypatch.setattr(observer.c, "api", api)
    monkeypatch.setattr(observer.c, "pages", lambda *_: [])
    records, failures = observer.discover_records("example/gwexpy")
    assert list(records) == ["v0.2.7"]
    assert failures[0]["tag"] == "v0.2.6"
    assert failures[0]["status"] == "blocked"


def test_feedstock_recipe_reads_revision_data_without_evaluating_jinja(monkeypatch):
    import base64

    import post_release_observers as observer

    monkeypatch.setattr(
        observer.c, "pages", lambda *_: [{"filename": "recipe/meta.yaml"}]
    )
    recipe = (
        '{% set version = "0.2.7" %}\nsource:\n  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n  sha256: '
        + "d" * 64
        + "\n"
    )
    calls = []

    def api(repo, path):
        calls.append((repo, path))
        return {"content": base64.b64encode(recipe.encode()).decode()}

    monkeypatch.setattr(observer.c, "api", api)
    result = observer.feedstock_recipe(feedstock_pull(), record())
    assert result["version"] == "0.2.7"
    assert result["source_sha256"] == "d" * 64
    assert calls[0][1].endswith("?ref=" + "f" * 40)


@pytest.mark.parametrize("title", ["gwexpy v0.2.70", "gwexpy v0.2.7rc1"])
def test_feedstock_does_not_confuse_neighbor_or_prerelease_versions(monkeypatch, title):
    import post_release_observers as observer

    pull = feedstock_pull()
    pull["title"] = title
    monkeypatch.setattr(observer.c, "pages", lambda *_: [pull])
    monkeypatch.setattr(observer, "feedstock_recipe", lambda *_: {"version": "0.2.7"})
    assert observer.feedstock(record())["status"] == "pending_external"


def test_human_complete_cannot_mask_new_zenodo_identity_conflict(monkeypatch):
    import post_release_observers as observer

    r = record()
    payload = dict(
        id=999,
        doi="10.5281/zenodo.999",
        conceptdoi="10.5281/zenodo.19059422",
        metadata=dict(version="0.2.7", publication_date="2026-11-01"),
        files=[
            dict(key="example/gwexpy-v0.2.7.zip", checksum="md5:" + "a" * 32, size=123)
        ],
    )
    observer.update(
        r,
        "zenodo",
        "complete",
        "https://zenodo.org/records/999",
        observer.zenodo(r, [payload]),
        "2026-11-01T00:00:00Z",
    )
    r["items"]["zenodo"]["human_decision"] = {
        "status": "complete",
        "approved_by": "owner",
    }
    prior = copy.deepcopy(r)
    monkeypatch.setattr(observer, "discover_records", lambda *args: ({"v0.2.7": r}, []))
    monkeypatch.setattr(observer, "github", lambda _: {"readback_succeeded": True})
    monkeypatch.setattr(observer, "zenodo_records", lambda *_: [payload])
    monkeypatch.setattr(observer, "public_json", lambda _: {"files": []})
    monkeypatch.setattr(
        observer,
        "feedstock",
        lambda _: {"status": "pending_external", "reason": "Absent"},
    )
    payload["files"][0]["checksum"] = "md5:" + "b" * 32
    result = controller.merge_manifest(prior, observer.observe_with_durable(record()))
    item = result["items"]["zenodo"]
    assert item["status"] == "blocked"
    assert item["readback_succeeded"] is False
    assert item["human_decision"] == prior["items"]["zenodo"]["human_decision"]
    assert (
        len(
            [
                e
                for e in item["evidence"]
                if '"readback_succeeded": true' in e["observation"]
            ]
        )
        == 1
    )
    observer.update(
        result,
        "zenodo",
        "complete",
        "https://zenodo.org/records/999",
        {"readback_succeeded": True},
        "2026-11-02T00:00:00Z",
    )
    assert result["items"]["zenodo"]["status"] == "blocked"


@pytest.mark.parametrize(
    "version,active_hash",
    [("0.2.70", "d" * 64), ("0.2.7rc1", "d" * 64), ("0.2.7", "e" * 64)],
)
def test_feedstock_recipe_rejects_prefix_version_and_commented_hash(
    monkeypatch, version, active_hash
):
    import base64

    import post_release_observers as observer

    recipe = (
        '{% set version = "'
        + version
        + '" %}\nsource:\n  sha256: '
        + active_hash
        + "\n# sha256: "
        + "d" * 64
        + "\n"
    )
    monkeypatch.setattr(
        observer.c, "pages", lambda *_: [{"filename": "recipe/meta.yaml"}]
    )
    monkeypatch.setattr(
        observer.c,
        "api",
        lambda *_: {"content": base64.b64encode(recipe.encode()).decode()},
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        observer.feedstock_recipe(feedstock_pull(), record())


def test_discovery_null_evidence_url_isolates_version(monkeypatch):
    import post_release_observers as observer

    malformed = record("0.2.6")
    malformed["items"]["publication_identity"]["evidence"][0]["url"] = None
    test_discovery_malformed_record_does_not_hide_other_version(monkeypatch, malformed)
    monkeypatch.setattr(observer, "observe", lambda r: r)
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
    written = []
    monkeypatch.setattr(
        observer.c, "persist", lambda repo, r: written.append(r["version"])
    )
    results = observer.resume("example/gwexpy")
    assert written == ["0.2.7"]
    assert results[0]["tag"] == "v0.2.6" and results[0]["status"] == "blocked"


def test_discovery_missing_overall_status_isolates_version(monkeypatch):
    malformed = record("0.2.6")
    del malformed["post_release_status"]
    test_discovery_malformed_record_does_not_hide_other_version(monkeypatch, malformed)


def test_resume_missing_status_does_not_abort_subsequent_version(monkeypatch):
    import post_release_observers as observer

    malformed = record("0.2.6")
    del malformed["post_release_status"]
    monkeypatch.setattr(
        observer,
        "discover_records",
        lambda _: ({"v0.2.6": malformed, "v0.2.7": record()}, []),
    )
    monkeypatch.setattr(observer, "observe", lambda r: r)
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
    written = []
    monkeypatch.setattr(
        observer.c, "persist", lambda repo, r: written.append(r["version"])
    )
    result = observer.resume("example/gwexpy")
    assert written == ["0.2.7"]
    assert result[0]["tag"] == "v0.2.6" and result[0]["status"] == "blocked"


def recipe_result(monkeypatch, text):
    import base64

    import post_release_observers as observer

    monkeypatch.setattr(
        observer.c, "pages", lambda *_: [{"filename": "recipe/meta.yaml"}]
    )
    monkeypatch.setattr(
        observer.c,
        "api",
        lambda *_: {"content": base64.b64encode(text.encode()).decode()},
    )
    return observer.feedstock_recipe(feedstock_pull(), record())


@pytest.mark.parametrize("indicator", ["|2", ">2", "|2-", "|-2", ">+2"])
def test_recipe_indented_scalar_metadata_cannot_supply_active_identity(
    monkeypatch, indicator
):
    text = (
        "description: "
        + indicator
        + "\n  version: 0.2.7\n  source:\n    sha256: "
        + "d" * 64
        + "\nversion: 0.2.7\nsource:\n  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n  sha256: "
        + "e" * 64
        + "\n"
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


@pytest.mark.parametrize(
    "url",
    [
        "https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.70.tar.gz",
        "https://evil.example/gwexpy-0.2.7.tar.gz",
        "https://pypi.org/packages/source/o/other/other-0.2.7.tar.gz",
    ],
)
def test_recipe_hash_must_bind_exact_sdist_url(monkeypatch, url):
    text = "version: 0.2.7\nsource:\n  url: " + url + "\n  sha256: " + "d" * 64 + "\n"
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


def test_recipe_multiple_source_mappings_cannot_lend_expected_hash(monkeypatch):
    text = (
        "version: 0.2.7\nsource:\n  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n  sha256: "
        + "e" * 64
        + "\nextra:\n  source:\n    url: https://evil.example/other.tar.gz\n    sha256: "
        + "d" * 64
        + "\n"
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


def test_recipe_valid_literal_url_and_single_version(monkeypatch):
    url = "https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz"
    result = recipe_result(
        monkeypatch,
        "version: 0.2.7\nsource:\n  url: " + url + "\n  sha256: " + "d" * 64 + "\n",
    )
    assert result["source_url"] == url


def test_recipe_inactive_jinja_source_cannot_hide_active_inline_source(monkeypatch):
    text = (
        '{% set version = "0.2.7" %}\n'
        "{% if false %}\n"
        "source:\n"
        "  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n"
        "  sha256: " + "d" * 64 + "\n{% endif %}\n"
        'source: {url: "https://evil.example/wrong.tar.gz", sha256: "'
        + "e"
        * 64
        + '"}\n'
        "package:\n  name: gwexpy\n  version: {{ version }}\n"
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


def test_recipe_v1_context_version_template_binds_exact_sdist(monkeypatch):
    text = (
        'context:\n  version: "0.2.7"\n'
        "package:\n  name: gwexpy\n  version: ${{ version }}\n"
        "source:\n"
        "  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-${{ version }}.tar.gz\n"
        "  sha256: " + "d" * 64 + "\n"
    )
    result = recipe_result(monkeypatch, text)
    assert result["version"] == "0.2.7"
    assert result["source_url"].endswith("gwexpy-0.2.7.tar.gz")


def test_recipe_jinja_comment_source_cannot_hide_quoted_active_source(monkeypatch):
    text = (
        "version: 0.2.7\n"
        "{#\n"
        "source:\n"
        "  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n"
        "  sha256: " + "d" * 64 + "\n#}\n"
        '"source": {url: "https://evil.example/wrong.tar.gz", sha256: "'
        + "e" * 64
        + '"}\n'
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


def test_recipe_rejects_jinja_control_hidden_in_yaml_comment(monkeypatch):
    text = (
        '{% set version = "0.2.7" %}\n'
        "# {% if false %}\n"
        "source:\n"
        "  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n"
        "  sha256: " + "d" * 64 + "\n# {% endif %}\n"
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


def test_recipe_duplicate_quoted_source_key_is_rejected(monkeypatch):
    text = (
        "version: 0.2.7\nsource:\n"
        "  url: https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.7.tar.gz\n"
        "  sha256: "
        + "d" * 64
        + '\n"source": {url: https://evil.example/wrong.tar.gz, sha256: '
        + "e" * 64
        + "}\n"
    )
    with pytest.raises(ValueError, match="Feedstock recipe"):
        recipe_result(monkeypatch, text)


@pytest.mark.parametrize("merged", [True, False])
def test_docs_pages_readback_requires_exact_revision_workflow_and_merged_pr(
    monkeypatch, merged
):
    import base64
    import hashlib
    import json

    import post_release_observers as observer

    r = record("0.2.7")
    revision = "f" * 40
    assets = {
        "docs_redesign/_static/images/development-activity-v0.2.7.svg": b"en svg",
        "docs_redesign/_static/images/development-activity-v0.2.7-ja.svg": b"ja svg",
        "docs_redesign/_static/downloads/development-activity-v0.2.7-weekly.csv": b"csv",
    }
    changelog = (
        "Published on 2026-10-27 UTC: [PyPI 0.2.7](/project)\n"
        "## v0.2.7 weekly development activity\n"
        "/_static/downloads/development-activity-v0.2.7-weekly.csv"
    ).encode()
    release_status = b'{"latest_release":"0.2.7"}'

    def api(_repo, path, **_kwargs):
        if not path:
            return {"default_branch": "main"}
        if path.endswith("release_status.json?ref=" + revision):
            content = release_status
        elif path.endswith("about/changelog.md?ref=" + revision):
            content = changelog
        elif path.endswith(".svg?ref=" + revision) or path.endswith(
            ".csv?ref=" + revision
        ):
            content = assets[path.split("?", 1)[0].replace("contents/", "")]
        else:
            raise AssertionError(path)
        return {
            "path": path.split("?", 1)[0].removeprefix("contents/"),
            "encoding": "base64",
            "content": base64.b64encode(content).decode(),
        }

    monkeypatch.setattr(observer.c, "api", api)
    build_info = json.dumps({"source_revision": revision, "dirty": False}).encode()
    page_content = {
        "about/changelog.html": b"PyPI 0.2.7 weekly development activity",
        "ja/about/changelog.html": "PyPI 0.2.7 公開".encode(),
    }

    def public_bytes(url):
        path = url.split("/gwexpy/docs/", 1)[1]
        if path in ("build-info.json", "ja/build-info.json"):
            return build_info
        if path.startswith("_static/"):
            return assets["docs_redesign/" + path]
        return page_content[path]

    monkeypatch.setattr(observer, "public_bytes", public_bytes)

    def pages(_repo, path, key=None):
        if path.startswith("actions/workflows/"):
            return [
                dict(
                    id=77,
                    head_sha=revision,
                    head_branch="main",
                    run_attempt=1,
                    status="completed",
                    conclusion="success",
                    html_url="https://github.com/example/gwexpy/actions/runs/77",
                )
            ]
        if path == "actions/runs/77/attempts/1/jobs":
            return [
                dict(
                    name="Publish verified Docs artifact",
                    status="completed",
                    conclusion="success",
                    steps=[
                        dict(
                            name="Verify published EN and JA HTML",
                            status="completed",
                            conclusion="success",
                        ),
                        dict(
                            name="Verify published EN and JA artifact hashes",
                            status="completed",
                            conclusion="success",
                        ),
                    ],
                )
            ]
        if path == f"commits/{revision}/pulls" and merged:
            return [
                dict(
                    number=42,
                    html_url="https://github.com/example/gwexpy/pull/42",
                    merged_at="2026-10-27T12:00:00Z",
                    base={"ref": "main", "repo": {"full_name": "example/gwexpy"}},
                )
            ]
        if path == f"commits/{revision}/pulls":
            return []
        raise AssertionError((path, key))

    monkeypatch.setattr(observer.c, "pages", pages)
    result = observer.docs_pages(r)
    assert result["status"] == "complete"
    assert result["docs_status"] == ("complete" if merged else "pending_review")
    assert result["source_revision"] == revision
    assert result["pages_workflow_run_id"] == 77
    assert result["artifact_manifest_readback"] == "success"
    assert result["merged_pull_request"] == (
        {"number": 42, "url": "https://github.com/example/gwexpy/pull/42"}
        if merged
        else None
    )
    assert result["asset_sha256"] == {
        path: hashlib.sha256(content).hexdigest() for path, content in assets.items()
    }


def test_docs_pages_records_publication_wait_with_current_latest_release(monkeypatch):
    import base64
    import json

    import post_release_observers as observer

    r = record("0.2.6")
    revision = "a" * 40
    monkeypatch.setattr(
        observer,
        "public_bytes",
        lambda _url: json.dumps({"source_revision": revision, "dirty": False}).encode(),
    )
    monkeypatch.setattr(
        observer.c,
        "api",
        lambda _repo, path: {
            "path": "docs_redesign/release_status.json",
            "encoding": "base64",
            "content": base64.b64encode(b'{"latest_release":"0.2.5"}').decode(),
        },
    )
    result = observer.docs_pages(r)
    assert result["status"] == "pending_external"
    assert result["docs_status"] == "pending_review"
    assert result["published_latest_release"] == "0.2.5"


def test_docs_and_pages_resume_states_advance_only_with_review_evidence(monkeypatch):
    import post_release_observers as observer

    r = record("0.2.7")
    baseline = copy.deepcopy(r)
    observation = dict(readback_succeeded=True, source_revision="f" * 40)
    result = dict(
        status="complete",
        docs_status="pending_review",
        merged_pull_request=None,
        **observation,
    )
    monkeypatch.setattr(observer, "docs_pages", lambda _record: result)
    observer.observe_docs_pages(r)
    assert r["items"]["pages"]["status"] == "complete"
    assert r["items"]["docs"]["status"] == "pending_review"
    assert r["post_release_status"] == "pending_review"

    result = dict(
        status="complete",
        docs_status="complete",
        merged_pull_request={"number": 42},
        **observation,
    )
    observer.observe_docs_pages(r)
    first = copy.deepcopy(r)
    observer.observe_docs_pages(r)
    assert r["items"]["docs"]["status"] == "complete"
    assert r["items"]["pages"]["status"] == "complete"
    assert r["items"]["docs"]["evidence"] == first["items"]["docs"]["evidence"]
    assert r["items"]["pages"]["evidence"] == first["items"]["pages"]["evidence"]
    merged = controller.merge_manifest(baseline, r)
    assert merged["items"]["docs"]["status"] == "complete"
    assert merged["items"]["pages"]["status"] == "complete"


@pytest.mark.parametrize("snapshot", ["null", "[]", "7"])
def test_daily_malformed_zenodo_snapshot_blocks_only_affected_version(
    monkeypatch, snapshot
):
    import post_release_observers as observer

    bad, good = record("0.2.6"), record()
    bad["items"]["zenodo"]["evidence"] = [
        dict(
            url="https://zenodo.org/records/1",
            sha256=None,
            observed_at_utc="2026-11-01T00:00:00Z",
            observation=snapshot,
        )
    ]
    monkeypatch.setattr(
        observer, "discover_records", lambda _: ({"v0.2.6": bad, "v0.2.7": good}, [])
    )
    monkeypatch.setattr(observer, "github", lambda _: {"readback_succeeded": True})
    monkeypatch.setattr(
        observer,
        "feedstock",
        lambda _: {"status": "pending_external", "reason": "Absent"},
    )
    monkeypatch.setattr(observer, "public_json", lambda _: {"files": []})

    def records(r, url):
        version = r["version"]
        return [
            dict(
                id=1,
                doi="10.5281/zenodo.1",
                conceptdoi="10.5281/zenodo.19059422",
                metadata=dict(version=version, publication_date="2026-11-01"),
                files=[
                    dict(
                        key="example/gwexpy-v" + version + ".zip",
                        checksum="md5:" + "a" * 32,
                        size=123,
                    )
                ],
            )
        ]

    monkeypatch.setattr(observer, "zenodo_records", records)
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
    written = {}
    monkeypatch.setattr(
        observer.c, "persist", lambda repo, r: written.update({r["version"]: r})
    )
    result = observer.resume("example/gwexpy")
    assert len(result) == 2
    assert written["0.2.6"]["items"]["zenodo"]["status"] == "blocked"
    assert written["0.2.6"]["items"]["zenodo"]["readback_succeeded"] is False
    assert written["0.2.7"]["items"]["zenodo"]["readback_succeeded"] is True


def test_validate_rejects_complete_over_pending_items():
    r = record()
    r["post_release_status"] = "complete"
    with pytest.raises(ValueError, match="complete"):
        controller.validate_record(r)


def test_validate_rejects_pending_over_complete_items():
    r = record()
    for item in r["items"].values():
        item.update(
            status="complete",
            evidence=[
                dict(
                    url="https://example.org/evidence",
                    sha256=None,
                    observed_at_utc="2026-10-27T01:00:00Z",
                    observation="Verified objective",
                )
            ],
            pending_reason="",
            resume_condition="",
        )
    r["post_release_status"] = "pending_review"
    with pytest.raises(ValueError, match="overall status"):
        controller.validate_record(r)


def test_resume_normalizes_complete_over_pending_then_observes(monkeypatch):
    import post_release_observers as observer

    r = record()
    r["post_release_status"] = "complete"
    monkeypatch.setattr(observer, "discover_records", lambda _: ({"v0.2.7": r}, []))
    seen = []

    def observe(r):
        seen.append(r["post_release_status"])
        r["items"]["pypi"]["pending_reason"] = "Fresh API observation"
        return r

    monkeypatch.setattr(observer, "observe", observe)
    monkeypatch.setattr(observer, "observe_docs_pages", lambda _record: None)
    written = []
    monkeypatch.setattr(observer.c, "persist", lambda repo, r: written.append(r))
    observer.resume("example/gwexpy")
    assert seen == ["pending_review"]
    assert written[0]["items"]["pypi"]["pending_reason"] == "Fresh API observation"
