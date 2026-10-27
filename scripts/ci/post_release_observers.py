"""Read-only external observers. All mutable review state remains in the controller."""

from __future__ import annotations

import base64
import copy
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

import post_release as c

SERVICES = ("github", "pypi", "zenodo", "conda_feedstock", "conda_package")


def public_json(url: str) -> Any:
    """GET public metadata without credentials or repository code execution."""
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def update(
    record: dict[str, Any],
    name: str,
    status: str,
    url: str,
    observation: dict[str, Any],
    observed: str,
) -> None:
    """Append deduplicated evidence; blocked and human-owned states are sticky."""
    item = record["items"][name]
    serialized = json.dumps(observation, sort_keys=True)
    entry = dict(
        url=url,
        sha256=c.sha256(serialized.encode()),
        observed_at_utc=observed,
        observation=serialized,
    )
    changed = False
    if not any(
        all(old.get(k) == entry[k] for k in ("url", "sha256", "observation"))
        for old in item["evidence"]
    ):
        item["evidence"].append(entry)
        changed = True
    if status == "blocked":
        # Verification failure invalidates readback even after human approval.
        # Retain the human decision as history, never as a safety override.
        item.update(
            status="blocked",
            observer=name,
            readback_succeeded=False,
            pending_reason=observation.get("reason", "Identity verification failed"),
            resume_condition="Owner must resolve the identity conflict before resuming",
        )
    if (
        item["status"] != "blocked"
        and not item.get("human_decision")
        and (
            item["status"] in ("pending_external", "retryable_failure")
            or item.get("observer") == name
        )
    ):
        item.update(
            status=status,
            observer=name,
            pending_reason=""
            if status == "complete"
            else observation.get("reason", "External review outstanding"),
            resume_condition=""
            if status == "complete"
            else "Re-observe same immutable identity; owner resolves blocked/review decisions",
        )
        item["readback_succeeded"] = observation.get("readback_succeeded", False)
    if changed:
        record["recorded_at_utc"] = observed


def pypi(record: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Require exactly the two frozen distributions and retain upload times."""
    expected = {f["name"]: f for f in record["release"]["distributions"]}
    files = data["urls"]
    c.require(
        data["info"]["version"] == record["version"]
        and len(files) == 2
        and {f["filename"] for f in files} == set(expected),
        "PyPI file identity mismatch",
    )
    result = []
    for f in files:
        e = expected[f["filename"]]
        c.require(
            f["digests"]["sha256"] == e["sha256"]
            and f["size"] == e["size_in_bytes"]
            and f["url"].startswith("https://files.pythonhosted.org/")
            and f["upload_time_iso_8601"].endswith("Z")
            and not f.get("yanked"),
            "PyPI file identity mismatch",
        )
        result.append(
            {
                k: f[k]
                for k in ("filename", "digests", "size", "url", "upload_time_iso_8601")
            }
        )
    return dict(
        readback_succeeded=True, files=sorted(result, key=lambda f: f["filename"])
    )


def zenodo_records(record: dict[str, Any], url: str) -> list[dict[str, Any]]:
    """Paginate anonymous concept records and select the exact version locally."""
    matches: list[dict[str, Any]] = []
    for page in range(1, 1001):
        hits = public_json(url + "&page=" + str(page))["hits"]["hits"]
        matches.extend(
            hit
            for hit in hits
            if hit.get("metadata", {}).get("version")
            in (record["version"], "v" + record["version"])
        )
        if len(hits) < 25:
            return matches
    raise ValueError("Zenodo pagination limit exceeded")


def zenodo(record: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind version DOI and archive; date disposition is a separate human decision."""
    c.require(len(hits) == 1, "Zenodo version record missing or ambiguous")
    data = hits[0]
    version = record["version"]
    c.require(
        data["metadata"]["version"] in (version, "v" + version),
        "Zenodo version mismatch",
    )
    c.require(
        data["doi"] == f"10.5281/zenodo.{data['id']}"
        and data["conceptdoi"] == "10.5281/zenodo.19059422",
        "Zenodo DOI mismatch",
    )
    files = data["files"]
    c.require(
        len(files) == 1
        and files[0]["key"].endswith("-v" + version + ".zip")
        and re.fullmatch(r"md5:[0-9a-f]{32}", files[0]["checksum"])
        and files[0]["size"] > 0,
        "Zenodo archive identity mismatch",
    )
    c.require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["metadata"]["publication_date"]),
        "Zenodo date missing",
    )
    result = dict(
        readback_succeeded=True,
        record_id=data["id"],
        doi=data["doi"],
        conceptdoi=data["conceptdoi"],
        publication_date=data["metadata"]["publication_date"],
        files=files,
        created=data.get("created"),
        updated=data.get("updated"),
        reason="Owner-approved Zenodo date disposition remains outstanding",
    )
    for entry in record["items"]["zenodo"]["evidence"]:
        try:
            previous = json.loads(entry["observation"])
        except (ValueError, TypeError):
            continue
        if previous.get("readback_succeeded"):
            c.require(
                all(
                    previous.get(key) == result[key]
                    for key in (
                        "record_id",
                        "doi",
                        "conceptdoi",
                        "publication_date",
                        "files",
                    )
                ),
                "Zenodo previously observed archive/date identity changed",
            )
    return result


def conda_package(record: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Require a matching build in the final solver-visible channel repodata."""
    files = [f for f in data["files"] if f.get("version") == record["version"]]
    available = []
    missing = []
    repodata: dict[str, Any] = {}
    for entry in files:
        attrs = entry.get("attrs", {})
        subdir = attrs.get("subdir", "")
        basename = entry.get("basename", "")
        c.require(
            re.fullmatch(r"[a-z0-9_-]+", subdir)
            and basename.startswith(subdir + "/")
            and len(basename.split("/")) == 2
            and entry.get("download_url"),
            "Conda package identity incomplete",
        )
        name = basename.split("/")[1]
        url = f"https://conda.anaconda.org/conda-forge/{subdir}/repodata.json"
        if subdir not in repodata:
            repodata[subdir] = public_json(url)
        index = repodata[subdir]
        package = index.get("packages", {}).get(name) or index.get(
            "packages.conda", {}
        ).get(name)
        if not package:
            missing.append(basename)
            continue
        c.require(
            package.get("name") == "gwexpy"
            and package.get("version") == record["version"]
            and package.get("build") == attrs.get("build")
            and package.get("build_number") == attrs.get("build_number")
            and package.get("size") == entry.get("size")
            and bool(entry.get("sha256"))
            and package.get("sha256") == entry["sha256"],
            "Conda inventory/repodata build identity mismatch",
        )
        available.append(
            dict(
                basename=basename,
                download_url=entry["download_url"],
                labels=entry.get("labels", []),
                repodata_url=url,
                repodata=package,
                sha256=entry["sha256"],
                size=entry["size"],
            )
        )
    if not available:
        return dict(
            status="pending_external",
            reason="Exact build absent from final conda-forge repodata",
            inventory_files=[f.get("basename") for f in files],
            missing_from_repodata=missing,
        )
    return dict(
        status="complete",
        readback_succeeded=True,
        files=available,
        missing_from_repodata=missing,
    )


def feedstock_recipe(pull: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Read recipe bytes at the immutable PR revision; never evaluate YAML/Jinja."""
    repo = "conda-forge/gwexpy-feedstock"
    changed = c.pages(repo, f"pulls/{pull['number']}/files")
    paths = [
        f["filename"]
        for f in changed
        if f["filename"] in ("recipe/meta.yaml", "recipe/recipe.yaml")
    ]
    c.require(len(paths) == 1, "Feedstock recipe change missing or ambiguous")
    head = pull["head"]
    data = c.api(head["repo"]["full_name"], f"contents/{paths[0]}?ref={head['sha']}")
    raw = base64.b64decode(data["content"])
    text = raw.decode()
    # A deliberately restricted literal reader, never a Jinja evaluator.
    # Ignore comments and YAML block scalars; hashes must be active source fields.
    versions: list[str] = []
    hashes: list[str] = []
    source_indent: int | None = None
    block_indent: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        if re.search(r":\s*[|>][+-]?\s*$", line):
            block_indent = indent
            continue
        declaration = re.fullmatch(
            r'\s*\{%-?\s*set\s+version\s*=\s*(["\'])([^"\']+)\1\s*-?%\}\s*', line
        )
        literal = re.fullmatch(
            r'\s*version:\s*(?:"([^"\']+)"|\'([^"\']+)\'|([0-9][0-9A-Za-z.+-]*))\s*',
            line,
        )
        if declaration:
            versions.append(declaration[2])
        elif literal:
            versions.append(
                next(group for group in literal.groups() if group is not None)
            )
        if source_indent is not None and indent <= source_indent:
            source_indent = None
        if re.fullmatch(r"\s*source:\s*", line):
            source_indent = indent
        if source_indent is not None and indent > source_indent:
            digest = re.fullmatch(
                r'\s*(?:-\s+)?sha256:\s*(?:"([0-9a-f]{64})"|\'([0-9a-f]{64})\'|([0-9a-f]{64}))\s*',
                line,
            )
            if digest:
                hashes.append(
                    next(group for group in digest.groups() if group is not None)
                )
    version_text = versions[0] if versions else ""
    c.require(
        bool(versions) and all(version == record["version"] for version in versions),
        "Feedstock recipe version mismatch or not literal",
    )
    expected = next(
        f["sha256"]
        for f in record["release"]["distributions"]
        if f["name"].endswith(".tar.gz")
    )
    c.require(
        expected in hashes, "Feedstock recipe source SHA256 mismatch or unavailable"
    )
    return dict(
        path=paths[0],
        version=version_text,
        source_sha256=expected,
        recipe_sha256=c.sha256(raw),
        head_sha=head["sha"],
    )


def feedstock(record: dict[str, Any]) -> dict[str, Any]:
    """Find one exact-version PR on the feedstock, retaining revision evidence."""
    repo = "conda-forge/gwexpy-feedstock"
    version = record["version"]
    pulls = c.pages(repo, "pulls?state=all")
    matches = [
        p
        for p in pulls
        if p.get("base", {}).get("repo", {}).get("full_name") == repo
        and re.search(
            r"(?<![0-9A-Za-z.])v?" + re.escape(version) + r"(?![0-9A-Za-z.])",
            p.get("title", ""),
        )
        and re.search(
            r"(?<![0-9A-Za-z.])" + re.escape(version) + r"(?![0-9A-Za-z.])",
            p.get("head", {}).get("ref", ""),
        )
    ]
    c.require(len(matches) <= 1, "Feedstock exact-version PR ambiguous")
    if not matches:
        return dict(
            status="pending_external", reason="Exact-version feedstock PR absent"
        )
    pull = matches[0]
    c.require(
        re.fullmatch(r"[0-9a-f]{40}", pull["head"]["sha"]), "Feedstock revision missing"
    )
    c.require(
        pull["state"] == "open" or pull.get("merged_at"), "Feedstock PR closed unmerged"
    )
    recipe = feedstock_recipe(pull, record)
    return dict(
        status="complete" if pull.get("merged_at") else "pending_review",
        pull_url=pull["html_url"],
        number=pull["number"],
        author=pull["user"]["login"],
        head_branch=pull["head"]["ref"],
        head_sha=pull["head"]["sha"],
        recipe=recipe,
        merged_at=pull.get("merged_at"),
        updated_at=pull["updated_at"],
        reason="Feedstock maintainer review outstanding",
    )


def github(record: dict[str, Any]) -> dict[str, Any]:
    """Rebind tag/source and verify frozen assets and source text hashes."""
    repo = record["publication"]["repository"]
    identity = record["release"]
    tag = identity["tag"]
    ref = c.api(repo, f"git/ref/tags/{tag}")["object"]
    c.require(
        ref["type"] == "tag" and ref["sha"] == identity["tag_object_sha"],
        "Tag identity changed",
    )
    obj = c.api(repo, f"git/tags/{ref['sha']}")
    c.require(
        obj["object"]["sha"] == identity["source_sha"]
        and obj["object"]["type"] == "commit",
        "Source identity changed",
    )
    release = c.api(repo, f"releases/tags/{tag}")
    c.require(
        not release["draft"]
        and not release["prerelease"]
        and release["tag_name"] == tag,
        "GitHub final Release mismatch",
    )
    assets = {f["name"]: f for f in release["assets"]}
    c.require(
        len(assets) == len(release["assets"]) and "promotion-manifest.json" in assets,
        "Duplicate/missing assets",
    )

    def read(name: str) -> bytes:
        return c.api(repo, f"releases/assets/{assets[name]['id']}", raw=True)

    raw = read("promotion-manifest.json")
    c.require(
        c.sha256(raw) == identity["promotion_manifest_sha256"],
        "Promotion manifest changed",
    )
    manifest = json.loads(raw)
    c.require(
        manifest["source_sha"] == identity["source_sha"]
        and manifest["tag"] == tag
        and manifest.get("version") == record["version"],
        "Manifest source mismatch",
    )
    c.validate_release_asset_names(manifest, set(assets))
    _, expected = c._expected_files(manifest)
    c.require(
        {
            f["name"]: (f["sha256"], f["size_in_bytes"])
            for f in identity["distributions"]
        }
        == expected,
        "Frozen distribution identity mismatch",
    )
    for artifact in manifest["artifacts"]:
        if artifact["role"] == "payload" or artifact["role"].startswith("sidecar:"):
            for f in artifact["files"]:
                content = read(f["name"])
                c.require(
                    len(content) == f["size_in_bytes"]
                    and c.sha256(content) == f["sha256"],
                    "Release asset identity mismatch",
                )
    frozen = {}
    for key in ("release_notes", "review_evidence"):
        entry = manifest[key]
        path = entry["path"]
        c.require(
            not path.startswith("/") and ".." not in path.split("/"),
            "Unsafe frozen source path",
        )
        data = c.api(repo, f"contents/{path}?ref={identity['source_sha']}")
        content = base64.b64decode(data["content"])
        c.require(c.sha256(content) == entry["sha256"], "Frozen source hash mismatch")
        frozen[key] = entry
        if key == "release_notes":
            c.require(
                path == f"release_notes/{tag}.md"
                and release["body"].strip() == content.decode().strip(),
                "Frozen Release notes mismatch",
            )
    commit = c.api(repo, f"commits/{identity['source_sha']}")
    return dict(
        readback_succeeded=True,
        published_at=release["published_at"],
        created_at=release["created_at"],
        tagger_date=obj["tagger"]["date"],
        source_committer_date=commit["commit"]["committer"]["date"],
        assets=[
            {
                key: entry[key]
                for key in (
                    "name",
                    "size",
                    "digest",
                    "browser_download_url",
                    "created_at",
                    "updated_at",
                )
                if key in entry
            }
            for entry in release["assets"]
        ],
        frozen=frozen,
    )


def observe(record: dict[str, Any]) -> dict[str, Any]:
    """Isolate service failures so successful independent reads remain usable."""
    record = copy.deepcopy(record)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = record["version"]
    repo = record["publication"]["repository"]
    query = urllib.parse.urlencode(
        {
            "q": "conceptrecid:19059422",
            "all_versions": "true",
            "size": 25,
        }
    )
    urls = dict(
        github=f"https://github.com/{repo}/releases/tag/v{version}",
        pypi=f"https://pypi.org/pypi/gwexpy/{version}/json",
        zenodo="https://zenodo.org/api/records?" + query,
        conda_feedstock="https://api.github.com/repos/conda-forge/gwexpy-feedstock/pulls",
        conda_package="https://api.anaconda.org/package/conda-forge/gwexpy",
    )
    for name in SERVICES:
        url = urls[name]
        try:
            if name == "github":
                result = github(record)
            elif name == "pypi":
                result = pypi(record, public_json(url))
            elif name == "zenodo":
                hits = zenodo_records(record, url)
                if not hits:
                    update(
                        record,
                        name,
                        "pending_external",
                        url,
                        dict(reason="Version Zenodo record absent"),
                        now,
                    )
                    continue
                result = zenodo(record, hits)
            elif name == "conda_package":
                result = conda_package(record, public_json(url))
            else:
                result = feedstock(record)
            status = result.get(
                "status", "pending_review" if name == "zenodo" else "complete"
            )
            update(record, name, status, url, result, now)
        except urllib.error.HTTPError as error:
            status = "pending_external" if error.code == 404 else "retryable_failure"
            update(record, name, status, url, dict(reason=f"HTTP {error.code}"), now)
        except (OSError, subprocess.CalledProcessError) as error:
            update(
                record,
                name,
                "retryable_failure",
                url,
                dict(reason=f"Transport failure: {type(error).__name__}"),
                now,
            )
        except (ValueError, KeyError, TypeError) as error:
            update(record, name, "blocked", url, dict(reason=str(error)), now)
    components = {
        name: record["items"][name]["status"]
        for name in ("conda_feedstock", "conda_package")
    }
    update(
        record,
        "conda",
        next(s for s in c.STATUSES if s in components.values()),
        urls["conda_package"],
        dict(
            components=components,
            reason="See separate feedstock PR and actual channel-package states",
        ),
        now,
    )
    record["items"]["conda"]["components"] = components
    record["post_release_status"] = next(
        s for s in c.STATUSES if any(i["status"] == s for i in record["items"].values())
    )
    c.validate_record(record)
    return record


def discover_records(
    repository: str, tag: str | None = None
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Discover durable branch data, isolating invalid entries by version."""
    default = c.api(repository, "")["default_branch"]
    refs = [default]
    for pull in c.pages(repository, "pulls?state=open"):
        head = pull["head"]
        if (
            (head.get("repo") or {}).get("full_name") == repository
            and re.fullmatch("post-release/" + c.TAG_PATTERN, head["ref"])
            and pull["base"]["ref"] == default
            and (tag is None or head["ref"] == "post-release/" + tag)
        ):
            refs.append(head["ref"])
    records: dict[str, dict[str, Any]] = {}
    failures = []
    for ref in refs:
        try:
            head_sha = c.api(repository, "git/ref/heads/" + ref)["object"]["sha"]
            commit = c.api(repository, "git/commits/" + head_sha)
            tree = c.api(repository, f"git/trees/{commit['tree']['sha']}?recursive=1")
            c.require(
                not tree.get("truncated"), "Cannot discover manifests in truncated tree"
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.CalledProcessError,
        ) as error:
            failures.append(failure(tag or ref, error))
            continue
        for entry in tree["tree"]:
            match = re.fullmatch(
                r"docs/developers/plans/manifests/audit-manifest-("
                + c.TAG_PATTERN
                + r")-post-release\.yaml",
                entry["path"],
            )
            if (
                not match
                or (tag is not None and tag != match[1])
                or (ref != default and ref != "post-release/" + match[1])
            ):
                continue
            try:
                c.require(
                    entry["type"] == "blob" and entry["mode"] == "100644",
                    "Unsafe durable manifest",
                )
                blob = c.api(repository, "git/blobs/" + entry["sha"])
                record = json.loads(base64.b64decode(blob["content"]))
                c.require(isinstance(record, dict), "Durable record must be an object")
                c.require(
                    isinstance(record.get("items"), dict)
                    and all(
                        isinstance(item, dict) for item in record["items"].values()
                    ),
                    "Durable record objectives must be objects",
                )
                c.validate_record(record)
                c.require(
                    record["publication"]["repository"] == repository
                    and record["release"]["tag"] == match[1],
                    "Durable record identity mismatch",
                )
                records[match[1]] = c.merge_manifest(records.get(match[1]), record)
            except (
                ValueError,
                KeyError,
                TypeError,
                OSError,
                subprocess.CalledProcessError,
            ) as error:
                failures.append(failure(match[1], error))
    return records, failures


def failure(tag: str, error: Exception) -> dict[str, str]:
    """Return a safe per-version diagnostic without inventing accepted identity."""
    return dict(
        tag=tag,
        status="retryable_failure"
        if isinstance(error, (OSError, subprocess.CalledProcessError))
        else "blocked",
        reason=f"{type(error).__name__}: {error}",
    )


def observe_with_durable(record: dict[str, Any]) -> dict[str, Any]:
    """Load prior identity before manual/automatic readback, including Zenodo."""
    records, failures = discover_records(
        record["publication"]["repository"], record["release"]["tag"]
    )
    c.require(
        not failures,
        "Durable evidence could not be safely loaded: " + json.dumps(failures),
    )
    prior = records.get(record["release"]["tag"])
    return observe(c.merge_manifest(prior, record))


def resume(repository: str) -> list[dict[str, str]]:
    """Resume versions independently, returning diagnostics for unsafe records."""
    records, results = discover_records(repository)
    failed = {r["tag"] for r in results}
    for tag, record in records.items():
        if tag in failed:
            continue
        try:
            c.validate_record(record)
            if record["post_release_status"] == "complete":
                continue
            for name in ("conda_feedstock", "conda_package"):
                if name not in record["items"]:
                    record["items"][name] = copy.deepcopy(record["items"]["conda"])
                    record["items"][name].update(
                        status="pending_external",
                        evidence=[],
                        pending_reason="Component readback missing",
                        resume_condition="Run component observer",
                    )
            observed = observe(record)
            c.persist(repository, observed)
            results.append(
                dict(
                    tag=tag,
                    status=observed["post_release_status"],
                    reason="Observed and persisted",
                )
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.CalledProcessError,
        ) as error:
            results.append(failure(tag, error))
    return results
