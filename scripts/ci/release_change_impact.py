#!/usr/bin/env python3
"""Report path and evidence impact without changing release authorization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from release_contract import release_contract

SCHEMA = "gwexpy-release-change-impact-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
RELEASE_TEST_PATHS = (
    "tests/test_publish_release_workflow.py",
    "tests/test_release_contracts.py",
    "tests/test_release_evidence.py",
    "tests/test_release_gate_workflow_contract.py",
    "tests/test_release_go.py",
    "tests/test_release_human_approval.py",
    "tests/test_release_payload.py",
    "tests/test_release_promotion.py",
    "tests/test_release_review_evidence.py",
    "tests/test_validate_release_script.py",
    "tests/test_v026_review_evidence_v2.py",
)
PROPOSED_LANE_NAMES = frozenset(
    {"scientific-data-model", "release-security", "documentation"}
)

# These categories describe effects. Contract review lanes remain the source of
# truth for which reviewers cover a path; an unknown path never implies safety.
CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "scientific_product": ("gwexpy",),
    "scientific_tests": ("benchmarks",),
    "build_input": ("requirements",),
    "release_control": (
        ".github",
        "scripts",
        ".agent",
        ".clinerules",
        *RELEASE_TEST_PATHS,
    ),
    "public_documentation": (
        "docs",
        "docs_redesign",
        "release_notes",
        "tests/docs",
    ),
    "administrative_evidence": (
        "docs/developers/plans",
        "docs/developers/reports",
        "deliverables",
    ),
}
CATEGORY_EXACT: dict[str, tuple[str, ...]] = {
    "build_input": (
        "environment.yml",
        "MANIFEST.in",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
    ),
    "release_control": ("RELEASING.md",),
    "public_documentation": (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "MAINTAINERS.md",
        "README.md",
        "ROADMAP.md",
        "SECURITY.md",
    ),
    "administrative_evidence": (".zenodo.json", "CITATION.cff"),
}
PROPOSED_VALIDATIONS: dict[str, tuple[str, ...]] = {
    "scientific_product": ("relevant_scientific_tests",),
    "scientific_tests": ("relevant_scientific_tests",),
    "build_input": ("dependency_and_build_checks", "relevant_scientific_tests"),
    "release_control": ("workflow_and_release_control_checks",),
    "public_documentation": ("documentation_build_and_links",),
    "administrative_evidence": ("structured_record_syntax_check",),
}

SCIENTIFIC_INPUT_PATHS = (
    "gwexpy",
    "tests",
    "benchmarks",
    "environment.yml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "MANIFEST.in",
    "requirements.txt",
    "requirements",
    "requirements-dev.txt",
)


class ChangeImpactError(ValueError):
    """Raised when the requested change-impact report cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChangeImpactError(message)


def _scope_covers(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope.rstrip("/") + "/")


def classify_path(path: str) -> list[str]:
    """Return every known effect category for a normalized repository path."""
    _require(
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and "\x00" not in path
        and "\n" not in path
        and "\r" not in path
        and all(part not in {"", ".", ".."} for part in path.split("/")),
        "changed path is not a normalized repository-relative path",
    )
    categories = {
        category
        for category, prefixes in CATEGORY_PREFIXES.items()
        if any(_scope_covers(path, prefix) for prefix in prefixes)
    }
    structured_admin_record = path.endswith(
        (".json", ".yaml", ".yml", ".csv", ".cff")
    ) and any(
        _scope_covers(path, prefix)
        for prefix in (
            "docs/developers/plans",
            "docs/developers/reports",
            "deliverables",
        )
    )
    if structured_admin_record:
        categories.discard("public_documentation")
        categories.add("administrative_evidence")
    categories.update(
        category for category, files in CATEGORY_EXACT.items() if path in files
    )
    if path.startswith("tests/") and not any(
        _scope_covers(path, prefix)
        for prefix in (
            "tests/docs",
            *RELEASE_TEST_PATHS,
        )
    ):
        categories.add("scientific_tests")
    if path.startswith("requirements/"):
        categories.add("build_input")
    return sorted(categories)


def changed_paths(repo_root: Path, base: str, candidate: str) -> list[str]:
    """List both endpoints of renames as ordinary removed/added paths."""
    for label, revision in (("base", base), ("candidate", candidate)):
        _require(
            isinstance(revision, str) and SHA40.fullmatch(revision) is not None,
            f"{label} must be a full lowercase commit SHA",
        )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, candidate],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(
        ancestry.returncode == 0,
        "prior reviewed base must be an ancestor of the candidate",
    )
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            f"{base}..{candidate}",
            "--",
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, "cannot enumerate changed paths")
    raw_paths = result.stdout
    _require(
        not raw_paths or raw_paths.endswith(b"\0"),
        "git returned a malformed changed-path list",
    )
    try:
        paths = [
            part.decode("utf-8", errors="strict") for part in raw_paths.split(b"\0")
        ]
    except UnicodeDecodeError as exc:
        raise ChangeImpactError("changed paths are not valid UTF-8") from exc
    _require(
        paths[-1:] == [""] and all(paths[:-1]),
        "git returned an invalid changed-path list",
    )
    return sorted(set(paths[:-1]), key=lambda item: item.encode("utf-8"))


def _tree_digest(repo_root: Path, commit: str, paths: list[str]) -> str:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, "cannot calculate a scoped tree digest")
    return hashlib.sha256(result.stdout).hexdigest()


def _scientific_input_digest(repo_root: Path, commit: str) -> str:
    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
            "--",
            *SCIENTIFIC_INPUT_PATHS,
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, "cannot calculate scientific-input digest")
    included = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        try:
            _metadata, path_bytes = entry.split(b"\t", 1)
            path = path_bytes.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ChangeImpactError("git returned a malformed tree entry") from exc
        categories = classify_path(path)
        if categories and set(categories).intersection(
            {"scientific_product", "scientific_tests", "build_input"}
        ):
            included.append(entry + b"\0")
    return hashlib.sha256(b"".join(included)).hexdigest()


def _lanes_for_path(path: str, lanes: dict[str, list[str]]) -> list[str]:
    return sorted(
        name
        for name, scopes in lanes.items()
        if any(_scope_covers(path, scope) for scope in scopes)
    )


def _validate_changed_admin_records(
    repo_root: Path, candidate: str, records: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Parse changed structured evidence without interpreting its policy meaning."""
    validated = []
    for record in records:
        path = record["path"]
        if "administrative_evidence" not in record["categories"] or not path.endswith(
            (".json", ".yaml", ".yml", ".csv", ".cff")
        ):
            continue
        result = subprocess.run(
            ["git", "show", f"{candidate}:{path}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            validated.append({"path": path, "status": "deleted_requires_manual_review"})
            continue
        try:
            content = result.stdout.decode("utf-8", errors="strict")
            suffix = Path(path).suffix.lower()
            if suffix == ".json":
                parsed = json.loads(content)
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("JSON evidence must be an object or array")
            elif suffix == ".csv":
                rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
                if (
                    not rows
                    or not rows[0]
                    or any(len(row) != len(rows[0]) for row in rows)
                ):
                    raise ValueError(
                        "CSV evidence is empty or has inconsistent columns"
                    )
            else:
                import yaml

                try:
                    parsed = yaml.safe_load(content)
                except yaml.YAMLError as exc:
                    raise ChangeImpactError(
                        f"structured evidence is invalid: {path}"
                    ) from exc
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("YAML evidence must be a mapping or sequence")
        except (UnicodeDecodeError, ValueError, csv.Error) as exc:
            raise ChangeImpactError(f"structured evidence is invalid: {path}") from exc
        validated.append({"path": path, "status": "syntactically_valid"})
    return validated


def analyze_change_impact(
    repo_root: Path, *, tag: str, base: str, candidate: str
) -> dict[str, Any]:
    """Build an advisory report; never transfer evidence or authorize release."""
    contract = release_contract(tag)
    raw_lanes = contract.get("review_lanes")
    if (
        not isinstance(raw_lanes, dict)
        or not raw_lanes
        or not all(
            isinstance(name, str)
            and isinstance(scopes, list)
            and all(isinstance(scope, str) and scope for scope in scopes)
            for name, scopes in raw_lanes.items()
        )
    ):
        raise ChangeImpactError("release contract review lanes are malformed")
    lanes = cast(dict[str, list[str]], raw_lanes)
    paths = changed_paths(repo_root, base, candidate)
    records = []
    unmapped_paths = []
    for path in paths:
        categories = classify_path(path)
        assigned_lanes = _lanes_for_path(path, lanes)
        if not categories or not assigned_lanes:
            unmapped_paths.append(path)
        records.append(
            {
                "path": path,
                "categories": categories or ["unclassified"],
                "configured_review_lanes": assigned_lanes,
            }
        )

    all_lanes = sorted(lanes)
    affected_lanes = sorted(
        {name for record in records for name in record["configured_review_lanes"]}
    )
    if unmapped_paths:
        affected_lanes = all_lanes
    structured_records = _validate_changed_admin_records(repo_root, candidate, records)
    structured_records_valid = all(
        item["status"] == "syntactically_valid" for item in structured_records
    )

    lane_digests = {}
    for name, scopes in sorted(lanes.items()):
        before = _tree_digest(repo_root, base, scopes)
        after = _tree_digest(repo_root, candidate, scopes)
        lane_digests[name] = {
            "base_tree_sha256": before,
            "candidate_tree_sha256": after,
            "unchanged": before == after,
        }

    science_before = _scientific_input_digest(repo_root, base)
    science_after = _scientific_input_digest(repo_root, candidate)
    science_inputs_unchanged = science_before == science_after
    source_changed = base != candidate
    proposed_lanes: set[str] = set()
    if any("unclassified" in record["categories"] for record in records) or not set(
        lanes
    ).issubset(PROPOSED_LANE_NAMES):
        proposed_lanes.update(all_lanes)
    else:
        if (
            any(
                set(record["categories"]).intersection(
                    {"scientific_product", "scientific_tests", "build_input"}
                )
                for record in records
            )
            and "scientific-data-model" in lanes
        ):
            proposed_lanes.add("scientific-data-model")
        if (
            any(
                set(record["categories"]).intersection(
                    {"build_input", "release_control"}
                )
                for record in records
            )
            and "release-security" in lanes
        ):
            proposed_lanes.add("release-security")
        if (
            any(
                set(record["categories"]).intersection({"public_documentation"})
                for record in records
            )
            and "documentation" in lanes
        ):
            proposed_lanes.add("documentation")
    proposed_validations = {
        validation
        for record in records
        for category in record["categories"]
        for validation in PROPOSED_VALIDATIONS.get(category, ())
    }
    if unmapped_paths:
        proposed_validations.add("all_configured_checks_require_scope_review")
    return {
        "schema": SCHEMA,
        "mode": "advisory_read_only",
        "release_tag": tag,
        "base_sha": base,
        "candidate_sha": candidate,
        "changed_paths": records,
        "unmapped_paths": unmapped_paths,
        "changed_structured_records": structured_records,
        "current_contract_affected_lanes": affected_lanes,
        "proposed_impact_review_lanes": sorted(proposed_lanes),
        "proposed_local_validation_scopes": sorted(proposed_validations),
        "all_configured_review_lanes": all_lanes,
        "lane_scope_digests": lane_digests,
        "scientific_inputs": {
            "base_tree_sha256": science_before,
            "candidate_tree_sha256": science_after,
            "unchanged": science_inputs_unchanged,
            "reuse_candidate_only_after_contract_adoption": (
                science_inputs_unchanged
                and not unmapped_paths
                and structured_records_valid
            ),
        },
        "active_contract_effect": {
            "review_lanes": "all_configured_lanes_remain_required_for_exact_source",
            "source_approval": "invalidated_for_new_source_sha"
            if source_changed
            else "unchanged_source_sha",
            "candidate_artifacts_and_qualification": "must_bind_new_candidate_run_and_exact_artifacts"
            if source_changed
            else "no_source_change",
            "release_owner_go": "must_bind_new_candidate_run_manifest_and_hashes"
            if source_changed
            else "no_source_change",
            "review_evidence_transfer": "not_authorized_by_this_report",
        },
        "adoption_required_before_reuse": True,
        "publication_authorized": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--base", required=True, help="prior reviewed full commit SHA")
    parser.add_argument(
        "--candidate", required=True, help="new candidate full commit SHA"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = analyze_change_impact(
            args.repo_root.resolve(),
            tag=args.release_tag,
            base=args.base,
            candidate=args.candidate,
        )
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, subprocess.SubprocessError, ChangeImpactError, ValueError) as exc:
        print(f"release impact report rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
