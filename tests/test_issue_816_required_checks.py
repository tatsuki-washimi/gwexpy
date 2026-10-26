"""Regression contract for the required-check producers fixed by Issue #816."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
EVIDENCE = (
    ROOT
    / "docs"
    / "developers"
    / "plans"
    / "manifests"
    / "issue-816-required-checks-v1.json"
)


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def load_evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def matrix_cells(job: dict[str, Any]) -> list[dict[str, str] | None]:
    matrix = job.get("strategy", {}).get("matrix")
    if matrix is None:
        return [None]
    axes = {
        key: values
        for key, values in matrix.items()
        if key != "include" and isinstance(values, list)
    }
    return [
        dict(zip(axes, values, strict=True))
        for values in itertools.product(*axes.values())
    ]


def render_job_name(job: dict[str, Any], cell: dict[str, str] | None) -> str:
    template = job.get("name", "")
    if cell is None:
        return template
    return re.sub(
        r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}",
        lambda match: cell[match.group(1)],
        template,
    )


def test_evidence_maps_exactly_the_23_required_contexts_to_workflow_jobs() -> None:
    evidence = load_evidence()
    assert evidence["schema_version"] == 1
    assert evidence["issue"] == 816
    assert evidence["ruleset"] == 25554248
    assert evidence["protected_branches"] == ["main", "maint/0.2"]
    assert len(evidence["required_contexts"]) == 23
    assert len({item["context"] for item in evidence["required_contexts"]}) == 23

    actual: dict[tuple[str, str, str | None], str] = {}
    for item in evidence["required_contexts"]:
        workflow_file = item["workflow_file"]
        workflow = load_yaml(WORKFLOW_ROOT / workflow_file)
        assert workflow["name"] == evidence["producers"][workflow_file]["workflow"]
        job = workflow["jobs"][item["job"]]
        matching_cells = [cell for cell in matrix_cells(job) if cell == item["matrix"]]
        assert len(matching_cells) == 1
        job_name = render_job_name(job, matching_cells[0])
        key = (workflow_file, item["job"], json.dumps(item["matrix"], sort_keys=True))
        actual[key] = job_name
        assert item["context"] == actual[key]


def test_all_producers_run_on_each_protected_pr_and_push_branch() -> None:
    evidence = load_evidence()
    protected = evidence["protected_branches"]
    for workflow_file, producer in evidence["producers"].items():
        workflow = load_yaml(WORKFLOW_ROOT / workflow_file)
        triggers = workflow["on"]
        pull_request = triggers["pull_request"]
        assert pull_request["branches"] == producer["pull_request_branches"]
        assert set(protected).issubset(pull_request["branches"])
        assert "paths" not in pull_request
        assert "paths-ignore" not in pull_request
        if producer["push_branches"]:
            assert triggers["push"]["branches"] == producer["push_branches"]
            assert set(protected).issubset(triggers["push"]["branches"])
        else:
            assert "push" not in triggers
        assert set(protected).issubset(producer["pull_request_branches"])
        assert producer["old_pull_request_path_behavior"]
        assert producer["new_pull_request_path_behavior"].startswith("No path filter;")


def test_required_jobs_have_no_job_level_conditions_or_dependency_skips() -> None:
    evidence = load_evidence()
    for item in evidence["required_contexts"]:
        workflow = load_yaml(WORKFLOW_ROOT / item["workflow_file"])
        job = workflow["jobs"][item["job"]]
        assert "if" not in job, item["context"]
        assert "needs" not in job, item["context"]
