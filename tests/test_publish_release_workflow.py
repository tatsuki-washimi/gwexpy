"""Static security contract for the release workflow."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "publish-release.yml"
)


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_workflow_replaces_legacy_identity_and_is_manual_dry_run_only():
    workflow = read_workflow()
    assert not (WORKFLOW.parent / "release.yml").exists()
    assert "workflow_dispatch:" in workflow
    assert "release_ref:" in workflow
    assert "expected_tag:" in workflow
    assert "      publish:" not in workflow
    assert "workflow_dispatch must run with --ref main" in workflow


def test_all_actions_are_full_sha_pinned_and_publish_job_is_minimal():
    workflow = read_workflow()
    uses = re.findall(r"^\s*uses:\s*([^\s]+)$", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in uses)

    publish = workflow.split("\n  publish:\n", maxsplit=1)[1]
    publish_uses = re.findall(r"^\s*uses:\s*([^\s]+)$", publish, flags=re.MULTILINE)
    assert publish_uses == [
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "pypa/gh-action-pypi-publish@a892a5a61159132606e93a2fa6f4358831b04d26",
    ]
    assert "id-token: write" in publish
    assert workflow.count("id-token: write") == 1


def test_verify_separates_validator_and_source_trees_and_publish_requires_tag_push():
    """The verify job keeps validator code and validated source apart.

    This asserts separation and SHA pinning only. It deliberately does *not*
    assert that the `control` checkout is an independent trust boundary: on a
    tag push `github.workflow_sha` is the revision the tag points at, not a
    protected `main`, so a tag carrying a rewritten workflow would supply its
    own validator. The controls that bound that risk are the tag rulesets,
    the `pypi` environment, and the PyPI Trusted Publisher binding -- all
    configured outside this repository and documented in RELEASING.md.
    """
    workflow = read_workflow()
    assert "ref: ${{ github.workflow_sha }}" in workflow
    assert "path: control" in workflow
    assert "path: source" in workflow
    assert "--repo-root source" in workflow
    assert "source_sha" in workflow
    assert "scripts/validate_release.py" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "twine check --strict" in workflow
    assert "sys.prefix" in workflow


def test_verify_pins_python_before_running_the_validator():
    """The validator needs Python 3.11+ (`datetime.UTC`), so verify pins it.

    Without an explicit `setup-python`, the validator would run on whatever
    interpreter the runner image ships, letting an image update silently
    break release verification while build/smoke stay pinned.
    """
    workflow = read_workflow()
    verify = workflow.split("\n  verify:\n", maxsplit=1)[1].split("\n  build:\n")[0]
    setup_python = verify.index("actions/setup-python@")
    # The invocation path, not the bare script name: the latter also appears
    # in the explanatory comment above the setup-python step.
    validator = verify.index("python control/scripts/validate_release.py")
    assert setup_python < validator
    assert verify.count('python-version: "3.11"') == 1
