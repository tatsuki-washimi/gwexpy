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


def test_releasing_doc_separates_enforced_controls_from_operational_rules():
    """Immutability is a maintainer rule until the ruleset enforces it.

    A security-boundary document that lists an unenforced convention beside
    configured controls overstates the guarantee, so the two must stay in
    separate sections and the ruleset rules that would enforce tag
    immutability must be named explicitly.
    """
    # Collapse wrapping so the prose assertions below do not depend on where
    # the source lines happen to break.
    releasing = re.sub(
        r"\s+", " ", (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    )
    enforced = releasing.index("### Enforced by configuration")
    operational = releasing.index("### Operational rules, not enforced")
    assert enforced < operational
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule in releasing[enforced:operational]
    assert "not a guarantee the platform provides" in releasing[operational:]


def releasing_readback(*, collapse: bool) -> str:
    """Return the readback section, optionally with wrapping collapsed.

    Prose assertions use the collapsed form so they do not depend on line
    breaks; table rows are read from the raw form, because a Markdown row is
    a single line and collapsing would merge the two rulesets' rows together.
    """
    text = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    if collapse:
        text = re.sub(r"\s+", " ", text)
    return text[text.index("### Readback commands") :]


def test_releasing_doc_requires_full_ruleset_readback_fields():
    """`rules[].type` alone does not prove a ruleset constrains anything.

    An `evaluate` ruleset, one targeting branches, or one whose conditions
    miss `refs/tags/v*` can carry exactly the right rules and still permit the
    tag operations they name. The readback checklist must therefore pin every
    field an auditor has to look at.
    """
    readback = releasing_readback(collapse=True)
    for field in (
        "`enforcement`",
        "`active`",
        "`target`",
        "`conditions.ref_name.include`",
        "`refs/tags/v*`",
        "`bypass_actors`",
    ):
        assert field in readback, field
    # The Trusted Publisher binding has no GitHub API readback, so the doc
    # must say where to confirm it instead of implying the commands cover it.
    assert "not readable through the GitHub API" in readback


def test_readback_requires_opposite_bypass_actor_policies_per_ruleset():
    """A single `bypass_actors` rule for both rulesets is wrong either way.

    `creation`, `update`, and `deletion` restrict an operation *to* the
    bypass actors rather than forbidding it, so the two rulesets need
    opposite states: `release-tags-create-admin-only` must enumerate the
    permitted creators (empty locks out the release), while
    `release-tags-integrity` must be empty (any entry can move or delete a
    published tag). This asserts each ruleset's own row, so swapping the two
    policies fails rather than passing on a shared substring.
    """
    rows = {
        name: next(
            line
            for line in releasing_readback(collapse=False).splitlines()
            if line.startswith(f"| `{name}`")
        )
        for name in ("release-tags-create-admin-only", "release-tags-integrity")
    }
    creation, integrity = rows.values()

    # Each ruleset names only its own rules; a row listing the other's rules
    # would mean the responsibilities have been merged or transposed.
    assert "`creation`" in creation
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule not in creation, rule
    for rule in ("`update`", "`deletion`", "`non_fast_forward`", "`tag_name_pattern`"):
        assert rule in integrity, rule
    assert "`creation`" not in integrity

    # Opposite bypass requirements, each stated with its failure mode.
    assert "enumerated" in creation
    assert "An empty list here means" in creation
    assert "Empty." in integrity
    assert "may move or delete a published release tag" in integrity

    prose = releasing_readback(collapse=True)
    # Enumerated actors are only auditable with the fields that identify them
    # and say whether the bypass is conditional.
    for field in ("`actor_type`", "`actor_id`", "`bypass_mode`"):
        assert field in prose, field
    assert "break-glass" in prose


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
