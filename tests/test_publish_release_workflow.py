"""Static security contract for the release workflow."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

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
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
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
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule in integrity, rule
    assert "`creation`" not in integrity
    # `tag_name_pattern` belongs to neither row: the API refuses the rule type
    # on this repository, so requiring it here would fail every readback on a
    # check that can never pass. See the dedicated test below.
    assert "`tag_name_pattern`" not in integrity

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


def test_readback_explains_the_unavailable_tag_name_rule_and_its_substitute():
    """An absent control must be explained, not silently dropped.

    `tag_name_pattern` cannot be configured on this repository -- the API
    rejects the rule type regardless of parameters -- so the readback table
    omits it. Omitting it without a reason would read as an oversight to the
    next auditor, who would then either re-attempt the impossible change or
    record a false finding. The doc must therefore say it is unavailable and
    name what enforces the tag name in its place, so the absence is auditable
    as a deliberate state rather than a gap.
    """
    prose = releasing_readback(collapse=True)
    assert "`tag_name_pattern` is not available on this repository" in prose
    # The evidence, so a future reader need not rediscover it by trying again.
    assert "Invalid rule 'tag_name_pattern'" in prose
    # Each substitute control, named where an auditor can verify it.
    assert "release-tags-create-admin-only" in prose
    assert "RELEASE_TAG_PATTERN" in prose
    assert "`github.ref_name`" in prose
    # The residual gap is stated rather than implied to be closed.
    assert "residual gap" in prose.lower()


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


def test_release_smoke_covers_both_artifacts_on_python_311_and_312():
    workflow = read_workflow()
    smoke = workflow.split("\n  smoke:\n", maxsplit=1)[1].split("\n  publish:\n")[0]

    for token in (
        'python-version: ["3.11", "3.12"]',
        "distribution: [wheel, sdist]",
        "${{ matrix.python-version }}",
        "${{ matrix.distribution }}",
        "gwexpy.register_all()",
        "LICENSE.sha256",
        "distribution-sha256.json",
        "retention-days: 90",
    ):
        assert token in workflow if token == "retention-days: 90" else token in smoke


def test_release_smoke_executes_with_license_sidecar_path(tmp_path):
    """The shell/Python boundary passes the sidecar path, not its hash value."""
    workflow = read_workflow()
    smoke = workflow.split("\n  smoke:\n", maxsplit=1)[1].split(
        "\n  publish:\n", maxsplit=1
    )[0]

    invocation = re.search(
        r'"\$smoke_dir/venv/bin/python" - "\$artifact" "\$(?P<name>[a-z_]+)" "\$REPORT" <<\'PY\'',
        smoke,
    )
    assert invocation is not None
    argument_name = invocation.group("name")

    license_bytes = b"release-license\n"
    license_digest = hashlib.sha256(license_bytes).hexdigest()
    license_sidecar = tmp_path / "LICENSE.sha256"
    license_sidecar.write_text(license_digest + "\n", encoding="ascii")
    artifact = tmp_path / "gwexpy-0.1.13-py3-none-any.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("gwexpy-0.1.13.dist-info/licenses/LICENSE.txt", license_bytes)

    if f'{argument_name}="$(cat ' in smoke:
        license_argument = license_digest
    else:
        assert (
            f'{argument_name}="${{{{ runner.temp }}}}/release-sidecars/LICENSE.sha256"'
            in smoke
        )
        license_argument = str(license_sidecar)

    embedded = re.search(
        r"<<'PY'\n(?P<body>.*?)\n          PY",
        smoke,
        flags=re.DOTALL,
    )
    assert embedded is not None
    script = textwrap.dedent(embedded.group("body"))
    script = script.split(
        'assert gwexpy.__version__ == os.environ["EXPECTED_VERSION"]', maxsplit=1
    )[0]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(artifact),
            license_argument,
            str(tmp_path / "report.json"),
        ],
        input=script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert argument_name == "expected_license_hash_file"


def test_releasing_manual_dispatch_supplies_v0114_review_evidence():
    releasing = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    assert (
        "-f review_evidence="
        "docs/developers/plans/manifests/"
        "audit-manifest-v0.1.14-release-readiness.yaml"
    ) in releasing


def test_workflow_is_payload_only_locked_and_collects_same_run_evidence():
    workflow = read_workflow()
    assert "--require-hashes -r requirements/release-build.txt" in workflow
    assert "python -m build --no-isolation" in workflow
    assert "pip install --upgrade pip build twine" not in workflow
    assert "release-payload-${{ needs.verify.outputs.source_sha }}" in workflow
    assert "release-sidecars-${{ needs.verify.outputs.source_sha }}" in workflow
    publish = workflow.split("\n  publish:\n", maxsplit=1)[1]
    assert "release-payload-${{ needs.verify.outputs.source_sha }}" in publish
    assert "release-sidecars-${{ needs.verify.outputs.source_sha }}" not in publish
    assert 'find "$artifact_dir"' not in workflow
    assert "--frozen-tip" in workflow
    assert "--review-evidence" in workflow
    assert "review_evidence:" in workflow
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.review_evidence" in workflow
    )
    assert "artifact_prefix: ${{ steps.validate.outputs.artifact_prefix }}" in workflow
    assert 'print "artifact_prefix=" $2' in workflow
    assert "assemble_release_evidence.py" in workflow
    assert (
        "name: ${{ needs.verify.outputs.artifact_prefix }}-"
        "${{ needs.verify.outputs.source_sha }}"
    ) in workflow
    assert "audit-manifest-v0.1.13-sol-followup.yaml" not in workflow
    assert "v0113-integration-evidence-" not in workflow


def test_workflow_dispatch_inputs_are_preserved_for_manual_candidates():
    workflow = read_workflow()
    dispatch = workflow.split("  workflow_dispatch:\n", maxsplit=1)[1].split(
        "\npermissions:", maxsplit=1
    )[0]

    for name in ("release_ref", "expected_tag", "review_evidence"):
        assert f"      {name}:" in dispatch
    assert dispatch.count("required: true") == 3


def test_workflow_fetches_the_exact_tags_contract_protected_refs():
    workflow = read_workflow()
    fetch = workflow.split(
        "      - name: Fetch frozen protected branch tips\n", maxsplit=1
    )[1].split("\n      - name: Validate metadata", maxsplit=1)[0]

    assert "python control/scripts/ci/release_contract.py --protected-ref" in fetch
    assert "$EXPECTED_TAG" in fetch
    assert "release contract lookup failed" in fetch
    assert "+refs/heads/${protected_ref}:refs/remotes/origin/${protected_ref}" in fetch
    assert "maint/0.1" not in fetch


def test_workflow_contract_revision_disagreement_fails_closed(tmp_path: Path):
    """The workflow producer and validator consumer must use one revision.

    A hypothetical mixed revision fetches v0.2.0's `maint/0.2` from one
    control tree while a validator from another requires `maint/0.3`.  The
    validator must reject the missing second revision's ref rather than
    accepting the successfully fetched first revision's refs.
    """
    root = WORKFLOW.parents[2]

    def write_control_revision(name: str, maintenance_ref: str) -> Path:
        control = tmp_path / name / "scripts"
        ci = control / "ci"
        ci.mkdir(parents=True)
        for filename in ("release_contract.py", "release_contracts.json"):
            shutil.copy2(root / "scripts" / "ci" / filename, ci / filename)
        shutil.copy2(root / "scripts" / "validate_release.py", control)
        contracts_path = ci / "release_contracts.json"
        contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
        contracts["releases"]["v0.2.0"]["protected_refs"] = [
            "main",
            maintenance_ref,
        ]
        contracts_path.write_text(json.dumps(contracts), encoding="utf-8")
        return control

    producer = write_control_revision("producer", "maint/0.2")
    consumer = write_control_revision("consumer", "maint/0.3")
    emitted = subprocess.run(
        [
            sys.executable,
            str(producer / "ci" / "release_contract.py"),
            "--protected-ref",
            "v0.2.0",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert emitted == ["main", "maint/0.2"]

    repo = tmp_path / "source"
    repo.mkdir()
    for args in (
        ("init", "-b", "main"),
        ("config", "user.name", "Release Test"),
        ("config", "user.email", "release-test@example.invalid"),
        ("commit", "--allow-empty", "-m", "source"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    for ref in emitted:
        subprocess.run(
            ["git", "update-ref", f"refs/remotes/origin/{ref}", source_sha],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    spec = importlib.util.spec_from_file_location(
        "consumer_validate_release", consumer / "validate_release.py"
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    with pytest.raises(validator.ReleaseValidationError, match="origin/maint/0.3"):
        validator.validate_frozen_tip(repo, source_sha, expected_tag="v0.2.0")
