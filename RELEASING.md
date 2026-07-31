# Releasing GWexpy

The only release workflow is `publish-release.yml`.  Manual dispatches are
dry-runs and must be launched with `--ref main`, which the workflow enforces:

```bash
gh workflow run publish-release.yml --ref main \
  -f release_ref=<existing-final-tag-or-40-character-candidate-sha> \
  -f expected_tag=<final-version-tag>
```

`release_ref` is either an existing annotated final-release tag (strict mode)
or a lowercase 40-character candidate SHA.  Branch names, abbreviated SHAs,
and arbitrary ref expressions are rejected.  Candidate mode requires that the
expected tag does not exist; it verifies package, CITATION, Zenodo, and
CHANGELOG version/date consistency but cannot verify a tagger date.

Before pushing a final tag, run a candidate dry-run against the release SHA,
then create an annotated tag locally.  Verify the tag name, peeled commit SHA,
and UTC tagger date against the release metadata.  Signed tags are recommended.
Push the tag only after the candidate result, target-SHA CI, physics sign-off,
PyPI publisher, environment, and ruleset readbacks are all approved.

After a tag push, the strict workflow must pass verify, build, smoke, and
publish.  Confirm the PyPI distribution/version, GitHub Release, Zenodo, and
conda follow-up state before declaring release acceptance.

## Where the trust boundary actually is

The `verify` job checks out two trees: the workflow revision (`control`) and
the revision under test (`source`).  That separation keeps the validator from
grading its own working tree, and it pins the source to an exact SHA that
`build`, `smoke`, and `publish` all reuse.

**The dual checkout is not, by itself, protection against a modified tag
revision.**  On a `workflow_dispatch` run the control revision is pinned to
`main` by the `--ref main` enforcement above.  On a *tag push*,
`github.workflow_sha` is the revision the tag points at -- it identifies the
commit containing the workflow file, not an independently protected `main`.
A tag carrying a rewritten workflow would therefore run its own validator.
Do not read the two checkouts as a protected-controller architecture.

What actually bounds that risk is configured outside this repository, and
must be verified by readback before every release:

- **Protected release tag rulesets** — `release-tags-create-admin-only` and
  `release-tags-integrity` on `refs/tags/v*`, restricting who may create a
  release tag and enforcing the final-release SemVer pattern.
- **Tag creation permission** — only admins may create `v*` tags, so an
  arbitrary contributor cannot start a publishing run at all.
- **Immutable tag operation** — published release tags are never deleted,
  moved, or retargeted, so a verified SHA cannot be swapped afterwards.
- **GitHub Environment protection** — the `pypi` environment permits `v*`
  tags only, denies branch deployments, and gates the publish job.
- **PyPI Trusted Publisher binding** — bound to `publish-release.yml`
  specifically, so no other workflow in this repository can mint a token.
- **Exact SHA validation** — `validate_release.py` resolves the release ref
  to a 40-character SHA and rejects branch names, abbreviated SHAs, and
  arbitrary ref expressions; downstream jobs consume that SHA, not the ref.

A release is only as protected as the weakest of those six.  Treating the
workflow file as the boundary would overstate the guarantee.

## Publisher rotation and protection recovery

The former `release.yml` Trusted Publisher binding must be removed before the
binding for `publish-release.yml` is added.  A tag on an old commit can still
start the historical workflow, but it cannot obtain a PyPI token after that
old binding has been removed.

The `pypi` environment permits `v*` tags only and does not permit branch
deployments.  The `release-tags-create-admin-only` and
`release-tags-integrity` rulesets use target selector `refs/tags/v*`; the
latter additionally enforces the final-release SemVer metadata restriction:

```text
^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$
```

After creation, record both ruleset IDs and their GET responses in the release
audit manifest, then replace the placeholders below in a docs-only PR.

```text
release-tags-create-admin-only: <RULESET_ID>
release-tags-integrity: <RULESET_ID>
```

For an emergency recovery, first save the current ruleset JSON, then disable
the affected ruleset through the GitHub ruleset API.  Delete it only when
recreation is required; restore the saved configuration and read it back before
resuming releases.  Never delete or retarget an already published release tag.
