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
must be verified by readback before every release.  Keep two categories
apart: what configuration *enforces*, and what people merely *agree to do*.

### Enforced by configuration

Each of these is a setting whose state can be read back from an API and
recorded.  If a readback does not show it, it is not in effect.

- **Protected release tag rulesets** — `release-tags-create-admin-only` and
  `release-tags-integrity` on `refs/tags/v*`, restricting who may create a
  release tag and enforcing the final-release SemVer pattern.
- **Tag creation permission** — only admins may create `v*` tags, so an
  arbitrary contributor cannot start a publishing run at all.
- **Tag update and deletion restriction** — the `release-tags-integrity`
  ruleset must include the `update`, `deletion`, and `non_fast_forward`
  rules.  Only with those does a published tag actually become immutable,
  so that a verified SHA cannot be swapped after the fact.
- **GitHub Environment protection** — the `pypi` environment permits `v*`
  tags only, denies branch deployments, and gates the publish job.
- **PyPI Trusted Publisher binding** — bound to `publish-release.yml`
  specifically, so no other workflow in this repository can mint a token.
- **Exact SHA validation** — `validate_release.py` resolves the release ref
  to a 40-character SHA and rejects branch names, abbreviated SHAs, and
  arbitrary ref expressions; `build`, `smoke`, and `publish` then consume
  that SHA rather than the ref.

### Operational rules, not enforced

These depend on maintainer discipline.  They are worth stating, but they
must not be counted as controls when assessing the boundary.

- **Never retarget a published release tag.**  Until the `update`,
  `deletion`, and `non_fast_forward` rules above are actually present in the
  ruleset readback, tag immutability is a rule maintainers follow, not a
  guarantee the platform provides.
- **Run the candidate dry-run before tagging**, and read the result rather
  than assuming it.

A release is only as protected as the weakest of the *enforced* controls.
Treating the workflow file, or an unenforced operational rule, as the
boundary would overstate the guarantee.

### Readback commands

Record the full response of each in the release audit manifest:

```bash
gh api repos/:owner/:repo/rulesets
gh api repos/:owner/:repo/rulesets/<RULESET_ID>
gh api repos/:owner/:repo/environments/pypi
```

An empty `rulesets` response means no tag protection exists at all; in that
state every bullet under "Enforced by configuration" that names a ruleset is
unmet, regardless of what this document says.

`rules[].type` alone is not sufficient evidence.  A ruleset can carry exactly
the right rules and still constrain nothing, so the detail response must be
checked on every one of the following before a tag is pushed:

| Field | Required value | Why it matters |
|---|---|---|
| `enforcement` | `active` | `evaluate` and `disabled` report violations without blocking them, so the rules never take effect. |
| `target` | `tag` | A ruleset targeting branches does not constrain tag operations at all. |
| `conditions.ref_name.include` | `refs/tags/v*` | A different pattern leaves release tags outside the ruleset. |
| `rules[].type` | `creation`, `update`, `deletion`, `non_fast_forward`, `tag_name_pattern` | Without `update`/`deletion`/`non_fast_forward`, only the tag *name* is constrained, not whether an existing tag can be moved or removed. |
| `bypass_actors` | empty, or only actors that must not be able to defeat immutability | A bypass entry silently reinstates exactly the operations the rules above forbid, for the actors listed. |

Read `environments/pypi` back for the same reason: confirm the deployment
branch/tag policy actually restricts to `v*` tags and denies branch
deployments, rather than assuming it from the environment's existence.

The PyPI Trusted Publisher binding is not readable through the GitHub API.
Confirm it in the PyPI project's publishing settings: the binding must name
`publish-release.yml`, and the former `release.yml` binding must be absent.
Record both observations, including the absence, in the audit manifest.

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

`release-tags-integrity` must also carry the `update`, `deletion`, and
`non_fast_forward` rules.  Without them the ruleset constrains only what a
tag may be *named*, not whether an existing one can be moved or removed, and
the "Immutable tag operation" property claimed above does not hold.

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
