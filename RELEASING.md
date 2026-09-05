# Releasing GWexpy

The only release workflow is `publish-release.yml`.  Manual dispatches are
dry-runs and must be launched with `--ref main`, which the workflow enforces:

```bash
gh workflow run publish-release.yml --ref main \
  -f release_ref=<existing-final-tag-or-40-character-candidate-sha> \
  -f expected_tag=<final-version-tag> \
  -f review_evidence=docs/developers/plans/manifests/audit-manifest-v0.2.3-release-readiness.yaml
```

The accepted tag-specific plan, evidence schema/path, review lanes, S-to-R
paths, payload/integration schemas, artifact prefix, and protected refs are defined only in
`scripts/ci/release_contracts.json`.  A syntactically valid SemVer tag that is
not listed there is unsupported and fails closed.  Keep the v0.1.13 entry
unchanged as the historical release contract; add a new entry for each future
release instead of rewriting an older one.

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

## Frozen source, payload, and evidence

Both a candidate dispatch and a tag push run the frozen-tip validator.  It
loads the exact expected tag's `protected_refs` from the release contract and
requires every fetched `origin/<protected-ref>` tip to equal the validated
40-character source SHA.  The frozen v0.1.13 and v0.1.14 contracts require
`main` and `maint/0.1`; v0.2.0 and v0.2.2 require exactly `main` and
`maint/0.2`, as does v0.2.3.
A missing protected-ref fetch or any moved tip is a release failure; it is
never ignored as optional.

The PyPI upload artifact is a fresh `release-payload-<source-sha>` directory
containing exactly one normalized wheel and one sdist.  `LICENSE.sha256` and
the distribution SHA-256 manifest live only in the separate
`release-sidecars-<source-sha>` artifact.  Smoke tests select the manifest's
exact filename and reject missing, extra, symlinked, non-regular, substituted,
or version-mismatched payload files.  The publish job downloads the payload
artifact only.

The four required smoke reports are named exactly
`python-3.11-wheel.json`, `python-3.11-sdist.json`,
`python-3.12-wheel.json`, and `python-3.12-sdist.json`.  Their collector
accepts only typed release facts (not logs, URLs, credentials, or raw review
text) and emits a single allowlisted aggregate artifact whose name is selected from the exact release contract:
`v0113-integration-evidence-<40-character-source-sha>`,
`v0114-integration-evidence-<40-character-source-sha>`, or
`v020-integration-evidence-<40-character-source-sha>`, or
`v022-integration-evidence-<40-character-source-sha>`, or
`v023-integration-evidence-<40-character-source-sha>`.  It is retained for
90 days.  Record its artifact ID, API digest, `created_at`, and
`expires_at` in UTC; acceptance requires
`expires_at - created_at >= 90 days - 5 minutes`.
Repository retention policy may cap the configured duration, and run/artifact
deletion or expiry invalidates the evidence.

The measured `90 days - 5 minutes` threshold above applies to the integration
aggregate. All release artifact uploads request `retention-days: 90`, but the
current contract specifies no corresponding measured minimum for the payload,
sidecars, individual cell reports, or qualification aggregate. Record each
artifact's actual API timestamps and expiry separately; do not claim that the
integration threshold was verified for every artifact. The source validator
and evidence collectors do not read GitHub artifact expiry, so this acceptance
check is an external API readback before publication.

For v0.2.2, the historical same-build qualification also used a 19-cell
matrix. Every cell verified `distribution-sha256.json` before installation
and emitted its source SHA, version, and wheel/sdist digests. The historical
aggregate is named
`v022-qualification-evidence-<40-character-source-sha>`.

For v0.2.3, the same 19 named cells additionally record JUnit outcomes and
bind optional skips to the reviewed baseline in
`scripts/ci/v023_qualification_expected_skips.json`. A new or changed skip is
a release HOLD. The `qualification_evidence` job rejects a cell, digest,
source, payload, or skip mismatch and writes
`v023-qualification-evidence-<40-character-source-sha>`.
For both releases, the publish job depends on the version-specific
qualification aggregate as well as the four-cell smoke aggregate, so it
cannot publish a payload different from the qualified bytes.

Terra review evidence is advisory orchestration metadata, not identity proof,
legal approval, or publication authorization.  It contains the reviewed
commit, canonical `git ls-tree -r -z --full-tree` scope digest, sanitized
finding IDs, and verdict only.  Raw reports are not collected.  Human approval
and protected-environment controls remain the only publication authorization.

The coordinator-owned path selected for the expected tag by
`scripts/ci/release_contracts.json` is the sole release-gate review-evidence
path for tag runs.  For v0.1.14 it is
`docs/developers/plans/manifests/audit-manifest-v0.1.14-release-readiness.yaml`;
for v0.2.3 it is
`docs/developers/plans/manifests/audit-manifest-v0.2.3-release-readiness.yaml`.
At source commit `S`, the v0.2.3 source-preparation form deliberately starts
with the configured schema and an empty `entries` array, so it establishes the
review scope path but does not authorize a release. The executable gate rejects
that placeholder. Only the coordinator may replace it in `R` after all three
same-candidate lane reviews are `APPROVED`.

For v0.2.3, `S` and `R` must be distinct commits. Before accepting the allowed
`S..R` delta, the validator reads the configured evidence path directly from
`S` and requires the byte-exact empty placeholder. Validation fails closed if
the placeholder is absent, already populated, malformed, or contains extra YAML.

The selected evidence file must contain
exactly one top-level `review_evidence_json: |` block whose content is the
strict JSON review-evidence schema; no YAML text is allowed outside that block,
and duplicate JSON keys are rejected.  The document is size-bounded, `model`
and finding IDs are short identifiers, and `effort` is an allowlisted value so
the S-to-R evidence commit cannot carry raw review prose in approved fields.
Manual dispatches may name that same repository-relative path explicitly.  The
reviewed commit is `S`, while the validator binds it to `R` only when the
`S..R` diff contains only the tag-specific coordinator paths allowlisted in
the same contract.  Any plan
delta must be byte-identical except for existing checkbox transitions from
`[ ]` to `[x]`; frozen-tip validation happens only after this binding.

## Locked release build toolchain

`requirements/release-build.txt` is the Ubuntu/CPython 3.11 release-builder
lock.  Generate it from a disposable environment with `pip-compile
--generate-hashes`, including `pip`, `build`, `twine`, `setuptools`, `wheel`,
and every transitive dependency.  Review the exact versions and SHA-256 hashes
before replacing the file.  The workflow installs it with
`python -m pip install --require-hashes -r requirements/release-build.txt`
and builds with `python -m build --no-isolation`; it must never substitute a
floating `pip install --upgrade pip build twine` command.

Changing this lock invalidates the release artifact, all four smoke results,
the evidence aggregate, and the advisory review.  Regenerate the artifacts and
repeat the complete release review against the new source SHA.

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
  arbitrary contributor cannot start a publishing run at all.  This is the
  `creation` rule on `release-tags-create-admin-only`, whose `bypass_actors`
  list *is* the set of permitted creators rather than a set of exceptions.
- **Tag update and deletion restriction** — the `release-tags-integrity`
  ruleset must include the `update`, `deletion`, and `non_fast_forward`
  rules **and carry no bypass actors**.  Only then does a published tag
  actually become immutable, so that a verified SHA cannot be swapped after
  the fact.
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
checked on every one of the following before a tag is pushed.

These three are required identically on **both** rulesets:

| Field | Required value | Why it matters |
|---|---|---|
| `enforcement` | `active` | `evaluate` and `disabled` report violations without blocking them, so the rules never take effect. |
| `target` | `tag` | A ruleset targeting branches does not constrain tag operations at all. |
| `conditions.ref_name.include` | `refs/tags/v*` | A different pattern leaves release tags outside the ruleset. |

The remaining two fields are **not** shared, and the correct `bypass_actors`
state is the opposite in each ruleset.  A `creation`, `update`, or `deletion`
rule does not forbid its operation outright — it restricts that operation to
the actors listed in `bypass_actors`.  So "a bypass entry is a finding" is
right for one of these rulesets and wrong for the other, and applying it
uniformly misreads one of them every time:

| Ruleset | Required `rules[].type` | Required `bypass_actors` |
|---|---|---|
| `release-tags-create-admin-only` | `creation` | Exactly the actors permitted to create a release tag, enumerated. An empty list here means *no one* — not even an admin — can create a `v*` tag, so the release cannot be published at all. |
| `release-tags-integrity` | `update`, `deletion`, `non_fast_forward` | Empty. Any actor listed here may move or delete a published release tag, which is exactly the immutability this ruleset exists to provide. |

### `tag_name_pattern` is not available on this repository

An earlier revision of this document listed `tag_name_pattern` as a fourth
required rule on `release-tags-integrity`. It cannot be configured here: the
API rejects the rule type with `422 Validation Failed / Invalid rule
'tag_name_pattern'` regardless of the parameters supplied (verified 2026-08-08
against four payload shapes while preparing v0.1.13). The repository is
user-owned; the REST reference documents the type, but this repository cannot
accept it. Do not record its absence as a configuration defect, and do not
treat a readback without it as a failed check.

What enforces the release tag name instead:

- `release-tags-create-admin-only` already restricts creation of any
  `refs/tags/v*` to the enumerated bypass actors, so an arbitrary contributor
  cannot create a release tag of any name.
- `scripts/validate_release.py` applies the same final-release SemVer regex
  (`RELEASE_TAG_PATTERN`) to `expected_tag`, which on a tag push is
  `github.ref_name`. A malformed tag fails the `verify` job, so `build`,
  `smoke`, and `publish` never run and nothing is published.
- The `pypi` environment permits deployment only from `v*` tags.

The residual gap is cosmetic: a permitted creator can create a badly named
`v*` tag, which then fails CI and publishes nothing.

For each entry that legitimately appears on `release-tags-create-admin-only`,
record `actor_type`, `actor_id`, and `bypass_mode`.  `bypass_mode` is what
decides whether the bypass applies unconditionally, and the `pull_request`
mode available to branch rulesets does not exist for tags, so an entry that
looks gated on a branch ruleset is not gated here.

A non-empty `bypass_actors` on `release-tags-integrity` is a break-glass
exception, not a working configuration.  If one is present, record it as
weakening tag immutability — with the actor, the reason, and when it will be
removed — and do not report the "Immutable tag operation" property as held
while it stands.

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
`release-tags-integrity` rulesets use target selector `refs/tags/v*`.

The final-release SemVer restriction

```text
^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$
```

is enforced by `scripts/validate_release.py`, not by the ruleset -- see
"`tag_name_pattern` is not available on this repository" above.

`release-tags-integrity` must carry the `update`, `deletion`, and
`non_fast_forward` rules, with an empty `bypass_actors`.  Without those rules
the ruleset does not constrain whether an existing tag can be moved or removed;
with a bypass actor it names the operations but still permits them for that
actor.  In either case the "Immutable tag operation" property claimed above
does not hold.

`release-tags-create-admin-only` is the mirror image: its `creation` rule is
what makes tag creation an enumerated permission, so its `bypass_actors` must
list every actor allowed to create a release tag.  Leaving it empty locks out
the release itself.

Record both ruleset IDs and their GET responses in the release audit manifest.
The current IDs, read back on 2026-08-08 while preparing v0.1.13:

```text
release-tags-create-admin-only: 20117036
release-tags-integrity: 20117136
```

For an emergency recovery, first save the current ruleset JSON, then disable
the affected ruleset through the GitHub ruleset API.  Delete it only when
recreation is required; restore the saved configuration and read it back before
resuming releases.  Never delete or retarget an already published release tag.
