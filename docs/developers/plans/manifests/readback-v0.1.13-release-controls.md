# v0.1.13 release control readback

Recorded 2026-08-08. Release SHA **R = `f7f836eec7e6247a01e9a1b61cc1a2121235e58d`**
(`origin/main` == `origin/maint/0.1` == R at the time of reading).

## Enforced by configuration

### ruleset `release-tags-create-admin-only` (id 20117036)

| Field | Value | RELEASING.md requirement | Met |
|---|---|---|---|
| `enforcement` | `active` | `active` | yes |
| `target` | `tag` | `tag` | yes |
| `conditions.ref_name.include` | `["refs/tags/v*"]` | `refs/tags/v*` | yes |
| `rules[].type` | `["creation"]` | `creation` | yes |
| `bypass_actors` | `[{actor_type: RepositoryRole, actor_id: 5, bypass_mode: always}]` | enumerated permitted creators, non-empty | yes |

### ruleset `release-tags-integrity` (id 20117136)

| Field | Value | RELEASING.md requirement | Met |
|---|---|---|---|
| `enforcement` | `active` | `active` | yes |
| `target` | `tag` | `tag` | yes |
| `conditions.ref_name.include` | `["refs/tags/v*"]` | `refs/tags/v*` | yes |
| `rules[].type` | `["update", "deletion", "non_fast_forward"]` | `update`, `deletion`, `non_fast_forward`, **`tag_name_pattern`** | **partial — `tag_name_pattern` absent** |
| `bypass_actors` | `[]` | empty | yes |

**Finding**: the SemVer `tag_name_pattern` rule that RELEASING.md lines 125 and
158-162 list as required is not present, and **cannot be added to this
repository**. Four payload shapes (full parameters, no `name`, minimal, and a
`starts_with` operator) were each rejected with an identical
`422 Validation Failed / Invalid rule 'tag_name_pattern'`, so the rule *type* is
unavailable rather than the payload being malformed. The repository is
user-owned, public, `owner.plan: null`. The GitHub REST reference documents the
type but the API refuses it here. The ruleset was verified unchanged after each
attempt.

Tag immutability (`update`/`deletion`/`non_fast_forward`, empty bypass list) is
unaffected and remains in effect.

**Compensating controls for the tag-name format**, all verified:

1. `release-tags-create-admin-only` restricts creation of any `refs/tags/v*` to
   the enumerated bypass actor (RepositoryRole 5), so a non-admin cannot create
   a release tag at all.
2. `scripts/validate_release.py` enforces the *same* regex
   (`RELEASE_TAG_PATTERN`, lines 15-17) that the ruleset would have used. On a
   tag push `EXPECTED_TAG` is `github.ref_name`, so a malformed tag fails the
   `verify` job and never reaches `publish`. Measured:

   | `expected_tag` | validator result |
   |---|---|
   | `v0.1.13rc1` | `expected_tag is not a final SemVer tag` |
   | `v0.1` | `expected_tag is not a final SemVer tag` |
   | `0.1.13` | `expected_tag is not a final SemVer tag` |

3. `environments/pypi` permits deployment only from `v*` tags.

Residual risk: an admin can create a badly named `v*` tag, which then fails CI
and publishes nothing. That is cosmetic, not an integrity gap.

**Not fixed before this release, deliberately**: `RELEASING.md` is inside lane
A's reviewed scope, so amending its wording before tagging would change lane A's
`scope_digest` and invalidate the review evidence, forcing a genuine lane A
re-review. Correct the wording in the v0.1.14 cycle, through review.

### environment `pypi`

| Field | Value |
|---|---|
| `protection_rules` | `["branch_policy"]` |
| `deployment_branch_policy.protected_branches` | `false` |
| `deployment_branch_policy.custom_branch_policies` | `true` |
| policies | exactly one: `{name: "v*", type: "tag"}` |

Tag-only deployment to `v*` is in effect; branch deployments are denied.

### Exact SHA validation

`scripts/validate_release.py --repo-root . --release-ref f7f836eec... --expected-tag v0.1.13 --review-evidence ... --frozen-tip` returns:

```
mode=candidate
source_sha=f7f836eec7e6247a01e9a1b61cc1a2121235e58d
version=0.1.13
release_date=2026-08-08
```

## Not readable through the GitHub API — requires maintainer confirmation

- **PyPI Trusted Publisher binding**: must name `publish-release.yml`; the
  former `release.yml` binding must be absent. Both observations, including
  the absence, need to be recorded from the PyPI project's publishing
  settings before the tag is pushed.

## Review evidence

`docs/developers/plans/manifests/audit-manifest-v0.1.13-sol-followup.yaml`
carries lane A (release tooling) and lane B (public docs), both
`verdict: APPROVED`, `model: gpt-5.6-terra`, reviewed 2026-08-07T20:10:28Z.

`reviewed_commit` was repointed from `2780a8f7f` to `9f3559d51` in commit
`f7f836eec`, because completing the changelog after the reviews put
`CHANGELOG.md` into the S-to-R delta. Every path in both lanes' scopes is
byte-identical between the two commits, so both `scope_digest` values are
unchanged and still bind each verdict to the content actually read;
`validate_release_review_evidence.py` confirms this at the new commit. The
tree at `9f3559d51` is identical to the tree at the merge commit `cf2ef50e3`.

**Not claimed**: no new review was run. `timestamp_utc` and
`raw_report_sha256` still refer to the 2026-08-07 reviews. The evidence
schema fixes lane A/B scope paths in the validator and has no lane covering
`CHANGELOG.md`, so the changelog completion (`9f3559d51`, +141 lines, docs
only, no source change) carries no independent reviewer verdict.

## Measured CI results

At `9f3559d51` (run 31238158048), all 14 required checks pass:

| Gate | Result |
|---|---|
| PR-fast (ruff, mypy, pytest, smoke build) | 6283 passed, 484 skipped, 28 deselected, 7 xfailed |
| Core I/O contract | 824 passed, 180 skipped |
| GWF I/O release | 20 passed |
| NetCDF I/O release | 45 passed |
| ROOT interoperability release | 15 passed |
| Interop MNE | 69 passed |
| I/O conformance | 51 passed, 18 skipped |

The "7894 passed" figure in PR #643's body comes from an earlier full-suite
run, not from this SHA.

## Milestone reconciliation (completed 2026-08-08, before tagging)

Milestone v0.1.13 (id 8) now reads `open_issues: 0`, `closed_issues: 25`.

Closed with a comment naming the resolving commit or PR:

- Landed on `main` before the RC: #511 (PR #621), #605 (PR #626), #608 (PR #627),
  #610 (PR #622), #611 (PR #624)
- Landed in PR #643: #451, #559, #578, #579, #593, #614, #615, #617, #618, #620
- PR #453 closed as superseded by `9acbfe465` inside PR #643

No issue was closed without a cited fix.

## GitHub Release creation

`publish-release.yml` does **not** create a GitHub Release: every `permissions`
block in it is `contents: read` and it contains no release step. The v0.1.12
release was created by `tatsuki-washimi` with no attached assets. GitHub Release
creation for v0.1.13 is therefore a manual step after the tag push and a
successful PyPI publish.

## Distribution state at time of writing

| Channel | Version |
|---|---|
| source (`main`, `maint/0.1`) | 0.1.13 |
| GitHub Release (latest) | 0.1.12 |
| PyPI | 0.1.12 (tag push pending) |
| conda-forge `main` recipe | 0.1.11 (`recipe/recipe.yaml`, rattler-build v1) |
| conda-forge open bot PR | #7, "gwexpy v0.1.12", MERGEABLE |

conda-forge is not part of this release transaction. After PyPI carries 0.1.13,
supersede feedstock PR #7 by updating the recipe directly to 0.1.13 rather than
merging the 0.1.12 PR first.

## PyPI Trusted Publisher (maintainer readback, 2026-08-08)

Confirmed visually in the PyPI project's Publishing settings, since this
binding is not readable through the GitHub API.

"Manage current publishers" lists exactly one entry:

| Field | Value |
|---|---|
| Publisher | GitHub |
| Repository | `tatsuki-washimi/gwexpy` |
| Workflow | `publish-release.yml` |
| Environment name | `pypi` |

Both required observations hold: the `publish-release.yml` binding is present,
and the former `release.yml` binding is **absent** (no second publisher is
listed). The environment name matches the `environment: pypi` declared by the
publish job, so the GitHub environment's `v*`-tag-only deployment policy gates
the token.

