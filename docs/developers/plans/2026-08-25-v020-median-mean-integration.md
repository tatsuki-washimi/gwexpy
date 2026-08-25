# v0.2.0 median-mean integration plan

Status: planned

## Objective and scope

Include the already reviewed `agent/v020-wave2-median-mean` series in the
isolated v0.2.0 integration candidate.  The source series is exactly the
three commits `2cc96203b8c55e1e0402a3a4f69360373d2fb42f`,
`a5aa3a91ad4b9ec6d17b87932ccf45a30eebe263`, and
`86d1caa8107431691db755868bc24894ed00b740`, whose common base is
`6bca889a7b719f49be4f535f360b7dd63f613283`.

The candidate must provide:

- the public `TimeSeries.psd()` / `TimeSeries.asd()` `method="median-mean"`
  dispatch and public `median_bias(n)`;
- LAL/FINDCHIRP-compatible numerical coverage and conditional PyCBC oracle
  coverage retained from the reviewed series;
- a GWexpy-owned spectral metadata contract: `name`, `channel`, and `epoch`
  are restored at the public `psd()` / `asd()` boundary, while the backend
  remains authoritative for numeric values, units, and the frequency axis;
- explicit metadata-contract regression coverage for both `median-mean` and
  one standard GWpy method (`welch`), so restoration is not presented as a
  median-mean-only workaround.

This scope does not change a GWpy normative guarantee.  It makes the
metadata-rich GWpy data model a stable GWexpy public contract across backend
and subclass-conversion boundaries.

Out of scope: release/version changes, tags, publication, remote/GitHub
mutation, the unrelated PR #686 merge itself, and any alteration to the
original dirty worktree.

## Detailed roadmap

### Phase 1 — preflight and series application

Status: planned

1. Record clean integration status, exact `HEAD`, and the original-worktree
   porcelain hash as read-only evidence.
2. Confirm source ancestry and inspect each of the three commits.
3. Write the backend-independent metadata-restoration regression before
   importing production code and record its expected RED result against the
   pre-#686 candidate.
4. Cherry-pick those commits in their source order onto the integration
   branch.  If a conflict occurs, stop, preserve the conflict evidence, and
   resolve only the overlap required to retain both the previously integrated
   feature and the median-mean contract.
5. Record the resulting commit mapping and any resolution in the v0.2
   integration audit manifest.

### Phase 2 — contract completion

Status: planned

1. Add a concise, backend-independent metadata preservation regression that
   covers `psd()` and `asd()` with `method="welch"` as well as
   `method="median-mean"`.
2. Verify output type, `name`, `channel`, `epoch`, unit transformation, and
   frequency-grid metadata.  Do not assert that GWpy itself guarantees all
   three restored metadata attributes.
3. Update public/developer release documentation only if existing wording
   incorrectly treats this as a GWpy guarantee rather than a GWexpy contract.

### Phase 3 — verification and review

Status: planned

Run in the existing GWexpy conda environment with `PYTHONPATH=$PWD`:

1. Focused spectral contract tests, including `tests/signal/test_median_bias.py`
   and `tests/signal/test_median_mean_dispatch.py`.
2. Relevant TimeSeries spectral tests and the existing v0.2 focused regression
   matrix affected by the cherry-pick.
3. `ruff check` and `ruff format --check` for changed files, `mypy` for changed
   production modules, and `git diff --check`.
4. Build a wheel/sdist and run the existing import smoke only if packaging
   metadata or public exports change.
5. Request independent spec and quality review after local verification.

Any command exceeding 180 seconds is recorded as a timeout and retried in
bounded test-file/node shards; it is never counted as a pass merely because
it produced no failure trace.

### Phase 4 — release-candidate handoff

Status: planned

1. Update the integration audit manifest with command, result, revision, and
   review evidence.
2. Reconfirm a clean integration worktree and the untouched original-worktree
   observation, while recording concurrent external drift truthfully if seen.
3. Present the exact candidate SHA for the separately authorized GitHub CI,
   waiver, versioning, tagging, and publication phases.  These actions are
   not authorized by this plan.

## Model, skills, and effort estimate

- Task type: narrow cross-branch integration plus public metadata contract
  hardening.
- Recommended execution: current coding agent with `gwexpy_conda_jobs`,
  `lint_check`, and review agents for independent confirmation.
- Estimated wall-clock time: 30–60 minutes locally, excluding CI queue time
  and human release decisions.
- Estimated quota: medium.  Main uncertainty is conflict resolution against
  the integration branch and optional LAL/PyCBC availability.

## Required human and release gates

The #686 human physics/numerical comment is substantive sign-off.  The
remaining governance decision is whether that comment is accepted as formal
release-gate evidence; this plan neither changes nor substitutes that policy.
Latest-GWpy-4.x CI, baseline doctest/JA-Sphinx waiver decisions, versioning,
and publication remain separate required release steps.
