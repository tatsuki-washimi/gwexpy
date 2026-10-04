# Release Artifact Build-Once Promotion Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development` (if subagents are available) or `superpowers:executing-plans` to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor future GWexpy releases so candidate-qualified sdist and wheel bytes are promoted unchanged to GitHub Release and PyPI.

**Architecture:** A `workflow_dispatch` candidate run builds once, selects the applicable qualification and evidence contracts from the opted-in release contract, qualifies the uploaded payload, and creates one promotion manifest that binds existing Actions artifact IDs, run metadata, gate evidence, and file hashes. A tag run validates the completed candidate, manifest, exact annotated-tag binding, release-owner GO, and candidate artifacts, then publishes by exact artifact ID without building again.

**Tech Stack:** Python standard library, GitHub Actions YAML and REST API, PyPI JSON API, pytest under the `gwexpy` conda environment.

---

## Scope and file map

This plan applies to releases after v0.2.5.

It does not alter the published v0.2.5 tag, GitHub Release, or PyPI files.

The next release contract must opt into the promotion schema and name its release tracking issue and release owner before a candidate can run.

| File | Responsibility |
|---|---|
| `scripts/ci/release_contract.py` | Load and validate the optional promotion and qualification profiles while preserving every frozen legacy release contract. |
| `scripts/ci/release_contracts.json` | Store promotion settings only when a future version and its tracking issue are selected. Do not add a speculative v0.2.6 contract. |
| `scripts/ci/qualification_evidence.py` | Resolve future qualification evidence schemas and skip-baseline rules from the opted-in release contract; retain the frozen legacy mappings. |
| `scripts/ci/diaggui_qualification_evidence.py` | Resolve DiagGUI payload and evidence schemas for promotion-enabled versions without a v0.2.4/v0.2.5-only selector. |
| `scripts/ci/v025_cross_format_io_evidence.py` | Generalize the existing v0.2.5 cross-format evidence contract so a future release profile can require the same gate suite without hard-coded version checks. |
| `scripts/ci/validate_release_review_evidence.py` | Validate the future version's configured S approval record while preserving legacy v0.2.4/v0.2.5 formats. |
| `scripts/ci/verify_release_human_approval.py` | Verify the configured S approval comment for promotion-enabled releases; keep this distinct from release GO. |
| `scripts/ci/release_promotion.py` | Canonical manifest serialization, strict manifest validation, candidate run validation, exact Actions artifact metadata validation, and tag record parsing. |
| `scripts/ci/verify_release_go.py` | Read one configured issue comment and verify the release-owner GO record and its chronology. Keep S approval validation separate. |
| `scripts/ci/validate_github_release_readback.py` | Require exact GitHub Release metadata and the five expected assets, including the promotion manifest. |
| `scripts/ci/validate_pypi_readback.py` | Require exactly the candidate sdist and wheel in normal closure and compare API metadata with downloaded file bytes. |
| `.github/workflows/publish-release.yml` | Separate candidate qualification from tag promotion, download original payload and sidecar artifacts by exact ID, and add PyPI readback. |
| `RELEASING.md` | Document candidate dispatch, GO record, tag message, artifact retention, promotion, readback, and partial upload recovery. |
| `tests/test_release_contracts.py` | Preserve current release contract compatibility and validate the future promotion block. |
| `tests/test_qualification_skip_evidence.py` | Keep frozen historical qualification behavior and exercise a synthetic future qualification contract. |
| `tests/test_diaggui_qualification_evidence.py` | Preserve v0.2.4/v0.2.5 schemas and test configured future-version selection. |
| `tests/test_v025_cross_format_io_evidence.py` | Preserve v0.2.5 evidence and test version-neutral future profile behavior. |
| `tests/test_release_human_approval.py` | Keep existing approval records valid and test configured future S approval separately from GO. |
| `tests/test_release_review_evidence.py` | Keep frozen review-evidence formats valid and test the configured future S-approval evidence contract. |
| `tests/test_release_promotion.py` | Test manifest, run, tag, and Actions artifact identity rules. |
| `tests/test_release_go.py` | Test canonical GO comments, configured author/issue, and timestamp ordering. |
| `tests/test_pypi_readback.py` | Test exact PyPI file-set and hash validation. |
| `tests/test_publish_release_workflow.py` | Test the dispatch/tag job graphs, least privilege, exact artifact IDs, five GitHub assets, and PyPI readback dependency. |
| `tests/test_release_gate_workflow_contract.py` | Assert future promotion contracts select all configured qualification jobs for candidate runs and that tag runs do not build. |
| `tests/fixtures/release/promotion-contract-v1.json` | Supply a synthetic future contract for unit tests without authorizing a real future version. |
| `tests/fixtures/release/promotion-manifest-v1.json` | Supply a strict manifest fixture with separate Actions artifact and package digests. |

Use no new runtime dependencies.

Run commands from the root of this worktree with `conda run -n gwexpy pytest ...` and `conda run -n gwexpy ruff check ...`. Capture the command's actual exit status before marking a step complete or committing. Do not use a helper that derives its repository root from its own location: the installed wrapper lives outside this worktree and would test a different checkout. If a command must run detached, start it in a task-specific `tmux` session with an explicit `cd` to this worktree, then collect its completed log and exit status before proceeding.

## Task 1: Extend release contracts without changing frozen entries

**Files:**
- Modify: `scripts/ci/release_contract.py`
- Modify: `tests/test_release_contracts.py`
- Create: `tests/fixtures/release/promotion-contract-v1.json`

- [ ] **Step 1: Add failing tests for the promotion contract shape**

Add tests for a valid promotion contract block containing the schema identifier, canonical workflow path, release GO issue number, release-owner login, S-approval schema and approver, artifact name patterns, a qualification profile, evidence schema identifiers, and version-specific required job names.

Add rejection tests for missing fields, invalid issue identifiers, invalid owner login, duplicate or unsorted job names, and a noncanonical workflow path.

Assert the seven currently configured release contracts still load with their existing values and no promotion block.

- [ ] **Step 2: Run the contract tests and confirm the new cases fail**

Run: `conda run -n gwexpy pytest tests/test_release_contracts.py -q`

Expected: new promotion-block tests fail because the loader rejects the new block; all existing frozen-contract tests continue to pass.

- [ ] **Step 3: Implement the optional promotion contract validator**

Extend `release_contract.py` to accept either the existing legacy shape or that shape with one strict `promotion` object.

The promotion object must require the schema, `.github/workflows/publish-release.yml`, positive issue number, safe GO owner login, a separate configured S-approval format and approver, artifact naming rules, a named qualification profile, evidence schema identifiers, and a sorted unique list of required jobs. Validate that the profile maps to the complete applicable gate/evidence set; contracts may not silently omit an applicable qualification gate.

Keep all current version-specific contract records and behaviors unchanged. The synthetic future fixture must demonstrate that a version later than v0.2.5 can opt into promotion without adding a real release contract.

Keep all current entries byte-for-byte unchanged in `release_contracts.json`.

Do not add an unselected future version or placeholder issue number.

- [ ] **Step 4: Rerun the contract tests**

Run: `conda run -n gwexpy pytest tests/test_release_contracts.py -q`

Expected: legacy and promotion fixture cases pass; malformed promotion blocks fail closed.

- [ ] **Step 5: Commit the contract change**

```bash
git add scripts/ci/release_contract.py tests/test_release_contracts.py tests/fixtures/release/promotion-contract-v1.json
git commit -m "feat(release): define promotion contract schema"
```

## Task 2: Add strict promotion manifest and artifact identity validation

**Files:**
- Create: `scripts/ci/release_promotion.py`
- Create: `tests/test_release_promotion.py`
- Create: `tests/fixtures/release/promotion-manifest-v1.json`
- Modify: `tests/test_release_contracts.py`

- [ ] **Step 1: Write failing tests for canonical manifest bytes**

Test sorted UTF-8 JSON with one trailing LF, rejection of duplicate keys, missing or unknown fields, invalid full SHAs, invalid SHA-256 values, nonpositive artifact IDs, unsafe filenames, and inconsistent candidate run IDs.

Test that every payload, sidecar, and required evidence artifact records `artifact_id`, `name`, GitHub artifact `digest`, `size_in_bytes`, and `run_id`, separately from each contained file's safe name, byte size, and SHA-256. Include the workflow ID/path, event, dispatch ref, workflow ref/SHA, S and R SHAs, promotion contract identifier/hash, review-evidence path/hash, release-note path/hash, and exact gate/evidence set in the fixture and schema assertions.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `conda run -n gwexpy pytest tests/test_release_promotion.py -q`

Expected: collection or assertions fail because the promotion validator does not exist.

- [ ] **Step 3: Implement manifest build and validation functions**

Implement pure functions in `release_promotion.py` to serialize the manifest, load it with duplicate-key rejection, validate the exact schema, and compare it with the release contract and expected repository, version, tag, S, R, and run metadata.

Represent each existing Actions artifact with `artifact_id`, `name`, GitHub artifact digest, `size_in_bytes`, and `run_id`.

Represent the contents independently with safe filename, file size, and SHA-256.

The manifest-job validator checks `run_attempt == 1`, every contract-required gate is completed with `success`, all existing artifacts have valid same-run metadata, and the payload/sidecar/evidence contents match their file hashes. It must not require the overall workflow run to be complete: the manifest job runs before its own workflow run can complete.

Reject expired artifacts, wrong run IDs, wrong names, wrong IDs, wrong GitHub digests, wrong artifact sizes, missing or extra required gate evidence, and mismatched package hashes. The manifest builder reads and rehashes existing payload, sidecar, and evidence artifacts only; its sole new upload is the promotion manifest. Record existing artifact IDs as publication authority.

The script must not upload or rewrite payload, sidecar, or evidence artifacts.

- [ ] **Step 4: Add failing and passing artifact metadata cases**

Exercise each metadata mismatch independently so one invalid field cannot be hidden by another mismatch.

Verify the promotion manifest artifact is selected by its fixed `release-promotion-manifest-<R SHA>` name within the candidate run, and its API metadata and raw-byte SHA-256 are checked. Do not place the promotion manifest artifact's own ID or digest inside the manifest; that would be self-referential.

- [ ] **Step 5: Rerun focused tests and lint the new module**

Run: `conda run -n gwexpy pytest tests/test_release_promotion.py tests/test_release_contracts.py -q`

Run: `conda run -n gwexpy ruff check scripts/ci/release_promotion.py tests/test_release_promotion.py tests/test_release_contracts.py`

Expected: all focused tests pass and Ruff reports no findings.

- [ ] **Step 6: Commit the manifest validator**

```bash
git add scripts/ci/release_promotion.py tests/test_release_promotion.py tests/fixtures/release/promotion-manifest-v1.json tests/test_release_contracts.py
git commit -m "feat(release): validate promotion manifests and artifacts"
```

## Task 3: Verify canonical release GO and tag binding

**Files:**
- Modify: `scripts/ci/release_promotion.py`
- Create: `scripts/ci/verify_release_go.py`
- Create: `tests/test_release_go.py`
- Modify: `tests/test_release_promotion.py`

- [ ] **Step 1: Add failing GO and tag parser tests**

Test the exact `GWEXPY-RELEASE-GO-v1` field order and values, configured tracking issue, configured release-owner login, immutable comment timestamp, candidate completion, and promotion manifest artifact creation time. Explicitly reject a GO comment made after all gate jobs succeed but before the manifest artifact is uploaded, and a GO comment made after manifest upload but before the candidate run completes.

Test the exact `GWEXPY-PROMOTION-v1` tag body, annotated-tag requirement, tag name, repository, R SHA, candidate run ID, manifest SHA, and GO comment ID.

Reject extra lines, duplicate or unknown fields, lightweight tags, timestamps before candidate completion, timestamps before manifest upload, a candidate with a non-success conclusion, and a tag-time candidate run that is not `status=completed` and `conclusion=success`.

- [ ] **Step 2: Confirm the new tests fail**

Run: `conda run -n gwexpy pytest tests/test_release_go.py tests/test_release_promotion.py -q`

Expected: the new parser and comment-verification tests fail before their implementations exist.

- [ ] **Step 3: Implement local GO and tag validators**

Add pure parsers to `release_promotion.py` for the fixed tag record and GO body.

Implement `verify_release_go.py` with an injectable HTTP/API boundary so tests do not access GitHub.

Verify the comment ID resolves to the release contract's issue, the author is the configured release owner, `updated_at == created_at`, the body exactly matches the canonical GO record, and `created_at` is later than both candidate `completed_at` and promotion manifest artifact `created_at`. The tag-time validator additionally requires the candidate run API record to be `status=completed`, `conclusion=success`, and `run_attempt=1`; the manifest-job validator uses only its available gate and artifact completion facts.

Keep `verify_release_human_approval.py` responsible only for S approval.

- [ ] **Step 4: Rerun the GO and tag tests**

Run: `conda run -n gwexpy pytest tests/test_release_go.py tests/test_release_promotion.py -q`

Expected: valid records pass; all malformed, wrong-identity, and early-timestamp cases fail closed.

- [ ] **Step 5: Commit the authorization validators**

```bash
git add scripts/ci/release_promotion.py scripts/ci/verify_release_go.py tests/test_release_go.py tests/test_release_promotion.py
git commit -m "feat(release): bind GO to candidate promotion"
```

## Task 4: Make candidate run build once and emit only a promotion manifest

**Files:**
- Modify: `.github/workflows/publish-release.yml`
- Modify: `scripts/ci/release_promotion.py`
- Modify: `scripts/ci/qualification_evidence.py`
- Modify: `scripts/ci/diaggui_qualification_evidence.py`
- Modify: `scripts/ci/v025_cross_format_io_evidence.py`
- Modify: `scripts/ci/validate_release_review_evidence.py`
- Modify: `scripts/ci/verify_release_human_approval.py`
- Modify: `tests/test_publish_release_workflow.py`
- Modify: `tests/test_release_gate_workflow_contract.py`
- Modify: `tests/test_release_human_approval.py`
- Modify: `tests/test_release_review_evidence.py`
- Modify: `tests/test_release_promotion.py`
- Modify: `tests/test_qualification_skip_evidence.py`
- Modify: `tests/test_diaggui_qualification_evidence.py`
- Modify: `tests/test_v025_cross_format_io_evidence.py`

- [ ] **Step 1: Add a failing candidate workflow contract**

Add a synthetic future release contract and assert the resolved qualification profile selects the complete applicable gate set for that version. Test that the profile drives `qualify`, the qualification evidence aggregate, DiagGUI, cross-format I/O, and every other configured gate/evidence contract without hard-coded `0.2.5` equality checks. Existing historical version paths must retain their current gate sets and evidence formats.

Assert the existing `verify` job validates configured S approval for a promotion-enabled future contract before `build` starts. A missing or failed S approval must fail `verify`, and the build job's dependency/condition must keep package construction skipped. Keep the current legacy approval paths unchanged.

Add a workflow test in `tests/test_publish_release_workflow.py` that checks the future-contract S-approval validation is in `verify`, `build` requires successful `verify`, and a failed/missing approval therefore prevents build. Add a synthetic missing/invalid-approval case to the approval validator tests.

Assert the build job and every candidate qualification job run only for `workflow_dispatch`; assert tag runs contain no candidate build/qualification job. Put candidate-versus-tag graph assertions in both `tests/test_publish_release_workflow.py` and `tests/test_release_gate_workflow_contract.py` as required by the design spec.

Assert all qualification lanes consume the same build job's uploaded payload and sidecars.

Assert the final manifest job depends on every profile-applicable gate and has only `contents: read` and `actions: read` permissions. Assert it runs only after every configured gate succeeds.

Assert the finalizer uploads `release-promotion-manifest-<R SHA>` once and never uploads package, sidecar, or evidence files.

- [ ] **Step 2: Run the workflow contract test and confirm it fails**

Run: `conda run -n gwexpy pytest tests/test_publish_release_workflow.py -q`

Expected: the dispatch/tag job graph assertion fails against the current workflow, which builds on tag pushes.

- [ ] **Step 3: Add the candidate manifest job**

- **3a. Resolve qualification profiles and evidence schemas.** Make `qualification_evidence.py` resolve payload/evidence schema and expected-skip baseline metadata from the release contract; make `diaggui_qualification_evidence.py` consume configured version-specific schemas while keeping its test node lists and gate semantics stable; generalize the v0.2.5 cross-format evidence module so its schema/version comes from the selected profile rather than a `VERSION == "0.2.5"` restriction. Add synthetic future-version tests for each selector. Keep v0.2.5-only historical 74-case evidence tied to its explicit historical contract and do not silently impose that frozen baseline on later versions. The profile enumerates required jobs and their evidence artifact contracts; unknown jobs, missing schemas, or skipped required lanes fail closed.

- **3b. Gate build on source approval.** Extend the `verify` job to invoke the configured S review-evidence and human-approval validators for promotion-enabled contracts before the `build` job. Require build to depend on successful `verify`; a missing, malformed, wrong-author, or failed S approval must fail `verify` and prevent package construction. Add contract/workflow tests for this dependency and synthetic future approval tests in `tests/test_release_human_approval.py` and `tests/test_release_review_evidence.py`.

- **3c. Preserve source approval through qualification.** Keep the candidate `evidence` gate's source-review validation and bind its evidence artifact, review-document path/hash, and approval identity to the manifest. This must not replace the pre-build `verify` check. Release GO remains a later, separate decision.

- **3d. Build once and finalize read-only evidence.** Keep one build job for `workflow_dispatch` only. After all profile-required gates succeed, list artifacts for the current run through the Actions API, select each expected existing artifact once, download it read-only by artifact ID, and pass its REST metadata and contents to `release_promotion.py`. Require every manifest-bound aggregate evidence artifact to have 90-day retention. Upload exactly one new Actions artifact from this job: the promotion manifest with the fixed name. Do not set overwrite behavior or upload any payload, sidecar, or evidence under a second artifact name.

- [ ] **Step 4: Run the candidate-specific contract tests**

Run: `conda run -n gwexpy pytest tests/test_publish_release_workflow.py tests/test_release_gate_workflow_contract.py tests/test_release_human_approval.py tests/test_release_review_evidence.py tests/test_qualification_skip_evidence.py tests/test_diaggui_qualification_evidence.py tests/test_v025_cross_format_io_evidence.py tests/test_release_promotion.py -q`

Expected: candidate dispatch builds once, all gates consume that payload, and the manifest is the only finalizer upload.

- [ ] **Step 5: Commit the candidate workflow change**

```bash
git add .github/workflows/publish-release.yml scripts/ci/release_promotion.py scripts/ci/qualification_evidence.py scripts/ci/diaggui_qualification_evidence.py scripts/ci/v025_cross_format_io_evidence.py scripts/ci/validate_release_review_evidence.py scripts/ci/verify_release_human_approval.py tests/test_publish_release_workflow.py tests/test_release_gate_workflow_contract.py tests/test_release_human_approval.py tests/test_release_review_evidence.py tests/test_release_promotion.py tests/test_qualification_skip_evidence.py tests/test_diaggui_qualification_evidence.py tests/test_v025_cross_format_io_evidence.py
git commit -m "feat(release): emit manifest after candidate qualification"
```

## Task 5: Promote exact candidate artifact IDs from tag runs

**Files:**
- Modify: `.github/workflows/publish-release.yml`
- Modify: `scripts/ci/release_promotion.py`
- Modify: `scripts/ci/verify_release_go.py`
- Modify: `scripts/ci/validate_github_release_readback.py`
- Modify: `tests/test_publish_release_workflow.py`
- Modify: `tests/test_release_promotion.py`

- [ ] **Step 1: Add failing tag-run and Release asset tests**

Assert the tag graph has no build or candidate qualification jobs and creates a Release only after promotion verification.

Assert candidate run `workflow_id`, `path`, event, ref, workflow SHA, source SHA, version, completion state, conclusion, and attempt match the contract and tag.

Assert promotion-enabled future candidates validate the configured S approval before manifest creation, and tag-time verification rechecks the exact S-approval evidence/comment bound by the manifest. S approval remains separate from release GO.

Assert payload, sidecar, and evidence downloads specify manifest artifact IDs and compare REST metadata fields `run_id`, `name`, `expired`, `digest`, and `size_in_bytes`.

Assert Release readback requires exactly the five named assets, verifies their bytes and SHA-256, and rejects every extra or duplicate asset.

- [ ] **Step 2: Confirm the tag and asset tests fail**

Run: `conda run -n gwexpy pytest tests/test_publish_release_workflow.py tests/test_release_promotion.py -q`

Expected: tests fail because the current tag path rebuilds and the readback validator accepts four assets.

- [ ] **Step 3: Implement tag promotion verification**

Add a tag-only `promotion_verify` job that parses the annotated tag record, validates candidate run and manifest, verifies the exact GO comment, and validates artifact metadata before any Release creation.

At tag time, verify that the manifest binds a successful candidate `evidence` gate and the exact configured S-approval evidence/comment for R before publication. S approval and release GO remain separate checks; neither implies the other.

Use `actions: read` for cross-run artifact metadata and download, `issues: read` for the GO comment, and `contents: read` for source/tag inspection.

Use the manifest's exact payload and sidecar artifact IDs as download selectors in the verification, GitHub Release, and PyPI jobs; use exact evidence artifact IDs in verification. Select the promotion manifest artifact by candidate run ID plus its fixed name, then validate its API metadata and raw-byte SHA-256.

Each publisher job must independently download the same exact candidate artifact IDs and recheck member file hashes because runners do not share a filesystem.

- [ ] **Step 4: Remove build and qualification from the tag publication path**

Set candidate build and qualification jobs to dispatch-only.

Make `github_release` depend on source/tag validation and `promotion_verify`, not skipped candidate jobs.

Make `publish` depend on successful GitHub Release creation and readback plus promotion verification.

Keep GitHub Release `contents: write` separate from PyPI `id-token: write`.

Pin every external action to a full commit SHA.

- [ ] **Step 5: Expand GitHub Release readback to five exact assets**

Update `validate_github_release_readback.py` to validate sdist, wheel, `distribution-sha256.json`, `LICENSE.sha256`, and promotion manifest as the complete asset set.

For an existing Release, continue only when target, notes, the exact five asset names, and every asset byte match the candidate; reject unknown or extra assets. Add tests for exact existing Release continuation, conflicting target/body/asset rejection, and duplicate/extra assets.

- [ ] **Step 6: Rerun focused tests and Ruff**

Run: `conda run -n gwexpy pytest tests/test_publish_release_workflow.py tests/test_release_promotion.py -q`

Run: `conda run -n gwexpy ruff check scripts/ci/release_promotion.py scripts/ci/verify_release_go.py scripts/ci/validate_github_release_readback.py tests/test_publish_release_workflow.py tests/test_release_promotion.py`

Expected: the tag graph contains no build, only exact artifact IDs can reach publishers, five asset readback passes, and all other artifact sets fail.

- [ ] **Step 7: Commit tag promotion and Release readback**

```bash
git add .github/workflows/publish-release.yml scripts/ci/release_promotion.py scripts/ci/verify_release_go.py scripts/ci/validate_github_release_readback.py tests/test_publish_release_workflow.py tests/test_release_promotion.py
git commit -m "feat(release): promote exact candidate artifacts on tag"
```

## Task 6: Add exact PyPI readback and preserve partial-upload recovery

**Files:**
- Create: `scripts/ci/validate_pypi_readback.py`
- Create: `tests/test_pypi_readback.py`
- Modify: `.github/workflows/publish-release.yml`
- Modify: `tests/test_publish_release_workflow.py`
- Modify: `RELEASING.md`

- [ ] **Step 1: Add failing PyPI response tests**

Test a valid PyPI response with exactly the manifest wheel and sdist, correct advertised hashes, and matching downloaded bytes.

Reject missing files, extra files, duplicate filenames, wrong hashes, wrong version, invalid URLs, and downloaded bytes that do not match the JSON digest.

- [ ] **Step 2: Run PyPI tests and confirm they fail**

Run: `conda run -n gwexpy pytest tests/test_pypi_readback.py -q`

Expected: tests fail because the PyPI readback validator does not exist.

- [ ] **Step 3: Implement a testable PyPI readback validator**

Implement a pure validator for JSON metadata and local downloaded files.

Keep HTTP requests in a small injectable workflow-facing layer so tests remain network-free.

Require the file set to be exactly the sdist and wheel for normal closure. Test retryable transport/404/5xx outcomes separately from nonretryable version, filename, and digest mismatches; only the former can use the bounded retry window.

- [ ] **Step 4: Add post-publish workflow readback**

Add a read-only `pypi_readback` job after `publish`.

For transient 404, 5xx, or network errors, retry for at most 15 minutes.

Fail immediately on filename, version, or hash mismatch; fail closure if the retry window expires.

Download the PyPI files and compare actual bytes with the promotion manifest and GitHub Release assets.

- [ ] **Step 5: Update partial upload recovery text and tests**

Update the `RELEASING.md` partial-upload section to preserve HOLD, same-candidate artifact identity, exact PyPI file/hash comparison, explicit release-owner review, and missing-file-only completion.

Document that the normal successful closure requires exactly two PyPI files even when an approved partial recovery was needed.

Add workflow and validator tests proving an extra PyPI file or a missing final file prevents closure.

- [ ] **Step 6: Run focused PyPI and workflow tests**

Run: `conda run -n gwexpy pytest tests/test_pypi_readback.py tests/test_publish_release_workflow.py -q`

Run: `conda run -n gwexpy ruff check scripts/ci/validate_pypi_readback.py tests/test_pypi_readback.py tests/test_publish_release_workflow.py`

Expected: the readback graph follows PyPI upload; transient lookup failures are bounded; the exact two-file closure passes.

- [ ] **Step 7: Commit PyPI readback and recovery documentation**

```bash
git add scripts/ci/validate_pypi_readback.py tests/test_pypi_readback.py .github/workflows/publish-release.yml tests/test_publish_release_workflow.py RELEASING.md
git commit -m "feat(release): verify PyPI promotion readback"
```

## Task 7: Complete release documentation and regression coverage

**Files:**
- Modify: `RELEASING.md`
- Modify: `tests/test_publish_release_workflow.py`
- Modify: `tests/test_release_gate_workflow_contract.py`
- Modify: `tests/test_release_contracts.py`
- Modify: `tests/test_release_human_approval.py`
- Modify: `tests/test_release_review_evidence.py`
- Modify: `tests/test_qualification_skip_evidence.py`
- Modify: `tests/test_diaggui_qualification_evidence.py`
- Modify: `tests/test_v025_cross_format_io_evidence.py`
- Modify: `tests/test_release_promotion.py`
- Modify: `tests/test_release_go.py`
- Modify: `tests/test_pypi_readback.py`

- [ ] **Step 1: Document the future release procedure**

Document how to run a candidate from `main`, wait for completed successful run and manifest upload, record the exact GO comment, construct the strict annotated tag body, and inspect the resulting GitHub Release and PyPI readback.

State that v0.2.5 remains immutable and this workflow applies only after a release contract opts into the promotion schema.

- [ ] **Step 2: Add workflow failure-injection contract cases**

Require rejection for GO after gate success but before manifest upload, GO after manifest upload but before overall candidate completion, wrong workflow ID/path, run attempt other than one, expired or wrong artifact ID, cross-run artifact, wrong GitHub artifact digest/size, missing gate, and failed or skipped gate. Also assert that every synthetic future-profile lane is required and that a promotion-enabled future contract cannot bypass S approval.

Require failure before GitHub Release creation for every prepublication mismatch.

- [ ] **Step 3: Add exact set and privilege regression cases**

Require exact five GitHub Release assets and exact two PyPI files for normal closure. Verify that an existing Release with the exact target, notes, assets, and bytes is idempotently accepted, while any conflict or extra asset fails. Verify retryable PyPI lookup/network failures are bounded and nonretryable identity/hash mismatches fail immediately.

Assert verification and publisher jobs have only the permissions required for `contents: read`, `actions: read`, `issues: read` where applicable, `contents: write` for GitHub Release, and `id-token: write` for PyPI.

Assert PyPI readback uses no publishing credential and runs after the publish job.

- [ ] **Step 4: Run the full release-control test slice**

Run: `conda run -n gwexpy pytest tests/test_release_contracts.py tests/test_release_promotion.py tests/test_release_go.py tests/test_release_human_approval.py tests/test_release_review_evidence.py tests/test_qualification_skip_evidence.py tests/test_diaggui_qualification_evidence.py tests/test_v025_cross_format_io_evidence.py tests/test_pypi_readback.py tests/test_publish_release_workflow.py tests/test_release_gate_workflow_contract.py -q`

Expected: all release-control contract tests pass; no test contacts GitHub or PyPI.

- [ ] **Step 5: Run Ruff on changed Python files and check the final diff**

Run: `conda run -n gwexpy ruff check scripts/ci/release_contract.py scripts/ci/qualification_evidence.py scripts/ci/diaggui_qualification_evidence.py scripts/ci/v025_cross_format_io_evidence.py scripts/ci/validate_release_review_evidence.py scripts/ci/verify_release_human_approval.py scripts/ci/release_promotion.py scripts/ci/verify_release_go.py scripts/ci/validate_github_release_readback.py scripts/ci/validate_pypi_readback.py tests/test_release_contracts.py tests/test_release_human_approval.py tests/test_release_review_evidence.py tests/test_qualification_skip_evidence.py tests/test_diaggui_qualification_evidence.py tests/test_v025_cross_format_io_evidence.py tests/test_release_promotion.py tests/test_release_go.py tests/test_pypi_readback.py tests/test_publish_release_workflow.py tests/test_release_gate_workflow_contract.py`

Run: `git diff --check`

Expected: Ruff and whitespace checks pass; `git status --short` contains only the planned files.

- [ ] **Step 6: Commit final documentation and regression contracts**

```bash
git add RELEASING.md tests/test_publish_release_workflow.py tests/test_release_gate_workflow_contract.py tests/test_release_contracts.py tests/test_release_human_approval.py tests/test_release_review_evidence.py tests/test_qualification_skip_evidence.py tests/test_diaggui_qualification_evidence.py tests/test_v025_cross_format_io_evidence.py tests/test_release_promotion.py tests/test_release_go.py tests/test_pypi_readback.py
git commit -m "docs(release): document candidate artifact promotion"
```

## Release safety conditions

Do not create or move a release tag, dispatch a candidate run, publish to GitHub or PyPI, upload a GitHub issue comment, alter Trusted Publisher settings, or modify v0.2.5 assets while executing this code plan.

The plan implements workflow behavior and tests only.

The next real release requires a new S review cycle, a release contract entry with real issue/owner values, candidate qualification, an exact GO, and the normal publication authorization.
