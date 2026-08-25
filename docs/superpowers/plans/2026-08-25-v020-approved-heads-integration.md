# v0.2.0 Approved Heads Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the independently approved v0.2.0 implementation heads into the isolated integration branch without changing release, remote, or GitHub state.

**Architecture:** Preserve each reviewed commit series and integrate from the shared bootstrap base. Resolve only the two known overlap surfaces: GPS versus HDF5 pickle compatibility, and GPS versus GWF public reader dispatch. Validate each series immediately, then run combined and broad gates before any PR update.

**Tech Stack:** Git worktrees, Python 3.12, pytest, Ruff, MyPy, Sphinx, GWpy/Astropy/HDF5, multiprocessing spawn.

---

## Scope and fixed boundaries

- Integration worktree: `/home/washimi/.paseo/worktrees/1ee8a2ux/v020-integration`
- Required starting HEAD: `6bca889a7b719f49be4f535f360b7dd63f613283`
- Required branch: `agent/v020-integration`
- Use RTK-prefixed shell commands only.
- Set environment variables through RTK, for example:
  `rtk env PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/gwexpy-v020-integration-mpl pytest -q ...`.
- Do not install packages or mutate the shared environment.
- Do not push, create or update PRs, merge GitHub PRs, tag, bump versions, publish, close issues, or alter release state.
- Do not integrate PR #686 / median-mean without separate explicit merge authorization.
- Do not adopt SeriesMatrix B1 runtime. Preserve `adopted: false`.
- Stop rather than infer a physics or data-model decision.

The five series are integrated in the conflict-aware order shown below. Within
each series, commits must retain their original chronological order. The five
independent series are not globally reordered by wall-clock commit timestamps.

## Command and evidence policy

- Create `docs/developers/plans/manifests/audit-manifest-v020-approved-heads-integration.yaml` and record every gate with `command`, `status`, `result`, and, for `timeout` or `skipped`, a concrete `reason`.
- "Every gate" includes baseline HEAD/branch/status, approved-head ancestry, each cherry-pick range and conflict resolution, focused and broad tests, static/docs gates, final status, and the before/after original-worktree comparison.
- Commit this plan locally before the first approved range is cherry-picked. Commit the completed audit manifest locally after final validation. Both files are tracked integration evidence; neither is permitted to remain untracked at final handoff.
- Use a 180-second harness bound for one pytest or Sphinx invocation. If pytest times out, record it as `timeout`, split it by test file or node, and record every shard separately. If a per-language Sphinx build times out, record it as `timeout`, retry that language once with `-j 1` and a fresh `/tmp` output directory, and record the retry separately. A second Sphinx timeout remains `timeout`; it is not converted to `pass` by other docs tests.
- Use non-mutating verification commands. In particular, run `rtk env PYTHONPATH=$PWD ruff format --check gwexpy tests`, never an auto-format command, during validation.
- Required static commands are:
  - `rtk env PYTHONPATH=$PWD ruff check gwexpy tests`
  - `rtk env PYTHONPATH=$PWD ruff format --check gwexpy tests`
  - `rtk env PYTHONPATH=$PWD mypy gwexpy/`
  - `rtk git diff --check`
- Required broad test command is `rtk env PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/gwexpy-v020-integration-mpl pytest -q`. Apply the shard rule above if it exceeds 180 seconds.
- Required documentation commands are:
  - `rtk env PYTHONPATH=$PWD python -m sphinx -W --keep-going -b html -D language=en docs /tmp/gwexpy-v020-integration-docs-en`
  - `rtk env PYTHONPATH=$PWD python -m sphinx -W --keep-going -b html -D language=ja docs /tmp/gwexpy-v020-integration-docs-ja`
  - `rtk env PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/gwexpy-v020-integration-mpl pytest -q --doctest-modules gwexpy`
- A timed-out Sphinx language retry uses the same command with `-j 1` and output directory suffixed `-serial`.
- Before the audit commit, parse the completed integration manifest with `rtk env PYTHONPATH=$PWD python -c 'from pathlib import Path; import yaml; yaml.safe_load(Path("docs/developers/plans/manifests/audit-manifest-v020-approved-heads-integration.yaml").read_text())'` and record the result.
- A required command that cannot run must be recorded as `skipped` or `timeout` with the exact reason. It must not be omitted or described as passing.

## Approved inputs

| Series | Approved head | Range base | Review state |
|---|---|---|---|
| Exact GPS / MNE | `5f64fbd9d202e5729d529bed8f1891afbf4201ca` | `6bca889a7b719f49be4f535f360b7dd63f613283` | Luna and Sol approved |
| Provenance + Wave 3 HDF5 | `e8bbab002bd946353b7e013b67367236131b6d3c` | `6bca889a7b719f49be4f535f360b7dd63f613283` | Luna and Sol approved |
| GWF parallel I/O | `fdfadf6be98e8bf04b20bd3189d7d6d5d2be1aee` | `6bca889a7b719f49be4f535f360b7dd63f613283` | Luna and Sol approved |
| Coupling schema | `73f4bbfccf142d92e83ab2a28d74d6848a7f4837` | `6bca889a7b719f49be4f535f360b7dd63f613283` | Luna and Sol approved |
| SeriesMatrix B0 | `c7890cbc5e9759e37647a6972ca13a82e891aee8` | `6a13900672900551ccaf1b18fe78b9ce6f062e29` | Luna and Sol approved; human D21 still separate |

### Task 1: Verify the isolated baseline

**Files:**
- Inspect only: repository and worktree Git state

- [ ] Confirm the integration branch and exact starting HEAD. The expected baseline is a clean tracked/index state plus the single untracked integration-plan file; no other untracked path is permitted.
- [ ] Capture the original `/home/washimi/work/gwexpy` worktree's `git status --porcelain=v1` without modifying it. Re-run the same command after integration and require byte-for-byte identical output.
- [ ] Confirm every approved head resolves locally and each listed base is its ancestor.
- [ ] Record the ordered commit lists for every range.
- [ ] Run a small bootstrap/import baseline without changing dependencies.
- [ ] Stop and report if the baseline differs from the explicitly allowed state.
- [ ] Stage only this plan and create local commit `docs(plan): record v0.2 approved-heads integration`. Confirm the integration worktree is then fully clean before cherry-picking approved ranges.

### Task 2: Integrate Exact GPS / MNE

**Files:**
- Integrate range: `6bca889a..5f64fbd9`
- Validate: `gwexpy/timeseries/`, `gwexpy/interop/mne_.py`, pickle compatibility tests

- [ ] Cherry-pick the GPS range in original chronological order.
- [ ] Confirm there are no unresolved conflicts or unrelated changes.
- [ ] Run exact GPS, MNE interop, copy/pickle, and import-order tests.
- [ ] Run Ruff, MyPy on changed production modules, and `git diff --check`.

### Task 3: Integrate Provenance and Wave 3 HDF5

**Files:**
- Integrate range: `6bca889a..e8bbab00`
- Expected overlap: `gwexpy/io/pickle_compat.py`, `tests/io/test_pickle_compat.py`

- [ ] Cherry-pick the cumulative provenance/HDF5 range in chronological order.
- [ ] If pickle conflicts occur, preserve both exact-GPS portability and provenance-free GWpy-only portability. Do not choose one side wholesale.
- [ ] Preserve canonical sidecar paths, bounded rollback errors, deterministic process locking, append semantics, and truthful audit evidence.
- [ ] Run provenance, HDF5, Spectrogram, statistics, GWpy/pickle, and deterministic spawn-lock tests.
- [ ] Run docs contract tests, YAML parsing, Ruff, MyPy, the exact Sphinx commands in the command policy, and `git diff --check`; record pass/fail/timeout/skipped for each.

### Task 4: Integrate GWF parallel I/O

**Files:**
- Integrate range: `6bca889a..fdfadf6b`
- Expected overlap: `gwexpy/timeseries/timeseries.py`, `gwexpy/timeseries/collections.py`

- [ ] Cherry-pick the GWF range in chronological order.
- [ ] Resolve reader conflicts by preserving exact GPS epoch behavior and all four GWF reader surfaces (`TimeSeries`, `TimeSeriesDict`, `StateVector`, `StateVectorDict`).
- [ ] Preserve one-time PathLike snapshots, fail-closed local-source preflight, decoded-span merge ordering, worker exception identity, metadata independence, and daemon rejection.
- [ ] Run exact GPS plus GWF four-reader contract tests together before proceeding.
- [ ] Run adjacent/native backend tests. Record each optional backend as pass, fail, timeout, or skipped with the dependency reason; then run Ruff, MyPy, and `git diff --check`.

### Task 5: Integrate Coupling schema

**Files:**
- Integrate range: `6bca889a..73f4bbfc`
- Validate: coupling implementation, adapters, docs, import topology

- [ ] Cherry-pick the Coupling range in chronological order.
- [ ] Preserve canonical ns/Hz units, binary64 fail-closed policy, exact directional 32-ULP rule, typed empty schemas, metadata carriers, and omitted `significance`.
- [ ] Run coupling, relevant analysis, import-order, docs contract, Ruff, MyPy, the exact Sphinx commands in the command policy, and diff checks; record each result.
- [ ] Record the separate human physics/data-model gate without claiming a new formula review.

### Task 6: Integrate SeriesMatrix B0

**Files:**
- Integrate range: `6a139006..c7890cbc`
- Validate: runtime, 480-cell manifest, B0/B1 evidence

- [ ] Cherry-pick only the listed SeriesMatrix range; do not reintroduce older bootstrap state.
- [ ] Preserve the exact 480-cell ledger, all six scalar collision selectors, source immutability, and `adopted: false`.
- [ ] Confirm 478 appears only as superseded history and D21 remains proposed/pending human sign-off.
- [ ] Run manifest, operator, indexing, benchmark-contract, Ruff, MyPy, and diff checks.

### Task 7: Validate the combined integration

**Files:**
- Test the complete integration tree
- Create: `docs/developers/plans/manifests/audit-manifest-v020-approved-heads-integration.yaml`

- [ ] Confirm each listed base is an ancestor of its approved head, then record deterministic content-preservation evidence for each cherry-picked approved range. Original approved head SHAs are not required to be integration-HEAD ancestors.
- [ ] Run a focused cross-feature suite covering GPS, pickle, provenance/HDF5, GWF, Coupling, SeriesMatrix, bootstrap/import order, and docs contracts.
- [ ] Run the exact full MyPy, Ruff check, and non-mutating Ruff format-check commands from the command policy.
- [ ] Run the broad pytest suite in bounded shards if one invocation exceeds the harness limit; report every shard and timeout honestly.
- [ ] Run the exact EN/JA Sphinx and doctest commands from the command policy. Record timeout or skipped status with reasons instead of omitting unavailable gates.
- [ ] Do not change unrelated baseline formatting solely to make a repository-wide command green; report the existing `tests/docs/test_root_roadmap_contract.py` discrepancy separately.
- [ ] Review the final Git graph, conflicts/resolutions, status, and diff checks.
- [ ] Recheck `/home/washimi/work/gwexpy` status and prove it is byte-for-byte unchanged from the captured baseline.
- [ ] Complete the audit manifest with baseline, ancestry, cherry-pick/conflict, all verification, original-worktree comparison, and a `pre_audit_commit_status` record. At this phase the only permitted change is the audit manifest itself; record that exact status rather than claiming the tree is already clean.
- [ ] Run and record the required YAML parse command. Stage only the manifest and create local commit `docs(audit): record v0.2 approved-heads integration`.
- [ ] After the audit commit, run `rtk git status --porcelain=v1` and require empty output. This post-commit result is recorded in the integration agent's final report and independently rechecked by the Luna/Sol reviewers; do not amend the manifest and recreate the same self-reference.
- [ ] Confirm the final integration worktree has no staged, modified, or untracked files; the committed plan and manifest are part of the clean branch history.
- [ ] Leave a clean local integration branch with no remote mutation.

### Task 8: Independent integration review

**Files:**
- Review the final integration range and evidence; no writes

- [ ] Request Luna specification review of the combined tree.
- [ ] After Luna approval, request Sol quality/adversarial review.
- [ ] Return any Critical or Important findings to the integration implementer and repeat review.
- [ ] Do not update PR #679 until both reviews approve and human-gate scope is reconciled.
